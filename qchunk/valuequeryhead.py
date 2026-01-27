"""Value Query Head that reuses the SmolVLA text decoder to score action chunks."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence
import time
import sys

import torch
from torch import nn
from transformers import AutoModelForImageTextToText
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.smolvla.modeling import make_att_2d_masks
import numpy as np


def _build_attention_mask(bool_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert a boolean attention mask to the additive format Llama expects."""

    if bool_mask.ndim != 3:
        raise ValueError(f"Expected (batch, seq, seq) mask, got {bool_mask.shape}")
    float_mask = torch.zeros(
        bool_mask.shape[0],
        1,
        bool_mask.shape[1],
        bool_mask.shape[2],
        dtype=dtype,
        device=bool_mask.device,
    )
    neg_value = torch.finfo(dtype).min
    float_mask.masked_fill_(~bool_mask.unsqueeze(1), neg_value)
    return float_mask


@dataclass
class ValueQueryHeadConfig:
    """Minimal configuration for the Value Query Head."""

    chunk_size: int
    action_dim: int
    num_backbone_layers: int = 2
    critic_hidden_dims: Sequence[int] = (512, 512)
    vlm_model_name: str | None = None
    head_type: str = "mlp"
    head_num_layers: int = 2
    head_mlp_dims: Sequence[int] = (512, 512)
    att_mode: str = "causal"


class Q_Former_Backbone(nn.Module):
    """Thin stack of decoder layers initialised from a SmolVLM text model."""

    def __init__(
        self,
        num_layers: int,
        model_id: str | None = None,
        text_config: LlamaConfig | None = None,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        if model_id is not None:
            base_model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )
            text_model = base_model.model.text_model
            total_layers = len(text_model.layers)
            if num_layers > total_layers:
                raise ValueError(f"Requested {num_layers} layers but model only has {total_layers}.")
            self.layers = nn.ModuleList([copy.deepcopy(text_model.layers[idx]) for idx in range(num_layers)])
            self.norm = copy.deepcopy(text_model.norm)
            self.text_config = text_model.config
            del base_model
        else:
            if text_config is None:
                raise ValueError("Either `model_id` or `text_config` must be provided.")
            self.text_config = text_config
            self.layers = nn.ModuleList([LlamaDecoderLayer(text_config, layer_idx=i) for i in range(num_layers)])
            self.norm = LlamaRMSNorm(text_config.hidden_size, text_config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(self.text_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        output = hidden_states
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            output = layer(
                output,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                past_key_values=None,
                cache_position=None,
                output_attentions=False,
                use_cache=False,
            )
        return self.norm(output)


class ConcatMLPHead(nn.Module):
    """Simple MLP critic that consumes query embeddings and flattened action chunks."""

    def __init__(self, query_dim: int, action_dim: int, hidden_dims: Iterable[int] = (512, 512)):
        super().__init__()
        input_dim = query_dim + action_dim
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, query_emb: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim < 2:
            raise ValueError("Expected at least 2D action tensor.")
        flat_actions = actions.view(actions.shape[0], -1)
        critic_input = torch.cat([query_emb, flat_actions], dim=-1)
        return self.net(critic_input)


class ActionTokenizer(nn.Module):
    """Project each action step into the model hidden space for transformer heads."""

    def __init__(self, action_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(action_dim, hidden_dim)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim != 3:
            raise ValueError("Expected (batch, chunk, action_dim) action tensor.")
        return self.proj(actions)


class TransformerCriticHead(nn.Module):
    """Transformer-based head that produces a value embedding before scoring."""

    def __init__(
        self,
        *,
        hidden_dim: Optional[int],
        action_dim: int,
        num_layers: int = 2,
        mlp_hidden_dims: Sequence[int] = (512, 512),
        model_id: str | None = None,
        text_config: LlamaConfig | None = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        att_mode: str = "causal",  # causal / bi-level
        bias_init_enabled: bool = False,
        bias_init_value: float = 0.0,
    ) -> None:
        super().__init__()
        self.bias_init_enabled = bias_init_enabled
        self.bias_init_value = bias_init_value
        self.decoder = Q_Former_Backbone(
            num_layers=num_layers,
            model_id=model_id,
            text_config=text_config,
            torch_dtype=torch_dtype,
        )
        decoder_hidden = self.decoder.text_config.hidden_size
        if hidden_dim is not None and decoder_hidden != hidden_dim:
            raise ValueError("hidden_dim must match text hidden size")
        self.hidden_dim = decoder_hidden
        self.action_tokens = ActionTokenizer(action_dim, self.hidden_dim)
        self.query_token = nn.Parameter(torch.zeros(self.hidden_dim))
        self.value_token = nn.Parameter(torch.zeros(self.hidden_dim))
        layers: list[nn.Module] = []
        prev_dim = self.hidden_dim
        total_layers = len(mlp_hidden_dims)
        for idx, hidden in enumerate(mlp_hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden))
            if idx + 1 < total_layers:
                layers.append(nn.SiLU())
                layers.append(nn.LayerNorm(hidden))
            prev_dim = hidden
        layers.append(nn.Linear(prev_dim, 1))
        self.value_head = nn.Sequential(*layers)
        self.att_mode = att_mode
        self.prefix_proj: nn.Linear | None = None
    def forward(
        self,
        *,
        prefix_embs: Optional[torch.Tensor],
        pad_masks: Optional[torch.Tensor],
        att_masks: Optional[torch.Tensor],
        query_emb: Optional[torch.Tensor],
        actions: torch.Tensor,
        actions_is_pad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = actions.shape[0]
        device = actions.device
        dtype = self.value_token.dtype

        token_segments = []
        pad_segments = []
        att_segments = []

        if prefix_embs is not None and pad_masks is not None and att_masks is not None:
            token_segments.append(prefix_embs)
            pad_segments.append(pad_masks)
            att_segments.append(att_masks)

        if query_emb is not None:
            token_segments.append(query_emb.unsqueeze(1))
            pad_segments.append(torch.ones(batch_size, 1, dtype=torch.bool, device=device))
            att_segments.append(torch.zeros(batch_size, 1, dtype=torch.bool, device=device))

        action_tokens = self.action_tokens(actions)
        token_segments.append(action_tokens)
        if actions_is_pad is None:
            actions_is_pad = torch.zeros(batch_size, action_tokens.shape[1], dtype=torch.bool, device=device)
        pad_segments.append(~actions_is_pad.bool())
        if self.att_mode == "bi-level":
            att_segments.append(torch.zeros(batch_size, action_tokens.shape[1], dtype=torch.bool, device=device))
        else: att_segments.append(torch.ones(batch_size, action_tokens.shape[1], dtype=torch.bool, device=device))
        
        value_tokens = self.value_token.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, 1, -1)
        token_segments.append(value_tokens)
        pad_segments.append(torch.ones(batch_size, 1, dtype=torch.bool, device=device))
        att_segments.append(torch.ones(batch_size, 1, dtype=torch.bool, device=device)) # 可改

        tokens = torch.cat(token_segments, dim=1).to(dtype)
        pad_mask = torch.cat(pad_segments, dim=1)
        att_mask = torch.cat(att_segments, dim=1)

        att_2d_masks = make_att_2d_masks(pad_mask, att_mask)
        attention_mask = _build_attention_mask(att_2d_masks, dtype=tokens.dtype)
        position_ids = torch.cumsum(pad_mask.long(), dim=1) - 1
        position_ids = position_ids.clamp_min(0)

        outputs = self.decoder(tokens, attention_mask=attention_mask, position_ids=position_ids)
        value_indices = pad_mask.sum(dim=1).long() - 1
        batch_indices = torch.arange(batch_size, device=device)
        value_emb = outputs[batch_indices, value_indices]
        return self.value_head(value_emb)


class ValueQueryHead(nn.Module):
    """Project SmolVLA embeddings through a small decoder and score action chunks."""

    def __init__(self, config: ValueQueryHeadConfig, text_config: LlamaConfig | None = None):
        super().__init__()
        self.config = config
        self.head_type = config.head_type if hasattr(config, "head_type") else "mlp"
        backbone = Q_Former_Backbone(
            num_layers=config.num_backbone_layers,
            model_id=config.vlm_model_name,
            text_config=text_config,
            torch_dtype=torch.bfloat16,
        )
        self.backbone = backbone
        hidden_size = backbone.text_config.hidden_size
        self.query_embedding = nn.Parameter(torch.zeros(hidden_size))
        critic_input_dim = config.chunk_size * config.action_dim
        if self.head_type == "transformer":
            self.critic_head = TransformerCriticHead(
                hidden_dim=hidden_size,
                action_dim=config.action_dim,
                num_layers=config.head_num_layers,
                mlp_hidden_dims=config.head_mlp_dims,
                model_id=config.vlm_model_name,
                text_config=text_config,
                att_mode=getattr(config, "att_mode", "causal"),
            )
        else:
            self.critic_head = ConcatMLPHead(
                query_dim=hidden_size,
                action_dim=critic_input_dim,
                hidden_dims=config.critic_hidden_dims,
            )

    def encode_from_embeddings(
        self,
        prefix_embs: torch.Tensor,
        pad_masks: torch.Tensor,
        att_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Return query embeddings built from prefix representations."""

        augmented_embs, aug_pad_masks, aug_att_masks = self._append_query_token_my(prefix_embs, pad_masks, att_masks) ##直接插入
        att_2d_masks = make_att_2d_masks(aug_pad_masks, aug_att_masks)
        
        attention_mask = _build_attention_mask(att_2d_masks, dtype=augmented_embs.dtype)
        position_ids = torch.cumsum(aug_pad_masks.long(), dim=1) - 1
        position_ids = position_ids.clamp_min(0)

        target_dtype = next(self.backbone.parameters()).dtype
        augmented_embs = augmented_embs.to(target_dtype)
        attention_mask = attention_mask.to(target_dtype)
        outputs = self.backbone(augmented_embs, attention_mask=attention_mask, position_ids=position_ids)
        # query_indices = aug_pad_masks.sum(dim=1).long() - 1
        query_idx = aug_pad_masks.shape[1] - 1
        batch_indices = torch.arange(outputs.shape[0], device=outputs.device)
        return outputs[batch_indices, torch.full_like(batch_indices, query_idx)]
        # return outputs[batch_indices, query_indices]

    def forward_from_embeddings(
        self,
        prefix_embs: torch.Tensor,
        pad_masks: torch.Tensor,
        att_masks: torch.Tensor,
        actions: torch.Tensor,
        actions_is_pad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query_emb = self.encode_from_embeddings(prefix_embs, pad_masks, att_masks)
        if isinstance(self.critic_head, ConcatMLPHead):
            return self.critic_head(query_emb, actions)
        return self.critic_head(
            prefix_embs=None,
            pad_masks=None,
            att_masks=None,
            query_emb=query_emb,
            actions=actions,
            actions_is_pad=actions_is_pad,
        )

    def _append_query_token(
        self,
        embs: torch.Tensor,
        pad_masks: torch.Tensor,
        att_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Insert the learnable query token right after the valid prefix tokens."""

        batch_size, seq_len, hidden = embs.shape
        device = embs.device
        dtype = embs.dtype

        new_seq_len = seq_len + 1
        new_embs = torch.zeros(batch_size, new_seq_len, hidden, device=device, dtype=dtype)
        new_pad_masks = torch.zeros(batch_size, new_seq_len, device=device, dtype=pad_masks.dtype)
        new_att_masks = torch.zeros(batch_size, new_seq_len, device=device, dtype=att_masks.dtype)

        seq_lengths = pad_masks.long().sum(dim=1)
        query = self.query_embedding.to(device=device, dtype=dtype)

        for b_idx in range(batch_size):
            length = seq_lengths[b_idx].item()
            if length > 0:
                new_embs[b_idx, :length] = embs[b_idx, :length]
                new_att_masks[b_idx, :length] = att_masks[b_idx, :length]
                new_pad_masks[b_idx, :length] = True
            new_embs[b_idx, length] = query
            new_pad_masks[b_idx, length] = True
            if length < seq_len:
                remainder = seq_len - length
                new_embs[b_idx, length + 1 : length + 1 + remainder] = embs[b_idx, length:]
                new_att_masks[b_idx, length + 1 : length + 1 + remainder] = att_masks[b_idx, length:]

        # Alternative vectorized approach (kept for reference) that shifts padded indices using
        # advanced indexing instead of looping per batch:
        # batch_idx = torch.arange(batch_size, device=device).view(-1, 1)
        # seq_idx = torch.arange(seq_len, device=device).view(1, -1).expand(batch_size, -1)
        # mask = seq_idx >= seq_lengths.unsqueeze(1)
        # new_seq_idx = seq_idx + mask.long()
        # new_embs[batch_idx, new_seq_idx] = embs
        # new_pad_masks[batch_idx, new_seq_idx] = pad_masks
        # new_att_masks[batch_idx, new_seq_idx] = att_masks
        # new_embs[torch.arange(batch_size, device=device), seq_lengths] = query
        # new_pad_masks[torch.arange(batch_size, device=device), seq_lengths] = True
        # new_att_masks[torch.arange(batch_size, device=device), seq_lengths] = False
        return new_embs, new_pad_masks.bool(), new_att_masks.bool()
    def _append_query_token_my(
        self,
        embs: torch.Tensor,
        pad_masks: torch.Tensor,
        att_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Insert the learnable query token right after the valid prefix tokens."""

        batch_size, seq_len, hidden = embs.shape
        device = embs.device
        dtype = embs.dtype

        new_seq_len = seq_len + 1
        new_embs = torch.zeros(batch_size, new_seq_len, hidden, device=device, dtype=dtype)
        new_pad_masks = torch.zeros(batch_size, new_seq_len, device=device, dtype=pad_masks.dtype)
        new_att_masks = torch.zeros(batch_size, new_seq_len, device=device, dtype=att_masks.dtype)

        # copy the original tokens/masks
        new_embs[:, :seq_len] = embs
        new_pad_masks[:, :seq_len] = pad_masks
        new_att_masks[:, :seq_len] = att_masks

        # insert query at the new last position
        query = self.query_embedding.to(device=device, dtype=dtype)
        new_embs[:, -1] = query
        new_pad_masks[:, -1] = True

        # ensure the query forms a new block so previous tokens cannot attend to it,
        # but the query can attend to all previous blocks.
        new_att_masks[:, -1] = True

        return new_embs, new_pad_masks.bool(), new_att_masks.bool()


@dataclass
class ValueHeadConfig:
    chunk_size: int
    action_dim: int
    num_head_layers: int = 2
    head_mlp_dims: Sequence[int] = (512, 512)
    vlm_model_name: str | None = None
    att_mode: str = "causal"
    use_raw_state_fusion: bool = False
    raw_state_dim: int = 8
    bias_init_enabled: bool = False
    bias_init_value: float = 0.0


class Qchunk_Former(nn.Module):
    """Critic that consumes prefix tokens directly via a transformer head."""

    def __init__(
        self,
        config: ValueHeadConfig,
        text_config: LlamaConfig | None = None,
    ):
        super().__init__()
        if text_config is None and config.vlm_model_name is None:
            raise ValueError("TransformerValueCriticHead requires text_config or vlm_model_name.")
        head_cls = Q_Former
        self.head = head_cls(
            hidden_dim=(text_config.hidden_size if text_config is not None else None),
            action_dim=config.action_dim,
            num_layers=config.num_head_layers,
            mlp_hidden_dims=config.head_mlp_dims,
            model_id=config.vlm_model_name,
            text_config=text_config,
            att_mode=config.att_mode,
            bias_init_enabled=getattr(config, "bias_init_enabled", False),
            bias_init_value=getattr(config, "bias_init_value", 0.0),
        )
        # Raw state fusion (optional, backward compatible)
        self.use_raw_state_fusion = getattr(config, "use_raw_state_fusion", False)
        self.raw_state_dim = getattr(config, "raw_state_dim", 8)
        self.hidden_size = text_config.hidden_size if text_config is not None else None
        if self.hidden_size is None and self.use_raw_state_fusion:
            raise ValueError("Cannot enable raw_state_fusion without a known hidden size.")
        if self.use_raw_state_fusion:
            # VLM State: 960 -> 512 (高带宽保持语义)
            self.vlm_proj = nn.Sequential(
                nn.Linear(self.hidden_size, 512),
                nn.LayerNorm(512),
                nn.GELU(),
            )
            # Action: 7 -> 256
            self.act_proj = nn.Linear(config.action_dim, 256)
            # Raw State: raw_state_dim -> 192
            self.raw_proj = nn.Sequential(
                nn.Linear(self.raw_state_dim, 128),
                nn.GELU(),
                nn.Linear(128, 192),
                nn.LayerNorm(192),
            )
            self.fusion = nn.Sequential(
                nn.Linear(512 + 192 + 256, self.hidden_size),  # 512(VLM) + 192(Raw) + 256(Action) = 960
                nn.LayerNorm(self.hidden_size),
                nn.Dropout(0.1),
            )

    def forward_from_embeddings(
        self,
        prefix_embs: torch.Tensor,
        pad_masks: torch.Tensor,
        att_masks: torch.Tensor,
        actions: torch.Tensor,
        actions_is_pad: torch.Tensor | None = None,
        raw_state: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if self.use_raw_state_fusion:
            device = actions.device
            batch_size, chunk_len, _ = actions.shape
            if raw_state is None:
                raw_state = torch.zeros(batch_size, self.raw_state_dim, device=device, dtype=actions.dtype)
            elif raw_state.dim() > 2:
                # Collated states may carry an extra time dim; take the first slice to align with batch.
                raw_state = raw_state[:, 0]
            raw_state = raw_state.to(device=device, dtype=actions.dtype)
            act_emb = self.act_proj(actions)
            # pick last valid prefix token based on masks (True=valid, handles internal padding)
            seq_len = prefix_embs.shape[1]
            if pad_masks is not None:
                positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(prefix_embs.shape[0], -1)
                masked_pos = positions.masked_fill(~pad_masks.bool(), -1)
                last_idx = masked_pos.max(dim=1).values
            elif att_masks is not None:
                positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(prefix_embs.shape[0], -1)
                masked_pos = positions.masked_fill(~att_masks.bool(), -1)
                last_idx = masked_pos.max(dim=1).values
            else:
                last_idx = torch.full((prefix_embs.shape[0],), seq_len - 1, device=device, dtype=torch.long)
            last_idx = last_idx.clamp(min=0, max=seq_len - 1)
            batch_idx = torch.arange(prefix_embs.shape[0], device=device)
            vlm_state = prefix_embs[batch_idx, last_idx]
            vlm_emb = self.vlm_proj(vlm_state)
            raw_emb = self.raw_proj(raw_state)

            vlm_rep = vlm_emb.unsqueeze(1).expand(-1, chunk_len, -1)
            raw_rep = raw_emb.unsqueeze(1).expand(-1, chunk_len, -1)
            fused = torch.cat([vlm_rep, raw_rep, act_emb], dim=-1)
            fused = self.fusion(fused).to(prefix_embs.dtype)
            if actions_is_pad is None:
                actions_is_pad = torch.zeros(batch_size, chunk_len, dtype=torch.bool, device=device)
            return self.head(
                prefix_embs=prefix_embs,
                pad_masks=pad_masks,
                att_masks=att_masks,
                query_emb=None,
                actions=None,
                actions_is_pad=actions_is_pad,
                inputs_embeds=fused,
                **kwargs,
            )

        return self.head(
            prefix_embs=prefix_embs,
            pad_masks=pad_masks,
            att_masks=att_masks,
            query_emb=None,
            actions=actions,
            actions_is_pad=actions_is_pad,
            **kwargs,
        )


# Backward-compatible alias for older imports.
MYQueryValueHeadCritic = Qchunk_Former


class Q_Former(nn.Module):
    """Transformer-based head that produces a  value embedding before scoring."""

    def __init__(
        self,
        *,
        hidden_dim: Optional[int],
        action_dim: int,
        num_layers: int = 2,
        mlp_hidden_dims: Sequence[int] = (512, 512),
        model_id: str | None = None,
        text_config: LlamaConfig | None = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        att_mode: str = "causal",  # causal / bi-level
        bias_init_enabled: bool = False,
        bias_init_value: float = 0.0,
    ) -> None:
        super().__init__()
        self.bias_init_enabled = bias_init_enabled
        self.bias_init_value = bias_init_value
        self.decoder = Q_Former_Backbone(
            num_layers=num_layers,
            model_id=model_id,
            text_config=text_config,
            torch_dtype=torch_dtype,
        )
        decoder_hidden = self.decoder.text_config.hidden_size
        if hidden_dim is not None and decoder_hidden != hidden_dim:
            raise ValueError("hidden_dim must match text hidden size")
        self.hidden_dim = decoder_hidden
        self.action_tokens = ActionTokenizer(action_dim, self.hidden_dim)
        self.query_token = nn.Parameter(torch.zeros(self.hidden_dim))
        self.value_token = nn.Parameter(torch.zeros(self.hidden_dim))
        layers: list[nn.Module] = []
        prev_dim = self.hidden_dim
        total_layers = len(mlp_hidden_dims)
        # print("total_layers", total_layers)
        for idx, hidden in enumerate(mlp_hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden))
            if idx  < total_layers:
                
                
                layers.append(nn.GELU())
                layers.append(nn.LayerNorm(hidden))

                # layers.append(nn.GELU())
                # layers.append(nn.LayerNorm(hidden))

            prev_dim = hidden
        final_layer = nn.Linear(prev_dim, 1)
        if bias_init_enabled and final_layer.bias is not None:
            nn.init.constant_(final_layer.bias, bias_init_value)
            print(f">>> Initializing value head bias to {bias_init_value}")
        layers.append(final_layer)
        self.value_head = nn.Sequential(*layers)
        self.att_mode = att_mode
    def prepare_mask_emb(
        self,
        prefix_embs,
        actions,
        pad_masks,
        att_masks,
        actions_is_pad,
        action_embeds: torch.Tensor | None = None,
    ):
        # print(self.att_mode)
        action_tokens = action_embeds if action_embeds is not None else self.action_tokens(actions)
        batch_size = action_tokens.shape[0]
        device = action_tokens.device
        dtype = self.value_token.dtype
        token_segments = []
        pad_segments = []
        att_segments = []
        token_segments.append(prefix_embs)

        token_segments.append(action_tokens)
        value_tokens = self.value_token.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, 1, -1)
        token_segments.append(value_tokens)

        pad_segments.append(pad_masks)
      
        pad_segments.append(~actions_is_pad.bool()) # for action token
        pad_segments.append(torch.ones(batch_size, 1, dtype=torch.bool, device=device)) # for value token
        # 需要得到 query 的位置 ，value的位置

        att_segments.append(att_masks)
        if self.att_mode == "causal":
            att_segments.append(torch.ones(batch_size, action_tokens.shape[1], dtype=torch.bool, device=device))
        # att_segments.append(~actions_is_pad.bool())
        else:
            att_segments.append(torch.ones(batch_size, 1, dtype=torch.bool, device=device))
            att_segments.append(torch.zeros(batch_size, action_tokens.shape[1]-1, dtype=torch.bool, device=device))
        att_segments.append(torch.ones(batch_size, 1, dtype=torch.bool, device=device)) # for value token
                
        tokens = torch.cat(token_segments, dim=1).to(dtype)
        pad_mask = torch.cat(pad_segments, dim=1)
        att_mask = torch.cat(att_segments, dim=1)

        att_2d_masks = make_att_2d_masks(pad_mask, att_mask)


        attention_mask = _build_attention_mask(att_2d_masks, dtype=tokens.dtype)
        # attention_mask打印
        position_ids = torch.cumsum(pad_mask.long(), dim=1) - 1
        position_ids = position_ids.clamp_min(0)


        return tokens , attention_mask ,  position_ids
    
    def forward(
        self,
        *,
        prefix_embs: Optional[torch.Tensor], # output of vlm
        pad_masks: Optional[torch.Tensor],   # original input of vlm
        att_masks: Optional[torch.Tensor],   # original input of vlm
        query_emb: Optional[torch.Tensor],
        actions: torch.Tensor | None = None,               # output of vla or from a batch
        actions_is_pad : Optional[torch.Tensor] = None,          # depedend on where the action from
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            action_tokens = inputs_embeds
            if actions_is_pad is None:
                actions_is_pad = torch.zeros(
                    action_tokens.shape[0],
                    action_tokens.shape[1],
                    dtype=torch.bool,
                    device=action_tokens.device,
                )
        else:
            if actions is None:
                raise ValueError("Either `actions` or `inputs_embeds` must be provided.")
            if actions_is_pad is None:
                actions_is_pad = torch.zeros(
                    actions.shape[0],
                    actions.shape[1],
                    dtype=torch.bool,
                    device=actions.device,
                )
            action_tokens = None
        actions_arg = actions if action_tokens is None else None
        tokens , attention_mask ,  position_ids = self.prepare_mask_emb(
            prefix_embs,
            actions_arg,
            pad_masks,
            att_masks,
            actions_is_pad,
            action_embeds=action_tokens,
        )

        outputs = self.decoder(tokens, attention_mask=attention_mask, position_ids=position_ids)
        value_emb = outputs[:, -1, :]
        return self.value_head(value_emb)
