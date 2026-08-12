"""Self-contained critic module that owns actor/target/optimizer (no external trainer)."""

from __future__ import annotations

import copy
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from models.smolvla.modeling_smolvla import make_att_2d_masks

from qchunk.critic_adapters import (
    MLPCriticAdapter,
    PolicyEmbeddings,
    ValueHeadCriticAdapter,
)
from qchunk.critic_utils import (
    aggregate_q,
    discounted_chunk_returns,
    extract_future_batch,
    get_raw_state,
    get_tensor_from_batch,
    repeat_batch,
    soft_update_target,
)
from qchunk.networks import CriticBackbone
from qchunk.ood_calql_utils import (
    compute_calql_loss,
    compute_explicit_penalty_loss,
    prepare_cal_ood_actions,
    prepare_erg_ood_actions,
)
from qchunk.valuequeryhead import Qchunk_Former, ValueHeadConfig
from qchunk.valuequeryhead import CriticLanguageEmbedding, CriticVisionEncoder


class PolicyEncoderFn:
    def __call__(self, policy: Any, batch: Dict[str, torch.Tensor]) -> PolicyEmbeddings:
        raise NotImplementedError


class QChunkedCritic:
    """Owns actor + critic/target/opt/scheduler; exposes update & best-of-n."""

    def __init__(
        self,
        *,
        actor: Any,
        critic: nn.Module,
        target_critic: nn.Module,
        state_encoder: nn.Module | None,
        target_state_encoder: nn.Module | None,
        critic_vision_encoder: nn.Module | None,
        target_critic_vision_encoder: nn.Module | None,
        critic_lang_embedding: nn.Module | None,
        vision_proj: nn.Module | None,
        target_vision_proj: nn.Module | None,
        lang_proj: nn.Module | None,
        target_lang_proj: nn.Module | None,
        use_critic_state_encoder: bool,
        use_independent_critic_encoder: bool,
        optimizer: torch.optim.Optimizer,
        cfg: Any,
        device: torch.device,
        chunk_size: int,
        q_chunk_len: int,
        action_step_dim: int,
        encoder_fn: PolicyEncoderFn,
        scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
        action_stats: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.state_encoder = state_encoder
        self.target_state_encoder = target_state_encoder
        self.critic_vision_encoder = critic_vision_encoder
        self.target_critic_vision_encoder = target_critic_vision_encoder
        self.critic_lang_embedding = critic_lang_embedding
        self.vision_proj = vision_proj
        self.target_vision_proj = target_vision_proj
        self.lang_proj = lang_proj
        self.target_lang_proj = target_lang_proj
        self.use_critic_state_encoder = bool(use_critic_state_encoder and state_encoder is not None)
        self.use_independent_critic_encoder = bool(use_independent_critic_encoder and critic_vision_encoder is not None and critic_lang_embedding is not None)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.device = device
        self.chunk_size = chunk_size
        self.q_chunk_len = int(q_chunk_len)
        if self.q_chunk_len <= 0:
            raise ValueError(f"q_chunk_len must be > 0, got {self.q_chunk_len}.")
        if self.q_chunk_len > self.chunk_size:
            raise ValueError(
                f"q_chunk_len ({self.q_chunk_len}) cannot exceed chunk_size ({self.chunk_size})."
            )
        self.action_step_dim = action_step_dim
        self.encode_policy = encoder_fn
        self.action_stats = action_stats or {}
        self._raw_state_dim = getattr(cfg, "raw_state_dim", 8)
        self._state_encoder_in_dim = (
            int(getattr(self.state_encoder, "in_features"))
            if self.state_encoder is not None and hasattr(self.state_encoder, "in_features")
            else int(self._raw_state_dim)
        )
        self._last_target_head_means: Optional[list[float]] = None
        self._shape_debug_done = False

    @staticmethod
    def _masked_mean(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1).to(tensor.dtype)
        summed = (tensor * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return summed / denom

    def _prepare_state_encoder_input(
        self,
        raw_state: torch.Tensor,
        *,
        use_target: bool,
    ) -> torch.Tensor:
        encoder = self.target_state_encoder if use_target else self.state_encoder
        if encoder is None:
            return raw_state
        if raw_state.ndim > 2:
            raw_state = raw_state[:, 0]
        raw_state = raw_state.to(device=self.device)
        in_dim = int(self._state_encoder_in_dim)
        if raw_state.shape[-1] < in_dim:
            pad_dim = in_dim - raw_state.shape[-1]
            pad = torch.zeros(raw_state.shape[0], pad_dim, device=self.device, dtype=raw_state.dtype)
            raw_state = torch.cat([raw_state, pad], dim=-1)
        elif raw_state.shape[-1] > in_dim:
            raw_state = raw_state[..., :in_dim]
        target_dtype = next(encoder.parameters()).dtype
        return raw_state.to(dtype=target_dtype)

    @staticmethod
    def _iter_policy_params(policy: Any) -> list[nn.Parameter]:
        return list(policy.parameters()) if hasattr(policy, "parameters") else []

    def _resolve_backup_action_samples(self) -> int:
        preferred = getattr(self.cfg, "backup_action_samples", None)
        if preferred is None:
            preferred = getattr(self.cfg, "action_samples", 2)
        return max(1, int(preferred))

    @staticmethod
    def _pad_prefix_to_length(
        prefix_embs: torch.Tensor,
        pad_masks: torch.Tensor,
        att_masks: torch.Tensor,
        *,
        prefix_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if prefix_embs.shape[1] >= prefix_length:
            return prefix_embs, pad_masks, att_masks
        pad_len = int(prefix_length - prefix_embs.shape[1])
        if pad_len <= 0:
            return prefix_embs, pad_masks, att_masks
        bsz, _, hidden = prefix_embs.shape
        device = prefix_embs.device
        emb_pad = torch.zeros(bsz, pad_len, hidden, dtype=prefix_embs.dtype, device=device)
        mask_pad = torch.zeros(bsz, pad_len, dtype=torch.bool, device=device)
        return (
            torch.cat([prefix_embs, emb_pad], dim=1),
            torch.cat([pad_masks.bool(), mask_pad], dim=1),
            torch.cat([att_masks.bool(), mask_pad], dim=1),
        )

    def _encode_policy_independent(
        self,
        policy: Any,
        batch: Dict[str, torch.Tensor],
        raw_state: torch.Tensor,
        *,
        use_target: bool,
        track_state_grad: bool,
    ) -> PolicyEmbeddings:
        vision_encoder = self.target_critic_vision_encoder if use_target else self.critic_vision_encoder
        if vision_encoder is None or self.critic_lang_embedding is None:
            return self._encode_policy_pre_state_replace(
                policy,
                batch,
                raw_state,
                use_target=use_target,
                track_state_grad=track_state_grad,
            )
        if not (
            hasattr(policy, "_prepare_batch")
            and hasattr(policy, "prepare_images")
            and hasattr(policy, "model")
        ):
            return self.encode_policy(policy, batch).to(self.device)

        state_encoder = self.target_state_encoder if use_target else self.state_encoder
        if state_encoder is None:
            return self.encode_policy(policy, batch).to(self.device)

        was_training = bool(getattr(policy, "training", False))
        policy.eval()
        try:
            processed_batch = policy._prepare_batch({k: v for k, v in batch.items()})
            images, img_masks = policy.prepare_images(processed_batch)
            lang_tokens = processed_batch[f"{OBS_LANGUAGE_TOKENS}"].to(self.device).long()
            lang_masks = processed_batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"].to(self.device).bool()
            batch_size = lang_tokens.shape[0]
            device = self.device

            model = getattr(policy, "model", None)
            add_image_special_tokens = bool(getattr(model, "add_image_special_tokens", False))
            image_start_token = getattr(model, "global_image_start_token", None)
            image_end_token = getattr(model, "image_end_token", None)
            prefix_length = int(getattr(model, "prefix_length", 0) or 0)

            lang_embedding = self.critic_lang_embedding
            vision_proj = self.target_vision_proj if use_target else self.vision_proj
            lang_proj = self.target_lang_proj if use_target else self.lang_proj
            target_dtype = next(state_encoder.parameters()).dtype

            token_segments: list[torch.Tensor] = []
            pad_segments: list[torch.Tensor] = []
            att_segments: list[torch.Tensor] = []

            grad_ctx = torch.enable_grad() if track_state_grad else torch.no_grad()
            with grad_ctx:
                for image, img_mask in zip(images, img_masks, strict=False):
                    image = image.to(device=device)
                    img_mask = img_mask.to(device=device).bool().view(batch_size, -1).any(dim=1)
                    if add_image_special_tokens and image_start_token is not None:
                        start_ids = image_start_token.to(device=device).long().unsqueeze(0).expand(batch_size, -1)
                        start_emb = lang_embedding(start_ids).to(dtype=target_dtype)
                        token_segments.append(start_emb)
                        pad_segments.append(torch.ones(batch_size, start_emb.shape[1], dtype=torch.bool, device=device))
                        att_segments.append(torch.zeros(batch_size, start_emb.shape[1], dtype=torch.bool, device=device))

                    image_tokens = vision_encoder(image).to(dtype=target_dtype)
                    if vision_proj is not None:
                        image_tokens = vision_proj(image_tokens)
                    img_dim = image_tokens.shape[-1]
                    image_tokens = image_tokens * torch.tensor(float(img_dim) ** 0.5, dtype=image_tokens.dtype, device=device)
                    num_img_tokens = image_tokens.shape[1]
                    token_segments.append(image_tokens)
                    pad_segments.append(img_mask[:, None].expand(batch_size, num_img_tokens))
                    att_segments.append(torch.zeros(batch_size, num_img_tokens, dtype=torch.bool, device=device))

                    if add_image_special_tokens and image_end_token is not None:
                        end_ids = image_end_token.to(device=device).long().unsqueeze(0).expand(batch_size, -1)
                        end_emb = lang_embedding(end_ids).to(dtype=target_dtype)
                        token_segments.append(end_emb)
                        pad_segments.append(torch.ones(batch_size, end_emb.shape[1], dtype=torch.bool, device=device))
                        att_segments.append(torch.zeros(batch_size, end_emb.shape[1], dtype=torch.bool, device=device))

                lang_emb = lang_embedding(lang_tokens).to(dtype=target_dtype)
                if lang_proj is not None:
                    lang_emb = lang_proj(lang_emb)
                lang_dim = lang_emb.shape[-1]
                lang_emb = lang_emb * torch.tensor(float(lang_dim) ** 0.5, dtype=lang_emb.dtype, device=device)
                token_segments.append(lang_emb)
                pad_segments.append(lang_masks)
                att_segments.append(torch.zeros(batch_size, lang_emb.shape[1], dtype=torch.bool, device=device))

                state_in = self._prepare_state_encoder_input(raw_state, use_target=use_target)
                state_emb = state_encoder(state_in).to(dtype=target_dtype)
                if state_emb.ndim == 2:
                    state_emb = state_emb.unsqueeze(1)
                token_segments.append(state_emb)
                pad_segments.append(torch.ones(batch_size, state_emb.shape[1], dtype=torch.bool, device=device))
                att_segments.append(torch.ones(batch_size, state_emb.shape[1], dtype=torch.bool, device=device))

                prefix_embs = torch.cat(token_segments, dim=1)
                pad_masks = torch.cat(pad_segments, dim=1).bool()
                att_masks = torch.cat(att_segments, dim=1).bool()
                if prefix_length > 0:
                    prefix_embs, pad_masks, att_masks = self._pad_prefix_to_length(
                        prefix_embs,
                        pad_masks,
                        att_masks,
                        prefix_length=prefix_length,
                    )

                prefix_outputs = prefix_embs.to(torch.float32)
                content_mask = (~att_masks) & pad_masks
                state_mask = att_masks & pad_masks
                pooled = torch.cat(
                    [
                        self._masked_mean(prefix_outputs, content_mask),
                        self._masked_mean(prefix_outputs, state_mask),
                    ],
                    dim=-1,
                )
                if track_state_grad:
                    return PolicyEmbeddings(
                        pooled=pooled,
                        prefix_outs=prefix_outputs,
                        pad_masks=pad_masks,
                        att_masks=att_masks,
                    )
                return PolicyEmbeddings(
                    pooled=pooled.detach(),
                    prefix_outs=prefix_outputs.detach(),
                    pad_masks=pad_masks.detach(),
                    att_masks=att_masks.detach(),
                )
        finally:
            policy.train(was_training)

    def _encode_policy_pre_state_replace(
        self,
        policy: Any,
        batch: Dict[str, torch.Tensor],
        raw_state: torch.Tensor,
        *,
        use_target: bool,
        track_state_grad: bool,
    ) -> PolicyEmbeddings:
        """
        Build critic embeddings by replacing state tokens *before* VLM forward.

        This keeps state token contextualization inside VLM while using critic-side state encoder.
        """
        if not self.use_critic_state_encoder:
            return self.encode_policy(policy, batch).to(self.device)

        if not (
            hasattr(policy, "_prepare_batch")
            and hasattr(policy, "prepare_images")
            and hasattr(policy, "prepare_state")
            and hasattr(policy, "model")
            and hasattr(policy.model, "embed_prefix")
            and hasattr(policy.model, "vlm_with_expert")
        ):
            # Safe fallback for non-SmolVLA policies.
            return self.encode_policy(policy, batch).to(self.device)

        encoder = self.target_state_encoder if use_target else self.state_encoder
        if encoder is None:
            return self.encode_policy(policy, batch).to(self.device)
        use_vlm_backbone_encode = bool(getattr(self.cfg, "use_vlm_backbone_encode", True))

        was_training = bool(getattr(policy, "training", False))
        params = self._iter_policy_params(policy)
        prev_requires = [p.requires_grad for p in params]
        freeze_policy = track_state_grad and len(params) > 0

        grad_ctx = torch.enable_grad() if track_state_grad else torch.no_grad()
        policy.eval()
        try:
            with grad_ctx:
                if freeze_policy:
                    for p in params:
                        p.requires_grad_(False)

                processed_batch = policy._prepare_batch({k: v for k, v in batch.items()})
                images, img_masks = policy.prepare_images(processed_batch)
                state = policy.prepare_state(processed_batch)
                lang_tokens = processed_batch[f"{OBS_LANGUAGE_TOKENS}"]
                lang_masks = processed_batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

                prefix_embs, pad_masks, att_masks = policy.model.embed_prefix(
                    images,
                    img_masks,
                    lang_tokens,
                    lang_masks,
                    state=state,
                )

                # Replace state token(s) BEFORE VLM forward so they can absorb global context.
                state_mask = pad_masks.bool() & att_masks.bool()
                if bool(state_mask.any()):
                    state_in = self._prepare_state_encoder_input(raw_state, use_target=use_target)
                    state_emb = encoder(state_in)
                    state_tokens = state_emb.to(dtype=prefix_embs.dtype).unsqueeze(1).expand(-1, prefix_embs.shape[1], -1)
                    prefix_embs = torch.where(state_mask.unsqueeze(-1), state_tokens, prefix_embs)

                if use_vlm_backbone_encode:
                    att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
                    position_ids = torch.cumsum(pad_masks, dim=1) - 1
                    outputs_embeds, _ = policy.model.vlm_with_expert.forward(
                        attention_mask=att_2d_masks,
                        position_ids=position_ids,
                        past_key_values=None,
                        inputs_embeds=[prefix_embs, None],
                        use_cache=False,
                        fill_kv_cache=True,
                    )
                    prefix_outputs = outputs_embeds[0].to(torch.float32)
                else:
                    prefix_outputs = prefix_embs.to(torch.float32)

                pad_mask_bool = pad_masks.bool()
                att_mask_bool = att_masks.bool()
                content_mask = (~att_mask_bool) & pad_mask_bool
                state_mask = att_mask_bool & pad_mask_bool
                pooled = torch.cat(
                    [
                        self._masked_mean(prefix_outputs, content_mask),
                        self._masked_mean(prefix_outputs, state_mask),
                    ],
                    dim=-1,
                )

                if track_state_grad:
                    return PolicyEmbeddings(
                        pooled=pooled,
                        prefix_outs=prefix_outputs,
                        pad_masks=pad_masks,
                        att_masks=att_masks,
                    )
                return PolicyEmbeddings(
                    pooled=pooled.detach(),
                    prefix_outs=prefix_outputs.detach(),
                    pad_masks=pad_masks.detach(),
                    att_masks=att_masks.detach(),
                )
        finally:
            if freeze_policy:
                for p, req in zip(params, prev_requires, strict=False):
                    p.requires_grad_(req)
            policy.train(was_training)

    def encode_policy_for_critic(
        self,
        policy: Any,
        batch: Dict[str, torch.Tensor],
        raw_state: torch.Tensor,
        *,
        use_target: bool,
        track_state_grad: bool = False,
    ) -> PolicyEmbeddings:
        if self.use_independent_critic_encoder:
            return self._encode_policy_independent(
                policy,
                batch,
                raw_state,
                use_target=use_target,
                track_state_grad=track_state_grad,
            )
        return self._encode_policy_pre_state_replace(
            policy,
            batch,
            raw_state,
            use_target=use_target,
            track_state_grad=track_state_grad,
        )

    # ------------------------------------------------------------------ #
    # Builders
    # ------------------------------------------------------------------ #
    @classmethod
    def _build_critic_module(
        cls,
        policy: Any,
        cfg: Any,
        q_chunk_len: int,
        action_step_dim: int,
        obs_dim: int,
        device: torch.device,
    ):
        critic_type = str(getattr(cfg, "critic_type", "mlp")).lower()
        if critic_type in {"my_value_query_head", "my_value_head", "value_head"}:
            critic_type = "q_chunk_former"
        if critic_type == "mlp":
            backbone = CriticBackbone(
                obs_dim=obs_dim,
                action_dim=action_step_dim * q_chunk_len,
                hidden_sizes=getattr(cfg, "hidden_dims", (512, 512)),
            )
            critic = MLPCriticAdapter(backbone).to(device)
        elif critic_type == "q_chunk_former":
            text_config = getattr(getattr(getattr(policy, "model", None), "vlm_with_expert", None), "config", None)
            text_config = getattr(text_config, "text_config", None)
            num_head_layers = getattr(cfg, "qformer_num_backbone_layers", None)
            if num_head_layers is None:
                num_head_layers = getattr(cfg, "value_head_num_layers", getattr(cfg, "head_num_layers", 2))
            vh_config = ValueHeadConfig(
                chunk_size=q_chunk_len,
                action_dim=action_step_dim,
                num_head_layers=num_head_layers,
                head_mlp_dims=getattr(cfg, "value_head_mlp_dims", getattr(cfg, "head_mlp_dims", (512, 512))),
                vlm_model_name=getattr(cfg, "value_head_vlm_model_name", getattr(cfg, "vqh_vlm_model_name", None)),
                att_mode=getattr(cfg, "att_mode", "causal"),
                use_raw_state_fusion=getattr(cfg, "use_raw_state_fusion", False),
                fuse_vlm_state=bool(getattr(cfg, "use_vlm_backbone_encode", True)),
                raw_state_dim=getattr(cfg, "raw_state_dim", 8),
                bias_init_enabled=getattr(cfg, "value_head_bias_init_enabled", False),
                bias_init_value=getattr(cfg, "value_head_bias_init_value", 0.0),
            )
            num_q_heads = max(1, int(getattr(cfg, "num_q_heads", 1)))
            heads = [
                Qchunk_Former(vh_config, text_config=text_config)
                for _ in range(num_q_heads)
            ]
            critic = ValueHeadCriticAdapter(heads).to(device)
        else:
            raise ValueError(f"Unknown critic_type {critic_type} (expected 'mlp' or 'q_chunk_former').")
        return critic

    @classmethod
    def build(
        cls,
        *,
        policy_path: Path,
        policy_cfg: Any,
        critic_cfg: Any,
        sample_batch: Dict[str, torch.Tensor],
        ds_meta: Any = None,
        device: str = "cuda",
        encoder_fn: Optional[PolicyEncoderFn] = None,
        actor: Any = None,
        freeze_actor: bool = True,
    ) -> "QChunkedCritic":
        from scripts.train_qchunk_offline import encode_policy_observations
        from utils.local_policy import make_local_smolvla_policy

        device_str = str(device)
        device_obj = torch.device(device_str)

        # Keep config serializable (draccus expects plain python types).
        policy_cfg.device = device_str
        policy_cfg.pretrained_path = str(policy_path)
        if actor is None:
            actor = make_local_smolvla_policy(cfg=policy_cfg, ds_meta=ds_meta)
        if freeze_actor:
            actor.eval()
            for p in actor.parameters():
                p.requires_grad = False

        actions = sample_batch["action"]
        chunk_size = actions.shape[-2]
        action_step_dim = actions.shape[-1]
        q_chunk_len = getattr(critic_cfg, "q_chunk_len", None)
        if q_chunk_len is None:
            raise ValueError("critic_cfg.q_chunk_len must be set to use the critic.")
        q_chunk_len = int(q_chunk_len)
        if q_chunk_len <= 0:
            raise ValueError(f"critic_cfg.q_chunk_len must be > 0, got {q_chunk_len}.")
        if q_chunk_len > chunk_size:
            raise ValueError(f"critic_cfg.q_chunk_len ({q_chunk_len}) cannot exceed chunk_size ({chunk_size}).")
        encoder = encoder_fn or encode_policy_observations
        encoding: PolicyEmbeddings = encoder(actor, sample_batch)
        obs_dim = encoding.pooled.shape[-1]
        critic = cls._build_critic_module(actor, critic_cfg, q_chunk_len, action_step_dim, obs_dim, device_obj)
        target_critic = copy.deepcopy(critic).to(device_obj)
        use_independent_critic_encoder = bool(getattr(critic_cfg, "use_independent_critic_encoder", False))
        use_critic_state_encoder = bool(getattr(critic_cfg, "use_critic_state_encoder", False))
        if use_independent_critic_encoder and not use_critic_state_encoder:
            logging.warning(
                "use_independent_critic_encoder=True requires a critic state encoder; forcing use_critic_state_encoder=True."
            )
            use_critic_state_encoder = True
        state_encoder: nn.Module | None = None
        target_state_encoder: nn.Module | None = None
        if use_critic_state_encoder:
            actor_state_proj = getattr(getattr(actor, "model", None), "state_proj", None)
            if isinstance(actor_state_proj, nn.Module):
                state_encoder = copy.deepcopy(actor_state_proj).to(device_obj)
            else:
                hidden_size = int(encoding.prefix_outs.shape[-1])
                raw_state_dim = int(getattr(critic_cfg, "raw_state_dim", 8))
                state_encoder = nn.Linear(raw_state_dim, hidden_size).to(device_obj)
                logging.warning(
                    "Critic state encoder initialized from scratch (actor.state_proj missing). "
                    "raw_state_dim=%s hidden_size=%s",
                    raw_state_dim,
                    hidden_size,
                )
            target_state_encoder = copy.deepcopy(state_encoder).to(device_obj)
            target_state_encoder.eval()
            for p in target_state_encoder.parameters():
                p.requires_grad = False

        critic_vision_encoder: nn.Module | None = None
        target_critic_vision_encoder: nn.Module | None = None
        critic_lang_embedding: nn.Module | None = None
        vision_proj: nn.Module | None = None
        target_vision_proj: nn.Module | None = None
        lang_proj: nn.Module | None = None
        target_lang_proj: nn.Module | None = None
        if use_independent_critic_encoder:
            vlm_with_expert = getattr(getattr(actor, "model", None), "vlm_with_expert", None)
            get_vlm_model = getattr(vlm_with_expert, "get_vlm_model", None)
            vlm_model = get_vlm_model() if callable(get_vlm_model) else None
            if vlm_model is None:
                logging.warning(
                    "use_independent_critic_encoder=True but actor VLM model is unavailable; "
                    "falling back to actor-coupled critic encoding."
                )
                use_independent_critic_encoder = False
            else:
                critic_vision_freeze_layers = max(0, int(getattr(critic_cfg, "critic_vision_freeze_layers", 0)))
                critic_vision_encoder = CriticVisionEncoder(
                    vlm_model,
                    freeze_bottom_layers=critic_vision_freeze_layers,
                ).to(device_obj)
                target_critic_vision_encoder = copy.deepcopy(critic_vision_encoder).to(device_obj)
                target_critic_vision_encoder.eval()
                for p in target_critic_vision_encoder.parameters():
                    p.requires_grad = False

                lang_embed = vlm_model.text_model.get_input_embeddings()
                critic_lang_embedding = CriticLanguageEmbedding(lang_embed).to(device_obj)
                critic_lang_embedding.eval()

                hidden_size = int(encoding.prefix_outs.shape[-1])
                sample_visual_dim: int | None = None
                probe_error: Exception | None = None
                try:
                    processed_batch = actor._prepare_batch({k: v for k, v in sample_batch.items()})
                    sample_images, _ = actor.prepare_images(processed_batch)
                    if len(sample_images) > 0:
                        with torch.no_grad():
                            sample_visual = critic_vision_encoder(sample_images[0].to(device_obj))
                        sample_visual_dim = int(sample_visual.shape[-1])
                except Exception as exc:
                    probe_error = exc

                if sample_visual_dim is None:
                    if probe_error is not None:
                        raise RuntimeError(
                            "use_independent_critic_encoder=True requires probing critic vision output dim, "
                            "but the probe failed. Please check actor._prepare_batch / actor.prepare_images "
                            "and sample_batch image fields."
                        ) from probe_error
                    raise RuntimeError(
                        "use_independent_critic_encoder=True requires at least one image tensor to probe "
                        "critic vision output dim, but actor.prepare_images returned no images."
                    )

                if sample_visual_dim != hidden_size:
                    vision_proj = nn.Linear(sample_visual_dim, hidden_size).to(device_obj)
                    target_vision_proj = copy.deepcopy(vision_proj).to(device_obj)
                    target_vision_proj.eval()
                    for p in target_vision_proj.parameters():
                        p.requires_grad = False

                lang_weight = getattr(getattr(critic_lang_embedding, "embed_tokens", None), "weight", None)
                if lang_weight is not None:
                    lang_dim = int(lang_weight.shape[-1])
                    if lang_dim != hidden_size:
                        lang_proj = nn.Linear(lang_dim, hidden_size).to(device_obj)
                        target_lang_proj = copy.deepcopy(lang_proj).to(device_obj)
                        target_lang_proj.eval()
                        for p in target_lang_proj.parameters():
                            p.requires_grad = False

        optimizer_params = list(critic.parameters())
        if state_encoder is not None:
            optimizer_params.extend(list(state_encoder.parameters()))
        if critic_vision_encoder is not None:
            optimizer_params.extend(list(critic_vision_encoder.parameters()))
        if vision_proj is not None:
            optimizer_params.extend(list(vision_proj.parameters()))
        if lang_proj is not None:
            optimizer_params.extend(list(lang_proj.parameters()))
        optimizer = torch.optim.Adam(
            optimizer_params,
            lr=getattr(critic_cfg, "lr", 3e-4),
            betas=getattr(critic_cfg, "betas", (0.9, 0.999)),
            weight_decay=getattr(critic_cfg, "weight_decay", 0.0),
        )
        scheduler = None
        warmup_steps = max(getattr(critic_cfg, "lr_warmup_steps", 0), 0)
        total_steps = getattr(critic_cfg, "lr_total_steps", None)
        final_lr = max(getattr(critic_cfg, "lr_final", 0.0), 0.0)
        base_lr = getattr(critic_cfg, "lr", 3e-4)
        if warmup_steps > 0 or (total_steps and final_lr >= 0):
            min_factor = 0.0
            if base_lr > 0:
                min_factor = min(final_lr / base_lr, 1.0)

            def _lr_lambda(step: int) -> float:
                step_idx = step + 1
                if warmup_steps > 0 and step_idx <= warmup_steps:
                    return step_idx / warmup_steps
                if total_steps and total_steps > warmup_steps:
                    progress = (step_idx - warmup_steps) / (total_steps - warmup_steps)
                    progress = max(0.0, min(progress, 1.0))
                    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                    return min_factor + (1.0 - min_factor) * cosine
                return 1.0

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

        return cls(
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            state_encoder=state_encoder,
            target_state_encoder=target_state_encoder,
            critic_vision_encoder=critic_vision_encoder,
            target_critic_vision_encoder=target_critic_vision_encoder,
            critic_lang_embedding=critic_lang_embedding,
            vision_proj=vision_proj,
            target_vision_proj=target_vision_proj,
            lang_proj=lang_proj,
            target_lang_proj=target_lang_proj,
            use_critic_state_encoder=use_critic_state_encoder,
            use_independent_critic_encoder=use_independent_critic_encoder,
            optimizer=optimizer,
            cfg=critic_cfg,
            device=device_obj,
            chunk_size=chunk_size,
            q_chunk_len=q_chunk_len,
            action_step_dim=action_step_dim,
            encoder_fn=encoder,
            scheduler=scheduler,
            action_stats=(getattr(ds_meta, "stats", {}) or {}).get("action") if hasattr(ds_meta, "stats") else None,
        )

    # ------------------------------------------------------------------ #
    # Inference (best-of-n)
    # ------------------------------------------------------------------ #
    def predict_best_of_n(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        action_samples: Optional[int] = None,
        forced_mask: Optional[torch.Tensor] = None,
        return_candidates: bool = False,
        actor: Optional[Any] = None,
        encoding: Optional[PolicyEmbeddings] = None,
    ):
        first_tensor = next((value for value in batch.values() if isinstance(value, torch.Tensor)), None)
        if first_tensor is None:
            raise ValueError("Batch does not contain tensor observations.")
        batch_size = first_tensor.shape[0]
        actor_for_selection = actor if actor is not None else self.actor
        raw_state = get_raw_state(batch, batch_size, getattr(self.cfg, "raw_state_dim", self._raw_state_dim), self.device)
        if encoding is None:
            encoding = self.encode_policy_for_critic(
                actor_for_selection,
                batch,
                raw_state,
                use_target=True,
                track_state_grad=False,
            )
        else:
            encoding = encoding.to(self.device)
        q_chunk_len = self.q_chunk_len
        action_samples = action_samples or self._resolve_backup_action_samples()
        if action_samples <= 1:
            actions = actor_for_selection.predict_action_chunk(batch)
            if actions.shape[1] < q_chunk_len:
                raise ValueError(
                    f"Policy chunk ({actions.shape[1]}) shorter than q_chunk_len ({q_chunk_len})."
                )
            if actions.shape[1] != q_chunk_len:
                actions = actions[:, :q_chunk_len]
            if forced_mask is not None:
                eval_mask = forced_mask.to(self.device).bool()
                if eval_mask.shape[-1] > q_chunk_len:
                    eval_mask = eval_mask[..., :q_chunk_len]
            else:
                eval_mask = torch.zeros(actions.shape[:2], dtype=torch.bool, device=self.device)
            q_values = self.target_critic(encoding, actions, action_mask=eval_mask, raw_state=raw_state)
            best_q = aggregate_q(q_values, getattr(self.cfg, "q_aggregation", "mean"))
            if return_candidates:
                return actions, best_q, actions.unsqueeze(1)
            return actions, best_q

        expanded_batch = repeat_batch(batch, batch_size, action_samples)
        with torch.no_grad():
            expanded_actions = actor_for_selection.predict_action_chunk(expanded_batch)
        if expanded_actions.shape[1] < q_chunk_len:
            raise ValueError(
                f"Policy chunk ({expanded_actions.shape[1]}) shorter than q_chunk_len ({q_chunk_len})."
            )
        if expanded_actions.shape[1] != q_chunk_len:
            expanded_actions = expanded_actions[:, :q_chunk_len]
        action_dim = expanded_actions.shape[-1]
        stacked = expanded_actions.view(batch_size, action_samples, q_chunk_len, action_dim)
        flat_actions = stacked.view(-1, q_chunk_len, action_dim)
        repeated_encoding = encoding.repeat(action_samples)
        eval_mask = torch.zeros(flat_actions.shape[:2], dtype=torch.bool, device=self.device)
        if forced_mask is not None:
            forced = forced_mask.to(self.device).bool()
            if forced.shape[-1] > q_chunk_len:
                forced = forced[..., :q_chunk_len]
            forced = forced.view(batch_size, q_chunk_len)
            eval_mask = forced.unsqueeze(1).expand(-1, action_samples, -1).reshape(-1, q_chunk_len)
        raw_state_rep = raw_state.repeat_interleave(action_samples, dim=0)
        q_values = self.target_critic(repeated_encoding, flat_actions, action_mask=eval_mask, raw_state=raw_state_rep)
        q = aggregate_q(q_values, getattr(self.cfg, "q_aggregation", "mean")).view(batch_size, action_samples, -1)
        best_idx = torch.argmax(q, dim=1).squeeze(-1)
        batch_idx = torch.arange(batch_size, device=self.device)
        best_actions = stacked[batch_idx, best_idx]
        best_q = q[batch_idx, best_idx]
        if best_q.ndim == 1:
            best_q = best_q.unsqueeze(-1)
        if return_candidates:
            return best_actions, best_q, stacked
        return best_actions, best_q

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def precompute_next_target_encoding(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        target_policy: Optional[Any] = None,
    ) -> PolicyEmbeddings:
        next_batch = extract_future_batch(batch)
        if next_batch is None:
            raise ValueError("Future batch missing `next_observations`.")
        first_tensor = next((value for value in next_batch.values() if isinstance(value, torch.Tensor)), None)
        if first_tensor is None:
            raise ValueError("Future batch does not contain tensor observations.")
        target_actor = target_policy if target_policy is not None else self.actor
        next_raw_state = get_raw_state(
            next_batch,
            int(first_tensor.shape[0]),
            getattr(self.cfg, "raw_state_dim", self._raw_state_dim),
            self.device,
        )
        return self.encode_policy_for_critic(
            target_actor,
            next_batch,
            next_raw_state,
            use_target=True,
            track_state_grad=False,
        )

    def update(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        current_step: Optional[int] = None,
        ood_warmup_steps: int = 0,
        ood_policy: Optional[Any] = None,
        target_action_samples: Optional[int] = None,
        target_policy: Optional[Any] = None,
        cached_next_encoding: Optional[PolicyEmbeddings] = None,
        step_scheduler: bool = True,
    ) -> Dict[str, float]:
        self.critic.train()
        if self.state_encoder is not None:
            self.state_encoder.train()
        if self.critic_vision_encoder is not None:
            self.critic_vision_encoder.train()
        if self.target_critic_vision_encoder is not None:
            self.target_critic_vision_encoder.eval()
        if self.critic_lang_embedding is not None:
            self.critic_lang_embedding.eval()
        if self.vision_proj is not None:
            self.vision_proj.train()
        if self.target_vision_proj is not None:
            self.target_vision_proj.eval()
        if self.lang_proj is not None:
            self.lang_proj.train()
        if self.target_lang_proj is not None:
            self.target_lang_proj.eval()
        start = time.perf_counter()
        actions = batch["action"].to(self.device)
        q_chunk_len = self.q_chunk_len
        action_mask = get_tensor_from_batch(batch, ["actions_is_pad", "action_is_pad"], default_shape=actions.shape[:2], device=self.device)

        if actions.shape[-2] > q_chunk_len:
            actions = actions[:, :q_chunk_len]
        if action_mask.shape[-1] > q_chunk_len:
            action_mask = action_mask[..., :q_chunk_len]
        action_mask = action_mask.to(self.device).bool()

        critic_input_actions = actions.clone()
        critic_input_mask = action_mask.clone()
        if self.critic.training:
            dropout_prob = getattr(self.cfg, "mask_dropout_prob", 0.5)
            keep_mask = torch.rand_like(critic_input_mask.float()) > dropout_prob
            critic_input_mask = critic_input_mask & keep_mask

        rewards = get_tensor_from_batch(batch, ["rewards"], default_shape=actions.shape[:2], device=self.device)
        if rewards.ndim > 1 and rewards.shape[-1] > q_chunk_len:
            rewards = rewards[..., :q_chunk_len]
        rewards = rewards.reshape(actions.shape[0], -1).to(self.device)

        reward_is_pad = get_tensor_from_batch(batch, ["reward_is_pad"], default_shape=actions.shape[:2], device=self.device)
        if reward_is_pad.shape[-1] > q_chunk_len:
            reward_is_pad = reward_is_pad[..., :q_chunk_len]

        returns = discounted_chunk_returns(rewards, reward_is_pad, getattr(self.cfg, "discount", 0.99))
        raw_state = get_raw_state(batch, actions.shape[0], getattr(self.cfg, "raw_state_dim", self._raw_state_dim), self.device)
        encoding = self.encode_policy_for_critic(
            self.actor,
            batch,
            raw_state,
            use_target=False,
            track_state_grad=bool(self.use_critic_state_encoder or self.use_independent_critic_encoder),
        )
        next_batch = extract_future_batch(batch)
        if next_batch is None:
            raise ValueError("Future batch missing `next_observations`.")

        next_pad = next_batch.get("next_observation_is_pad")
        if next_pad is None:
            raise ValueError("`next_observation_is_pad` missing in future batch.")
        next_pad = next_pad.to(self.device).bool()

        valid_lens = next_batch.get("next_obs_valid_chunk_len")
        if valid_lens is None:
            raise ValueError("`next_obs_valid_chunk_len` missing in future batch.")
        valid_lens = valid_lens.to(self.device).long()
        if valid_lens.ndim > 1:
            valid_lens = valid_lens.squeeze(-1)
        valid_lens = torch.clamp(valid_lens, min=0, max=q_chunk_len)
        pad_flat = next_pad.view(valid_lens.shape[0], -1).any(dim=1)
        valid_lens = valid_lens * (~pad_flat).long()
        range_tensor = torch.arange(q_chunk_len, device=self.device).unsqueeze(0).expand(valid_lens.shape[0], -1)
        gt_next_pad_mask = range_tensor >= valid_lens.unsqueeze(1)

        target_samples = (
            int(target_action_samples)
            if target_action_samples is not None
            else self._resolve_backup_action_samples()
        )
        target_samples = max(1, target_samples)
        target_actor = target_policy if target_policy is not None else self.actor

        with torch.no_grad():
            if cached_next_encoding is not None:
                next_encoding = cached_next_encoding.to(self.device)
            else:
                next_raw_state = get_raw_state(
                    next_batch,
                    valid_lens.shape[0],
                    getattr(self.cfg, "raw_state_dim", self._raw_state_dim),
                    self.device,
                )
                next_encoding = self.encode_policy_for_critic(
                    target_actor,
                    next_batch,
                    next_raw_state,
                    use_target=True,
                    track_state_grad=False,
                )
            next_action, next_q, next_action_candidates = self.predict_best_of_n(
                next_batch,
                action_samples=target_samples,
                forced_mask=gt_next_pad_mask,
                return_candidates=True,
                actor=target_actor,
                encoding=next_encoding,
            )

        mask = (~next_pad).to(next_q.dtype)
        bootstrap_discount = getattr(self.cfg, "discount", 0.99) ** q_chunk_len
        targets = returns + mask * bootstrap_discount * next_q

        action_mask = get_tensor_from_batch(batch, ["actions_is_pad", "action_is_pad"], default_shape=actions.shape[:2], device=self.device)
        if action_mask.shape[-1] > q_chunk_len:
            action_mask = action_mask[..., :q_chunk_len]
        action_mask = action_mask.to(self.device).bool()

        q_values = self.critic(encoding, critic_input_actions, action_mask=critic_input_mask, raw_state=raw_state)

        calql_loss = torch.tensor(0.0, device=self.device)
        calql_metrics: Dict[str, float] = {}
        ood_loss = torch.tensor(0.0, device=self.device)
        ood_metrics: Dict[str, float] = {}
        q_ood_tensor = None
        ood_payload = None
        calql_payload = None
        use_calql = getattr(self.cfg, "use_calql", False)
        use_ood_reg = getattr(self.cfg, "use_ood_reg", False)
        if current_step is not None and current_step < ood_warmup_steps:
            use_ood_reg = False

        tau_to_use = getattr(self.cfg, "tau", 0.005)
        tau_warmup = getattr(self.cfg, "tau_warmup", None)
        tau_warmup_steps = getattr(self.cfg, "tau_warmup_steps", 0)
        if tau_warmup is not None and current_step is not None and current_step < tau_warmup_steps:
            tau_to_use = tau_warmup
        if use_calql or use_ood_reg:
            ood_actor = ood_policy if ood_policy is not None else self.actor

            def _prepare(source: str) -> Dict[str, Any]:
                if source == "erg":
                    return prepare_erg_ood_actions(
                        self, ood_actor, batch, actions, next_action_candidates, action_mask, next_pad, raw_state
                    )
                if source == "cql":
                    return prepare_cal_ood_actions(
                        self, ood_actor, batch, actions, next_action_candidates, action_mask, next_pad, raw_state
                    )
                raise ValueError(f"Unknown action source '{source}' (expected 'erg' or 'cql').")

            ood_source = str(getattr(self.cfg, "ood_action_source", "erg")).lower().strip()
            payload = _prepare(ood_source)

            if use_ood_reg:
                ood_payload = payload
            if use_calql:
                calql_payload = payload

        if use_ood_reg and ood_payload is not None:
            ood_loss, ood_metrics, q_ood_tensor = compute_explicit_penalty_loss(
                self, encoding, targets, actions, ood_payload
            )

        if use_calql and calql_payload is not None:
            calql_loss, calql_metrics = compute_calql_loss(
                self, encoding, q_values, calql_payload
            )

        loss_mode = str(getattr(self.cfg, "critic_loss_mode", "mse")).lower()
        losses = None
        if loss_mode in ("mse", "mean"):
            current_q = aggregate_q(q_values, getattr(self.cfg, "q_aggregation", "mean"))
            td_loss = F.mse_loss(current_q, targets)
        elif loss_mode in ("per_head_mean",):
            losses = [F.mse_loss(q, targets) for q in q_values]
            td_loss = torch.stack(losses).mean()
            current_q = aggregate_q(q_values, getattr(self.cfg, "q_aggregation", "mean"))
        else:
            raise ValueError(f"Unknown critic_loss_mode '{loss_mode}'.")

        total_loss = td_loss
        if use_calql:
            alpha = getattr(self.cfg, "cql_alpha", 1.0)
            total_loss = total_loss + alpha * calql_loss
        if use_ood_reg:
            ood_alpha = getattr(self.cfg, "ood_alpha", getattr(self.cfg, "cql_alpha", 1.0))
            total_loss = total_loss + ood_alpha * ood_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        grad_clip = getattr(self.cfg, "grad_clip_norm", 10.0)
        grad_clip_warmup = getattr(self.cfg, "grad_clip_warmup", None)
        if current_step is not None and current_step < ood_warmup_steps and grad_clip_warmup is not None:
            grad_clip = grad_clip_warmup
        params_to_clip = list(self.critic.parameters())
        if self.state_encoder is not None:
            params_to_clip += list(self.state_encoder.parameters())
        if self.critic_vision_encoder is not None:
            params_to_clip += list(self.critic_vision_encoder.parameters())
        if self.vision_proj is not None:
            params_to_clip += list(self.vision_proj.parameters())
        if self.lang_proj is not None:
            params_to_clip += list(self.lang_proj.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(params_to_clip, grad_clip)
        self.optimizer.step()
        if step_scheduler and self.scheduler is not None:
            self.scheduler.step()
        soft_update_target(self.target_critic, self.critic, tau_to_use)
        if self.use_critic_state_encoder and self.state_encoder is not None and self.target_state_encoder is not None:
            soft_update_target(self.target_state_encoder, self.state_encoder, tau_to_use)
        if self.use_independent_critic_encoder and self.critic_vision_encoder is not None and self.target_critic_vision_encoder is not None:
            soft_update_target(self.target_critic_vision_encoder, self.critic_vision_encoder, tau_to_use)
        if self.use_independent_critic_encoder and self.vision_proj is not None and self.target_vision_proj is not None:
            soft_update_target(self.target_vision_proj, self.vision_proj, tau_to_use)
        if self.use_independent_critic_encoder and self.lang_proj is not None and self.target_lang_proj is not None:
            soft_update_target(self.target_lang_proj, self.lang_proj, tau_to_use)
        update_time = time.perf_counter() - start

        if isinstance(q_values, (list, tuple)) and len(q_values) > 1:
            q_avg = torch.stack(q_values, dim=0).mean(dim=0)
        else:
            q_avg = q_values[0] if isinstance(q_values, (list, tuple)) else q_values
        td_error = current_q.detach() - targets.detach()
        metrics: Dict[str, float] = {
            "critic_loss": float(total_loss.item()),
            "td_loss": float(td_loss.item()),
            "critic_q": float(q_avg.mean().item()),
            "critic_lr": float(self.optimizer.param_groups[0]["lr"]),
            "critic_update_s": update_time,
            "critic_grad_norm": float(grad_norm.detach().item()),
            "critic_target_action_samples": float(target_samples),
            "critic_cached_next_encoding": float(cached_next_encoding is not None),
            "td_error_mean": float(td_error.mean().item()),
            "td_error_abs_mean": float(td_error.abs().mean().item()),
            "loss/total": float(total_loss.item()),
            "loss/td": float(td_loss.item()),
            "loss/ood": float(ood_loss if use_ood_reg else 0.0),
            "train/grad_norm": float(grad_norm.detach().item()),
        }
        if losses is not None:
            for idx, loss in enumerate(losses):
                metrics[f"critic_loss_head{idx + 1}"] = float(loss.item())
        if isinstance(q_values, (list, tuple)):
            for idx, q in enumerate(q_values):
                metrics[f"critic_q_head{idx + 1}"] = float(q.mean().item())
        metrics.update(calql_metrics)
        metrics.update(ood_metrics)
        if q_ood_tensor is not None:
            with torch.no_grad():
                q_gt = current_q.detach().view(-1, 1)
                q_ood = q_ood_tensor.detach()
                gaps_mean = q_gt - q_ood.mean(dim=1, keepdim=True)
                gaps_hard = q_gt - q_ood.max(dim=1, keepdim=True).values
                gap_metrics = {
                    "q_val/gt": float(q_gt.mean().item()),
                    "gap/mean": float(gaps_mean.mean().item()),
                    "gap/min": float(gaps_mean.min().item()),
                    "gap/hard_min": float(gaps_hard.min().item()),
                    "gap/win_rate": float((gaps_mean > 0).float().mean().item()),
                }
            metrics.update(gap_metrics)
        if next_action_candidates is not None:
            with torch.no_grad():
                act_norm_policy = torch.norm(next_action_candidates[..., :6], dim=-1).mean()
                metrics["act_norm/policy_candidates"] = float(act_norm_policy.item())
        with torch.no_grad():
            act_norm_gt = torch.norm(critic_input_actions[..., :6], dim=-1).mean()
            metrics["act_norm/gt_clean"] = float(act_norm_gt.item())
        if use_ood_reg:
            metrics["ood_alpha"] = float(getattr(self.cfg, "ood_alpha", getattr(self.cfg, "cql_alpha", 1.0)))
            metrics["ood_include_current"] = float(ood_payload.get("ood_include_current", False)) if ood_payload else 0.0
            metrics["ood_include_random"] = float(ood_payload.get("ood_include_random", False)) if ood_payload else 0.0
            metrics["ood_include_next"] = float(ood_payload.get("ood_include_next", False)) if ood_payload else 0.0
        target_head_means = getattr(self, "_last_target_head_means", None)
        if target_head_means is not None:
            for idx, value in enumerate(target_head_means):
                metrics[f"target_q_head{idx + 1}"] = float(value)
        return metrics

    # ------------------------------------------------------------------ #
    # Checkpoint
    # ------------------------------------------------------------------ #
    def state_dict(self) -> Dict[str, Any]:
        return {
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "state_encoder": self.state_encoder.state_dict() if self.state_encoder is not None else None,
            "target_state_encoder": self.target_state_encoder.state_dict() if self.target_state_encoder is not None else None,
            "critic_vision_encoder": self.critic_vision_encoder.state_dict() if self.critic_vision_encoder is not None else None,
            "target_critic_vision_encoder": self.target_critic_vision_encoder.state_dict() if self.target_critic_vision_encoder is not None else None,
            "critic_lang_embedding": self.critic_lang_embedding.state_dict() if self.critic_lang_embedding is not None else None,
            "vision_proj": self.vision_proj.state_dict() if self.vision_proj is not None else None,
            "target_vision_proj": self.target_vision_proj.state_dict() if self.target_vision_proj is not None else None,
            "lang_proj": self.lang_proj.state_dict() if self.lang_proj is not None else None,
            "target_lang_proj": self.target_lang_proj.state_dict() if self.target_lang_proj is not None else None,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "cfg": getattr(self, "cfg", None),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.critic.load_state_dict(state["critic"])
        self.target_critic.load_state_dict(state["target_critic"])
        state_enc_state = state.get("state_encoder")
        target_state_enc_state = state.get("target_state_encoder")
        vision_state = state.get("critic_vision_encoder")
        target_vision_state = state.get("target_critic_vision_encoder")
        lang_state = state.get("critic_lang_embedding")
        vision_proj_state = state.get("vision_proj")
        target_vision_proj_state = state.get("target_vision_proj")
        lang_proj_state = state.get("lang_proj")
        target_lang_proj_state = state.get("target_lang_proj")
        if self.state_encoder is not None and state_enc_state is not None:
            self.state_encoder.load_state_dict(state_enc_state)
        if self.target_state_encoder is not None:
            if target_state_enc_state is not None:
                self.target_state_encoder.load_state_dict(target_state_enc_state)
            elif self.state_encoder is not None:
                self.target_state_encoder.load_state_dict(self.state_encoder.state_dict())
        if self.critic_vision_encoder is not None and vision_state is not None:
            self.critic_vision_encoder.load_state_dict(vision_state)
        if self.target_critic_vision_encoder is not None:
            if target_vision_state is not None:
                self.target_critic_vision_encoder.load_state_dict(target_vision_state)
            elif self.critic_vision_encoder is not None:
                self.target_critic_vision_encoder.load_state_dict(self.critic_vision_encoder.state_dict())
        if self.critic_lang_embedding is not None and lang_state is not None:
            self.critic_lang_embedding.load_state_dict(lang_state)
        if self.vision_proj is not None and vision_proj_state is not None:
            self.vision_proj.load_state_dict(vision_proj_state)
        if self.target_vision_proj is not None:
            if target_vision_proj_state is not None:
                self.target_vision_proj.load_state_dict(target_vision_proj_state)
            elif self.vision_proj is not None:
                self.target_vision_proj.load_state_dict(self.vision_proj.state_dict())
        if self.lang_proj is not None and lang_proj_state is not None:
            self.lang_proj.load_state_dict(lang_proj_state)
        if self.target_lang_proj is not None:
            if target_lang_proj_state is not None:
                self.target_lang_proj.load_state_dict(target_lang_proj_state)
            elif self.lang_proj is not None:
                self.target_lang_proj.load_state_dict(self.lang_proj.state_dict())
        opt_state = state.get("optimizer")
        if opt_state is not None:
            try:
                self.optimizer.load_state_dict(opt_state)
            except Exception as exc:
                logging.warning(
                    "Failed to load critic optimizer state (will re-init optimizer): %s: %s",
                    type(exc).__name__,
                    exc,
                )
        sched_state = state.get("scheduler")
        if self.scheduler is not None and sched_state is not None:
            self.scheduler.load_state_dict(sched_state)
