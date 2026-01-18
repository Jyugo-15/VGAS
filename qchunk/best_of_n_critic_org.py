"""Self-contained Best-of-N critic trainer for chunked action policies.

This module mirrors the helper classes that previously lived inside
`lerobot_ext.train_with_critic` so that unit tests can import and exercise the
critic logic without going through the full training script.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
import time
from typing import Any, Dict, Optional, Protocol, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from qchunk.networks import CriticBackbone
from qchunk.valuequeryhead import (
    ValueHeadConfig,
    ValueHeadCritic,
    ValueQueryHead,
    ValueQueryHeadConfig,
    MYQueryValueHeadCritic,
)

class PolicyEncoderFn(Protocol):
    """Signature for helpers that build policy embeddings from raw batches."""

    def __call__(self, policy: Any, batch: Dict[str, torch.Tensor]) -> "PolicyEmbeddings":
        """Return pooled + token embeddings detached from the policy."""



@dataclass
class PolicyEmbeddings:
    """Lightweight container mirroring the structure used in the train script."""

    pooled: torch.Tensor
    prefix_outs: torch.Tensor
    pad_masks: torch.Tensor
    att_masks: torch.Tensor

    def to(self, device: torch.device | str) -> "PolicyEmbeddings":
        return PolicyEmbeddings(
            pooled=self.pooled.to(device),
            prefix_outs=self.prefix_outs.to(device),
            pad_masks=self.pad_masks.to(device),
            att_masks=self.att_masks.to(device),
        )

    def repeat(self, repeats: int) -> "PolicyEmbeddings":
        def _repeat(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.repeat_interleave(repeats, dim=0)

        return PolicyEmbeddings(
            pooled=_repeat(self.pooled),
            prefix_outs=_repeat(self.prefix_outs),
            pad_masks=_repeat(self.pad_masks),
            att_masks=_repeat(self.att_masks),
        )


class MLPCriticAdapter(nn.Module):
    """Wrap the twin-MLP critic so it consumes `PolicyEmbeddings`."""

    def __init__(self, backbone: CriticBackbone):
        super().__init__()
        self.backbone = backbone

    def forward(
        self,
        encoding: PolicyEmbeddings,
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.backbone(encoding.pooled, actions)


class ValueQueryCriticAdapter(nn.Module):
    """Wrap one or more ValueQueryHead modules and expose a twin-Q style interface."""

    def __init__(self, heads: Sequence[ValueQueryHead] | ValueQueryHead):
        super().__init__()
        if isinstance(heads, ValueQueryHead):
            heads = [heads]
        if len(heads) == 0:
            raise ValueError("ValueQueryCriticAdapter requires at least one ValueQueryHead.")
        self.heads = nn.ModuleList(heads)

    def forward(
        self,
        encoding: PolicyEmbeddings,
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        return tuple(
            head.forward_from_embeddings(
                encoding.prefix_outs,
                encoding.pad_masks,
                encoding.att_masks,
                actions,
                action_mask,
            )
            for head in self.heads
        )


class ValueHeadCriticAdapter(nn.Module):
    """Wrap one or more ValueHeadCritic modules and expose a twin-Q style interface."""

    def __init__(self, heads: Sequence[ValueHeadCritic] | ValueHeadCritic):
        super().__init__()
        if isinstance(heads, ValueHeadCritic):
            heads = [heads]
        if len(heads) == 0:
            raise ValueError("ValueHeadCriticAdapter requires at least one head.")
        self.heads = nn.ModuleList(heads)

    def forward(
        self,
        encoding: PolicyEmbeddings,
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        outputs = tuple(
            head.forward_from_embeddings(
                encoding.prefix_outs,
                encoding.pad_masks,
                encoding.att_masks,
                actions,
                action_mask,
                **kwargs,
            )
            for head in self.heads
        )
        if len(outputs) == 1:
            return outputs[0], outputs[0]
        return outputs


def _discounted_chunk_returns(rewards: torch.Tensor, reward_is_pad: torch.Tensor, discount: float) -> torch.Tensor:
    """Compute gamma-discounted sums along the chunk dimension while ignoring padded rewards."""

    if rewards.ndim == 1:
        rewards = rewards.unsqueeze(0)
    if reward_is_pad.ndim == 1:
        reward_is_pad = reward_is_pad.unsqueeze(0)
    chunk_len = rewards.shape[-1]
    device = rewards.device
    if reward_is_pad.shape[-1] != chunk_len:
        raise ValueError(
            f"`reward_is_pad` last dimension ({reward_is_pad.shape[-1]}) must match rewards chunk length ({chunk_len})."
        )
    if reward_is_pad.shape[0] != rewards.shape[0]:
        raise ValueError(
            f"`reward_is_pad` batch size ({reward_is_pad.shape[0]}) must match rewards batch size ({rewards.shape[0]})."
        )

    discounts = torch.pow(
        torch.full((chunk_len,), discount, device=device, dtype=rewards.dtype),
        torch.arange(chunk_len, device=device, dtype=rewards.dtype),
    )
    valid_mask = (~reward_is_pad.to(device=device, dtype=torch.bool)).to(rewards.dtype)
    masked_rewards = rewards * valid_mask
    return torch.sum(discounts * masked_rewards, dim=-1, keepdim=True)


def _extract_future_batch(batch: Dict[str, Any]) -> Optional[Dict[str, Any]]:

    future = {}
    if "next_observations" in batch and isinstance(batch["next_observations"], dict):
        return batch["next_observations"]

class BestOfNCriticTrainer:
    """Maintain critic + target networks and expose best-of-n bootstrapping."""

    def __init__(
        self,
        critic: nn.Module,
        target_critic: nn.Module,
        optimizer: torch.optim.Optimizer,
        cfg: Any,
        device: torch.device,
        chunk_size: int,
        action_dim: int,
        encoder_fn: PolicyEncoderFn,
        action_stats: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.critic = critic
        self.target_critic = target_critic
        self.optimizer = optimizer
        self.cfg = cfg
        self.device = device
        self.chunk_size = chunk_size
        self.flat_action_dim = action_dim
        self.action_step_dim = action_dim // chunk_size
        self.action_distance_weights = getattr(cfg, "action_distance_weights", None)
        self.use_dual_noise_ood = getattr(cfg, "use_dual_noise_ood", False)
        self.use_dual_noise_ood = getattr(cfg, "use_dual_noise_ood", False)
        self.encode_policy = encoder_fn
        self.action_stats = action_stats
        self.critic_type = getattr(cfg, "critic_type", "mlp").lower()
        self.scheduler: torch.optim.lr_scheduler.LambdaLR | None = None
        self._last_target_head_means: Optional[list[float]] = None
        self._shape_debug_done: bool = False
        self._last_policy_act_norm: float | None = None
        self._last_gt_act_norm: float | None = None
        self._raw_state_dim = getattr(cfg, "raw_state_dim", 8)

    @classmethod
    def build(
        cls,
        policy: Any,
        batch: Dict[str, torch.Tensor],
        cfg: Any,
        device: torch.device,
        encoder_fn: PolicyEncoderFn,
        action_stats: Optional[Dict[str, torch.Tensor]] = None,
    ) -> "BestOfNCriticTrainer":
        if "action" not in batch:
            raise KeyError("Batch is missing 'action' tensor required for critic training.")
        actions = batch["action"]
        # Prefer explicit q_chunk_len; fall back to policy.n_action_steps, then observed action length
        target_chunk = getattr(cfg, "q_chunk_len", None)
        if target_chunk is None:
            target_chunk = getattr(getattr(policy, "config", None), "n_action_steps", None)
        if target_chunk is None:
            target_chunk = actions.shape[-2]
        target_chunk = min(target_chunk, actions.shape[-2])
        if actions.shape[-2] > target_chunk:
            actions = actions[:, :target_chunk]
        chunk_size = target_chunk
        action_step_dim = actions.shape[-1]
        flat_action_dim = action_step_dim * chunk_size
        with torch.no_grad():
            obs_encoding = encoder_fn(policy, batch)
        obs_dim = obs_encoding.pooled.shape[-1]

        critic_type = getattr(cfg, "critic_type", "mlp").lower()
        if critic_type == "mlp":
            backbone = CriticBackbone(obs_dim, flat_action_dim, hidden_sizes=getattr(cfg, "hidden_dims", (512, 512)))
            critic = MLPCriticAdapter(backbone).to(device)
        elif critic_type == "value_query_head":
            text_config = _get_text_config(policy)
            if text_config is None:
                raise ValueError("ValueQueryHead critic requires access to the policy text_config.")
            num_q_heads = getattr(cfg, "num_q_heads", 2)
            if num_q_heads < 1:
                raise ValueError("ValueQueryHead critic requires at least one head (num_q_heads >= 1).")
            vqh_config = ValueQueryHeadConfig(
                chunk_size=chunk_size,
                action_dim=action_step_dim,
                num_backbone_layers=getattr(cfg, "vqh_num_backbone_layers", 2),
                critic_hidden_dims=getattr(cfg, "vqh_hidden_dims", (512, 512)),
                vlm_model_name=getattr(cfg, "vqh_vlm_model_name", None),
                head_type=getattr(cfg, "head_type", "mlp"),
                head_num_layers=getattr(cfg, "head_num_layers", 2),
                head_mlp_dims=getattr(cfg, "head_mlp_dims", (512, 512)),
                att_mode=getattr(cfg, "att_mode", "causal"),
            )
            heads = [ValueQueryHead(vqh_config, text_config=text_config) for _ in range(num_q_heads)]
            critic = ValueQueryCriticAdapter(heads).to(device)
        elif critic_type == "value_head":
            text_config = _get_text_config(policy)
            if text_config is None:
                raise ValueError("ValueHead critic requires access to the policy text_config.")
            vh_config = ValueHeadConfig(
                chunk_size=chunk_size,
                action_dim=action_step_dim,
                num_head_layers=getattr(cfg, "value_head_num_layers", getattr(cfg, "head_num_layers", 2)),
                head_mlp_dims=getattr(cfg, "value_head_mlp_dims", getattr(cfg, "head_mlp_dims", (512, 512))),
                vlm_model_name=getattr(cfg, "value_head_vlm_model_name", getattr(cfg, "vqh_vlm_model_name", None)),
                att_mode=getattr(cfg, "att_mode", "causal"),
            )
            head = ValueHeadCritic(vh_config, text_config=text_config)
            critic = ValueHeadCriticAdapter(head).to(device)
        elif critic_type == "my_value_query_head":
            text_config = _get_text_config(policy)
            if text_config is None:
                raise ValueError("MY ValueQueryHead critic requires access to the policy text_config.")
            use_no_query_head = getattr(cfg, "use_no_query_head", False)

            vh_config = ValueHeadConfig(
                chunk_size=chunk_size,
                action_dim=action_step_dim,
                num_head_layers=getattr(cfg, "value_head_num_layers", getattr(cfg, "head_num_layers", 2)),
                head_mlp_dims=getattr(cfg, "value_head_mlp_dims", getattr(cfg, "head_mlp_dims", (512, 512))),
                vlm_model_name=getattr(cfg, "value_head_vlm_model_name", getattr(cfg, "vqh_vlm_model_name", None)),
                att_mode=getattr(cfg, "att_mode", "causal"),
                num_query_token=getattr(cfg, "num_query_token", 16),
                use_raw_state_fusion=getattr(cfg, "use_raw_state_fusion", False),
                raw_state_dim=getattr(cfg, "raw_state_dim", 8),
                bias_init_enabled=getattr(cfg, "value_head_bias_init_enabled", False),
                bias_init_value=getattr(cfg, "value_head_bias_init_value", 0.0),
            )
            num_q_heads = getattr(cfg, "num_q_heads", 1)
            if num_q_heads < 1:
                raise ValueError("Value head critic requires at least one head (num_q_heads >= 1).")
            heads = [
                MYQueryValueHeadCritic(vh_config, text_config=text_config, use_no_query_head=use_no_query_head)
                for _ in range(num_q_heads)
            ]
            critic = ValueHeadCriticAdapter(heads).to(device)
            print(critic)
            # head = MYQueryValueHeadCritic(vh_config, text_config=text_config) # 两层transformer,同时编码query embedding以及 value embedding
            # critic = ValueHeadCriticAdapter(head).to(device)
                    
        else:
            raise ValueError(f"Unknown critic_type '{getattr(cfg, 'critic_type', 'mlp')}'.")

        target_critic = copy.deepcopy(critic).to(device)
        critic.eval()
        target_critic.eval()
        optimizer = torch.optim.Adam(
            critic.parameters(),
            lr=getattr(cfg, "lr", 3e-4),
            betas=getattr(cfg, "betas", (0.9, 0.999)),
            weight_decay=getattr(cfg, "weight_decay", 0.0),
        )
        scheduler = None
        warmup_steps = max(getattr(cfg, "lr_warmup_steps", 0), 0)
        total_steps = getattr(cfg, "lr_total_steps", None)
        final_lr = max(getattr(cfg, "lr_final", 0.0), 0.0)
        base_lr = getattr(cfg, "lr", 3e-4)
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
        print("==================================crtic config==================================")
        print(cfg)
        trainer = cls(
            critic,
            target_critic,
            optimizer,
            cfg,
            device,
            chunk_size,
            flat_action_dim,
            encoder_fn,
            action_stats=action_stats,
        )
        trainer.scheduler = scheduler
        return trainer

    def state_dict(self) -> Dict[str, Any]:
        return {
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.critic.load_state_dict(state["critic"])
        self.target_critic.load_state_dict(state["target_critic"])
        self.optimizer.load_state_dict(state["optimizer"])
        sched_state = state.get("scheduler")
        if self.scheduler is not None and sched_state is not None:
            self.scheduler.load_state_dict(sched_state)

    def update(
        self,
        policy: Any,
        batch: Dict[str, torch.Tensor],
        current_step: int | None = None,
        ood_warmup_steps: int = 0,
    ) -> Dict[str, float]:
        
        self.critic.train()
        start = time.perf_counter()
        encoding = self.encode_policy(policy, batch).to(self.device)
        actions = batch["action"].to(self.device)
        action_mask = self._get_tensor(batch, ["actions_is_pad", "action_is_pad"], default_shape=actions.shape[:2])

        if actions.shape[-2] > self.chunk_size:
            actions = actions[:, : self.chunk_size]
        if action_mask.shape[-1] > self.chunk_size:
            action_mask = action_mask[..., : self.chunk_size]
        action_mask = action_mask.to(self.device).bool()


        critic_input_actions = actions.clone()
        # pad_indices = action_mask
        # if pad_indices.any():
        #     # scale 建议 0.1 (归一化空间)
        #     noise = torch.randn_like(critic_input_actions) * 0.1
        #     mask_broadcast = pad_indices.unsqueeze(-1)
        #     # 仅填充前6维，pad 位置用噪声，valid 保持原值
        #     critic_input_actions[..., :6] = torch.where(
        #         mask_broadcast,
        #         noise[..., :6],
        #         critic_input_actions[..., :6],
        #     )
        critic_input_mask = action_mask.clone()
        if self.critic.training:
            # 获取概率，默认 0.5
            dropout_prob = getattr(self.cfg, "mask_dropout_prob", 0.5)
            
            # 生成保留掩码: True=保持原样, False=强制变为Valid
            # rand_like 生成 [0, 1]，大于 0.5 的部分保持原样
            keep_mask = torch.rand_like(critic_input_mask.float()) > dropout_prob
            
            # 逻辑与操作:
            # Valid(F) & Keep(T/F) -> Valid(F) (有效动作永远有效)
            # Pad(T)   & Keep(T)   -> Pad(T)   (保持 Mask)
            # Pad(T)   & Keep(F)   -> Valid(F) (Mask 消失了！Critic 被迫看动作值)
            critic_input_mask = critic_input_mask & keep_mask
        # 给 来自数据集的动作加一点点噪声

        rewards = self._get_tensor(batch, ["rewards"], default_shape=actions.shape[:2])

        if rewards.ndim > 1 and rewards.shape[-1] > self.chunk_size:
            rewards = rewards[..., : self.chunk_size]
        rewards = rewards.reshape(actions.shape[0], -1) # rewards => [-1,-1,-1....] chunk_len
        rewards = rewards.to(self.device)
        
        reward_is_pad = self._get_tensor(batch, ["reward_is_pad"], default_shape=actions.shape[:2])
        if reward_is_pad.shape[-1] > self.chunk_size:
            reward_is_pad = reward_is_pad[..., : self.chunk_size]
       
       
        returns = _discounted_chunk_returns(rewards, reward_is_pad, getattr(self.cfg, "discount", 0.99))
        raw_state = self._get_raw_state(batch, actions.shape[0])
        next_batch = _extract_future_batch(batch)
        if next_batch is None:
            raise ValueError("Future batch missing `next_observations`.")

        next_pad = next_batch.get("next_observation_is_pad")  # (B, 1) expected
        if next_pad is None:
            raise ValueError("`next_observation_is_pad` missing in future batch.")
        next_pad = next_pad.to(self.device).bool()

        valid_lens = next_batch.get("next_obs_valid_chunk_len")
        if valid_lens is None:
            raise ValueError("`next_obs_valid_chunk_len` missing in future batch.")
        valid_lens = valid_lens.to(self.device).long()
        if valid_lens.ndim > 1:
            valid_lens = valid_lens.squeeze(-1)
        valid_lens = torch.clamp(valid_lens, min=0, max=self.chunk_size)
        # pad 样本的有效长度置 0，避免后续 mask 出现随机值
        pad_flat = next_pad.view(valid_lens.shape[0], -1).any(dim=1)
        valid_lens = valid_lens * (~pad_flat).long()
        range_tensor = torch.arange(self.chunk_size, device=self.device).unsqueeze(0).expand(valid_lens.shape[0], -1)
        gt_next_pad_mask = range_tensor >= valid_lens.unsqueeze(1)

        with torch.no_grad():
            next_encoding = self.encode_policy(policy, next_batch).to(self.device)
            next_action, next_q, next_action_candidates = self._best_of_n_actions_soon(
                policy,
                next_batch,
                next_encoding,
                return_candidates=True,
                forced_next_mask=gt_next_pad_mask,
            )

        #########################
        mask = (~next_pad).to(next_q.dtype)
        bootstrap_discount = getattr(self.cfg, "discount", 0.99) ** self.chunk_size
        targets = returns + mask * bootstrap_discount * next_q

        action_mask = self._get_tensor(batch, ["actions_is_pad", "action_is_pad"], default_shape=actions.shape[:2])
        if action_mask.shape[-1] > self.chunk_size:
            action_mask = action_mask[..., : self.chunk_size]
        action_mask = action_mask.to(self.device).bool()


        q_values = self.critic(encoding, critic_input_actions, action_mask=critic_input_mask, raw_state=raw_state)# 对原本 看不见的动作（超出episode 范围的）加了点噪声 

        # 显式初始化，避免未定义问题
        calql_loss, calql_metrics = 0.0, {}
        ood_loss, ood_metrics = 0.0, {}
        q_ood_tensor = None
        ood_payload = None
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
            ood_payload = self._prepare_ood_actions(
                policy, batch, actions, next_action_candidates, action_mask, next_pad, raw_state
            )

        if use_ood_reg:
            if ood_payload is None:
                raise ValueError("use_ood_reg=True 但未生成 ood_payload。")
            ood_loss, ood_metrics, q_ood_tensor = self._compute_explicit_penalty_loss(
                encoding,
                targets,    # y_t (GT Anchor)
                actions,    # GT Actions
                ood_payload,
            )

        if use_calql and ood_payload is not None:
            calql_loss, calql_metrics = self._compute_calql_loss(
                encoding, q_values, ood_payload
            )

        # 控制损失如何在多个 head 之间聚合：默认先聚合 Q 再算 MSE，per-head 方式则每个 head 单独算再平均
        loss_mode = getattr(self.cfg, "critic_loss_mode", "mse")
        losses = None
        if str(loss_mode).lower() in ("mse", "mean"):
            current_q = self._aggregate_q(*q_values)
            td_loss = F.mse_loss(current_q, targets)
        elif str(loss_mode).lower() in ("per_head_mean"):
            losses = [F.mse_loss(q, targets) for q in q_values]
            td_loss = torch.stack(losses).mean()
            current_q = self._aggregate_q(*q_values)
        else:
            raise ValueError(f"Unknown critic_loss_mode '{loss_mode}'.")

        # 叠加正则项
        total_loss = td_loss
        if use_calql:
            alpha = getattr(self.cfg, "cql_alpha", 1.0)
            total_loss = total_loss + alpha * calql_loss
        if use_ood_reg:
            ood_alpha = getattr(self.cfg, "ood_alpha", getattr(self.cfg, "cql_alpha", 1.0))
            total_loss = total_loss + ood_alpha * ood_loss

        ########################
        self.optimizer.zero_grad()
        total_loss.backward()
        grad_clip = getattr(self.cfg, "grad_clip_norm", 10.0)
        grad_clip_warmup = getattr(self.cfg, "grad_clip_warmup", None)
        if current_step is not None and current_step < ood_warmup_steps and grad_clip_warmup is not None:
            grad_clip = grad_clip_warmup
        grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), grad_clip)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self._soft_update_target(tau_to_use)
        update_time = time.perf_counter() - start

        if len(q_values) == 1:
            q_avg = q_values[0]
        else:
            q_avg = torch.stack(q_values, dim=0).mean(dim=0)
        td_error = current_q.detach() - targets.detach()
        metrics: Dict[str, float] = {
            "critic_loss": float(total_loss.item()),
            "td_loss": float(td_loss.item()),
            "critic_q": float(q_avg.mean().item()),
            "critic_lr": float(self.optimizer.param_groups[0]["lr"]),
            "critic_update_s": update_time,
            "critic_grad_norm": float(grad_norm.detach().item()),
            "td_error_mean": float(td_error.mean().item()),
            "td_error_abs_mean": float(td_error.abs().mean().item()),
            # Group A aliases for dashboard
            "loss/total": float(total_loss.item()),
            "loss/td": float(td_loss.item()),
            "loss/ood": float(ood_loss if use_ood_reg else 0.0),
            "train/grad_norm": float(grad_norm.detach().item()),
        }
        if losses is not None:
            for idx, loss in enumerate(losses):
                metrics[f"critic_loss_head{idx + 1}"] = float(loss.item())
        for idx, q in enumerate(q_values):
            metrics[f"critic_q_head{idx + 1}"] = float(q.mean().item())
        metrics.update(calql_metrics)
        metrics.update(ood_metrics)
        # Group C: gap metrics when OOD is enabled
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
        # 动作幅度监控：policy 候选 vs 清洗后的 GT
        if next_action_candidates is not None:
            with torch.no_grad():
                act_norm_policy = torch.norm(next_action_candidates[..., :6], dim=-1).mean()
                metrics["act_norm/policy_candidates"] = float(act_norm_policy.item())
        with torch.no_grad():
            act_norm_gt = torch.norm(critic_input_actions[..., :6], dim=-1).mean()
            metrics["act_norm/gt_clean"] = float(act_norm_gt.item())
        if use_ood_reg:
            metrics["ood_alpha"] = float(getattr(self.cfg, "ood_alpha", getattr(self.cfg, "cql_alpha", 1.0)))
            metrics["ood_m_actions"] = float(ood_payload["ood_m_actions"]) if ood_payload is not None else 0.0
            metrics["ood_include_current"] = float(ood_payload.get("ood_include_current", False)) if ood_payload else 0.0
            metrics["ood_include_random"] = float(ood_payload.get("ood_include_random", False)) if ood_payload else 0.0
            metrics["ood_include_next"] = float(ood_payload.get("ood_include_next", False)) if ood_payload else 0.0
        target_head_means = getattr(self, "_last_target_head_means", None)
        if target_head_means is not None:
            for idx, value in enumerate(target_head_means):
                metrics[f"target_q_head{idx + 1}"] = float(value)
        return metrics

    def _prepare_ood_actions(
        self,
        policy: Any,
        batch: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        next_action_candidates: torch.Tensor,
        action_mask: torch.Tensor,
        next_pad: torch.Tensor,
        raw_state: torch.Tensor,
    ) -> Dict[str, Any]:
        use_calql = getattr(self.cfg, "use_calql", False)
        # 默认：CalQL 保持原策略（current+random+next），纯 OOD 则只用 current
        include_current = getattr(self.cfg, "ood_include_current_actions", True)
        default_extra = True if use_calql else False
        include_random = getattr(self.cfg, "ood_include_random_actions", default_extra)
        include_next = getattr(self.cfg, "ood_include_next_actions", default_extra if use_calql else False)
        if use_calql:
            include_next = True  # CalQL 依赖 next

        if "mc_lower_bound" not in batch:
            raise ValueError("`mc_lower_bound` is required when use_calql=True.")
        mc_lower_bound = batch["mc_lower_bound"]
        if isinstance(mc_lower_bound, torch.Tensor) and mc_lower_bound.ndim > 1:
            mc_lower_bound = mc_lower_bound[:, 0]

        next_action_candidates = next_action_candidates.to(self.device)
        ood_m_actions = getattr(self.cfg, "cql_m_actions", None)
        if ood_m_actions is None:
            ood_m_actions = next_action_candidates.shape[1]
        ood_m_actions = max(1, min(ood_m_actions, next_action_candidates.shape[1]))
        # 统一的加噪 helper，可按统计量尺度添加噪声
        def _maybe_add_noise(actions_to_perturb: torch.Tensor, noise_std: float) -> torch.Tensor:
            if noise_std <= 0:
                return actions_to_perturb
            stats_for_noise = self.action_stats or {}
            act_mean_noise = stats_for_noise.get("mean")
            act_std_noise = stats_for_noise.get("std")
            if act_mean_noise is not None and act_std_noise is not None:
                mean = torch.as_tensor(act_mean_noise, device=self.device, dtype=actions.dtype).view(1, 1, 1, -1)
                std = torch.as_tensor(act_std_noise, device=self.device, dtype=actions.dtype).view(1, 1, 1, -1).clamp_min(1e-8)
                raw = actions_to_perturb * std + mean
                raw = raw + torch.randn_like(raw) * (noise_std * std)
                return (raw - mean) / std
            return actions_to_perturb + torch.randn_like(actions_to_perturb) * noise_std

        # 从 next_action_candidates 中采样（可选）
        ood_next_actions = None
        if include_next:
            sample_idx = torch.randperm(next_action_candidates.shape[1], device=self.device)[:ood_m_actions]
            ood_next_actions = next_action_candidates[:, sample_idx]  # (batch, m, chunk, action_dim)
            noise_std = getattr(self.cfg, "cql_next_noise_std", 0.05)
            if noise_std is None:
                noise_std = 0.05
            if noise_std > 0:
                ood_next_actions = _maybe_add_noise(ood_next_actions, noise_std)

        # 随机动作（可选）
        ood_random_actions = None
        if include_random:
            stats = self.action_stats or {}
            act_min = stats.get("min")
            act_max = stats.get("max")
            act_mean = stats.get("mean")
            act_std = stats.get("std")
            if act_min is not None and act_max is not None and act_mean is not None and act_std is not None:
                act_min = torch.as_tensor(act_min, device=self.device, dtype=actions.dtype).view(1, 1, 1, -1)
                act_max = torch.as_tensor(act_max, device=self.device, dtype=actions.dtype).view(1, 1, 1, -1)
                act_mean = torch.as_tensor(act_mean, device=self.device, dtype=actions.dtype).view(1, 1, 1, -1)
                act_std = torch.as_tensor(act_std, device=self.device, dtype=actions.dtype).view(1, 1, 1, -1)
                raw_random = torch.rand(
                    (actions.shape[0], ood_m_actions, self.chunk_size, self.action_step_dim),
                    device=self.device,
                    dtype=actions.dtype,
                ) * (act_max - act_min) + act_min
                ood_random_actions = (raw_random - act_mean) / act_std.clamp_min(1e-8)
            else:
                ood_random_actions = torch.rand(
                    (actions.shape[0], ood_m_actions, self.chunk_size, self.action_step_dim),
                    device=self.device,
                    dtype=actions.dtype,
                ) * 2.0 - 1.0  # bs * m * chunklen * actiondim (clip)
        # 当前 observation 的 policy 动作，尽量复用已有预测减少额外开销；否则重复 batch 采样多条

        with torch.no_grad():
            batch_size = actions.shape[0]
            expanded_batch = self._repeat_batch(batch, batch_size, ood_m_actions)
            expanded_actions = policy.predict_action_chunk(expanded_batch)
        cur = expanded_actions.to(self.device)
        if cur.shape[1] > self.chunk_size:
            cur = cur[:, : self.chunk_size]
        # 关键：还原回 (B, m, chunk, dim)
        cur = cur.view(batch_size, ood_m_actions, self.chunk_size, -1)
        ood_current_actions = cur
        # 给 current action 也加噪声，默认沿用 next 噪声幅度
        cur_noise_std = getattr(self.cfg, "cql_cur_noise_std", None)
        if cur_noise_std is None:
            cur_noise_std = noise_std
        if cur_noise_std > 0:
            ood_current_actions = _maybe_add_noise(ood_current_actions, cur_noise_std)
        cql_batch_action = actions
        return {
            "mc_lower_bound": mc_lower_bound,
            "ood_next_actions": ood_next_actions,
            "ood_random_actions": ood_random_actions,
            "ood_m_actions": ood_m_actions,
            "cql_batch_action": cql_batch_action,
            "ood_current_actions": ood_current_actions,
            "action_mask": action_mask.to(self.device).bool(),
            "next_valid_mask": (~next_pad).to(self.device).float(),
            "raw_state": raw_state.to(self.device),
            "ood_include_next": include_next,
            "ood_include_random": include_random,
            "ood_include_current": include_current,
        }

    def _compute_calql_loss(
        self,
        encoding: PolicyEmbeddings,
        q_values: tuple[torch.Tensor, ...],
        calql_payload: Dict[str, Any],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        # 当前 batch 动作的 Q
        q_batch = self._aggregate_q(*q_values).squeeze(-1) # bs * 1
        # if q_batch.ndim > 1:
        #     q_batch = q_batch.mean(dim=-1)  # 聚合 chunk 维度

        m = calql_payload["ood_m_actions"]
        mc_lower_bound = calql_payload["mc_lower_bound"].to(self.device)
        mc_lower_bound = mc_lower_bound.view(-1, 1)  # (B,1)
        action_mask = calql_payload["action_mask"].to(self.device).bool()
        next_valid_mask = calql_payload["next_valid_mask"].to(self.device)
        raw_state_base = calql_payload.get("raw_state")
        if raw_state_base is None:
            raw_state_base = torch.zeros(q_batch.shape[0], self._raw_state_dim, device=self.device, dtype=q_batch.dtype)
        raw_state_base = raw_state_base.to(self.device)

        def _eval_actions(act: torch.Tensor) -> torch.Tensor:
            num_actions = act.shape[1]
            flat = act.view(-1, self.chunk_size, self.action_step_dim)
            rep_encoding = encoding.repeat(num_actions)
            mask_rep = (
                action_mask.unsqueeze(1)
                .repeat(1, num_actions, 1)
                .view(-1, self.chunk_size)
            )
            raw_rep = raw_state_base.repeat_interleave(num_actions, dim=0)
            q = self._aggregate_q(*self.critic(rep_encoding, flat, action_mask=mask_rep, raw_state=raw_rep))
            q = q.view(act.shape[0], num_actions, -1).squeeze(-1)
            return q

        to_eval = [
            calql_payload["ood_next_actions"],
            calql_payload["ood_current_actions"],
        ]
        concatenated = torch.cat(to_eval, dim=1)
        evaluated = _eval_actions(concatenated)
        splits = [tensor.shape[1] for tensor in to_eval]
        q_next_raw, q_cur_raw = torch.split(evaluated, splits, dim=1) # 去掉了ran Q
        q_next = torch.maximum(q_next_raw, mc_lower_bound) # BS * m
        q_cur  = torch.maximum(q_cur_raw, mc_lower_bound)
        # mask 掉 next 已 pad 的样本：无效的 next 设为极小，避免 logsumexp 里贡献 exp(0)
        very_neg = -1e9
        next_valid = next_valid_mask.expand_as(q_next)
        q_next = q_next * next_valid + very_neg * (1.0 - next_valid)
        # 真实 batch 动作的 Q，压成 (B,1) 参与 logsumexp
        q_batch_col = q_batch
        if q_batch_col.ndim > 1:
            q_batch_col = q_batch_col.mean(dim=-1)
        q_batch_col = q_batch_col.view(-1, 1)
        
        # log-sum-exp 聚合 OOD actions
        temp = getattr(self.cfg, "cql_temp", 1.0)
        ood_actions_q = torch.cat([ q_next, q_cur, q_batch_col], dim=1)  # (B, 2m+1)
        cql_ood = torch.logsumexp(ood_actions_q / temp, dim=1) * temp
        cql_ood_mean = cql_ood.mean()
        cql_diff = cql_ood - q_batch
        calql_loss = cql_diff.mean()

        # stats for diagnostics
        valid_next = next_valid  # (B, m) already shaped
        valid_next_sum = valid_next.sum()
        if valid_next_sum.item() > 0:
            bound_rate_next = float(((q_next_raw < mc_lower_bound) * valid_next.bool()).sum().item() / valid_next_sum.item())
        else:
            bound_rate_next = 0.0
        bound_rate_cur = float((q_cur_raw < mc_lower_bound).float().mean().item())
        # q_rand_mean = float(q_rand.mean().item())
        q_next_mean = float(q_next_raw.mean().item())
        q_cur_mean = float(q_cur_raw.mean().item())

        calql_metrics = {
            "calql_loss": float(calql_loss.detach().cpu().item()),
            "calql_ood_q": float(cql_ood_mean.detach().cpu().item()),
            "calql_q_batch": float(q_batch.detach().mean().cpu().item()),
            "calql_q_diff": float(calql_loss.detach().cpu().item()),
            "calql_ood_q_mean": float(cql_ood_mean.detach().cpu().item()),
            # "calql_rand_q_mean": q_rand_mean,
            "calql_next_q_mean": q_next_mean,
            "calql_cur_q_mean": q_cur_mean,
            "calql_bound_rate_next": bound_rate_next,
            "calql_bound_rate_cur": bound_rate_cur,
        }
        return calql_loss, calql_metrics

    def _compute_weighted_distance(
                self, 
                pred_actions: torch.Tensor, 
                gt_actions: torch.Tensor, 
                pad_mask: torch.Tensor # True 表示是 Padding
            ) -> torch.Tensor:
                """
                计算加权距离，并忽略 padding 部分
                pred: (B, M, T, D)
                gt:   (B, T, D)
                pad_mask: (B, M, T) 或 (B, T) - True=Pad
                """
                # 1. 维度对齐
                if gt_actions.ndim == 3:
                    gt_actions = gt_actions.unsqueeze(1) # (B, 1, T, D)
                
                # 2. 计算基础差异
                diff = (pred_actions - gt_actions) ** 2
                
                weights_list = self.action_distance_weights
                if weights_list is None:
                    # 默认加权 (Grip > Pos > Rot)
                    weights_list = [5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0]
                if isinstance(weights_list, torch.Tensor):
                    weights_tensor = weights_list.to(device=pred_actions.device, dtype=pred_actions.dtype)
                    if weights_tensor.numel() != diff.shape[-1]:
                        raise ValueError(f"action_distance_weights length ({weights_tensor.numel()}) must match action dim ({diff.shape[-1]}).")
                    weights = weights_tensor.view(1, 1, 1, -1)
                else:
                    if len(weights_list) != diff.shape[-1]:
                        raise ValueError(f"action_distance_weights length ({len(weights_list)}) must match action dim ({diff.shape[-1]}).")
                    weights = torch.tensor(weights_list, device=pred_actions.device, dtype=pred_actions.dtype).view(1, 1, 1, -1)
                
                weighted_diff = diff * weights
                
                # 4. 在 Action Dim 求和 -> (B, M, T)
                dist_per_step = weighted_diff.sum(dim=-1)
                
                # 5. 处理 Mask
                #    确保 mask 维度匹配 (B, M, T)
                if pad_mask.ndim == 2:
                    pad_mask = pad_mask.unsqueeze(1)
                if pad_mask.shape[1] != dist_per_step.shape[1]:
                    # 如果 mask 的 M 维度是 1，广播它
                    pad_mask = pad_mask.expand_as(dist_per_step)

                valid_mask = ~pad_mask.bool()
                
                #    把 padding 部分的距离置为 0
                dist_masked = dist_per_step * valid_mask 
                
                # 6. 计算平均值 (Sum / Count)
                valid_count = valid_mask.sum(dim=-1).float()
                valid_count = torch.clamp(valid_count, min=1.0) # 防止除0
                
                dist = dist_masked.sum(dim=-1) / valid_count # (B, M)
                
                return dist

    def _compute_explicit_penalty_loss(
        self,
        encoding: PolicyEmbeddings,
        gt_bellman_target: torch.Tensor,  # y_t (B, 1)
        gt_actions: torch.Tensor,         # a_gt (B, T, D)
        calql_payload: Dict[str, Any],
        ) -> tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
        
        # --- 准备工作 ---
        # 获取 Frozen Policy 的动作 (这是最重要的评估对象)
        policy_actions = calql_payload["ood_current_actions"]
        raw_state_base = calql_payload.get("raw_state")
        
        # 获取 Pad Mask (True=Pad, False=Valid)
        gt_pad_mask = calql_payload["action_mask"] 
        
        B, T, D = gt_actions.shape
        

        m = policy_actions.shape[1]

        # =================================================================
        # 1. 构造负样本 (Negative Samples)
        # =================================================================

        if self.use_dual_noise_ood:
            # 方案 B：双噪声 GT（极小+小）
            noise_tiny = torch.randn_like(gt_actions) * 0.005
            bad_act_tiny = (gt_actions + noise_tiny).unsqueeze(1)

            noise_small = torch.randn_like(gt_actions) * 0.02
            bad_act_small = (gt_actions + noise_small).unsqueeze(1)

            ood_actions = torch.cat([
                policy_actions,
                bad_act_tiny,
                bad_act_small,
            ], dim=1)
            trunc_region = gt_pad_mask  # placeholder for mask shapes downstream
            dual_noise_mode = True
        else:
            # 方案 A：小噪声 + 截断
            noise_small = torch.randn_like(gt_actions) * 0.02
            bad_act_precision = (gt_actions + noise_small).unsqueeze(1)

            temp_trunc_act = gt_actions.clone()
            start_idx = int(T * 0.5)
            end_idx = int(T * 0.95)
            cut_indices = torch.randint(start_idx, max(start_idx + 1, end_idx), (B,), device=self.device)
            range_tensor = torch.arange(T, device=self.device).unsqueeze(0)
            trunc_region = range_tensor >= cut_indices.unsqueeze(1)
            temp_trunc_act[trunc_region] = 0.0
            bad_act_truncated = temp_trunc_act.unsqueeze(1)

            ood_actions = torch.cat([
                policy_actions, 
                bad_act_precision, 
                bad_act_truncated
            ], dim=1)
            dual_noise_mode = False
        
        # [关键修正] 动态获取总样本数 M_total
        B, M_total, T, D = ood_actions.shape
        m_policy = policy_actions.shape[1] # 获取 Policy 采样的个数 (例如 2)
        if raw_state_base is None:
            raw_state_base = torch.zeros(B, self._raw_state_dim, device=self.device, dtype=gt_actions.dtype)
        raw_state_base = raw_state_base.to(self.device)

        # =================================================================
        # 3. 构造 Mask 堆栈 (动态计算 repeat 次数)
        # =================================================================
        
        if dual_noise_mode:
            # 所有样本使用原始 Mask: [Policy(m), Tiny(1), Small(1)]
            num_standard_masks = m_policy + 2
            mask_standard = gt_pad_mask.unsqueeze(1).repeat(1, num_standard_masks, 1) # (B, m+2, T)
            ood_forward_mask = mask_standard
        else:
            # 前面 (m + 1) 个样本使用原始 Mask: [Policy(m), Precision(1)]
            num_standard_masks = m_policy + 1
            mask_standard = gt_pad_mask.unsqueeze(1).repeat(1, num_standard_masks, 1) # (B, m+1, T)
            
            # 最后一个是截断样本 Mask
            # 逻辑：原始Pad | 人为截断Pad
            mask_trunc_sample = (gt_pad_mask | trunc_region).unsqueeze(1) # (B, 1, T)
            
            # 拼起来: (B, M_total, T)
            ood_forward_mask = torch.cat([mask_standard, mask_trunc_sample], dim=1)
        
        # =================================================================
        # 4. 构造计算距离用的 Mask
        # =================================================================
        
        if dual_noise_mode:
            # 所有样本计算距离时，只忽略原始 Pad
            mask_dist_standard = gt_pad_mask.unsqueeze(1).repeat(1, num_standard_masks, 1)
            calc_dist_mask = mask_dist_standard
        else:
            # 前 (m + 1) 个样本计算距离时，忽略原始 Pad
            mask_dist_standard = gt_pad_mask.unsqueeze(1).repeat(1, num_standard_masks, 1)
            
            # 截断样本计算距离时，只忽略原始 Pad (截断区域视为 Valid)
            mask_dist_trunc = gt_pad_mask.unsqueeze(1)
            
            calc_dist_mask = torch.cat([mask_dist_standard, mask_dist_trunc], dim=1)

        # =================================================================
        # 5. 计算距离 & Target
        # =================================================================
        dist = self._compute_weighted_distance(ood_actions, gt_actions, pad_mask=calc_dist_mask)

        beta = getattr(self.cfg, "dist_penalty_beta", 0.5)
        
        # 扩展 GT target: (B, 1) -> (B, M_total)
        y_t = gt_bellman_target.detach().view(B, 1).expand(B, M_total)
        dist = torch.clamp(dist, max=5.0) 
        target_ood = y_t - beta * dist

        # =================================================================
        # 6. Critic Forward
        # =================================================================
        flat_ood = ood_actions.view(-1, T, D)
        
        # [关键修正] 展平 Mask
        flat_mask = ood_forward_mask.view(-1, T)

        # ood_actions flatten order (B, M_total, T, D) -> (B*M_total, T, D) aligns with repeat_interleave on batch dim.
        # Do not use .repeat(M_total, 1) here, which would interleave batches instead of candidates.
        raw_state_ood = raw_state_base.repeat_interleave(M_total, dim=0)
        
        # [关键修正] 动态 Repeat Encoding
        # 原来写死了 encoding.repeat(4)，现在改成 M_total
        rep_encoding = encoding.repeat(M_total) 

        q_ood_values = self.critic(rep_encoding, flat_ood, action_mask=flat_mask, raw_state=raw_state_ood)
        ood_losses = []
        for q_val in q_ood_values:
            # q_val: (B*M_total, 1) -> (B, M_total)
            q_view = q_val.view(B, M_total)
            # 计算当前 Head 的 MSE
            # loss = F.mse_loss(q_view, target_ood.detach()) # target 必须 detach
            loss = F.huber_loss(q_view, target_ood.detach(), delta=10.0)
            ood_losses.append(loss)

        loss_ood = torch.stack(ood_losses).mean()
        # 聚合 Q (B, M_total)
        

        # loss_ood = F.mse_loss(q_ood_pred, target_ood)

        with torch.no_grad():
            # 索引切片
            q_ood_pred_for_log = self._aggregate_q(*q_ood_values).view(B, M_total)
            idx_pol = m_policy
            idx_prec = m_policy + 1
            idx_trunc = idx_prec  # 截断样本从 idx_prec 开始
            q_policy_mean = q_ood_pred_for_log[:, :idx_pol].mean().item()
            q_prec_mean = q_ood_pred_for_log[:, idx_pol:idx_prec].mean().item()
            q_trunc_mean = q_ood_pred_for_log[:, idx_trunc:].mean().item()

            d_policy_mean = dist[:, :idx_pol].mean().item()
            d_prec_mean = dist[:, idx_pol:idx_prec].mean().item()
            d_trunc_mean = dist[:, idx_trunc:].mean().item()

            q_ood_avg = q_ood_pred_for_log.mean().item()
            q_ood_max = q_ood_pred_for_log.max(dim=1).values.mean().item()
            # 辅助：胜率与间隔
            q_gt = gt_bellman_target.detach().view(B, 1)
            q_policy_block = q_ood_pred_for_log[:, :idx_pol] if idx_pol > 0 else None
            dist_policy = dist[:, :idx_pol] if idx_pol > 0 else None
            q_trunc_block = q_ood_pred_for_log[:, idx_trunc:] if idx_trunc < q_ood_pred_for_log.shape[1] else None
            win_vs_policy = float(
                (q_gt > q_policy_block.mean(dim=1, keepdim=True)).float().mean().item()
            ) if q_policy_block is not None else 0.0
            win_vs_trunc = float(
                (q_gt > q_trunc_block.mean(dim=1, keepdim=True)).float().mean().item()
            ) if q_trunc_block is not None else 0.0
            win_policy_vs_trunc = float(
                (q_policy_block.mean(dim=1, keepdim=True) > q_trunc_block.mean(dim=1, keepdim=True)).float().mean().item()
            ) if (q_policy_block is not None and q_trunc_block is not None) else 0.0
            # policy 候选内部胜率分布（谁在 policy block 里得分最高）
            policy_win_rates: dict[str, float] = {}
            if q_policy_block is not None:
                win_idx = torch.argmax(q_policy_block, dim=1)
                for cand in range(q_policy_block.shape[1]):
                    policy_win_rates[f"win_rate/policy_cand{cand}"] = float(
                        (win_idx == cand).float().mean().item()
                    )
            gap_policy = float((q_gt.mean() - q_policy_block.mean()).item()) if q_policy_block is not None else 0.0
            gap_trunc = float((q_gt.mean() - q_trunc_block.mean()).item()) if q_trunc_block is not None else 0.0
            # 多头分歧
            head_diff_ood = 0.0
            if len(q_ood_values) > 1:
                head_diff_ood = float(
                    (q_ood_values[0].view(B, M_total) - q_ood_values[1].view(B, M_total)).abs().mean().item()
                )
            # 对齐指标：policy 候选内部，最像 GT 的是否也是 Q 最高
            align_top1 = 0.0
            align_rank = 0.0
            if q_policy_block is not None and m > 0 and dist_policy is not None:
                best_sim_idx = torch.argmin(dist_policy, dim=1)  # (B,)
                best_q_idx = torch.argmax(q_policy_block, dim=1)  # (B,)
                align_top1 = float((best_sim_idx == best_q_idx).float().mean().item())
                q_ranking = torch.argsort(q_policy_block, dim=1, descending=True)  # (B, m)
                rank_pos = torch.argmax((q_ranking == best_sim_idx.unsqueeze(1)).float(), dim=1)  # (B,)
                align_rank = float(rank_pos.float().mean().item())

        metrics = {
            "ood_loss": float(loss_ood.item()),
            "ood_q_mean": float(q_ood_pred_for_log.mean().item()),
            "ood_dist_mean": float(dist.mean().item()),
            "ood_target_mean": float(target_ood.mean().item()),
            "ood_beta": beta,
            "ood_dist_trunc": float(dist[:, -1].mean().item()),
            # 胜率与间隔
            "win_rate/gt_vs_policy": win_vs_policy,
            "win_rate/gt_vs_trunc": win_vs_trunc,
            "gap/gt_vs_policy": gap_policy,
            "gap/gt_vs_trunc": gap_trunc,
            "win_rate/policy_vs_trunc": win_policy_vs_trunc,
            "align/top1_policy_best_sim": align_top1,
            "align/rank_policy_best_sim": align_rank,
            # 多头分歧
            "stability/head_diff_ood": head_diff_ood,
            # Group B: Value ladder
            "q_val/ood_avg": q_ood_avg,
            "q_val/ood_policy": q_policy_mean,
            "q_val/ood_prec": q_prec_mean,
            "q_val/ood_trunc": q_trunc_mean,
            "q_val/ood_hardest": q_ood_max,
            # Group D: distance checks
            "dist/policy": d_policy_mean,
            "dist/prec": d_prec_mean,
            "dist/trunc": d_trunc_mean,
            # "param/beta": beta,
            # 原始惩罚刻度
            "debug/raw_penalty_score": float((beta * dist.mean()).item()),
        }
        metrics.update(policy_win_rates)
        return loss_ood, metrics, q_ood_pred_for_log.detach()

    def _soft_update_target(self, tau: float | None = None) -> None:
        tau = getattr(self.cfg, "tau", 0.005) if tau is None else tau
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters(), strict=True):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def _aggregate_q(self, *qs: torch.Tensor) -> torch.Tensor:
        if not qs:
            raise ValueError("At least one Q tensor is required for aggregation.")
        if len(qs) == 1:
            return qs[0]
        stacked = torch.stack(qs, dim=0)
        agg = str(getattr(self.cfg, "q_aggregation", "mean")).lower()
        if agg == "min":
            return torch.min(stacked, dim=0).values
        if agg == "max":
            return torch.max(stacked, dim=0).values
        return stacked.mean(dim=0)

    def _best_of_n_actions_soon(
        self,
        policy: Any,
        batch: Dict[str, torch.Tensor],
        obs_encoding: PolicyEmbeddings,
        return_candidates: bool = False,
        forced_next_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        first_tensor = next((value for value in batch.values() if isinstance(value, torch.Tensor)), None)
        if first_tensor is None:
            raise ValueError("Future batch does not contain tensor observations.")
        batch_size = first_tensor.shape[0]
        encoding = obs_encoding.to(self.device)
        raw_state = self._get_raw_state(batch, batch_size)
        action_samples = getattr(self.cfg, "action_samples", 2)
        if action_samples <= 1:
            actions = policy.predict_action_chunk(batch)
            zero_mask = torch.zeros(actions.shape[:2], dtype=torch.bool, device=self.device)
            eval_mask = zero_mask
            if forced_next_mask is not None:
                forced = forced_next_mask.to(self.device).bool()
                try:
                    forced = forced.view_as(zero_mask)
                except RuntimeError:
                    raise ValueError(
                        f"forced_next_mask shape {tuple(forced.shape)} incompatible with actions mask {tuple(zero_mask.shape)}"
                    )
                eval_mask = forced
            q_values = self.target_critic(encoding, actions, action_mask=eval_mask, raw_state=raw_state)
            self._last_target_head_means = [q.mean().item() for q in q_values]
            best_q = self._aggregate_q(*q_values)
            if return_candidates:
                candidates = actions.unsqueeze(1)  # (batch, 1, chunk, action_dim)
                return actions, best_q, candidates
            return actions, best_q

        expanded_batch = self._repeat_batch(batch, batch_size, action_samples)## b1, b2 = b1,b1,b2,b2
        
        with torch.no_grad():
            expanded_actions = policy.predict_action_chunk(expanded_batch)
        if expanded_actions.shape[1] != self.chunk_size:
            if expanded_actions.shape[1] < self.chunk_size:
                raise ValueError(
                    f"Policy chunk ({expanded_actions.shape[1]}) shorter than critic chunk_size ({self.chunk_size})."
                )
            expanded_actions = expanded_actions[:, : self.chunk_size] # 32 * 50 *7 => 32 * 10 * 7

        action_dim = expanded_actions.shape[-1]
        stacked = expanded_actions.view(batch_size, action_samples, self.chunk_size, action_dim) # b1_c1,b1_c2,b1_c3,b2_c1,b2_c2,b3_c3 => [[b1_c1,b1_c2,b1_c2],[b2_c1,b2_c2,b2_c3]]
        flat_actions = stacked.view(-1, self.chunk_size, action_dim)
        if not self._shape_debug_done:
            print("[BestOfN] stacked shape:", stacked.shape, "flat_actions shape:", flat_actions.shape)
            self._shape_debug_done = True

        repeated_encoding = encoding.repeat(action_samples) # b1,b2 => b1,b1,b2,b2
        zero_mask = torch.zeros(flat_actions.shape[:2], dtype=torch.bool, device=self.device) # zero mask代表全是有效的 action
        eval_mask = zero_mask
        if forced_next_mask is not None:
            forced = forced_next_mask.to(self.device).bool()
            try:
                forced = forced.view(batch_size, self.chunk_size)
            except RuntimeError:
                raise ValueError(
                    f"forced_next_mask shape {tuple(forced.shape)} incompatible with expected {(batch_size, self.chunk_size)}"
                )
            forced = forced.unsqueeze(1).expand(-1, action_samples, -1).reshape(-1, self.chunk_size)
            eval_mask = forced
        raw_state_rep = raw_state.repeat_interleave(action_samples, dim=0)
        q_values = self.target_critic(repeated_encoding, flat_actions, action_mask=eval_mask, raw_state=raw_state_rep) # (h1=> [b1_c1,b1_c2],[b2,c1],[b2,c2]) (h2=> [b1_c1,b1_c2],[b2,c1],[b2,b2])
        self._last_target_head_means = [q.mean().item() for q in q_values] 
        q = self._aggregate_q(*q_values) # 将不同q_network的q合在一起
        q = q.view(batch_size, action_samples, -1)
        best_indices = torch.argmax(q, dim=1).squeeze(-1)
        batch_indices = torch.arange(batch_size, device=self.device)
        best_actions = stacked[batch_indices, best_indices]
        best_q = q[ batch_indices, best_indices]
        if best_q.ndim == 1:
            best_q = best_q.unsqueeze(-1)
        if return_candidates:
            return best_actions, best_q, stacked
        return best_actions, best_q

    def _best_of_n_actions(
        self,
        policy: Any,
        batch: Dict[str, torch.Tensor],
        obs_encoding: PolicyEmbeddings,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        first_tensor = next((value for value in batch.values() if isinstance(value, torch.Tensor)), None)
        if first_tensor is None:
            raise ValueError("Future batch does not contain tensor observations.")
        batch_size = first_tensor.shape[0]
        encoding = obs_encoding.to(self.device)
        raw_state = self._get_raw_state(batch, batch_size)
        action_samples = getattr(self.cfg, "action_samples", 2)
        if action_samples <= 1:
            actions = policy.predict_action_chunk(batch)
            zero_mask = torch.zeros(actions.shape[:2], dtype=torch.bool, device=self.device)
            q_values = self.target_critic(encoding, actions, action_mask=zero_mask, raw_state=raw_state)
            self._last_target_head_means = [q.mean().item() for q in q_values]
            best_q = self._aggregate_q(*q_values)
            return actions, best_q

        candidates = []
        temperature = getattr(self.cfg, "temperature", 1.0)
        with torch.no_grad():
            # ********
            for _ in range(action_samples): # 先不管tempurature
                # noise = torch.randn(
                #     batch_size,
                #     self.chunk_size,
                #     self.action_step_dim,
                #     device=self.device,
                # ) * temperature
                candidate = policy.predict_action_chunk(batch)
                if candidate.shape[1] != self.chunk_size:
                    if candidate.shape[1] < self.chunk_size:
                        raise ValueError(
                            f"Policy chunk ({candidate.shape[1]}) shorter than critic chunk_size ({self.chunk_size})."
                        )
                    candidate = candidate[:, : self.chunk_size]
                if not self._shape_debug_done:
                    print(
                        "[BestOfN] policy candidate shape:",
                        candidate.shape,
                        "chunk_size=",
                        self.chunk_size,
                        "action_step_dim=",
                        self.action_step_dim,
                    )
                candidates.append(candidate)
            # # noise = 
            # policy.predict_action_chunk(batch)
        stacked = torch.stack(candidates, dim=0)  # (n_samples, batch, chunk, action_dim)
        flat_actions = stacked.view(-1, self.chunk_size, self.action_step_dim) #要改，感觉有问题
        if not self._shape_debug_done:
            print("[BestOfN] stacked shape:", stacked.shape, "flat_actions shape:", flat_actions.shape)
            self._shape_debug_done = True
        # （b0_sample0,b1_sample_0,b0_sample_1,b1_samlle1...b1_sample_n-1）
        repeated_encoding = encoding.repeat(action_samples)
        # (b0_ob,b1_ob,b0_ob,b1_ob...b1_ob）
        zero_mask = torch.zeros(flat_actions.shape[:2], dtype=torch.bool, device=self.device)
        raw_state_rep = raw_state.repeat_interleave(action_samples, dim=0)
        q_values = self.target_critic(repeated_encoding, flat_actions, action_mask=zero_mask, raw_state=raw_state_rep) # n_q_head * (bs* n_sample) *1 
        self._last_target_head_means = [q.mean().item() for q in q_values]
        q = self._aggregate_q(*q_values) # (bs* n_sample) *1
        q = q.view(action_samples, batch_size, 1) # n_sample * bs
        best_indices = torch.argmax(q, dim=0).squeeze(-1) # bs
        batch_indices = torch.arange(batch_size, device=self.device) # 0-bs-1
        best_actions = stacked[best_indices, batch_indices]
        best_q = q[best_indices, batch_indices]
        return best_actions, best_q

    def _get_tensor(
        self,
        batch: Dict[str, Any],
        keys: list[str],
        default_shape: tuple[int, ...],
    ) -> torch.Tensor:
        for key in keys:
            if key in batch and isinstance(batch[key], torch.Tensor):
                return batch[key]
        return torch.zeros(default_shape, device=self.device)

    def _get_raw_state(self, batch: Dict[str, Any], batch_size: int) -> torch.Tensor:
        raw = batch.get("observation.state")
        raw_dim = getattr(self.cfg, "raw_state_dim", self._raw_state_dim)
        if isinstance(raw, torch.Tensor):
            if raw.shape[0] != batch_size:
                raw = raw[:batch_size]
            return raw.to(self.device)
        return torch.zeros((batch_size, raw_dim), device=self.device, dtype=torch.float32)

    def _repeat_batch(
        self, batch: Dict[str, Any], batch_size: int, repeats: int
    ) -> Dict[str, Any]:
        if repeats <= 1:
            return batch
        expanded: Dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.shape[0] == batch_size:
                expanded[key] = torch.repeat_interleave(value, repeats, dim=0)
                m = value
                n = torch.repeat_interleave(value, repeats, dim=0)
                pass
            else:
                expanded[key] = value
        return expanded


__all__ = [
    "PolicyEmbeddings",
    "PolicyEncoderFn",
    "MLPCriticAdapter",
    "ValueQueryCriticAdapter",
    "ValueHeadCriticAdapter",
    "BestOfNCriticTrainer",
]
def _get_text_config(policy: Any):
    model = getattr(policy, "model", None)
    vlm = getattr(model, "vlm_with_expert", None)
    cfg = getattr(vlm, "config", None)
    return getattr(cfg, "text_config", None)
