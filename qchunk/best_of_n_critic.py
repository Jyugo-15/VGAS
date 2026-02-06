"""Self-contained Best-of-N critic trainer for chunked action policies.

This module mirrors the helper classes that previously lived inside
`lerobot_ext.train_with_critic` so that unit tests can import and exercise the
critic logic without going through the full training script.
"""

from __future__ import annotations

import copy
import logging
import math
import time
from typing import Any, Dict, Optional, Protocol

import torch
import torch.nn.functional as F
from torch import nn

from qchunk.networks import CriticBackbone
from qchunk.critic_adapters import (
    MLPCriticAdapter,
    ValueHeadCriticAdapter,
    PolicyEmbeddings,
)
from qchunk.valuequeryhead import (
    ValueHeadConfig,
    Qchunk_Former,
)
from qchunk.critic_utils import (
    discounted_chunk_returns,
    extract_future_batch,
    aggregate_q,
    soft_update_target,
    get_tensor_from_batch,
    get_raw_state,
    repeat_batch,
)
from qchunk.ood_calql_utils import (
    prepare_ood_actions,
    compute_calql_loss,
    compute_explicit_penalty_loss,
    compute_weighted_distance,
)

class PolicyEncoderFn(Protocol):
    """Signature for helpers that build policy embeddings from raw batches."""

    def __call__(self, policy: Any, batch: Dict[str, torch.Tensor]) -> "PolicyEmbeddings":
        """Return pooled + token embeddings detached from the policy."""



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
        if critic_type in {"my_value_query_head", "my_value_head", "value_head"}:
            critic_type = "q_chunk_former"
        if critic_type == "mlp":
            backbone = CriticBackbone(obs_dim, flat_action_dim, hidden_sizes=getattr(cfg, "hidden_dims", (512, 512)))
            critic = MLPCriticAdapter(backbone).to(device)
        elif critic_type == "q_chunk_former":
            text_config = _get_text_config(policy)
            if text_config is None:
                raise ValueError("QChunk Former critic requires access to the policy text_config.")
            num_head_layers = getattr(cfg, "qformer_num_backbone_layers", None)
            if num_head_layers is None:
                num_head_layers = getattr(cfg, "value_head_num_layers", getattr(cfg, "head_num_layers", 2))
            vh_config = ValueHeadConfig(
                chunk_size=chunk_size,
                action_dim=action_step_dim,
                num_head_layers=num_head_layers,
                head_mlp_dims=getattr(cfg, "value_head_mlp_dims", getattr(cfg, "head_mlp_dims", (512, 512))),
                vlm_model_name=getattr(cfg, "value_head_vlm_model_name", getattr(cfg, "vqh_vlm_model_name", None)),
                att_mode=getattr(cfg, "att_mode", "causal"),
                use_raw_state_fusion=getattr(cfg, "use_raw_state_fusion", False),
                raw_state_dim=getattr(cfg, "raw_state_dim", 8),
                bias_init_enabled=getattr(cfg, "value_head_bias_init_enabled", False),
                bias_init_value=getattr(cfg, "value_head_bias_init_value", 0.0),
            )
            num_q_heads = getattr(cfg, "num_q_heads", 1)
            if num_q_heads < 1:
                raise ValueError("Value head critic requires at least one head (num_q_heads >= 1).")
            heads = [
                Qchunk_Former(vh_config, text_config=text_config)
                for _ in range(num_q_heads)
            ]
            critic = ValueHeadCriticAdapter(heads).to(device)
                    
        else:
            raise ValueError(
                f"Unknown critic_type '{getattr(cfg, 'critic_type', 'mlp')}' (expected 'mlp' or 'q_chunk_former')."
            )

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
        # print("==================================crtic config==================================")
        # print(cfg)
        # print("==================================critic module==================================")
        # print(critic)
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
        def _strict_load(module: nn.Module, payload: Dict[str, Any] | None, label: str) -> None:
            if payload is None:
                raise ValueError(f"Missing {label} state in checkpoint.")
            if not isinstance(payload, dict):
                raise TypeError(f"Expected {label} state_dict dict, got {type(payload)}.")
            module.load_state_dict(payload, strict=True)

        if "critic" in state or "target_critic" in state:
            _strict_load(self.critic, state.get("critic"), "critic")
            _strict_load(self.target_critic, state.get("target_critic"), "target_critic")
        else:
            _strict_load(self.critic, state, "critic")
            _strict_load(self.target_critic, state, "target_critic")

        optim_state = state.get("optimizer")
        if optim_state is not None:
            try:
                self.optimizer.load_state_dict(optim_state)
            except Exception as exc:  # pragma: no cover - eval may ignore optimizer
                logging.warning("Skipping optimizer state load: %s", exc)
        sched_state = state.get("scheduler")
        if self.scheduler is not None and sched_state is not None:
            try:
                self.scheduler.load_state_dict(sched_state)
            except Exception as exc:  # pragma: no cover - eval may ignore scheduler
                logging.warning("Skipping scheduler state load: %s", exc)

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
        action_mask = get_tensor_from_batch(batch, ["actions_is_pad", "action_is_pad"], default_shape=actions.shape[:2], device=self.device)

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

        rewards = get_tensor_from_batch(batch, ["rewards"], default_shape=actions.shape[:2], device=self.device)

        if rewards.ndim > 1 and rewards.shape[-1] > self.chunk_size:
            rewards = rewards[..., : self.chunk_size]
        rewards = rewards.reshape(actions.shape[0], -1) # rewards => [-1,-1,-1....] chunk_len
        rewards = rewards.to(self.device)
        
        reward_is_pad = get_tensor_from_batch(batch, ["reward_is_pad"], default_shape=actions.shape[:2], device=self.device)
        if reward_is_pad.shape[-1] > self.chunk_size:
            reward_is_pad = reward_is_pad[..., : self.chunk_size]
       
       
        returns = discounted_chunk_returns(rewards, reward_is_pad, getattr(self.cfg, "discount", 0.99))
        raw_state = get_raw_state(batch, actions.shape[0], getattr(self.cfg, "raw_state_dim", self._raw_state_dim), self.device)
        next_batch = extract_future_batch(batch)
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

        action_mask = get_tensor_from_batch(batch, ["actions_is_pad", "action_is_pad"], default_shape=actions.shape[:2], device=self.device)
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
            ood_payload = prepare_ood_actions(
                policy, batch, actions, next_action_candidates, action_mask, next_pad, raw_state
            )

        if use_ood_reg:
            if ood_payload is None:
                raise ValueError("use_ood_reg=True 但未生成 ood_payload。")
            ood_loss, ood_metrics, q_ood_tensor = compute_explicit_penalty_loss(
                self, encoding, targets, actions, ood_payload
            )

        if use_calql and ood_payload is not None:
            calql_loss, calql_metrics = compute_calql_loss(self, encoding, q_values, ood_payload)

        # 控制损失如何在多个 head 之间聚合：默认先聚合 Q 再算 MSE，per-head 方式则每个 head 单独算再平均
        loss_mode = getattr(self.cfg, "critic_loss_mode", "mse")
        losses = None
        if str(loss_mode).lower() in ("mse", "mean"):
            current_q = aggregate_q(q_values, getattr(self.cfg, "q_aggregation", "mean"))
            td_loss = F.mse_loss(current_q, targets)
        elif str(loss_mode).lower() in ("per_head_mean"):
            losses = [F.mse_loss(q, targets) for q in q_values]
            td_loss = torch.stack(losses).mean()
            current_q = aggregate_q(q_values, getattr(self.cfg, "q_aggregation", "mean"))
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
        soft_update_target(self.target_critic, self.critic, tau_to_use)
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


    def _compute_calql_loss(
        self,
        encoding: PolicyEmbeddings,
        q_values: tuple[torch.Tensor, ...],
        calql_payload: Dict[str, Any],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        return compute_calql_loss(self, encoding, q_values, calql_payload)

    def _compute_weighted_distance(
        self,
        pred_actions: torch.Tensor,
        gt_actions: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return compute_weighted_distance(self, pred_actions, gt_actions, pad_mask=pad_mask)

    def _compute_explicit_penalty_loss(
        self,
        encoding: PolicyEmbeddings,
        gt_bellman_target: torch.Tensor,  # y_t (B, 1)
        gt_actions: torch.Tensor,         # a_gt (B, T, D)
        calql_payload: Dict[str, Any],
        ) -> tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
        return compute_explicit_penalty_loss(self, encoding, gt_bellman_target, gt_actions, calql_payload)

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
        raw_state = get_raw_state(batch, batch_size, getattr(self.cfg, "raw_state_dim", self._raw_state_dim), self.device)
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
            best_q = aggregate_q(q_values, getattr(self.cfg, "q_aggregation", "mean"))
            if return_candidates:
                candidates = actions.unsqueeze(1)  # (batch, 1, chunk, action_dim)
                return actions, best_q, candidates
            return actions, best_q

        expanded_batch = repeat_batch(batch, batch_size, action_samples)## b1, b2 = b1,b1,b2,b2
        
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
        q = aggregate_q(q_values, getattr(self.cfg, "q_aggregation", "mean")) # 将不同q_network的q合在一起
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




__all__ = [
    "PolicyEmbeddings",
    "PolicyEncoderFn",
    "MLPCriticAdapter",
    "ValueHeadCriticAdapter",
    "BestOfNCriticTrainer",
]
def _get_text_config(policy: Any):
    model = getattr(policy, "model", None)
    vlm = getattr(model, "vlm_with_expert", None)
    cfg = getattr(vlm, "config", None)
    return getattr(cfg, "text_config", None)
