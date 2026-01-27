"""Self-contained critic module that owns actor/target/optimizer (no external trainer)."""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

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
        self._last_target_head_means: Optional[list[float]] = None
        self._shape_debug_done = False

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
    ) -> "QChunkedCritic":
        from lerobot.policies.factory import make_policy
        from scripts.train_qchunk_offline import encode_policy_observations


        policy_cfg.device = device
        policy_cfg.pretrained_path = str(policy_path)
        if actor is None:
            actor = make_policy(cfg=policy_cfg, ds_meta=ds_meta, env_cfg=None)
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
        critic = cls._build_critic_module(actor, critic_cfg, q_chunk_len, action_step_dim, obs_dim, torch.device(device))
        target_critic = copy.deepcopy(critic).to(device)
        optimizer = torch.optim.Adam(
            critic.parameters(),
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
            optimizer=optimizer,
            cfg=critic_cfg,
            device=torch.device(device),
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
    ):
        first_tensor = next((value for value in batch.values() if isinstance(value, torch.Tensor)), None)
        if first_tensor is None:
            raise ValueError("Batch does not contain tensor observations.")
        batch_size = first_tensor.shape[0]
        encoding = self.encode_policy(self.actor, batch).to(self.device)
        raw_state = get_raw_state(batch, batch_size, getattr(self.cfg, "raw_state_dim", self._raw_state_dim), self.device)
        q_chunk_len = self.q_chunk_len
        action_samples = action_samples or getattr(self.cfg, "action_samples", 2)
        if action_samples <= 1:
            actions = self.actor.predict_action_chunk(batch)
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
            expanded_actions = self.actor.predict_action_chunk(expanded_batch)
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
    def update(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        current_step: Optional[int] = None,
        ood_warmup_steps: int = 0,
    ) -> Dict[str, float]:
        self.critic.train()
        start = time.perf_counter()
        encoding = self.encode_policy(self.actor, batch).to(self.device)
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

        with torch.no_grad():
            next_encoding = self.encode_policy(self.actor, next_batch).to(self.device)
            next_action, next_q, next_action_candidates = self.predict_best_of_n(
                next_batch,
                action_samples=getattr(self.cfg, "action_samples", 2),
                forced_mask=gt_next_pad_mask,
                return_candidates=True,
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
            def _prepare(source: str) -> Dict[str, Any]:
                if source == "erg":
                    return prepare_erg_ood_actions(
                        self, self.actor, batch, actions, next_action_candidates, action_mask, next_pad, raw_state
                    )
                if source == "cql":
                    return prepare_cal_ood_actions(
                        self, self.actor, batch, actions, next_action_candidates, action_mask, next_pad, raw_state
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
        grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), grad_clip)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        soft_update_target(self.target_critic, self.critic, tau_to_use)
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
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "cfg": getattr(self, "cfg", None),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.critic.load_state_dict(state["critic"])
        self.target_critic.load_state_dict(state["target_critic"])
        self.optimizer.load_state_dict(state["optimizer"])
        sched_state = state.get("scheduler")
        if self.scheduler is not None and sched_state is not None:
            self.scheduler.load_state_dict(sched_state)
