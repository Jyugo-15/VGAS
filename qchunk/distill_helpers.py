"""Utilities for teacher/student distillation in qchunk offline training."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.utils.constants import ACTION

from qchunk.critic_utils import aggregate_q, get_raw_state, get_tensor_from_batch, repeat_batch
from qchunk.qchunked_critic import QChunkedCritic
from utils.local_policy import make_local_smolvla_policy


def _filter_smolvla_config_payload(payload: dict, *, source: str) -> dict:
    """Drop config keys unsupported by the active SmolVLAConfig version."""
    valid_keys = set(getattr(SmolVLAConfig, "__dataclass_fields__", {}).keys())
    dropped = sorted(key for key in payload if key not in valid_keys)
    if dropped:
        logging.warning(
            "Ignoring unsupported SmolVLA config keys from %s: %s",
            source,
            ", ".join(dropped),
        )
    return {key: value for key, value in payload.items() if key in valid_keys}


def _load_teacher_config(teacher_path: str) -> SmolVLAConfig:
    """
    Load teacher config with compatibility fallback.

    Older/newer lerobot versions may not decode every `type` value or every config key.
    """
    try:
        teacher_cfg = PreTrainedConfig.from_pretrained(teacher_path)
        if not isinstance(teacher_cfg, SmolVLAConfig):
            raise TypeError(
                f"Teacher config must be SmolVLAConfig, got {type(teacher_cfg).__name__}."
            )
        return teacher_cfg
    except Exception as exc:
        config_path = Path(teacher_path)
        if config_path.is_dir():
            config_path = config_path / "config.json"
        if not config_path.exists():
            raise
        with config_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        payload.pop("type", None)
        payload = _filter_smolvla_config_payload(payload, source=str(config_path))
        logging.warning(
            "Falling back to direct SmolVLAConfig load for teacher from %s (%s: %s).",
            config_path,
            type(exc).__name__,
            exc,
        )
        return SmolVLAConfig(**payload)


def _sanitize_for_wandb(payload: dict) -> dict:
    """Convert tensors or unsupported types into plain python scalars for wandb logging."""
    sanitized: dict[str, float | int | str] = {}
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            tensor = value.detach()
            if tensor.numel() == 1:
                sanitized[key] = float(tensor.item())
            else:
                sanitized[key] = float(tensor.mean().item())
        elif isinstance(value, (float, int, str)):
            sanitized[key] = value
        elif value is None:
            continue
    return sanitized


def _freeze_policy_params(policy: PreTrainedPolicy) -> None:
    for param in policy.parameters():
        param.requires_grad_(False)
    policy.eval()


def _build_teacher_policy(
    cfg: Any,
    ds_meta: LeRobotDatasetMetadata,
) -> Optional[PreTrainedPolicy]:
    distill_cfg = getattr(cfg, "distill", None)
    if distill_cfg is None or not getattr(distill_cfg, "enable", False):
        return None

    teacher_path = distill_cfg.teacher_pretrained_path
    if not teacher_path and cfg.policy.pretrained_path:
        teacher_path = str(cfg.policy.pretrained_path)
    if not teacher_path:
        raise ValueError("Distillation is enabled but teacher_pretrained_path is not set.")

    teacher_cfg = _load_teacher_config(teacher_path)

    teacher_cfg.device = cfg.policy.device
    teacher_cfg.pretrained_path = teacher_path
    if distill_cfg.teacher_num_steps is not None:
        teacher_cfg.num_steps = int(distill_cfg.teacher_num_steps)

    teacher = make_local_smolvla_policy(cfg=teacher_cfg, ds_meta=ds_meta)
    if getattr(distill_cfg, "freeze_teacher", True):
        _freeze_policy_params(teacher)
    return teacher


@torch.no_grad()
def _select_best_teacher_actions(
    *,
    teacher: PreTrainedPolicy,
    critic: QChunkedCritic,
    batch: Dict[str, torch.Tensor],
    action_samples: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    first_tensor = next((value for value in batch.values() if isinstance(value, torch.Tensor)), None)
    if first_tensor is None:
        raise ValueError("Batch does not contain tensor observations.")

    batch_size = first_tensor.shape[0]
    q_chunk_len = critic.q_chunk_len
    action_samples = max(1, int(action_samples))

    if action_samples == 1:
        teacher_actions = teacher.predict_action_chunk(batch)
        if teacher_actions.shape[1] < q_chunk_len:
            raise ValueError(
                f"Teacher chunk ({teacher_actions.shape[1]}) shorter than q_chunk_len ({q_chunk_len})."
            )
        if teacher_actions.shape[1] > q_chunk_len:
            teacher_actions = teacher_actions[:, :q_chunk_len]
        return teacher_actions, {"distill/action_samples": 1.0}

    raw_state = get_raw_state(
        batch,
        batch_size,
        getattr(critic.cfg, "raw_state_dim", getattr(critic, "_raw_state_dim", 8)),
        critic.device,
    )
    encoding = critic.encode_policy_for_critic(
        critic.actor,
        batch,
        raw_state,
        use_target=False,
        track_state_grad=False,
    )
    expanded_batch = repeat_batch(batch, batch_size, action_samples)
    expanded_actions = teacher.predict_action_chunk(expanded_batch)
    if expanded_actions.shape[1] < q_chunk_len:
        raise ValueError(
            f"Teacher chunk ({expanded_actions.shape[1]}) shorter than q_chunk_len ({q_chunk_len})."
        )
    if expanded_actions.shape[1] > q_chunk_len:
        expanded_actions = expanded_actions[:, :q_chunk_len]

    action_dim = expanded_actions.shape[-1]
    candidates = expanded_actions.view(batch_size, action_samples, q_chunk_len, action_dim)
    flat_actions = candidates.view(-1, q_chunk_len, action_dim)
    repeated_encoding = encoding.repeat(action_samples)
    eval_mask = torch.zeros(flat_actions.shape[:2], dtype=torch.bool, device=critic.device)
    raw_state_rep = raw_state.repeat_interleave(action_samples, dim=0)

    q_values = critic.critic(
        repeated_encoding,
        flat_actions,
        action_mask=eval_mask,
        raw_state=raw_state_rep,
    )
    q_scores = aggregate_q(q_values, getattr(critic.cfg, "q_aggregation", "mean")).view(batch_size, action_samples, -1)
    best_idx = torch.argmax(q_scores, dim=1).squeeze(-1)
    batch_idx = torch.arange(batch_size, device=critic.device)
    best_actions = candidates[batch_idx, best_idx]
    best_q = q_scores[batch_idx, best_idx]
    if best_q.ndim > 1:
        best_q = best_q.squeeze(-1)
    metrics = {
        "distill/action_samples": float(action_samples),
        "distill/best_q_mean": float(best_q.mean().item()),
    }
    return best_actions, metrics


def _build_distill_actor_batch(
    *,
    batch: Dict[str, Any],
    selected_actions: torch.Tensor,
) -> Dict[str, Any]:
    actor_batch = dict(batch)
    if "action" not in actor_batch or not isinstance(actor_batch["action"], torch.Tensor):
        raise ValueError("Distillation actor batch requires tensor `action` in the input batch.")

    gt_action = actor_batch["action"]
    target_len = gt_action.shape[1]
    selected = selected_actions.detach().to(device=gt_action.device, dtype=gt_action.dtype)
    if selected.shape[0] != gt_action.shape[0]:
        raise ValueError(
            f"Selected action batch size ({selected.shape[0]}) does not match target batch size ({gt_action.shape[0]})."
        )
    if selected.shape[-1] != gt_action.shape[-1]:
        raise ValueError(
            f"Selected action dim ({selected.shape[-1]}) does not match target action dim ({gt_action.shape[-1]})."
        )
    if selected.shape[1] > target_len:
        raise ValueError(
            f"Selected action chunk ({selected.shape[1]}) longer than target action chunk ({target_len})."
        )

    action_pad = next(
        (
            value.to(device=gt_action.device, dtype=torch.bool).clone()
            for key in ("actions_is_pad", "action_is_pad", "actions_id_pad")
            if isinstance((value := actor_batch.get(key)), torch.Tensor)
        ),
        torch.zeros(gt_action.shape[:2], device=gt_action.device, dtype=torch.bool),
    )
    if action_pad.shape[:2] != gt_action.shape[:2]:
        raise ValueError(
            f"Action pad shape {tuple(action_pad.shape)} does not match target action shape prefix {tuple(gt_action.shape[:2])}."
        )

    merged = gt_action.detach().clone()
    merged[:, : selected.shape[1]] = selected
    if selected.shape[1] < target_len:
        action_pad[:, selected.shape[1] :] = True
    actor_batch["action"] = merged
    actor_batch[ACTION] = merged
    actor_batch["actions_is_pad"] = action_pad
    actor_batch["action_is_pad"] = action_pad
    actor_batch["actions_id_pad"] = action_pad
    return actor_batch


@torch.no_grad()
def _sample_policy_actions_single(
    *,
    policy: PreTrainedPolicy,
    batch: Dict[str, Any],
    q_chunk_len: int | None = None,
) -> torch.Tensor:
    actions = policy.predict_action_chunk(batch)
    if q_chunk_len is not None:
        if actions.shape[1] < q_chunk_len:
            raise ValueError(f"Policy chunk ({actions.shape[1]}) shorter than q_chunk_len ({q_chunk_len}).")
        if actions.shape[1] > q_chunk_len:
            actions = actions[:, :q_chunk_len]
    return actions


def _align_action_chunk_for_critic(
    *,
    actions: torch.Tensor,
    q_chunk_len: int,
    action_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Align arbitrary action chunks to critic input shape: (B, q_chunk_len, action_dim)."""
    aligned = actions.to(device=device, dtype=dtype)
    if aligned.ndim != 3:
        raise ValueError(f"Expected action tensor with shape (B, T, D), got {tuple(aligned.shape)}.")

    if aligned.shape[-1] < action_dim:
        pad_dim = action_dim - aligned.shape[-1]
        pad = torch.zeros(aligned.shape[0], aligned.shape[1], pad_dim, device=device, dtype=dtype)
        aligned = torch.cat([aligned, pad], dim=-1)
    elif aligned.shape[-1] > action_dim:
        aligned = aligned[..., :action_dim]

    if aligned.shape[1] < q_chunk_len:
        if aligned.shape[1] == 0:
            pad_steps = torch.zeros(aligned.shape[0], q_chunk_len, aligned.shape[-1], device=device, dtype=dtype)
            aligned = pad_steps
        else:
            pad_steps = q_chunk_len - aligned.shape[1]
            last_step = aligned[:, -1:, :].expand(-1, pad_steps, -1)
            aligned = torch.cat([aligned, last_step], dim=1)
    elif aligned.shape[1] > q_chunk_len:
        aligned = aligned[:, :q_chunk_len]
    return aligned


def _align_action_mask_for_critic(
    *,
    mask: torch.Tensor,
    q_chunk_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Align action padding masks to critic chunk shape: (B, q_chunk_len)."""
    aligned = mask.to(device=device, dtype=torch.bool)
    if aligned.ndim != 2:
        raise ValueError(f"Expected action mask with shape (B, T), got {tuple(aligned.shape)}.")
    if aligned.shape[1] < q_chunk_len:
        pad_steps = q_chunk_len - aligned.shape[1]
        pad = torch.ones(aligned.shape[0], pad_steps, device=device, dtype=torch.bool)
        aligned = torch.cat([aligned, pad], dim=1)
    elif aligned.shape[1] > q_chunk_len:
        aligned = aligned[:, :q_chunk_len]
    return aligned


def _resolve_normalized_action_clip_bounds(
    *,
    critic: QChunkedCritic,
    action_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve clip bounds in normalized action space from dataset stats; fall back to [-1, 1]."""
    stats = getattr(critic, "action_stats", {}) or {}
    act_min = stats.get("min")
    act_max = stats.get("max")
    act_mean = stats.get("mean")
    act_std = stats.get("std")

    if act_min is not None and act_max is not None and act_mean is not None and act_std is not None:
        min_t = torch.as_tensor(act_min, device=device, dtype=dtype).view(-1)
        max_t = torch.as_tensor(act_max, device=device, dtype=dtype).view(-1)
        mean_t = torch.as_tensor(act_mean, device=device, dtype=dtype).view(-1)
        std_t = torch.as_tensor(act_std, device=device, dtype=dtype).view(-1).clamp_min(1e-6)
        if (
            min_t.numel() >= action_dim
            and max_t.numel() >= action_dim
            and mean_t.numel() >= action_dim
            and std_t.numel() >= action_dim
        ):
            norm_min = (min_t[:action_dim] - mean_t[:action_dim]) / std_t[:action_dim]
            norm_max = (max_t[:action_dim] - mean_t[:action_dim]) / std_t[:action_dim]
            low = torch.minimum(norm_min, norm_max).view(1, 1, 1, -1)
            high = torch.maximum(norm_min, norm_max).view(1, 1, 1, -1)
            return low, high
    if act_min is not None and act_max is not None:
        min_t = torch.as_tensor(act_min, device=device, dtype=dtype).view(-1)
        max_t = torch.as_tensor(act_max, device=device, dtype=dtype).view(-1)
        if min_t.numel() >= action_dim and max_t.numel() >= action_dim:
            low = torch.minimum(min_t[:action_dim], max_t[:action_dim]).view(1, 1, 1, -1)
            high = torch.maximum(min_t[:action_dim], max_t[:action_dim]).view(1, 1, 1, -1)
            return low, high

    low = torch.full((1, 1, 1, action_dim), -1.0, device=device, dtype=dtype)
    high = torch.full((1, 1, 1, action_dim), 1.0, device=device, dtype=dtype)
    return low, high


def _select_local_optimized_student_actions(
    *,
    student: PreTrainedPolicy,
    critic: QChunkedCritic,
    batch: Dict[str, Any],
    action_samples: int,
    local_samples: int,
    include_dataset_action_in_global_pool: bool,
    local_steps: int,
    local_lr: float,
    local_grad_normalize: bool = True,
    local_adv_weight: bool = False,
    local_adv_eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    first_tensor = next((value for value in batch.values() if isinstance(value, torch.Tensor)), None)
    if first_tensor is None:
        raise ValueError("Batch does not contain tensor observations.")

    batch_size = first_tensor.shape[0]
    q_chunk_len = critic.q_chunk_len
    action_samples = max(0, int(action_samples))
    top_m = max(1, int(local_samples))
    local_steps = max(0, int(local_steps))
    local_lr = float(local_lr)
    local_adv_eps = max(float(local_adv_eps), 1e-8)

    dataset_actions = batch.get("action")
    if action_samples == 0 and not include_dataset_action_in_global_pool:
        raise ValueError("phase2_global_samples=0 requires include_dataset_action_in_global_pool=True.")
    if action_samples == 0 and not isinstance(dataset_actions, torch.Tensor):
        raise ValueError("phase2_global_samples=0 requires a tensor batch['action'] dataset target.")

    if action_samples > 0:
        expanded_batch = repeat_batch(batch, batch_size, action_samples)
        with torch.no_grad():
            expanded_actions = student.predict_action_chunk(expanded_batch)
        if expanded_actions.shape[1] < q_chunk_len:
            raise ValueError(
                f"Student chunk ({expanded_actions.shape[1]}) shorter than q_chunk_len ({q_chunk_len})."
            )
        if expanded_actions.shape[1] > q_chunk_len:
            expanded_actions = expanded_actions[:, :q_chunk_len]

        action_dim = expanded_actions.shape[-1]
        candidates = expanded_actions.view(batch_size, action_samples, q_chunk_len, action_dim).to(critic.device)
        candidate_masks = torch.zeros(
            batch_size,
            action_samples,
            q_chunk_len,
            dtype=torch.bool,
            device=critic.device,
        )
    else:
        action_dim = int(dataset_actions.shape[-1])
        candidates = torch.empty(
            batch_size,
            0,
            q_chunk_len,
            action_dim,
            device=critic.device,
            dtype=dataset_actions.dtype,
        )
        candidate_masks = torch.empty(
            batch_size,
            0,
            q_chunk_len,
            dtype=torch.bool,
            device=critic.device,
        )
    used_dataset_action = 0.0
    dataset_candidate_mask: Optional[torch.Tensor] = None
    if include_dataset_action_in_global_pool:
        if isinstance(dataset_actions, torch.Tensor):
            dataset_actions = _align_action_chunk_for_critic(
                actions=dataset_actions,
                q_chunk_len=q_chunk_len,
                action_dim=action_dim,
                device=critic.device,
                dtype=candidates.dtype,
            )
            dataset_action_mask = get_tensor_from_batch(
                batch,
                ["actions_is_pad", "action_is_pad"],
                default_shape=dataset_actions.shape[:2],
                device=critic.device,
            )
            dataset_action_mask = _align_action_mask_for_critic(
                mask=dataset_action_mask,
                q_chunk_len=q_chunk_len,
                device=critic.device,
            )
            candidates = torch.cat([candidates, dataset_actions.unsqueeze(1)], dim=1)
            candidate_masks = torch.cat([candidate_masks, dataset_action_mask.unsqueeze(1)], dim=1)
            used_dataset_action = 1.0
            dataset_candidate_mask = torch.zeros(
                batch_size,
                candidates.shape[1],
                dtype=torch.bool,
                device=critic.device,
            )
            dataset_candidate_mask[:, -1] = True
    # Local action optimization should evaluate a deterministic critic (no dropout noise).
    modules_to_eval = [
        critic.critic,
        getattr(critic, "state_encoder", None),
        getattr(critic, "critic_vision_encoder", None),
        getattr(critic, "vision_proj", None),
        getattr(critic, "lang_proj", None),
    ]
    module_training_states: list[tuple[torch.nn.Module, bool]] = []
    for module in modules_to_eval:
        if isinstance(module, torch.nn.Module):
            module_training_states.append((module, module.training))
            module.eval()

    try:
        raw_state = get_raw_state(
            batch,
            batch_size,
            getattr(critic.cfg, "raw_state_dim", getattr(critic, "_raw_state_dim", 8)),
            critic.device,
        )
        encoding = critic.encode_policy_for_critic(
            critic.actor,
            batch,
            raw_state,
            use_target=False,
            track_state_grad=False,
        )

        num_candidates = candidates.shape[1]
        with torch.no_grad():
            flat_candidates = candidates.view(-1, q_chunk_len, action_dim)
            rep_encoding = encoding.repeat(num_candidates)
            eval_mask = candidate_masks.view(-1, q_chunk_len)
            raw_rep = raw_state.repeat_interleave(num_candidates, dim=0)
            q_values = critic.critic(rep_encoding, flat_candidates, action_mask=eval_mask, raw_state=raw_rep)
            q_scores = aggregate_q(q_values, getattr(critic.cfg, "q_aggregation", "mean")).view(batch_size, num_candidates, -1)
            if q_scores.shape[-1] > 1:
                q_scores = q_scores.mean(dim=-1, keepdim=True)
            q_scores = q_scores.squeeze(-1)

        top_m = min(top_m, num_candidates)
        top_values, top_idx = torch.topk(q_scores, k=top_m, dim=1)
        gather_idx = top_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, q_chunk_len, action_dim)
        top_actions = candidates.gather(1, gather_idx).detach()
        mask_gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, q_chunk_len)
        top_action_masks = candidate_masks.gather(1, mask_gather_idx).detach()
        global_gt_hit_rate = 0.0
        top_dataset_mask: Optional[torch.Tensor] = None
        if dataset_candidate_mask is not None:
            top_dataset_mask = dataset_candidate_mask.gather(1, top_idx)
            global_gt_hit_rate = float(top_dataset_mask.any(dim=1).float().mean().item())

        optimized = top_actions
        clip_low, clip_high = _resolve_normalized_action_clip_bounds(
            critic=critic,
            action_dim=action_dim,
            dtype=top_actions.dtype,
            device=critic.device,
        )
        best_local_actions = optimized.detach().clone()
        best_local_q = torch.full(
            (batch_size, top_m),
            -float("inf"),
            device=critic.device,
            dtype=top_actions.dtype,
        )
        grad_norm_meter = 0.0
        adv_weight_abs_meter = 0.0
        local_update_steps = 0

        def _score_local(actions: torch.Tensor) -> torch.Tensor:
            flat = actions.view(-1, q_chunk_len, action_dim)
            rep_encoding = encoding.repeat(top_m)
            raw_rep = raw_state.repeat_interleave(top_m, dim=0)
            eval_mask = top_action_masks.view(-1, q_chunk_len)
            q_values = critic.critic(rep_encoding, flat, action_mask=eval_mask, raw_state=raw_rep)
            q_local_values = aggregate_q(q_values, getattr(critic.cfg, "q_aggregation", "mean")).view(batch_size, top_m, -1)
            if q_local_values.shape[-1] > 1:
                q_local_values = q_local_values.mean(dim=-1, keepdim=True)
            return q_local_values.squeeze(-1)

        for _ in range(local_steps):
            optimized = optimized.detach().requires_grad_(True)
            q_local = _score_local(optimized)
            with torch.no_grad():
                improved = q_local.detach() > best_local_q
                best_local_q = torch.where(improved, q_local.detach(), best_local_q)
                best_local_actions = torch.where(
                    improved.unsqueeze(-1).unsqueeze(-1),
                    optimized.detach(),
                    best_local_actions,
                )
            grad = torch.autograd.grad(q_local.sum(), optimized, create_graph=False, retain_graph=False)[0]
            if local_grad_normalize:
                grad_norm = grad.flatten(2).norm(dim=-1, keepdim=True).clamp_min(local_adv_eps)
                grad = grad / grad_norm.unsqueeze(-1)
            with torch.no_grad():
                grad_norm_meter += float(grad.flatten(2).norm(dim=-1).mean().item())
            if local_adv_weight:
                with torch.no_grad():
                    q_ref = q_local.detach()
                    q_mean = q_ref.mean(dim=1, keepdim=True)
                    q_std = q_ref.std(dim=1, unbiased=False, keepdim=True).clamp_min(local_adv_eps)
                    adv_w = torch.tanh((q_ref - q_mean) / q_std)
                    adv_weight_abs_meter += float(adv_w.abs().mean().item())
                grad = grad * adv_w.unsqueeze(-1).unsqueeze(-1)
            optimized = optimized + local_lr * grad
            optimized = torch.max(torch.min(optimized, clip_high), clip_low)
            local_update_steps += 1

        with torch.no_grad():
            q_local_final = _score_local(optimized.detach())
            improved = q_local_final > best_local_q
            best_local_q = torch.where(improved, q_local_final, best_local_q)
            best_local_actions = torch.where(
                improved.unsqueeze(-1).unsqueeze(-1),
                optimized.detach(),
                best_local_actions,
            )
            best_idx = torch.argmax(best_local_q, dim=1)
            batch_idx = torch.arange(batch_size, device=critic.device)
            best_actions = best_local_actions[batch_idx, best_idx]
            final_gt_selected_rate = 0.0
            if top_dataset_mask is not None:
                final_gt_selected_rate = float(top_dataset_mask[batch_idx, best_idx].float().mean().item())

        metrics = {
            "distill/global_samples": float(action_samples),
            "distill/global_pool_size": float(num_candidates),
            "distill/global_used_dataset_action": float(used_dataset_action),
            "distill/global_gt_hit_rate": float(global_gt_hit_rate),
            "distill/local_samples": float(top_m),
            "distill/local_grad_normalize": float(local_grad_normalize),
            "distill/local_adv_weight": float(local_adv_weight),
            "distill/global_top_q_mean": float(top_values.mean().item()),
            "distill/local_best_q_mean": float(best_local_q[batch_idx, best_idx].mean().item()),
            "distill/final_gt_selected_rate": float(final_gt_selected_rate),
        }
        if local_update_steps > 0:
            metrics["distill/local_grad_norm_mean"] = grad_norm_meter / float(local_update_steps)
            if local_adv_weight:
                metrics["distill/local_adv_weight_abs_mean"] = adv_weight_abs_meter / float(local_update_steps)
        return best_actions, metrics
    finally:
        for module, was_training in module_training_states:
            module.train(was_training)
