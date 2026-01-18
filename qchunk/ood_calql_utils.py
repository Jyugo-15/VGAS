"""Helpers for OOD penalty and CalQL computations."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from qchunk.critic_utils import aggregate_q
from qchunk.critic_adapters import PolicyEmbeddings


def prepare_ood_actions(
    trainer: Any,
    policy: Any,
    batch: Dict[str, torch.Tensor],
    actions: torch.Tensor,
    next_action_candidates: torch.Tensor,
    action_mask: torch.Tensor,
    next_pad: torch.Tensor,
    raw_state: torch.Tensor,
) -> Dict[str, Any]:
    """Build OOD action payload (current/random/next) for OOD/CalQL losses."""
    cfg = trainer.cfg
    use_calql = getattr(cfg, "use_calql", False)
    include_current = getattr(cfg, "ood_include_current_actions", True)
    default_extra = True if use_calql else False
    include_random = getattr(cfg, "ood_include_random_actions", default_extra)
    include_next = getattr(cfg, "ood_include_next_actions", default_extra if use_calql else False)
    if use_calql:
        include_next = True

    if "mc_lower_bound" not in batch:
        raise ValueError("`mc_lower_bound` is required when use_calql=True.")
    mc_lower_bound = batch["mc_lower_bound"]
    if isinstance(mc_lower_bound, torch.Tensor) and mc_lower_bound.ndim > 1:
        mc_lower_bound = mc_lower_bound[:, 0]

    next_action_candidates = next_action_candidates.to(trainer.device)
    ood_m_actions = getattr(cfg, "cql_m_actions", None)
    if ood_m_actions is None:
        ood_m_actions = next_action_candidates.shape[1]
    ood_m_actions = max(1, min(ood_m_actions, next_action_candidates.shape[1]))

    action_stats = trainer.action_stats or {}

    def _maybe_add_noise(actions_to_perturb: torch.Tensor, noise_std: float) -> torch.Tensor:
        if noise_std <= 0:
            return actions_to_perturb
        stats_for_noise = action_stats or {}
        act_mean_noise = stats_for_noise.get("mean")
        act_std_noise = stats_for_noise.get("std")
        if act_mean_noise is not None and act_std_noise is not None:
            mean = torch.as_tensor(act_mean_noise, device=trainer.device, dtype=actions.dtype).view(1, 1, 1, -1)
            std = torch.as_tensor(act_std_noise, device=trainer.device, dtype=actions.dtype).view(1, 1, 1, -1).clamp_min(1e-8)
            raw = actions_to_perturb * std + mean
            raw = raw + torch.randn_like(raw) * (noise_std * std)
            return (raw - mean) / std
        return actions_to_perturb + torch.randn_like(actions_to_perturb) * noise_std

    ood_next_actions = None
    if include_next:
        sample_idx = torch.randperm(next_action_candidates.shape[1], device=trainer.device)[:ood_m_actions]
        ood_next_actions = next_action_candidates[:, sample_idx]
        noise_std = getattr(cfg, "cql_next_noise_std", 0.05) or 0.05
        if noise_std > 0:
            ood_next_actions = _maybe_add_noise(ood_next_actions, noise_std)

    ood_random_actions = None
    if include_random:
        stats = action_stats or {}
        act_min = stats.get("min")
        act_max = stats.get("max")
        act_mean = stats.get("mean")
        act_std = stats.get("std")
        if act_min is not None and act_max is not None and act_mean is not None and act_std is not None:
            act_min = torch.as_tensor(act_min, device=trainer.device, dtype=actions.dtype).view(1, 1, 1, -1)
            act_max = torch.as_tensor(act_max, device=trainer.device, dtype=actions.dtype).view(1, 1, 1, -1)
            act_mean = torch.as_tensor(act_mean, device=trainer.device, dtype=actions.dtype).view(1, 1, 1, -1)
            act_std = torch.as_tensor(act_std, device=trainer.device, dtype=actions.dtype).view(1, 1, 1, -1)
            raw_random = torch.rand(
                (actions.shape[0], ood_m_actions, trainer.chunk_size, trainer.action_step_dim),
                device=trainer.device,
                dtype=actions.dtype,
            ) * (act_max - act_min) + act_min
            ood_random_actions = (raw_random - act_mean) / act_std.clamp_min(1e-8)
        else:
            ood_random_actions = torch.rand(
                (actions.shape[0], ood_m_actions, trainer.chunk_size, trainer.action_step_dim),
                device=trainer.device,
                dtype=actions.dtype,
            ) * 2.0 - 1.0

    with torch.no_grad():
        batch_size = actions.shape[0]
        from qchunk.critic_utils import repeat_batch
        expanded_batch = repeat_batch(batch, batch_size, ood_m_actions)
        expanded_actions = policy.predict_action_chunk(expanded_batch)
    cur = expanded_actions.to(trainer.device)
    if cur.shape[1] > trainer.chunk_size:
        cur = cur[:, : trainer.chunk_size]
    cur = cur.view(batch_size, ood_m_actions, trainer.chunk_size, -1)
    ood_current_actions = cur
    cur_noise_std = getattr(cfg, "cql_cur_noise_std", None)
    if cur_noise_std is None:
        cur_noise_std = getattr(cfg, "cql_next_noise_std", 0.05) or 0.05
    if cur_noise_std > 0:
        ood_current_actions = _maybe_add_noise(ood_current_actions, cur_noise_std)

    return {
        "mc_lower_bound": mc_lower_bound,
        "ood_next_actions": ood_next_actions,
        "ood_random_actions": ood_random_actions,
        "ood_m_actions": ood_m_actions,
        "cql_batch_action": actions,
        "ood_current_actions": ood_current_actions,
        "action_mask": action_mask.to(trainer.device).bool(),
        "next_valid_mask": (~next_pad).to(trainer.device).float(),
        "raw_state": raw_state.to(trainer.device),
        "ood_include_next": include_next,
        "ood_include_random": include_random,
        "ood_include_current": include_current,
    }


def compute_calql_loss(
    trainer: Any, encoding: PolicyEmbeddings, q_values: tuple[torch.Tensor, ...], calql_payload: Dict[str, Any]
) -> tuple[torch.Tensor, Dict[str, float]]:
    q_batch = aggregate_q(q_values, getattr(trainer.cfg, "q_aggregation", "mean")).squeeze(-1)
    m = calql_payload["ood_m_actions"]
    mc_lower_bound = calql_payload["mc_lower_bound"].to(trainer.device)
    mc_lower_bound = mc_lower_bound.view(-1, 1)
    action_mask = calql_payload["action_mask"].to(trainer.device).bool()
    next_valid_mask = calql_payload["next_valid_mask"].to(trainer.device)
    raw_state_base = calql_payload.get("raw_state")
    if raw_state_base is None:
        raw_state_base = torch.zeros(q_batch.shape[0], trainer._raw_state_dim, device=trainer.device, dtype=q_batch.dtype)
    raw_state_base = raw_state_base.to(trainer.device)

    def _eval_actions(act: torch.Tensor) -> torch.Tensor:
        num_actions = act.shape[1]
        flat = act.view(-1, trainer.chunk_size, trainer.action_step_dim)
        rep_encoding = encoding.repeat(num_actions)
        mask_rep = action_mask.unsqueeze(1).repeat(1, num_actions, 1).view(-1, trainer.chunk_size)
        raw_rep = raw_state_base.repeat_interleave(num_actions, dim=0)
        q = aggregate_q(trainer.critic(rep_encoding, flat, action_mask=mask_rep, raw_state=raw_rep),
                        getattr(trainer.cfg, "q_aggregation", "mean"))
        q = q.view(act.shape[0], num_actions, -1).squeeze(-1)
        return q

    to_eval = [calql_payload["ood_next_actions"], calql_payload["ood_current_actions"]]
    concatenated = torch.cat(to_eval, dim=1)
    evaluated = _eval_actions(concatenated)
    splits = [tensor.shape[1] for tensor in to_eval]
    q_next_raw, q_cur_raw = torch.split(evaluated, splits, dim=1)
    q_next = torch.maximum(q_next_raw, mc_lower_bound)
    q_cur = torch.maximum(q_cur_raw, mc_lower_bound)
    very_neg = -1e9
    next_valid = next_valid_mask.expand_as(q_next)
    q_next = q_next * next_valid + very_neg * (1.0 - next_valid)
    q_batch_col = q_batch
    if q_batch_col.ndim > 1:
        q_batch_col = q_batch_col.mean(dim=-1)
    q_batch_col = q_batch_col.view(-1, 1)

    temp = getattr(trainer.cfg, "cql_temp", 1.0)
    ood_actions_q = torch.cat([q_next, q_cur, q_batch_col], dim=1)
    cql_ood = torch.logsumexp(ood_actions_q / temp, dim=1) * temp
    cql_diff = cql_ood - q_batch
    calql_loss = cql_diff.mean()

    metrics = {
        "calql_loss": float(calql_loss.item()),
        "calql_ood_q": float(cql_ood.mean().item()),
    }
    return calql_loss, metrics


def compute_weighted_distance(
    trainer: Any,
    actions: torch.Tensor,
    ood_actions: torch.Tensor,
    pad_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    weights = getattr(trainer.cfg, "action_distance_weights", None)
    if weights is None:
        weights = getattr(trainer, "action_distance_weights", None)
    if weights is None:
        weights = [1.0] * actions.shape[-1]
    w = torch.as_tensor(weights, device=actions.device, dtype=actions.dtype).view(1, 1, 1, -1)
    diff = (actions - ood_actions) ** 2
    weighted = diff * w
    dist_per_step = weighted.sum(dim=-1)
    if pad_mask is not None:
        if pad_mask.ndim == 2:
            pad_mask = pad_mask.unsqueeze(1)
        if pad_mask.shape[:3] != dist_per_step.shape:
            pad_mask = pad_mask.expand_as(dist_per_step)
        valid_mask = ~pad_mask.bool()
        dist_per_step = dist_per_step * valid_mask
        valid_count = valid_mask.sum(dim=-1).clamp_min(1.0)
    else:
        valid_count = torch.tensor(dist_per_step.shape[-1], device=actions.device, dtype=actions.dtype)
    dist = dist_per_step.sum(dim=-1) / valid_count
    return torch.sqrt(dist.clamp_min(1e-8))


def compute_explicit_penalty_loss(
    trainer: Any,
    encoding: PolicyEmbeddings,
    targets: torch.Tensor,
    actions: torch.Tensor,
    ood_payload: Dict[str, Any],
) -> tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    q_batch = aggregate_q(trainer.critic(encoding, actions, action_mask=ood_payload["action_mask"], raw_state=ood_payload["raw_state"]),
                          getattr(trainer.cfg, "q_aggregation", "mean"))
    losses = []
    q_ood_collect = []

    def _score(act: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        flat = act.view(-1, trainer.chunk_size, trainer.action_step_dim)
        rep_encoding = encoding.repeat(act.shape[1])
        mask_rep = mask.view(-1, trainer.chunk_size) if mask is not None else None
        return aggregate_q(trainer.critic(rep_encoding, flat, action_mask=mask_rep, raw_state=ood_payload["raw_state"].repeat_interleave(act.shape[1], dim=0)),
                           getattr(trainer.cfg, "q_aggregation", "mean")).view(act.shape[0], act.shape[1], -1)

    if ood_payload.get("ood_include_next", False):
        q_next = _score(ood_payload["ood_next_actions"], ood_payload["action_mask"].unsqueeze(1).expand(-1, ood_payload["ood_next_actions"].shape[1], -1))
        q_ood_collect.append(q_next)
    if ood_payload.get("ood_include_current", False):
        q_current = _score(ood_payload["ood_current_actions"], ood_payload["action_mask"].unsqueeze(1).expand(-1, ood_payload["ood_current_actions"].shape[1], -1))
        q_ood_collect.append(q_current)
    if ood_payload.get("ood_include_random", False) and ood_payload["ood_random_actions"] is not None:
        q_rand = _score(ood_payload["ood_random_actions"], None)
        q_ood_collect.append(q_rand)

    if not q_ood_collect:
        raise ValueError("No OOD actions provided for explicit penalty loss.")
    q_ood = torch.cat(q_ood_collect, dim=1)

    dist = compute_weighted_distance(trainer, actions.unsqueeze(1), ood_payload["ood_current_actions"])
    beta = getattr(trainer.cfg, "dist_penalty_beta", 0.5)
    target_ood = targets.unsqueeze(1) - beta * dist
    target_ood = target_ood.detach()

    q_ood_flat = q_ood.view(q_ood.shape[0], q_ood.shape[1], -1)
    loss_ood = torch.mean(torch.clamp(q_ood_flat - target_ood, min=0.0) ** 2)

    win_policy = float((q_batch > q_ood_flat.mean(dim=1)).float().mean().item())
    win_trunc = float((q_batch > q_ood_flat.max(dim=1).values).float().mean().item())

    metrics = {
        "ood_loss": float(loss_ood.item()),
        "ood_q_mean": float(q_ood.mean().item()),
        "ood_target_mean": float(target_ood.mean().item()),
        "win_rate/gt_vs_policy": win_policy,
        "win_rate/gt_vs_trunc": win_trunc,
    }
    return loss_ood, metrics, q_ood
