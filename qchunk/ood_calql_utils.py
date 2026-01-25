"""Helpers for OOD penalty and CalQL computations."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from qchunk.critic_utils import aggregate_q
from qchunk.critic_adapters import PolicyEmbeddings


def _get_q_chunk_len(trainer: Any) -> int:
    q_chunk_len = getattr(trainer, "q_chunk_len", None)
    if q_chunk_len is None:
        raise ValueError("trainer.q_chunk_len must be set for OOD/CalQL computations.")
    q_chunk_len = int(q_chunk_len)
    if q_chunk_len <= 0:
        raise ValueError(f"trainer.q_chunk_len must be > 0, got {q_chunk_len}.")
    return q_chunk_len


def prepare_cal_ood_actions(
    trainer: Any,
    policy: Any,
    batch: Dict[str, torch.Tensor],
    actions: torch.Tensor,
    next_action_candidates: torch.Tensor,
    action_mask: torch.Tensor,
    next_pad: torch.Tensor,
    raw_state: torch.Tensor,
) -> Dict[str, Any]:
    """Build OOD action payload (current/random/next) for CQL-style losses."""
    cfg = trainer.cfg
    use_calql = getattr(cfg, "use_calql", False)
    include_current = getattr(cfg, "ood_include_current_actions", True)
    default_extra = True if use_calql else False
    include_random = getattr(cfg, "ood_include_random_actions", default_extra)
    include_next = getattr(cfg, "ood_include_next_actions", default_extra if use_calql else False)
    if use_calql:
        include_next = True

    if use_calql:
        if "mc_lower_bound" not in batch:
            raise ValueError("`mc_lower_bound` is required when use_calql=True.")
        mc_lower_bound = batch["mc_lower_bound"]
        if isinstance(mc_lower_bound, torch.Tensor) and mc_lower_bound.ndim > 1:
            mc_lower_bound = mc_lower_bound[:, 0]
    else:
        mc_lower_bound = None

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
        chunk_len = _get_q_chunk_len(trainer)
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
                (actions.shape[0], ood_m_actions, chunk_len, trainer.action_step_dim),
                device=trainer.device,
                dtype=actions.dtype,
            ) * (act_max - act_min) + act_min
            ood_random_actions = (raw_random - act_mean) / act_std.clamp_min(1e-8)
        else:
            ood_random_actions = torch.rand(
                (actions.shape[0], ood_m_actions, chunk_len, trainer.action_step_dim),
                device=trainer.device,
                dtype=actions.dtype,
            ) * 2.0 - 1.0

    with torch.no_grad():
        batch_size = actions.shape[0]
        from qchunk.critic_utils import repeat_batch
        expanded_batch = repeat_batch(batch, batch_size, ood_m_actions)
        expanded_actions = policy.predict_action_chunk(expanded_batch)
    cur = expanded_actions.to(trainer.device)
    chunk_len = _get_q_chunk_len(trainer)
    if cur.shape[1] < chunk_len:
        raise ValueError(f"Policy chunk ({cur.shape[1]}) shorter than q_chunk_len ({chunk_len}).")
    if cur.shape[1] > chunk_len:
        cur = cur[:, :chunk_len]
    cur = cur.view(batch_size, ood_m_actions, chunk_len, -1)
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
        "ood_style": "cal",
    }


def _build_erg_ood_action_pool(
    trainer: Any,
    policy_actions: torch.Tensor,
    gt_actions: torch.Tensor,
    gt_pad_mask: torch.Tensor,
) -> Dict[str, Any]:
    """Assemble ERG-style OOD action pool (policy + noise + mix + trunc)."""
    B, T, D = gt_actions.shape
    m_policy = policy_actions.shape[1]
    device = gt_actions.device
    noise_actions = []
    mix_actions = None
    mix_sel_idx = None
    mix_alpha = None
    trunc_region = None

    use_dual_noise = bool(getattr(trainer.cfg, "use_dual_noise_ood", False))
    if use_dual_noise:
        noise_tiny = torch.randn_like(gt_actions) * 0.005
        bad_act_tiny = (gt_actions + noise_tiny).unsqueeze(1)
        noise_small = torch.randn_like(gt_actions) * 0.02
        bad_act_small = (gt_actions + noise_small).unsqueeze(1)
        noise_actions = [bad_act_tiny, bad_act_small]
        ood_actions = torch.cat([policy_actions, *noise_actions], dim=1)
        trunc_region = gt_pad_mask
        dual_noise_mode = True
    else:
        use_noise = bool(getattr(trainer.cfg, "use_ood_noise", True))
        use_trunc = bool(getattr(trainer.cfg, "use_ood_trunc", True))
        use_mix = bool(getattr(trainer.cfg, "use_ood_mix", False))

        if use_noise:
            noise_stds = list(getattr(trainer.cfg, "ood_noise_stds", (0.02,)))
            if len(noise_stds) == 0:
                noise_stds = [0.02]
            for std in noise_stds:
                noise = torch.randn_like(gt_actions) * float(std)
                noise_actions.append((gt_actions + noise).unsqueeze(1))

        temp_trunc_act = gt_actions.clone()
        normalized_zero = None
        stats = trainer.action_stats or {}
        act_mean = stats.get("mean")
        act_std = stats.get("std")
        if act_mean is not None and act_std is not None:
            mean_t = torch.as_tensor(act_mean, device=device, dtype=gt_actions.dtype)
            std_t = torch.as_tensor(act_std, device=device, dtype=gt_actions.dtype).clamp_min(1e-6)
            normalized_zero = -mean_t / std_t
            if normalized_zero.numel() != temp_trunc_act.shape[-1]:
                normalized_zero = None
        if normalized_zero is None:
            normalized_zero = torch.zeros(gt_actions.shape[-1], device=device, dtype=gt_actions.dtype)

        start_frac = float(getattr(trainer.cfg, "trunc_start_frac", 0.5))
        end_frac = float(getattr(trainer.cfg, "trunc_end_frac", 0.95))
        start_frac = max(0.0, min(1.0, start_frac))
        end_frac = max(start_frac, min(1.0, end_frac))
        start_idx = min(T - 1, max(0, int(T * start_frac)))
        end_idx = min(T, max(start_idx + 1, int(T * end_frac)))
        cut_indices = torch.randint(start_idx, end_idx, (B,), device=device)
        range_tensor = torch.arange(T, device=device).unsqueeze(0)
        trunc_region = range_tensor >= cut_indices.unsqueeze(1)

        delta_dims = min(6, temp_trunc_act.shape[-1])
        if delta_dims > 0:
            mask = trunc_region.unsqueeze(-1)
            fill = normalized_zero.view(1, 1, -1)[..., :delta_dims]
            temp_trunc_act[..., :delta_dims] = torch.where(
                mask.expand(-1, -1, delta_dims),
                fill.expand_as(temp_trunc_act[..., :delta_dims]),
                temp_trunc_act[..., :delta_dims],
            )
        bad_act_truncated = temp_trunc_act.unsqueeze(1) if use_trunc else None

        if use_mix and m_policy > 0:
            mix_ratio = float(getattr(trainer.cfg, "ood_mix_ratio", 0.5))
            mix_ratio = max(0.0, min(1.0, mix_ratio))
            k = int(m_policy * mix_ratio)
            if k > 0:
                k = min(k, m_policy)
                rand_scores = torch.rand(B, m_policy, device=device)
                sel_idx = torch.argsort(rand_scores, dim=1)[:, :k]
                sel_idx_exp = sel_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, T, D)
                pol_selected = policy_actions.gather(1, sel_idx_exp)
                alpha_low = float(getattr(trainer.cfg, "ood_mix_alpha_low", 0.3))
                alpha_high = float(getattr(trainer.cfg, "ood_mix_alpha_high", 0.7))
                alpha_low = max(0.0, min(alpha_low, 1.0))
                alpha_high = max(alpha_low, min(alpha_high, 1.0))
                alpha = torch.rand(B, k, 1, 1, device=device) * (alpha_high - alpha_low) + alpha_low
                mix_actions = alpha * gt_actions.unsqueeze(1) + (1.0 - alpha) * pol_selected
                mix_sel_idx = sel_idx
                mix_alpha = alpha

        ood_blocks = [policy_actions]
        if noise_actions:
            ood_blocks.extend(noise_actions)
        if mix_actions is not None:
            ood_blocks.append(mix_actions)
        if bad_act_truncated is not None:
            ood_blocks.append(bad_act_truncated)
        ood_actions = torch.cat(ood_blocks, dim=1)
        dual_noise_mode = False

    B, M_total, T, D = ood_actions.shape
    m_policy = policy_actions.shape[1]
    m_noise = len(noise_actions)
    m_mix = mix_actions.shape[1] if mix_actions is not None else 0
    use_trunc = (not dual_noise_mode) and bool(getattr(trainer.cfg, "use_ood_trunc", True))
    m_trunc = 1 if use_trunc else 0

    num_standard_masks = m_policy + m_noise + m_mix
    mask_standard = gt_pad_mask.unsqueeze(1).repeat(1, num_standard_masks, 1)
    if use_trunc:
        mask_trunc_sample = (gt_pad_mask | trunc_region).unsqueeze(1)
        ood_forward_mask = torch.cat([mask_standard, mask_trunc_sample], dim=1)
    else:
        ood_forward_mask = mask_standard

    mask_dist_standard = gt_pad_mask.unsqueeze(1).repeat(1, num_standard_masks, 1)
    if use_trunc:
        mask_dist_trunc = gt_pad_mask.unsqueeze(1)
        calc_dist_mask = torch.cat([mask_dist_standard, mask_dist_trunc], dim=1)
    else:
        calc_dist_mask = mask_dist_standard

    return {
        "ood_actions": ood_actions,
        "ood_forward_mask": ood_forward_mask,
        "calc_dist_mask": calc_dist_mask,
        "m_policy": m_policy,
        "m_noise": m_noise,
        "m_mix": m_mix,
        "m_trunc": m_trunc,
        "mix_sel_idx": mix_sel_idx,
        "mix_alpha": mix_alpha,
    }


def prepare_erg_ood_actions(
    trainer: Any,
    policy: Any,
    batch: Dict[str, torch.Tensor],
    actions: torch.Tensor,
    next_action_candidates: torch.Tensor,
    action_mask: torch.Tensor,
    next_pad: torch.Tensor,
    raw_state: torch.Tensor,
) -> Dict[str, Any]:
    """Build OOD action payload (ERG-style pool) for OOD/CalQL losses."""
    cfg = trainer.cfg
    action_mask = action_mask.to(trainer.device).bool()
    use_calql = getattr(cfg, "use_calql", False)
    if use_calql:
        if "mc_lower_bound" not in batch:
            raise ValueError("`mc_lower_bound` is required when use_calql=True.")
        mc_lower_bound = batch["mc_lower_bound"]
        if isinstance(mc_lower_bound, torch.Tensor) and mc_lower_bound.ndim > 1:
            mc_lower_bound = mc_lower_bound[:, 0]
    else:
        mc_lower_bound = None

    next_action_candidates = next_action_candidates.to(trainer.device)
    ood_m_actions = getattr(cfg, "cql_m_actions", None)
    if ood_m_actions is None:
        ood_m_actions = next_action_candidates.shape[1]
    ood_m_actions = max(1, min(ood_m_actions, next_action_candidates.shape[1]))

    with torch.no_grad():
        batch_size = actions.shape[0]
        from qchunk.critic_utils import repeat_batch
        expanded_batch = repeat_batch(batch, batch_size, ood_m_actions)
        expanded_actions = policy.predict_action_chunk(expanded_batch)
    cur = expanded_actions.to(trainer.device)
    chunk_len = _get_q_chunk_len(trainer)
    if cur.shape[1] < chunk_len:
        raise ValueError(f"Policy chunk ({cur.shape[1]}) shorter than q_chunk_len ({chunk_len}).")
    if cur.shape[1] > chunk_len:
        cur = cur[:, :chunk_len]
    cur = cur.view(batch_size, ood_m_actions, chunk_len, -1)
    ood_current_actions = cur

    pool = _build_erg_ood_action_pool(trainer, ood_current_actions, actions, action_mask)
    return {
        "mc_lower_bound": mc_lower_bound,
        "ood_actions": pool["ood_actions"],
        "ood_forward_mask": pool["ood_forward_mask"],
        "calc_dist_mask": pool["calc_dist_mask"],
        "ood_m_actions": ood_m_actions,
        "cql_batch_action": actions,
        "ood_current_actions": ood_current_actions,
        "ood_next_actions": None,
        "ood_random_actions": None,
        "action_mask": action_mask.to(trainer.device).bool(),
        "next_valid_mask": (~next_pad).to(trainer.device).float(),
        "raw_state": raw_state.to(trainer.device),
        "ood_include_next": False,
        "ood_include_random": bool(pool["m_noise"] > 0),
        "ood_include_current": True,
        "m_policy": pool["m_policy"],
        "m_noise": pool["m_noise"],
        "m_mix": pool["m_mix"],
        "m_trunc": pool["m_trunc"],
        "mix_sel_idx": pool["mix_sel_idx"],
        "mix_alpha": pool["mix_alpha"],
        "ood_style": "erg",
    }


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
    """Backward-compatible wrapper: defaults to ERG unless configured."""
    source = str(getattr(trainer.cfg, "ood_action_source", "erg")).lower().strip()
    if source == "cql":
        return prepare_cal_ood_actions(
            trainer, policy, batch, actions, next_action_candidates, action_mask, next_pad, raw_state
        )
    return prepare_erg_ood_actions(
        trainer, policy, batch, actions, next_action_candidates, action_mask, next_pad, raw_state
    )


def compute_calql_loss(
    trainer: Any, encoding: PolicyEmbeddings, q_values: tuple[torch.Tensor, ...], calql_payload: Dict[str, Any]
) -> tuple[torch.Tensor, Dict[str, float]]:
    q_batch = aggregate_q(q_values, getattr(trainer.cfg, "q_aggregation", "mean")).squeeze(-1)
    chunk_len = _get_q_chunk_len(trainer)
    if calql_payload.get("ood_actions") is not None:
        ood_actions = calql_payload["ood_actions"]
        num_actions = ood_actions.shape[1]
        raw_state_base = calql_payload.get("raw_state")
        if raw_state_base is None:
            raw_state_base = torch.zeros(
                q_batch.shape[0], trainer._raw_state_dim, device=trainer.device, dtype=q_batch.dtype
            )
        raw_state_base = raw_state_base.to(trainer.device)
        ood_forward_mask = calql_payload.get("ood_forward_mask")
        if ood_forward_mask is not None:
            mask_rep = ood_forward_mask.view(-1, chunk_len)
        else:
            action_mask = calql_payload["action_mask"].to(trainer.device).bool()
            mask_rep = (
                action_mask.unsqueeze(1)
                .repeat(1, num_actions, 1)
                .view(-1, chunk_len)
            )
        flat = ood_actions.view(-1, chunk_len, trainer.action_step_dim)
        rep_encoding = encoding.repeat(num_actions)
        raw_rep = raw_state_base.repeat_interleave(num_actions, dim=0)
        q_ood = aggregate_q(
            trainer.critic(rep_encoding, flat, action_mask=mask_rep, raw_state=raw_rep),
            getattr(trainer.cfg, "q_aggregation", "mean"),
        )
        q_ood = q_ood.view(ood_actions.shape[0], num_actions)
        mc_lower_bound = calql_payload.get("mc_lower_bound")
        if mc_lower_bound is not None:
            mc_lower_bound = mc_lower_bound.to(trainer.device).view(-1, 1)
            q_ood = torch.maximum(q_ood, mc_lower_bound)

        q_batch_col = q_batch
        if q_batch_col.ndim > 1:
            q_batch_col = q_batch_col.mean(dim=-1)
        q_batch_col = q_batch_col.view(-1, 1)
        temp = getattr(trainer.cfg, "cql_temp", 1.0)
        ood_actions_q = torch.cat([q_ood, q_batch_col], dim=1)
        cql_ood = torch.logsumexp(ood_actions_q / temp, dim=1) * temp
        calql_loss = (cql_ood - q_batch).mean()
        metrics = {
            "calql_loss": float(calql_loss.item()),
            "calql_ood_q": float(cql_ood.mean().item()),
            "calql_ood_pool_size": float(num_actions),
        }
        return calql_loss, metrics

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
        flat = act.view(-1, chunk_len, trainer.action_step_dim)
        rep_encoding = encoding.repeat(num_actions)
        mask_rep = action_mask.unsqueeze(1).repeat(1, num_actions, 1).view(-1, chunk_len)
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
    pred_actions = ood_actions
    gt_actions = actions
    if gt_actions.ndim == 3:
        gt_actions = gt_actions.unsqueeze(1)

    diff = (pred_actions - gt_actions) ** 2
    weights_list = getattr(trainer.cfg, "action_distance_weights", None)
    if weights_list is None:
        weights_list = getattr(trainer, "action_distance_weights", None)
    if weights_list is None:
        weights_list = [5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0]
    if isinstance(weights_list, torch.Tensor):
        weights_tensor = weights_list.to(device=pred_actions.device, dtype=pred_actions.dtype)
        if weights_tensor.numel() != diff.shape[-1]:
            raise ValueError(
                f"action_distance_weights length ({weights_tensor.numel()}) must match action dim ({diff.shape[-1]})."
            )
        weights = weights_tensor.view(1, 1, 1, -1)
    else:
        if len(weights_list) != diff.shape[-1]:
            raise ValueError(
                f"action_distance_weights length ({len(weights_list)}) must match action dim ({diff.shape[-1]})."
            )
        weights = torch.tensor(weights_list, device=pred_actions.device, dtype=pred_actions.dtype).view(1, 1, 1, -1)

    dist_per_step = (diff * weights).sum(dim=-1)
    if pad_mask is not None:
        if pad_mask.ndim == 2:
            pad_mask = pad_mask.unsqueeze(1)
        if pad_mask.shape[1] != dist_per_step.shape[1]:
            pad_mask = pad_mask.expand_as(dist_per_step)
        valid_mask = ~pad_mask.bool()
        dist_masked = dist_per_step * valid_mask
        valid_count = valid_mask.sum(dim=-1).float().clamp_min(1.0)
        dist = dist_masked.sum(dim=-1) / valid_count
    else:
        valid_count = torch.tensor(dist_per_step.shape[-1], device=pred_actions.device, dtype=pred_actions.dtype)
        dist = dist_per_step.sum(dim=-1) / valid_count
    return dist


def compute_explicit_penalty_loss(
    trainer: Any,
    encoding: PolicyEmbeddings,
    targets: torch.Tensor,
    actions: torch.Tensor,
    ood_payload: Dict[str, Any],
) -> tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    chunk_len = _get_q_chunk_len(trainer)
    q_batch = aggregate_q(trainer.critic(encoding, actions, action_mask=ood_payload["action_mask"], raw_state=ood_payload["raw_state"]),
                          getattr(trainer.cfg, "q_aggregation", "mean"))
    
    if ood_payload.get("ood_actions") is None:
        raise ValueError("ood_payload must include 'ood_actions' for explicit penalty loss.")
    ood_actions = ood_payload["ood_actions"]
    ood_forward_mask = ood_payload.get("ood_forward_mask")
    calc_dist_mask = ood_payload.get("calc_dist_mask")
    num_actions = ood_actions.shape[1]
    flat = ood_actions.view(-1, chunk_len, trainer.action_step_dim)
    rep_encoding = encoding.repeat(num_actions)
    raw_state_base = ood_payload.get("raw_state")
    raw_state_base = raw_state_base.to(trainer.device)
    raw_rep = raw_state_base.repeat_interleave(num_actions, dim=0)
    if ood_forward_mask is not None:
        mask_rep = ood_forward_mask.view(-1, chunk_len)
    else:
        mask_rep = None
    q_ood_values = trainer.critic(rep_encoding, flat, action_mask=mask_rep, raw_state=raw_rep)
    if not isinstance(q_ood_values, (list, tuple)):
        q_ood_values = (q_ood_values,)

    dist = compute_weighted_distance(trainer, actions, ood_actions, pad_mask=calc_dist_mask)
    beta = getattr(trainer.cfg, "dist_penalty_beta", 0.5)
    dist_clamp_max = getattr(trainer.cfg, "dist_clamp_max", 5.0)
    if dist_clamp_max is not None:
        dist = torch.clamp(dist, max=float(dist_clamp_max))
    target_ood = targets.view(-1, 1) - beta * dist
    target_ood = target_ood.detach()
    ood_losses = []
    for q_val in q_ood_values:
        q_view = q_val.view(ood_actions.shape[0], num_actions)
        loss = F.huber_loss(q_view, target_ood, delta=10.0)
        ood_losses.append(loss)
    loss_anchor = torch.stack(ood_losses).mean()
    loss_anchor_weight = float(getattr(trainer.cfg, "loss_anchor_weight", 1.0))
    q_ood = aggregate_q(q_ood_values, getattr(trainer.cfg, "q_aggregation", "mean")).view(
        ood_actions.shape[0], num_actions
    )
    
    d_diff = dist.unsqueeze(1) - dist.unsqueeze(2)
    q_diff = q_ood.unsqueeze(2) - q_ood.unsqueeze(1)
    target_diff = beta * d_diff
    loss_rank = F.mse_loss(q_diff, target_diff)
    loss_rank_weight = float(getattr(trainer.cfg, "loss_rank_weight", 1.0))
    loss_ood = loss_anchor * loss_anchor_weight + loss_rank_weight * loss_rank

    m_policy = int(ood_payload.get("m_policy", 0) or 0)
    m_noise = int(ood_payload.get("m_noise", 0) or 0)
    m_mix = int(ood_payload.get("m_mix", 0) or 0)
    m_trunc = int(ood_payload.get("m_trunc", 0) or 0)
    idx_trunc = m_policy + m_noise + m_mix
    if m_policy > 0:
        win_policy = float((q_batch.view(-1, 1) > q_ood[:, :m_policy].mean(dim=1, keepdim=True)).float().mean().item())
    else:
        win_policy = 0.0
    if m_trunc > 0 and idx_trunc < q_ood.shape[1]:
        q_trunc = q_ood[:, idx_trunc : idx_trunc + m_trunc]
        win_trunc = float((q_batch.view(-1, 1) > q_trunc.mean(dim=1, keepdim=True)).float().mean().item())
    else:
        win_trunc = float((q_batch.view(-1, 1) > q_ood.max(dim=1, keepdim=True).values).float().mean().item())

    metrics = {
        "ood_loss": float(loss_ood.item()),
        "ood_q_mean": float(q_ood.mean().item()),
        "ood_target_mean": float(target_ood.mean().item()),
        "win_rate/gt_vs_policy": win_policy,
        "win_rate/gt_vs_trunc": win_trunc,
    }
    if loss_rank is not None:
        metrics["ood_loss_pairwise_raw"] = float(loss_rank.item())
    return loss_ood, metrics, q_ood
