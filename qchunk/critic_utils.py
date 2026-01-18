"""Shared critic utilities."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch


def discounted_chunk_returns(rewards: torch.Tensor, reward_is_pad: torch.Tensor, discount: float) -> torch.Tensor:
    """Compute gamma-discounted sums along the chunk dimension while ignoring padded rewards."""
    if rewards.ndim == 1:
        rewards = rewards.unsqueeze(0)
    if reward_is_pad.ndim == 1:
        reward_is_pad = reward_is_pad.unsqueeze(0)
    chunk_len = rewards.shape[-1]
    device = rewards.device
    discounts = torch.pow(
        torch.full((chunk_len,), discount, device=device, dtype=rewards.dtype),
        torch.arange(chunk_len, device=device, dtype=rewards.dtype),
    )
    valid_mask = (~reward_is_pad.to(device=device, dtype=torch.bool)).to(rewards.dtype)
    masked_rewards = rewards * valid_mask
    return torch.sum(discounts * masked_rewards, dim=-1, keepdim=True)


def extract_future_batch(batch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return future observation batch if present."""
    if "next_observations" in batch and isinstance(batch["next_observations"], dict):
        return batch["next_observations"]
    return None


def aggregate_q(qs: tuple[torch.Tensor, ...], mode: str = "mean") -> torch.Tensor:
    """Aggregate multiple Q tensors according to mode."""
    if isinstance(qs, list):
        qs = tuple(qs)
    if not qs:
        raise ValueError("At least one Q tensor is required for aggregation.")
    if len(qs) == 1:
        return qs[0]
    stacked = torch.stack(qs, dim=0)
    agg = str(mode).lower()
    if agg == "min":
        return torch.min(stacked, dim=0).values
    if agg == "max":
        return torch.max(stacked, dim=0).values
    return stacked.mean(dim=0)


def soft_update_target(target: torch.nn.Module, online: torch.nn.Module, tau: float) -> None:
    """Polyak averaging target parameters."""
    for t_param, param in zip(target.parameters(), online.parameters(), strict=True):
        t_param.data.copy_(tau * param.data + (1.0 - tau) * t_param.data)


def get_tensor_from_batch(
    batch: Dict[str, Any],
    keys: list[str],
    default_shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    """Return the first tensor found by keys or a zeros fallback."""
    for key in keys:
        if key in batch and isinstance(batch[key], torch.Tensor):
            return batch[key]
    return torch.zeros(default_shape, device=device)


def get_raw_state(batch: Dict[str, Any], batch_size: int, raw_state_dim: int, device: torch.device) -> torch.Tensor:
    """Extract raw state or return zeros."""
    raw = batch.get("observation.state")
    if isinstance(raw, torch.Tensor):
        if raw.shape[0] != batch_size:
            raw = raw[:batch_size]
        return raw.to(device)
    return torch.zeros((batch_size, raw_state_dim), device=device, dtype=torch.float32)


def repeat_batch(batch: Dict[str, Any], batch_size: int, repeats: int) -> Dict[str, Any]:
    """Repeat batch along batch dimension for sampling."""
    if repeats <= 1:
        return batch
    expanded: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.shape[0] == batch_size:
            expanded[key] = torch.repeat_interleave(value, repeats, dim=0)
        else:
            expanded[key] = value
    return expanded
