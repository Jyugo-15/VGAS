"""Reward-aware wrapper around LeRobotDataset.

This module provides a thin subclass of ``LeRobotDataset`` that exposes
episode-level accessors (`get_episode` / `load_episode`) and surfaces any
``reward``/``terminal`` signals that were injected into the local dataset.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Dict, Iterable

import torch

DEFAULT_ACTION_KEY = "actions"

try:  # pragma: no cover - optional dependency resolved at runtime
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:  # pragma: no cover - surfaced in datamodule before use
    LeRobotDataset = None  # type: ignore[assignment]
else:  # pragma: no cover - optional dependency at runtime
    from lerobot.utils.constants import ACTION as DEFAULT_ACTION_KEY


def _stack_sequence(values: Iterable[Any]) -> Any:
    """Collate a list of per-frame values into batched tensors/lists."""

    values = list(values)
    if not values:
        return values

    first = values[0]
    if isinstance(first, torch.Tensor):
        stacked = torch.stack([torch.as_tensor(val) for val in values], dim=0)
        return stacked
    if isinstance(first, (float, int)):
        dtype = torch.float32 if isinstance(first, float) else torch.long
        return torch.tensor(values, dtype=dtype)
    if isinstance(first, bool):
        return torch.tensor(values, dtype=torch.bool)
    return values


if LeRobotDataset is None:  # pragma: no cover - handled upstream

    class RewardAugmentedLeRobotDataset:  # type: ignore[no-redef]
        def __init__(self, *_args, **_kwargs):
            raise ImportError(
                "LeRobotDataset is unavailable. Install lerobot or make sure it is on PYTHONPATH."
            )

else:

    class RewardAugmentedLeRobotDataset(LeRobotDataset):
        """Extension of ``LeRobotDataset`` that exposes rewards/dones per episode."""

        def __init__(
            self,
            *args,
            reward_key: str = "reward",
            terminal_key: str = "terminal",
            action_key: str = DEFAULT_ACTION_KEY,
            cache_size: int = 50,
            chunk_size: int = 1,
            q_chunk_len: int | None = None,
            include_future_observation: bool = True,
            max_action_dim: int | None = None,
            discount: float = 0.99,
            allow_missing_reward: bool = False,
            mlp_critic: bool = False,
            **kwargs,
        ) -> None:
            delta_timestamps = kwargs.get("delta_timestamps")
            mirrored_reward_delta = False
            if delta_timestamps is not None and isinstance(delta_timestamps, dict):
                reward_has_delta = reward_key in delta_timestamps
                if not reward_has_delta:
                    candidate_keys = (
                        action_key,
                        DEFAULT_ACTION_KEY,
                        "action",
                    )
                    action_deltas = None
                    for candidate in candidate_keys:
                        if candidate in delta_timestamps:
                            action_deltas = delta_timestamps[candidate]
                            break

                    if action_deltas is not None:
                        cloned_delta = {
                            key: list(value) if isinstance(value, (list, tuple)) else value
                            for key, value in delta_timestamps.items()
                        }
                        cloned_delta[reward_key] = list(action_deltas)
                        kwargs["delta_timestamps"] = cloned_delta
                        mirrored_reward_delta = True

            super().__init__(*args, **kwargs)
            self.reward_key = reward_key
            self.terminal_key = terminal_key
            self.action_key = action_key
            self._cache_size = max(1, cache_size)
            self._episode_cache: OrderedDict[int, Dict[str, Any]] = OrderedDict()
            self._mirrored_reward_delta = mirrored_reward_delta
            self.chunk_size = max(1, int(chunk_size))
            self.q_chunk_len = int(q_chunk_len) if q_chunk_len is not None else self.chunk_size
            self.include_future_observation = include_future_observation
            self.max_action_dim = max_action_dim
            self.discount = float(discount)
            self.allow_missing_reward = allow_missing_reward
            self.mlp_critic = bool(mlp_critic)

            if mirrored_reward_delta:
                logging.info(
                    "delta_timestamps for '%s' not provided; mirroring values from '%s'.",
                    reward_key,
                    action_key,
                )

            missing = [key for key in (self.reward_key, self.terminal_key) if key not in self.features]
            if missing:
                logging.warning(
                    "RewardAugmentedLeRobotDataset: dataset does not expose keys %s. "
                    "Falling back to zeros for the missing modalities.",
                    missing,
                )
            self._episode_ranges: list[tuple[int, int]] = []
            self._index_to_episode: list[int] = []
            self._window_indices: list[tuple[int, int]] = []
            self._build_episode_mappings()

        # ------------------------------------------------------------------ #
        # Episode helpers
        # ------------------------------------------------------------------ #
        def _build_episode_mappings(self) -> None:
            total_frames = 0
            for ep_idx, record in enumerate(self.meta.episodes):
                start = int(record["dataset_from_index"])
                end = int(record["dataset_to_index"])
                self._episode_ranges.append((start, end))
                length = max(0, end - start)
                total_frames += length
                self._index_to_episode.extend([ep_idx] * length)

                if self.chunk_size > 1:
                    if self.mlp_critic:
                        max_start = max(0, length - self.q_chunk_len + 1)
                    else:
                        max_start = max(1, length)
                    for offset in range(max_start):
                        self._window_indices.append((ep_idx, start + offset))

        def _episode_for_index(self, dataset_index: int) -> int:
            if 0 <= dataset_index < len(self._index_to_episode):
                return self._index_to_episode[dataset_index]
            raise IndexError(f"Dataset index {dataset_index} out of bounds.")

        def _episode_bounds(self, episode_index: int) -> tuple[int, int]:
            record = self.meta.episodes[int(episode_index)]
            start = int(record["dataset_from_index"])
            end = int(record["dataset_to_index"])
            return start, end

        def _materialize_episode(self, episode_index: int) -> Dict[str, Any]:
            start, end = self._episode_bounds(episode_index)
            if end <= start:
                raise ValueError(f"Episode {episode_index} is empty: start={start}, end={end}")

            # Use the parent class __getitem__ to preserve LeRobot's dynamic features
            # (delta timestamps, *_is_pad masks, transforms, etc.).
            frames = [super(RewardAugmentedLeRobotDataset, self).__getitem__(idx) for idx in range(start, end)]
            stacked: Dict[str, Any] = {}
            for key in frames[0]:
                stacked[key] = _stack_sequence(frame[key] for frame in frames)

            length = end - start
            if self.action_key not in stacked:
                raise KeyError(f"Dataset does not provide '{self.action_key}'; cannot build episodes.")
            actions = torch.as_tensor(stacked.get(self.action_key))
            if actions.dim() == 1:
                actions = actions.unsqueeze(-1)
            stacked["actions"] = actions

            raw_reward = stacked.pop(self.reward_key, None)
            if raw_reward is None:
                rewards = torch.zeros(length, dtype=torch.float32)
            else:
                rewards = torch.as_tensor(raw_reward, dtype=torch.float32)
                if rewards.shape[0] != length:
                    rewards = rewards.reshape(length, -1)
            stacked["rewards"] = rewards.squeeze(-1)

            raw_done = stacked.pop(self.terminal_key, None)
            if raw_done is None:
                dones = torch.zeros(length, dtype=torch.bool)
            else:
                dones = torch.as_tensor(raw_done).to(torch.bool)
            stacked["dones"] = dones

            # Discounted return from the current step to the end of the episode (Monte-Carlo return).
            mc_returns = torch.zeros(length, dtype=torch.float32)
            running = torch.zeros(1, dtype=torch.float32)
            for idx in reversed(range(length)):
                running = rewards.flatten()[idx] + self.discount * running * (1.0 - dones[idx].float())
                mc_returns[idx] = running
            stacked["mc_returns"] = mc_returns

            return stacked

        def _cache_episode(self, episode_index: int, episode: Dict[str, Any]) -> None:
            self._episode_cache[episode_index] = episode
            self._episode_cache.move_to_end(episode_index)
            while len(self._episode_cache) > self._cache_size:
                self._episode_cache.popitem(last=False)

        def get_episode(self, episode_index: int) -> Dict[str, Any]:
            """Return all frames for a given episode (cached)."""

            episode_index = int(episode_index)
            cached = self._episode_cache.get(episode_index)
            if cached is not None:
                return cached

            episode = self._materialize_episode(episode_index)
            self._cache_episode(episode_index, episode)
            return episode

        def load_episode(self, episode_index: int) -> Dict[str, Any]:
            """Alias used by downstream loaders."""

            return self.get_episode(episode_index)

        def get_episode_length(self, episode_index: int) -> int:
            record = self.meta.episodes[int(episode_index)]
            return int(record["length"])

        def clear_episode_cache(self) -> None:
            self._episode_cache.clear()

        def __len__(self) -> int:
            if self.chunk_size > 1 and self._window_indices:
                return len(self._window_indices)
            return super().__len__()

        def _gather_chunk(
            self,
            episode_index: int,
            start_index: int,
        ) -> dict[str, torch.Tensor]:
            actions: list[torch.Tensor] = []
            rewards: list[float] = []
            dones: list[bool] = []
            pads: list[bool] = []
            mc_lb = torch.zeros(self.chunk_size, dtype=torch.float32)

            start, end = self._episode_ranges[episode_index]
            clamped_start = max(start, min(start_index, end - 1))
            last_valid = max(start, end - 1)

            for offset in range(self.chunk_size):
                cursor = clamped_start + offset
                pad = cursor >= end
                if pad:
                    cursor = last_valid

                row = self.hf_dataset[int(cursor)]
                action_value = row.get(self.action_key)
                action_tensor = torch.as_tensor(action_value, dtype=torch.float32)
                if action_tensor.ndim == 0:
                    action_tensor = action_tensor.unsqueeze(-1)
                if pad:
                    action_tensor = action_tensor.clone()
                    delta_dims = min(6, action_tensor.shape[-1])  # zero motion deltas
                    if delta_dims > 0:
                        action_tensor[..., :delta_dims] = 0.0
                actions.append(action_tensor)

                reward_value = row.get(self.reward_key, 0.0)
                rewards.append(float(reward_value))

                done_value = row.get(self.terminal_key, False)
                dones.append(bool(done_value))
                pads.append(pad)
                mc_val = row.get("mc_returns", 0.0)
                try:
                    mc_lb[offset] = float(mc_val)
                except (TypeError, ValueError):
                    mc_lb[offset] = 0.0

            chunk = {
                "actions": torch.stack(actions, dim=0),
                "rewards": torch.tensor(rewards, dtype=torch.float32),
                "dones": torch.tensor(dones, dtype=torch.bool),
                "actions_is_pad": torch.tensor(pads, dtype=torch.bool),
            }
            chunk["masks"] = (~chunk["actions_is_pad"]).to(torch.float32)
            chunk["mc_lower_bound"] = mc_lb
            return chunk

        def _pad_action_chunk(self, actions: torch.Tensor) -> torch.Tensor:
            if self.max_action_dim is None:
                return actions
            current_dim = actions.shape[-1]
            if current_dim == self.max_action_dim:
                return actions
            if current_dim > self.max_action_dim:
                return actions[..., : self.max_action_dim]
            padded = torch.zeros(
                (*actions.shape[:-1], self.max_action_dim),
                dtype=actions.dtype,
                device=actions.device,
            )
            padded[..., :current_dim] = actions
            return padded

        def _extract_observations(self, sample: Dict[str, Any]) -> Dict[str, Any]:
            return {k: v for k, v in sample.items() if k.startswith("observation.")}

        def _next_observation_snapshot(self, episode_index: int, start_index: int) -> Dict[str, Any]:
            start, end = self._episode_ranges[episode_index]
            desired = start_index + self.q_chunk_len  # use separate critic/future chunk
            pad = desired >= end
            target = min(max(start, desired), max(start, end - 1))
            try:
                future = super(RewardAugmentedLeRobotDataset, self).__getitem__(target)
            except KeyError:
                if not self.allow_missing_reward:
                    raise
                row = self.hf_dataset[int(target)]
                future = dict(row)
                future.setdefault(self.reward_key, 0.0)
            obs = self._extract_observations(future)
            if pad:
                for key in list(obs.keys()):
                    if key.endswith("_is_pad") and isinstance(obs[key], torch.Tensor):
                        obs[key] = torch.ones_like(obs[key], dtype=torch.bool)
            obs["next_observation_is_pad"] = torch.tensor([pad], dtype=torch.bool)
            remaining = 0 if pad else max(0, end - desired)
            obs["next_obs_valid_chunk_len"] = torch.tensor([remaining], dtype=torch.int64)
            return obs

        def __getitem__(self, index: int) -> Dict[str, Any]:
            if self.chunk_size <= 1 or not self._window_indices:
                try:
                    return super(RewardAugmentedLeRobotDataset, self).__getitem__(index)
                except KeyError:
                    if not self.allow_missing_reward:
                        raise
                    row = self.hf_dataset[int(index)]
                    sample = dict(row)
                    sample.setdefault(self.reward_key, 0.0)
                    return sample

            episode_index, global_start = self._window_indices[index]
            try:
                base_sample = super(RewardAugmentedLeRobotDataset, self).__getitem__(global_start)
            except KeyError:
                if not self.allow_missing_reward:
                    raise
                row = self.hf_dataset[int(global_start)]
                base_sample = dict(row)
                base_sample.setdefault(self.reward_key, 0.0)
                try:
                    task_idx = base_sample["task_index"].item()
                    base_sample["task"] = self.meta.tasks.iloc[task_idx].name
                except Exception:
                    pass

            chunk = self._gather_chunk(episode_index, global_start)
            base_sample.update(chunk)
            base_sample[self.action_key] = chunk["actions"]
            if self.include_future_observation:
                base_sample["next_observations"] = self._next_observation_snapshot(episode_index, global_start)
            return base_sample
