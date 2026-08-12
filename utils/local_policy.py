"""Helpers to instantiate the repository-local SmolVLA policy.

This avoids routing policy construction through lerobot's policy registry while still
reusing lerobot feature-mapping utilities and processor pipelines.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.envs.configs import EnvConfig
from lerobot.envs.utils import env_to_policy_features

from models.smolvla.modeling_smolvla import SmolVLAPolicy


def _populate_policy_features(cfg: Any, ds_meta: LeRobotDatasetMetadata | None, env_cfg: EnvConfig | None) -> None:
    """Populate cfg.input_features/output_features when metadata is available."""

    if ds_meta is not None and env_cfg is not None:
        raise ValueError("Provide either ds_meta or env_cfg, not both.")

    if ds_meta is not None:
        features = dataset_to_policy_features(ds_meta.features)
    elif env_cfg is not None:
        features = env_to_policy_features(env_cfg)
    else:
        has_features = bool(getattr(cfg, "input_features", None)) and bool(getattr(cfg, "output_features", None))
        if not has_features:
            raise ValueError("Policy features are missing; provide ds_meta or env_cfg to infer them.")
        return

    cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    cfg.input_features = {key: ft for key, ft in features.items() if key not in cfg.output_features}


def make_local_smolvla_policy(
    *,
    cfg: Any,
    ds_meta: LeRobotDatasetMetadata | None = None,
    env_cfg: EnvConfig | None = None,
) -> SmolVLAPolicy:
    """Instantiate local SmolVLA policy from scratch or pretrained weights."""

    _populate_policy_features(cfg, ds_meta, env_cfg)

    pretrained_path = getattr(cfg, "pretrained_path", None)
    if pretrained_path:
        policy = SmolVLAPolicy.from_pretrained(
            pretrained_name_or_path=str(pretrained_path),
            config=cfg,
        )
    else:
        if env_cfg is not None:
            logging.warning(
                "Instantiating local SmolVLA from scratch using env features; "
                "normalization stats may be unset without dataset metadata."
            )
        policy = SmolVLAPolicy(config=cfg)

    policy.to(cfg.device)
    assert isinstance(policy, torch.nn.Module)
    return policy
