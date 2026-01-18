"""Utilities to create SmolVLA pre/post processors and map batches."""

from dataclasses import asdict, fields
from typing import Any, Dict, Mapping, Optional, Tuple
import warnings

import torch

from configs import SmolVLAQChunkConfig
from data import TrajectoryBatch
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from models.smolvla.processor import make_smolvla_pre_post_processors


def _serialize_smolvla_section(project_config: SmolVLAQChunkConfig) -> dict:
    """Filter project SmolVLA section to keys supported by SmolVLAConfig."""

    raw = asdict(project_config.smolvla)
    valid_keys = {field.name for field in fields(SmolVLAConfig)}
    filtered = {key: raw[key] for key in valid_keys.intersection(raw.keys())}
    ignored = set(raw) - valid_keys
    if ignored:
        warnings.warn(
            f"Ignoring unsupported SmolVLA config keys: {sorted(ignored)}",
            RuntimeWarning,
            stacklevel=2,
        )
    return filtered


def build_policy_processors(
    project_config: SmolVLAQChunkConfig,
    dataset_stats: Dict[str, Dict[str, torch.Tensor]] | None = None,
    smolvla_cfg: Optional[SmolVLAConfig] = None,
) -> Tuple[Any, Any]:
    """Create SmolVLA pre/post processor pipelines."""

    if smolvla_cfg is None:
        smolvla_cfg = SmolVLAConfig(**_serialize_smolvla_section(project_config))
    preproc, postproc = make_smolvla_pre_post_processors(smolvla_cfg, dataset_stats)
    return preproc, postproc


def to_policy_batch(
    batch: TrajectoryBatch,
    preprocessor: Any,
) -> Dict[str, torch.Tensor]:
    """Convert a trajectory batch to the format expected by SmolVLA."""

    features: Dict[str, Any] = {}
    features.update(batch.observations)
    features["actions"] = batch.actions
    features["rewards"] = batch.rewards
    features["masks"] = batch.masks

    processed = preprocessor(features)
    return processed
