"""Checkpoint management for multi-stage training."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch


@dataclass
class CheckpointManager:
    """Handles saving/loading actor & critic weights together."""

    directory: Path

    def __post_init__(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)

    def latest(self) -> Optional[Path]:
        checkpoints = sorted(self.directory.glob("step_*.pt"))
        return checkpoints[-1] if checkpoints else None

    def save(
        self,
        step: int,
        *,
        actor_state: Dict[str, Any],
        critic_state: Dict[str, Any],
        optimizer_state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        payload = {
            "step": step,
            "actor": actor_state,
            "critic": critic_state,
            "optimizer": optimizer_state,
            "metadata": metadata or {},
        }
        path = self.directory / f"step_{step:07d}.pt"
        torch.save(payload, path)
        return path

    def load(self, path: Optional[Path] = None) -> Dict[str, Any]:
        ckpt_path = path or self.latest()
        if ckpt_path is None:
            raise FileNotFoundError("No checkpoint available.")
        return torch.load(ckpt_path, map_location="cpu")
