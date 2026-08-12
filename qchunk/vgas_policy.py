"""Lightweight wrapper that bundles an actor and the Best-of-N critic."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch

from qchunk.qchunked_critic import PolicyEncoderFn, QChunkedCritic


class VGASPolicy:
    """
    Bundle an actor with a critic to expose unified inference/training APIs.

    - `predict_action_chunk` delegates to the underlying actor.
    - `predict_chunk_best_of_n` mirrors best-of-n selection.
    - `update_critic` updates the critic while treating the actor as a target/action source.
    """

    def __init__(
        self,
        actor: Any,
        critic: QChunkedCritic,
        encoder_fn: Optional[PolicyEncoderFn] = None,
    ) -> None:
        self.actor = actor
        self.critic = critic
        # Use critic's encoder by default to stay consistent with its cfg.
        self._encode = encoder_fn or getattr(critic, "_encode", None)

    @classmethod
    def from_pretrained(
        cls,
        *,
        policy_path: Path,
        policy_cfg: Any,
        critic_cfg: Any,
        sample_batch: Dict[str, torch.Tensor],
        ds_meta: Any = None,
        device: str = "cuda",
        encoder_fn: Optional[PolicyEncoderFn] = None,
    ) -> "VGASPolicy":
        """
        Construct a VGASPolicy by loading a frozen actor and building a critic.

        Args:
            policy_path: directory of the pretrained actor checkpoint (pretrained_model).
            policy_cfg: loaded SmolVLA config object (device/n_action_steps will be set).
            critic_cfg: CriticConfig used by QChunkedCritic.build.
            sample_batch: one preprocessed batch to bootstrap critic shapes.
            ds_meta: optional dataset metadata for feature inference.
            device: torch device string.
            encoder_fn: optional encoder override (defaults to encode_policy_observations).
        """
        # Lazy imports to avoid hard-coding dependencies at module import time.
        from utils.local_policy import make_local_smolvla_policy
        from scripts.train_qchunk_offline import encode_policy_observations

        policy_cfg.device = str(device)
        policy_cfg.pretrained_path = str(policy_path)
        actor = make_local_smolvla_policy(cfg=policy_cfg, ds_meta=ds_meta)
        actor.eval()
        for param in actor.parameters():
            param.requires_grad = False

        encoder = encoder_fn or encode_policy_observations
        critic = QChunkedCritic.build(
            policy_path=policy_path,
            policy_cfg=policy_cfg,
            critic_cfg=critic_cfg,
            sample_batch=sample_batch,
            ds_meta=ds_meta,
            device=device,
            encoder_fn=encoder,
            actor=actor,  # reuse loaded actor; do not rebuild inside the critic
        )
        return cls(actor=actor, critic=critic, encoder_fn=encoder)

    @property
    def device(self) -> torch.device:
        return torch.device(getattr(self.actor, "device", getattr(self.critic, "device", "cpu")))

    @property
    def has_critic(self) -> bool:
        return self.critic is not None

    def predict_action_chunk(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward-only actor chunk prediction."""
        return self.actor.predict_action_chunk(batch)

    def predict_chunk_best_of_n(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        action_samples: Optional[int] = None,
        forced_mask: Optional[torch.Tensor] = None,
        return_candidates: bool = False,
    ):
        """
        Sample candidate chunks from the actor and pick the best via the critic.

        Returns (best_actions, best_q) or (best_actions, best_q, candidates) when
        `return_candidates=True`.
        """
        if self.critic is None:
            raise RuntimeError("critic is not initialized.")
        return self.critic.predict_best_of_n(
            batch,
            action_samples=action_samples,
            forced_mask=forced_mask,
            return_candidates=return_candidates,
        )

    def update_critic(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        current_step: Optional[int] = None,
        ood_warmup_steps: int = 0,
        ood_policy: Optional[Any] = None,
        target_action_samples: Optional[int] = None,
        target_policy: Optional[Any] = None,
        cached_next_encoding: Optional[Any] = None,
        step_scheduler: bool = True,
    ) -> Dict[str, float]:
        """
        Update only the critic; this method never changes actor parameters.
        """
        return self.critic.update(
            batch,
            current_step=current_step,
            ood_warmup_steps=ood_warmup_steps,
            ood_policy=ood_policy,
            target_action_samples=target_action_samples,
            target_policy=target_policy,
            cached_next_encoding=cached_next_encoding,
            step_scheduler=step_scheduler,
        )

    # ------------------------------------------------------------------ #
    # State helpers
    # ------------------------------------------------------------------ #
    def critic_state_dict(self) -> Dict[str, Any]:
        """Return critic + target + optimizer state for checkpointing."""
        if self.critic is None:
            return {}
        return self.critic.state_dict()

    def load_critic_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore critic/target/optimizer state."""
        if self.critic is None:
            raise RuntimeError("critic is not initialized.")
        self.critic.load_state_dict(state)
