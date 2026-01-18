"""Adapters that unify various critic heads under a twin-Q style interface."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn

from qchunk.networks import CriticBackbone
from qchunk.valuequeryhead import ValueHeadCritic, ValueQueryHead

class PolicyEmbeddings:
    """Lightweight container mirroring the structure used in the train script."""

    def __init__(self, pooled: torch.Tensor, prefix_outs: torch.Tensor, pad_masks: torch.Tensor, att_masks: torch.Tensor):
        self.pooled = pooled
        self.prefix_outs = prefix_outs
        self.pad_masks = pad_masks
        self.att_masks = att_masks

    def to(self, device: torch.device | str) -> "PolicyEmbeddings":
        return PolicyEmbeddings(
            pooled=self.pooled.to(device),
            prefix_outs=self.prefix_outs.to(device),
            pad_masks=self.pad_masks.to(device),
            att_masks=self.att_masks.to(device),
        )

    def repeat(self, repeats: int) -> "PolicyEmbeddings":
        def _repeat(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.repeat_interleave(repeats, dim=0)

        return PolicyEmbeddings(
            pooled=_repeat(self.pooled),
            prefix_outs=_repeat(self.prefix_outs),
            pad_masks=_repeat(self.pad_masks),
            att_masks=_repeat(self.att_masks),
        )


class MLPCriticAdapter(nn.Module):
    """Wrap the twin-MLP critic so it consumes ``PolicyEmbeddings``."""

    def __init__(self, backbone: CriticBackbone):
        super().__init__()
        self.backbone = backbone

    def forward(
        self,
        encoding: PolicyEmbeddings,
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.backbone(encoding.pooled, actions)


class ValueQueryCriticAdapter(nn.Module):
    """Wrap one or more ValueQueryHead modules and expose a twin-Q interface."""

    def __init__(self, heads: Sequence[ValueQueryHead] | ValueQueryHead):
        super().__init__()
        if isinstance(heads, ValueQueryHead):
            heads = [heads]
        if len(heads) == 0:
            raise ValueError("ValueQueryCriticAdapter requires at least one ValueQueryHead.")
        self.heads = nn.ModuleList(heads)

    def forward(
        self,
        encoding: PolicyEmbeddings,
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        return tuple(
            head.forward_from_embeddings(
                encoding.prefix_outs,
                encoding.pad_masks,
                encoding.att_masks,
                actions,
                action_mask,
            )
            for head in self.heads
        )


class ValueHeadCriticAdapter(nn.Module):
    """Wrap one or more ValueHeadCritic modules and expose a twin-Q interface."""

    def __init__(self, heads: Sequence[ValueHeadCritic] | ValueHeadCritic):
        super().__init__()
        if isinstance(heads, ValueHeadCritic):
            heads = [heads]
        if len(heads) == 0:
            raise ValueError("ValueHeadCriticAdapter requires at least one head.")
        self.heads = nn.ModuleList(heads)

    def forward(
        self,
        encoding: PolicyEmbeddings,
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        outputs = tuple(
            head.forward_from_embeddings(
                encoding.prefix_outs,
                encoding.pad_masks,
                encoding.att_masks,
                actions,
                action_mask,
                **kwargs,
            )
            for head in self.heads
        )
        if len(outputs) == 1:
            return outputs[0], outputs[0]
        return outputs
