"""Neural network components used by Q-Chunking trainers."""

from typing import Dict, Sequence

import torch
from torch import nn


def _flatten_observations(observations: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Concatenate observation tensors along the feature dimension."""
    flats = []
    for key in sorted(observations.keys()):
        tensor = observations[key]
        if tensor.ndim > 2:
            tensor = tensor.flatten(start_dim=1)
        flats.append(tensor)
    return torch.cat(flats, dim=-1) if flats else torch.empty(0)


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: Sequence[int], activation=nn.ReLU):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden))
            layers.append(activation())
            prev_dim = hidden
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CriticBackbone(nn.Module):
    """Twin Q-network backbone."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: Sequence[int] = (512, 512)):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        input_dim = obs_dim + action_dim
        self.net1 = MLP(input_dim, 1, hidden_sizes)
        self.net2 = MLP(input_dim, 1, hidden_sizes)

    def forward(self, observations: Dict[str, torch.Tensor], actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(observations, dict):
            batch_obs = _flatten_observations(observations)
        else:
            batch_obs = observations.view(observations.shape[0], -1)
        batch_actions = actions.view(actions.shape[0], -1)
        x = torch.cat([batch_obs, batch_actions], dim=-1)
        return self.net1(x), self.net2(x)


class CriticHead(nn.Module):
    """Applies reduction over chunk dimension."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.reduction = reduction

    def forward(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean":
            return 0.5 * (q1 + q2)
        return q1 + q2


class ActorBackbone(nn.Module):
    """Simple residual MLP that can refine SmolVLA outputs if desired."""

    def __init__(self, action_dim: int, hidden_sizes: Sequence[int] = (256, 256)):
        super().__init__()
        self.mlp = MLP(action_dim, action_dim, hidden_sizes)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        refined = self.mlp(actions.view(actions.shape[0], -1))
        return refined.view_as(actions)


class ActorHead(nn.Module):
    """Combine base policy output with residual corrections."""

    def __init__(self, residual_scale: float = 0.1):
        super().__init__()
        self.residual_scale = residual_scale

    def forward(self, base_actions: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return base_actions + self.residual_scale * residual
