"""Proprioceptive conditioning for the Action Expert."""

import torch
from torch import nn


class StateEncoder(nn.Module):
    """Project the current proprioceptive state to one Action context token."""

    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.proj(state).unsqueeze(1)
