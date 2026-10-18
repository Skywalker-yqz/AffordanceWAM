"""Euler integration schedule for continuous action generation.

The sampling grid applies Wan's shifted flow-time parameterization:
``sigma = shift * s / (1 + (shift - 1) * s)``. Each Euler update is
``x_next = x + velocity * (sigma_next - sigma)``. Model training samples
continuous uniform flow times independently of this inference grid.
"""
from __future__ import annotations

from typing import Optional

import torch


class ActionScheduler:
    """Flow-matching noise scheduler for the continuous action stream."""

    num_train_timesteps: int = 1000

    def __init__(self, shift: float = 5.0) -> None:
        self.shift = float(shift)
        self.sigmas: Optional[torch.Tensor] = None
        self.timesteps: Optional[torch.Tensor] = None
        self.linear_timesteps_weights: Optional[torch.Tensor] = None
        self.training: bool = False

    # ------------------------------------------------------------------
    def set_timesteps(
        self,
        num_inference_steps: int = 1000,
        denoising_strength: float = 1.0,
        training: bool = False,
        shift: Optional[float] = None,
    ) -> "ActionScheduler":
        """Build the (descending) sigma / timestep grid.

        Training passes ``num_inference_steps == num_train_timesteps`` (1000) so
        the grid covers every train timestep and the BSMNTW weights are exact.
        """
        shift = self.shift if shift is None else float(shift)
        # sigma_min = 0, sigma_max = 1 (Wan convention).
        sigma_start = 1.0 * denoising_strength
        sigmas = torch.linspace(sigma_start, 0.0, num_inference_steps + 1)[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        self.sigmas = sigmas
        self.timesteps = sigmas * self.num_train_timesteps
        if training:
            self._set_training_weight()
            self.training = True
        else:
            self.training = False
        return self

    def _set_training_weight(self) -> None:
        steps = self.num_train_timesteps
        x = self.timesteps
        y = torch.exp(-2 * ((x - steps / 2) / steps) ** 2)
        y_shifted = y - y.min()
        bsmntw = y_shifted * (steps / y_shifted.sum())
        if len(self.timesteps) != steps:  # empirical re-norm for short grids
            bsmntw = bsmntw * (len(self.timesteps) / steps)
            bsmntw = bsmntw + bsmntw[1]
        self.linear_timesteps_weights = bsmntw

    # ------------------------------------------------------------------
    @staticmethod
    def add_noise(
        original: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """``x_t = (1 - sigma) * x0 + sigma * eps`` (sigma broadcast over x0)."""
        return (1 - sigma) * original + sigma * noise

    @staticmethod
    def training_target(original: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Flow-matching velocity target ``eps - x0``."""
        return noise - original

    def training_weight(self, timestep_ids: torch.Tensor) -> torch.Tensor:
        assert (
            self.linear_timesteps_weights is not None
        ), "set_timesteps(training=True) first"
        return self.linear_timesteps_weights[timestep_ids]

    @staticmethod
    def flow_step(
        pred: torch.Tensor,
        sigma,
        sigma_next,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """One Euler flow step ``x_next = x + v_pred * (sigma_next - sigma)``."""
        return sample + pred * (sigma_next - sigma)

    # ------------------------------------------------------------------
    def to(self, device) -> "ActionScheduler":
        for name in ("sigmas", "timesteps", "linear_timesteps_weights"):
            t = getattr(self, name)
            if t is not None:
                setattr(self, name, t.to(device))
        return self


__all__ = ["ActionScheduler"]
