"""Per-dimension normalization for actions and proprioceptive state.

``q99`` maps the [q01,q99] interval to [-1,1] and clips values to that range.
``mean_std`` applies (x - mean) / std. Inverse transforms restore control units.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

NORM_EPS: float = 1e-8
VALID_MODES = ("q99", "mean_std")


def _q99_scale_offset(stats: Dict, eps: float = NORM_EPS):
    q01 = np.asarray(stats["q01"], dtype=np.float64)
    q99 = np.asarray(stats["q99"], dtype=np.float64)
    rng = np.maximum(q99 - q01, eps)
    scale = 2.0 / rng
    offset = q01 + rng / 2.0  # center of [q01, q99]
    return scale, offset


def normalize(
    x: np.ndarray, stats: Dict, *, mode: str = "q99", eps: float = NORM_EPS
) -> np.ndarray:
    """Normalize ``[..., D]`` by per-dim ``stats`` under ``mode``."""
    x = np.asarray(x, dtype=np.float64)
    if mode == "q99":
        scale, offset = _q99_scale_offset(stats, eps)
        return np.clip((x - offset) * scale, -1.0, 1.0)
    if mode == "mean_std":
        mean = np.asarray(stats["mean"], dtype=np.float64)
        std = np.asarray(stats["std"], dtype=np.float64)
        return (x - mean) / (std + eps)
    raise ValueError(f"unknown norm mode {mode!r}; valid: {VALID_MODES}")


def denormalize(
    x: np.ndarray, stats: Dict, *, mode: str = "q99", eps: float = NORM_EPS
) -> np.ndarray:
    """Inverse of :func:`normalize` (note: q99 clip is not inverted for
    out-of-range values — clipped tails are unrecoverable by construction)."""
    x = np.asarray(x, dtype=np.float64)
    if mode == "q99":
        scale, offset = _q99_scale_offset(stats, eps)
        return x / scale + offset
    if mode == "mean_std":
        mean = np.asarray(stats["mean"], dtype=np.float64)
        std = np.asarray(stats["std"], dtype=np.float64)
        return x * (std + eps) + mean
    raise ValueError(f"unknown norm mode {mode!r}; valid: {VALID_MODES}")


__all__ = ["NORM_EPS", "VALID_MODES", "normalize", "denormalize"]
