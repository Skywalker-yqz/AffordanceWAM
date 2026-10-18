"""Canonical differentiable RGB/scalar-affordance overlay renderer.

The repository has two consumers of the same mathematical ``Overlay(R, F)``
operation:

* the dataset materialises an RGB ``uint8`` target; and
* the World objective compares continuous predicted pixels in autograd.

This module owns the continuous operation.  Colormaps are immutable 256-entry
LUTs sampled once from Matplotlib's named maps and evaluated with piecewise
linear interpolation.  Consequently the output remains differentiable with
respect to both RGB ``R`` and scalar affordance ``F``.  Dataset quantisation is
an explicit, separate final step.

Tensor layouts follow the repository's channel-first video convention:
``[3,H,W]``, ``[B,3,H,W]`` or ``[B,3,T,H,W]``.  Scalar affordance may omit or
retain its singleton channel in the corresponding layout.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Tuple

import numpy as np
import torch


OVERLAY_RENDERER_VERSION = "canonical-linear-lut256-v1"
COLORMAP_LUT_SIZE = 256
SUPPORTED_OVERLAY_COLORMAPS: Tuple[str, ...] = (
    "jet",
    "turbo",
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
)


@lru_cache(maxsize=len(SUPPORTED_OVERLAY_COLORMAPS))
def _cpu_colormap_lut(name: str) -> torch.Tensor:
    """Return the immutable canonical CPU FP32 LUT for ``name``."""

    if name not in SUPPORTED_OVERLAY_COLORMAPS:
        raise ValueError(
            f"unsupported overlay colormap {name!r}; expected one of "
            f"{SUPPORTED_OVERLAY_COLORMAPS}"
        )
    # Keep Matplotlib out of module import so lightweight config parsing does
    # not initialise its global state.  Sampling is deterministic for a given
    # renderer version and happens only once per colormap/process.
    import matplotlib.cm as cm

    samples = np.linspace(0.0, 1.0, COLORMAP_LUT_SIZE, dtype=np.float64)
    rgba = cm.get_cmap(name)(samples)
    rgb = np.ascontiguousarray(rgba[:, :3], dtype=np.float32)
    return torch.from_numpy(rgb)


_DEVICE_LUT_CACHE: dict[tuple[str, str, torch.dtype], torch.Tensor] = {}


def _colormap_lut(
    name: str, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    key = (name, str(device), dtype)
    lut = _DEVICE_LUT_CACHE.get(key)
    if lut is None:
        lut = _cpu_colormap_lut(name).to(device=device, dtype=dtype)
        _DEVICE_LUT_CACHE[key] = lut
    return lut


def get_overlay_colormap_lut(
    name: str, *, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Return a copy of one fixed canonical ``[256,3]`` LUT."""

    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError(f"colormap LUT dtype must be floating, got {dtype}")
    return _colormap_lut(name, device=torch.device(device), dtype=dtype).clone()


def _normalise_layout(
    rgb: torch.Tensor, scalar_affordance: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Move RGB channels last and remove scalar's optional channel."""

    if rgb.ndim == 3:
        if rgb.shape[0] != 3:
            raise ValueError(f"RGB image must be [3,H,W], got {tuple(rgb.shape)}")
        if scalar_affordance.ndim == 3 and scalar_affordance.shape[0] == 1:
            scalar_affordance = scalar_affordance[0]
        expected = (rgb.shape[1], rgb.shape[2])
        if tuple(scalar_affordance.shape) != expected:
            raise ValueError(
                f"scalar Afd must be {expected} or (1,{expected[0]},{expected[1]}), "
                f"got {tuple(scalar_affordance.shape)}"
            )
        return rgb.permute(1, 2, 0), scalar_affordance, "chw"

    if rgb.ndim == 4:
        if rgb.shape[1] != 3:
            raise ValueError(f"RGB batch must be [B,3,H,W], got {tuple(rgb.shape)}")
        if scalar_affordance.ndim == 4 and scalar_affordance.shape[1] == 1:
            scalar_affordance = scalar_affordance[:, 0]
        expected = (rgb.shape[0], rgb.shape[2], rgb.shape[3])
        if tuple(scalar_affordance.shape) != expected:
            raise ValueError(
                f"scalar Afd must be [B,H,W] or [B,1,H,W] matching RGB; "
                f"got {tuple(scalar_affordance.shape)} vs {tuple(rgb.shape)}"
            )
        return rgb.permute(0, 2, 3, 1), scalar_affordance, "bchw"

    if rgb.ndim == 5:
        if rgb.shape[1] != 3:
            raise ValueError(
                f"RGB video batch must be [B,3,T,H,W], got {tuple(rgb.shape)}"
            )
        if scalar_affordance.ndim == 5 and scalar_affordance.shape[1] == 1:
            scalar_affordance = scalar_affordance[:, 0]
        expected = (rgb.shape[0], rgb.shape[2], rgb.shape[3], rgb.shape[4])
        if tuple(scalar_affordance.shape) != expected:
            raise ValueError(
                "scalar Afd must be [B,T,H,W] or [B,1,T,H,W] matching RGB; "
                f"got {tuple(scalar_affordance.shape)} vs {tuple(rgb.shape)}"
            )
        return rgb.permute(0, 2, 3, 4, 1), scalar_affordance, "bcthw"

    raise ValueError(
        "RGB must be [3,H,W], [B,3,H,W] or [B,3,T,H,W], " f"got {tuple(rgb.shape)}"
    )


def _restore_layout(rgb_last: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "chw":
        return rgb_last.permute(2, 0, 1)
    if layout == "bchw":
        return rgb_last.permute(0, 3, 1, 2)
    if layout == "bcthw":
        return rgb_last.permute(0, 4, 1, 2, 3)
    raise AssertionError(f"unknown internal layout {layout!r}")


def _validate_unit_interval(name: str, value: torch.Tensor) -> None:
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point in [0,1], got {value.dtype}")
    if value.numel() == 0:
        raise ValueError(f"{name} must be non-empty")
    detached = value.detach()
    if not bool(torch.isfinite(detached).all()):
        raise ValueError(f"{name} contains NaN/Inf")
    if bool((detached < 0).any()) or bool((detached > 1).any()):
        raise ValueError(f"{name} must lie in [0,1]")


def render_overlay_tensor(
    rgb: torch.Tensor,
    scalar_affordance: torch.Tensor,
    *,
    alpha: float = 0.6,
    colormap_name: str = "jet",
) -> torch.Tensor:
    """Render a continuous differentiable RGB affordance overlay.

    Args:
        rgb: Floating RGB in ``[0,1]`` using CHW/BCHW/BCTHW layout.
        scalar_affordance: Floating heatmap in ``[0,1]`` with the matching
            layout and an optional singleton channel.
        alpha: Global blend opacity used on frames containing any non-zero Afd.
        colormap_name: One of :data:`SUPPORTED_OVERLAY_COLORMAPS`.

    Returns:
        A channel-first tensor in the same shape as ``rgb``.  FP16/BF16 inputs
        are evaluated and returned in FP32 so LUT interpolation and the pixel
        loss retain stable gradients.  A frame whose heatmap is entirely zero
        is exactly the original RGB frame.
    """

    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError(f"alpha must be a finite number in [0,1], got {alpha!r}")
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be a finite number in [0,1], got {alpha!r}")
    if colormap_name not in SUPPORTED_OVERLAY_COLORMAPS:
        raise ValueError(
            f"unsupported overlay colormap {colormap_name!r}; expected one of "
            f"{SUPPORTED_OVERLAY_COLORMAPS}"
        )
    if rgb.device != scalar_affordance.device:
        raise ValueError(
            f"RGB/scalar Afd must share a device, got {rgb.device} and "
            f"{scalar_affordance.device}"
        )
    _validate_unit_interval("RGB", rgb)
    _validate_unit_interval("scalar Afd", scalar_affordance)

    rgb_last, heatmap, layout = _normalise_layout(rgb, scalar_affordance)
    compute_dtype = (
        torch.float64
        if torch.float64 in (rgb.dtype, scalar_affordance.dtype)
        else torch.float32
    )
    rgb_last = rgb_last.to(dtype=compute_dtype)
    heatmap = heatmap.to(dtype=compute_dtype)
    lut = _colormap_lut(colormap_name, device=rgb.device, dtype=compute_dtype)

    # Piecewise-linear LUT lookup.  Index selection is piecewise constant, but
    # ``fraction`` carries the within-bin derivative back to scalar affordance.
    position = heatmap * float(COLORMAP_LUT_SIZE - 1)
    lower = position.floor().to(torch.long)
    upper = (lower + 1).clamp_max(COLORMAP_LUT_SIZE - 1)
    fraction = (position - lower.to(dtype=compute_dtype)).unsqueeze(-1)
    coloured = lut[lower] + fraction * (lut[upper] - lut[lower])

    blended = rgb_last * (1.0 - alpha) + coloured * alpha
    # Presence is intentionally per frame, matching the historical dataset
    # contract: a wholly zero frame is RGB passthrough; once any Afd is present,
    # the named colormap/alpha is applied over the full frame.
    present = heatmap.amax(dim=(-2, -1), keepdim=True) > 0
    rendered = torch.where(present.unsqueeze(-1), blended, rgb_last)
    return _restore_layout(rendered.clamp(0.0, 1.0), layout).contiguous()


def quantize_overlay_uint8(overlay: torch.Tensor) -> torch.Tensor:
    """Quantise a continuous ``[0,1]`` overlay at the dataset boundary."""

    if not overlay.is_floating_point():
        raise TypeError(f"overlay must be floating point, got {overlay.dtype}")
    if not bool(torch.isfinite(overlay.detach()).all()):
        raise ValueError("overlay contains NaN/Inf")
    return overlay.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)


__all__ = [
    "COLORMAP_LUT_SIZE",
    "OVERLAY_RENDERER_VERSION",
    "SUPPORTED_OVERLAY_COLORMAPS",
    "get_overlay_colormap_lut",
    "quantize_overlay_uint8",
    "render_overlay_tensor",
]
