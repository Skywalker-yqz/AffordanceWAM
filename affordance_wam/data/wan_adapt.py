"""Conversion of aligned RGB and Scalar Affordance tensors to model inputs."""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from affordance_wam.base_model import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    NUM_PIXEL_FRAMES,
    BaseModelSpec,
)

VIDEO_KEY = "video"
SCALAR_AFFORDANCE_KEY = "affordance"
AFFORDANCE_VIDEO_KEY = "affordance_video"


def wan_pixels_from_uint8(
    video_uint8: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Convert a ``uint8`` video tensor to the Wan VAE's ``[-1, 1]`` input space.

    Accepts ``[3, T, H, W]`` or ``[B, 3, T, H, W]`` (or already-float tensors,
    which are assumed normalised and only cast/moved). Mirrors the vendored
    ``preprocess_image`` affine ``x * (2/255) - 1`` exactly so latents match a
    PIL→preprocess round-trip.
    """
    x = video_uint8
    if device is not None:
        x = x.to(device)
    if x.dtype == torch.uint8:
        x = x.to(torch.float32) * (2.0 / 255.0) - 1.0
    return x.to(dtype=dtype).contiguous()


def wan_batch_to_pixels(
    batch: Dict[str, torch.Tensor],
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str | None = None,
    video_key: str = VIDEO_KEY,
    affordance_video_key: str = AFFORDANCE_VIDEO_KEY,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(rgb_pixels, overlay_pixels)`` in Wan ``[-1, 1]`` space.

    This helper is for Afd-present pretraining batches; V+L+A posttraining uses
    only RGB history and an internally predicted world condition.
    """
    if video_key not in batch:
        raise KeyError(f"batch missing {video_key!r}; got keys {list(batch)}")
    if affordance_video_key not in batch:
        raise KeyError(
            f"batch missing {affordance_video_key!r}; the AffordanceWAM objective "
            f"needs both the RGB and affordance video streams."
        )
    rgb = wan_pixels_from_uint8(batch[video_key], dtype=dtype, device=device)
    aff = wan_pixels_from_uint8(batch[affordance_video_key], dtype=dtype, device=device)
    return rgb, aff


def wan_scalar_affordance_from_batch(
    batch: Dict[str, torch.Tensor],
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
    key: str = SCALAR_AFFORDANCE_KEY,
) -> torch.Tensor:
    """Load genuine one-channel Afd as ``[B,1,T,H,W]`` in ``[0,1]``."""

    if key not in batch:
        raise KeyError(
            f"batch missing {key!r}; canonical AffordanceWAM requires scalar "
            "Afd [B,1,T,H,W] in addition to the RGB overlay"
        )
    x = batch[key]
    if x.ndim == 4:
        x = x.unsqueeze(0)
    if x.ndim != 5 or x.shape[1] != 1:
        raise ValueError(f"{key} must be [B,1,T,H,W], got {tuple(x.shape)}")
    if device is not None:
        x = x.to(device)
    if x.dtype == torch.uint8:
        x = x.float().div_(255.0)
    else:
        x = x.float()
    if not torch.isfinite(x).all():
        raise ValueError(f"{key} contains NaN/Inf")
    return x.clamp_(0.0, 1.0).to(dtype=dtype).contiguous()


def wan_batch_to_world_pixels(
    batch: Dict[str, torch.Tensor],
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the only trainable World streams: ``(RGB, scalar Afd)``."""

    if VIDEO_KEY not in batch:
        raise KeyError(f"batch missing {VIDEO_KEY!r}; got keys {list(batch)}")
    rgb = wan_pixels_from_uint8(batch[VIDEO_KEY], dtype=dtype, device=device)
    scalar = wan_scalar_affordance_from_batch(batch, dtype=torch.float32, device=device)
    return rgb, scalar


def wan_latent_geometry(
    spec: BaseModelSpec,
    *,
    num_frames: int = NUM_PIXEL_FRAMES,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
) -> Tuple[int, int, int]:
    """``(T_lat, H_lat, W_lat)`` the spec's VAE yields for a pixel window.

    Causal temporal compression: ``T_lat = (T-1)//tc + 1``. Spatial: floor
    division by the VAE spatial factor.
    """
    t_lat = (num_frames - 1) // spec.temporal_compression
    t_lat += 1 if spec.causal_temporal else 0
    return t_lat, height // spec.spatial_compression, width // spec.spatial_compression


__all__ = [
    "VIDEO_KEY",
    "SCALAR_AFFORDANCE_KEY",
    "AFFORDANCE_VIDEO_KEY",
    "wan_pixels_from_uint8",
    "wan_batch_to_pixels",
    "wan_batch_to_world_pixels",
    "wan_scalar_affordance_from_batch",
    "wan_latent_geometry",
]
