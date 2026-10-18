"""Inference for the two supported Wan World layouts.

The eight-slot layout derives overlay as deterministic post-processing.  The
eleven-slot layout can decode its learned overlay suffix or explicitly derive
one from decoded RGB/scalar for comparison.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from affordance_wam.data.wan_adapt import (
    AFFORDANCE_VIDEO_KEY,
    SCALAR_AFFORDANCE_KEY,
    VIDEO_KEY,
    wan_pixels_from_uint8,
)
from affordance_wam.inference.sampler import SampleOutputs
from affordance_wam.wan.overlay_renderer import render_overlay_tensor


def _history_pixel_frames(model: Any) -> int:
    return 1 + max(0, int(model.config.num_cond_latents) - 1) * int(
        model.spec.temporal_compression
    )


def _as_rgb_gt_uint8(value: Any) -> Optional[torch.Tensor]:
    if not torch.is_tensor(value):
        return None
    x = value.detach().cpu()
    if x.ndim == 4:
        x = x.unsqueeze(0)
    if x.dtype == torch.uint8:
        return x.clone()
    x = x.float()
    if x.numel() and float(x.min()) >= -1.01 and float(x.max()) <= 1.01:
        x = (x.clamp(-1, 1) + 1.0) * 127.5
    return x.clamp(0, 255).round().to(torch.uint8)


def _as_scalar_gt(value: Any) -> Optional[torch.Tensor]:
    if not torch.is_tensor(value):
        return None
    x = value.detach().cpu()
    if x.ndim == 4:
        x = x.unsqueeze(0)
    if x.ndim != 5 or x.shape[1] != 1:
        raise ValueError(f"scalar Afd GT must be [B,1,T,H,W], got {tuple(x.shape)}")
    return (x.float() / 255.0 if x.dtype == torch.uint8 else x.float()).clamp(0, 1)


def capture_world_ground_truth(
    data_batch: Dict[str, Any],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Capture optional RGB/scalar/heatmap targets for offline evaluation."""

    return (
        _as_rgb_gt_uint8(data_batch.get(VIDEO_KEY)),
        _as_scalar_gt(data_batch.get(SCALAR_AFFORDANCE_KEY)),
        _as_rgb_gt_uint8(data_batch.get(AFFORDANCE_VIDEO_KEY)),
    )


def _validate_sampling(model: Any, num_steps: int, shift: float) -> None:
    if int(num_steps) < 1 or float(shift) <= 0:
        raise ValueError("num_steps and shift must be positive")
    if len(model.slot_temporal_ids) != int(model.nominal_world_layout.latent_slots):
        raise ValueError("World slot configuration mismatch")


def resolve_world_shift(model: Any, shift: Optional[float]) -> float:
    value = float(model.config.flow_shift) if shift is None else float(shift)
    if value <= 0:
        raise ValueError(f"shift must be positive, got {value}")
    return value


@torch.no_grad()
def prepare_world_condition(
    model: Any, data_batch: Dict[str, Any]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode only the five deployment RGB history frames into two slots."""

    if VIDEO_KEY not in data_batch:
        raise KeyError(f"batch missing {VIDEO_KEY!r}")
    rgb = wan_pixels_from_uint8(
        data_batch[VIDEO_KEY], dtype=model.dtype, device=model.device
    )
    if rgb.ndim == 4:
        rgb = rgb.unsqueeze(0)
    if rgb.ndim != 5 or rgb.shape[1] != 3:
        raise ValueError(f"video must be [B,3,T,H,W], got {tuple(rgb.shape)}")
    frames = _history_pixel_frames(model)
    if rgb.shape[2] < frames:
        raise ValueError(f"video has {rgb.shape[2]} frames; need {frames}")
    history = model.encode_video(rgb[:, :, :frames]).float()
    if history.shape[2] != int(model.config.num_cond_latents):
        raise ValueError("RGB history VAE slot count mismatch")
    return rgb, history


@torch.no_grad()
def sample_world_latents(
    model: Any,
    data_batch: Dict[str, Any],
    *,
    guidance: float = 1.0,
    seed: int = 1,
    num_steps: int = 35,
    shift: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Denoise the configured eight- or eleven-slot World without future GT."""

    from affordance_wam.wan_vendor.shared.diffusion import FlowMatchScheduler

    effective_shift = resolve_world_shift(model, shift)
    _validate_sampling(model, num_steps, effective_shift)
    rgb, history = prepare_world_condition(model, data_batch)
    batch, channels, _t, height, width = history.shape
    ncond = int(model.config.num_cond_latents)
    context = model.encode_text(data_batch.get("ai_caption"), batch)
    do_cfg = abs(float(guidance) - 1.0) > 1e-6
    null_context = model.encode_text(None, batch) if do_cfg else None
    scheduler = FlowMatchScheduler("Wan")
    scheduler.set_timesteps(int(num_steps), training=False, shift=effective_shift)
    generator = torch.Generator(device=model.device).manual_seed(int(seed))
    world = torch.randn(
        batch,
        channels,
        len(model.slot_temporal_ids),
        height,
        width,
        generator=generator,
        device=model.device,
        dtype=torch.float32,
    )
    world[:, :, :ncond] = history
    for timestep in scheduler.timesteps:
        t_batch = timestep.to(model.device).repeat(batch)
        velocity = model._forward_world(world, t_batch, context).float()
        if do_cfg:
            null_velocity = model._forward_world(world, t_batch, null_context).float()
            velocity = null_velocity + float(guidance) * (velocity - null_velocity)
        world = scheduler.step(velocity, timestep, world)
        world[:, :, :ncond] = history
    return world, history, rgb


@torch.no_grad()
def decode_world_outputs(
    model: Any,
    world_latents: torch.Tensor,
    history_latent: torch.Tensor,
    *,
    rgb_gt: Optional[torch.Tensor] = None,
    overlay_gt: Optional[torch.Tensor] = None,
    scalar_gt: Optional[torch.Tensor] = None,
    render_overlay: bool = True,
    overlay_source: str = "auto",
    overlay_alpha: float = 0.6,
    overlay_colormap: str = "jet",
) -> SampleOutputs:
    """Decode RGB/scalar and optionally a predicted or derived overlay."""

    from affordance_wam.wan_vendor.encode import decode_latents

    del history_latent  # RGB latent already includes its two clean slots.
    cfg = model.config
    rgb_end = int(cfg.num_cond_latents) + int(cfg.num_rgb_future_latents)
    scalar_end = rgb_end + int(cfg.num_scalar_afd_future_latents)
    layout = model.nominal_world_layout
    expected = int(layout.latent_slots)
    if world_latents.ndim != 5 or world_latents.shape[2] != expected:
        raise ValueError(f"world latent must contain {expected} slots")
    rgb_pixel = decode_latents(
        world_latents[:, :, :rgb_end].to(model.dtype),
        vae=model.vae,
        device=model.device,
        tiled=True,
    )
    scalar_future = model.scalar_afd_codec.decode_future(
        world_latents[:, :, rgb_end:scalar_end].to(model.dtype)
    ).float()
    overlay_source = str(overlay_source).lower()
    if overlay_source not in ("auto", "predicted", "derived"):
        raise ValueError("overlay_source must be auto/predicted/derived")
    if overlay_source == "auto":
        overlay_source = "predicted" if layout.overlay_slots else "derived"
    if overlay_source == "predicted" and not layout.overlay_slots:
        raise ValueError(
            "overlay_source='predicted' requires world_mode=rgb_scalar_overlay"
        )
    overlay_future = None
    if render_overlay:
        history_frames = _history_pixel_frames(model)
        if overlay_source == "predicted":
            overlay_with_history = torch.cat(
                [
                    world_latents[:, :, : int(cfg.num_cond_latents)],
                    world_latents[:, :, scalar_end:expected],
                ],
                dim=2,
            )
            overlay_full = decode_latents(
                overlay_with_history.to(model.dtype),
                vae=model.vae,
                device=model.device,
                tiled=True,
            )
            overlay_future = overlay_full[:, :, history_frames:].float()
        else:
            rgb_future_unit = (
                rgb_pixel[:, :, history_frames:].float().clamp(-1, 1) + 1.0
            ) * 0.5
            overlay_future = (
                render_overlay_tensor(
                    rgb_future_unit,
                    scalar_future,
                    alpha=float(overlay_alpha),
                    colormap_name=str(overlay_colormap),
                )
                .mul(2.0)
                .sub(1.0)
            )
    return SampleOutputs(
        rgb_pixel=rgb_pixel,
        aff_future_pixel=overlay_future,
        samples_latent=world_latents,
        rgb_history_latent=world_latents[:, :, : int(cfg.num_cond_latents)],
        rgb_gt=rgb_gt,
        aff_gt=overlay_gt,
        scalar_affordance_future_pixel=scalar_future,
        scalar_affordance_gt=scalar_gt,
    )


@torch.no_grad()
def run_wan_inference(
    model: Any,
    data_batch: Dict[str, Any],
    *,
    guidance: float = 1.0,
    seed: int = 1,
    num_steps: int = 35,
    shift: Optional[float] = None,
    num_conditional_frames: Optional[int] = None,
    use_ema: bool = False,
    render_overlay: bool = True,
    overlay_source: str = "auto",
    overlay_alpha: float = 0.6,
    overlay_colormap: str = "jet",
) -> SampleOutputs:
    del use_ema
    frames = _history_pixel_frames(model)
    if num_conditional_frames is not None and int(num_conditional_frames) != frames:
        raise ValueError(f"Wan expects {frames} conditional frames")
    world, history, _ = sample_world_latents(
        model,
        data_batch,
        guidance=guidance,
        seed=seed,
        num_steps=num_steps,
        shift=shift,
    )
    rgb_gt, scalar_gt, overlay_gt = capture_world_ground_truth(data_batch)
    return decode_world_outputs(
        model,
        world,
        history,
        rgb_gt=rgb_gt,
        overlay_gt=overlay_gt,
        scalar_gt=scalar_gt,
        render_overlay=render_overlay,
        overlay_source=overlay_source,
        overlay_alpha=overlay_alpha,
        overlay_colormap=overlay_colormap,
    )


__all__ = [
    "capture_world_ground_truth",
    "decode_world_outputs",
    "prepare_world_condition",
    "resolve_world_shift",
    "run_wan_inference",
    "sample_world_latents",
]
