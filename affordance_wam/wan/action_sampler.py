"""Joint World/Action generation followed by frozen-World Action refinement."""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from affordance_wam.inference.sampler import SampleOutputs
from affordance_wam.wan.wan_sampler import (
    capture_world_ground_truth,
    decode_world_outputs,
    prepare_world_condition,
    resolve_world_shift,
    run_wan_inference,
)


def _positive_steps(name: str, value: int) -> int:
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _action_grid(model: Any, n_steps: int, *, strength: float = 1.0):
    from affordance_wam.action.scheduler import ActionScheduler

    if not 0.0 < float(strength) <= 1.0:
        raise ValueError(f"action denoising strength must be in (0,1], got {strength}")
    scheduler = ActionScheduler(
        shift=float(getattr(model.config, "action_flow_shift", 5.0))
    )
    scheduler.set_timesteps(
        _positive_steps("action_num_steps", n_steps),
        denoising_strength=float(strength),
        training=False,
    )
    return scheduler


def _joint_events(world_steps: int, action_steps: int):
    """Sorted union of both Euler grids as exact rational progress values."""

    events = {Fraction(i, world_steps) for i in range(world_steps)}
    events.update(Fraction(i, action_steps) for i in range(action_steps))
    return sorted(events)


def _next_sigma(sigmas: torch.Tensor, index: int, device: torch.device) -> torch.Tensor:
    if index + 1 >= len(sigmas):
        return torch.zeros((), device=device, dtype=sigmas.dtype)
    return sigmas[index + 1].to(device)


@torch.no_grad()
def _action_from_frozen_world(
    model: Any,
    world_latents: torch.Tensor,
    context: torch.Tensor,
    proprio_current: Optional[torch.Tensor],
    *,
    seed: int,
    num_steps: int,
    initial_action: Optional[torch.Tensor] = None,
    strength: float = 1.0,
) -> torch.Tensor:
    """Denoise/refine action against a single frozen-world K/V prefill."""

    from affordance_wam.wan import expert_blocks as experts

    device, dtype = model.device, model.dtype
    batch = world_latents.shape[0]
    context_video = model.dit.embed_text(context)
    clean_timestep = torch.zeros(batch, device=device, dtype=dtype)
    world_state = experts.video_prepare(
        model,
        world_latents.to(dtype),
        clean_timestep,
        context_video,
    )
    cache = experts.prefill_video_cache(model, world_state)
    scheduler = _action_grid(model, num_steps, strength=strength)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    noise = torch.randn(
        batch,
        model.config.action_horizon,
        model.config.action_dim,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    if initial_action is None:
        action = noise
    else:
        sigma0 = scheduler.sigmas[0].to(device=device, dtype=torch.float32)
        action = (1.0 - sigma0) * initial_action.float() + sigma0 * noise

    for index, timestep in enumerate(scheduler.timesteps):
        t_batch = timestep.to(device).repeat(batch)
        action_state = model.action_backbone.prepare(
            action.to(dtype), t_batch.to(dtype), context, proprio=proprio_current
        )
        velocity = experts.run_action_with_video_cache(
            model.action_backbone,
            action_state,
            cache,
        ).float()
        sigma = scheduler.sigmas[index].to(device)
        action = scheduler.flow_step(
            velocity, sigma, _next_sigma(scheduler.sigmas, index, device), action
        )
    return action


@torch.no_grad()
def run_wan_action_inference(
    model: Any,
    data_batch: Dict[str, Any],
    *,
    guidance: float = 1.0,
    seed: int = 1,
    num_steps: Optional[int] = None,
    action_num_steps: Optional[int] = None,
    action_refinement_steps: Optional[int] = None,
    action_refinement_strength: Optional[float] = None,
    shift: Optional[float] = None,
    use_ema: bool = False,
    render_overlay: bool = True,
    overlay_source: str = "auto",
    overlay_alpha: float = 0.6,
    overlay_colormap: str = "jet",
) -> Tuple[SampleOutputs, np.ndarray]:
    """Return RGB/scalar World and normalized action; overlay is optional."""

    del use_ema
    cfg = model.config
    world_shift = resolve_world_shift(model, shift)
    if (
        bool(getattr(cfg, "use_proprio", False))
        and data_batch.get("proprio_current") is None
    ):
        raise KeyError(
            "model.config.use_proprio=True requires deployment batch field "
            "'proprio_current' [B,state_dim]; refusing silent unconditioned action inference"
        )
    world_steps = _positive_steps(
        "num_steps",
        (
            num_steps
            if num_steps is not None
            else getattr(cfg, "inference_world_denoise_steps", 35)
        ),
    )
    action_steps = _positive_steps(
        "action_num_steps",
        (
            action_num_steps
            if action_num_steps is not None
            else getattr(cfg, "inference_action_denoise_steps", 35)
        ),
    )
    result = _infer_dual_self_attn(
        model,
        data_batch,
        guidance=guidance,
        seed=seed,
        world_steps=world_steps,
        action_steps=action_steps,
        refinement_steps=(
            cfg.action_refinement_steps
            if action_refinement_steps is None
            else action_refinement_steps
        ),
        refinement_strength=(
            cfg.action_refinement_strength
            if action_refinement_strength is None
            else action_refinement_strength
        ),
        shift=world_shift,
    )

    outputs, actions = result
    overlay_source = str(overlay_source).lower()
    if overlay_source not in ("auto", "predicted", "derived"):
        raise ValueError("overlay_source must be auto/predicted/derived")
    has_predicted_overlay = bool(model.nominal_world_layout.overlay_slots)
    if overlay_source == "auto":
        overlay_source = "predicted" if has_predicted_overlay else "derived"
    if overlay_source == "predicted" and not has_predicted_overlay:
        raise ValueError(
            "overlay_source='predicted' requires world_mode=rgb_scalar_overlay"
        )
    if not render_overlay:
        outputs.aff_future_pixel = None
    elif overlay_source == "derived":
        from affordance_wam.wan.overlay_renderer import render_overlay_tensor

        scalar = outputs.scalar_affordance_future_pixel
        if scalar is None:
            raise RuntimeError("scalar Afd output is required to render overlay")
        rgb_future = outputs.rgb_pixel[:, :, -scalar.shape[2] :].float()
        outputs.aff_future_pixel = (
            render_overlay_tensor(
                (rgb_future.clamp(-1, 1) + 1.0) * 0.5,
                scalar,
                alpha=float(overlay_alpha),
                colormap_name=str(overlay_colormap),
            )
            .mul(2.0)
            .sub(1.0)
        )
    return outputs, actions


@torch.no_grad()
def _infer_dual_self_attn(
    model: Any,
    data_batch: Dict[str, Any],
    *,
    guidance: float,
    seed: int,
    world_steps: int,
    action_steps: int,
    refinement_steps: int,
    refinement_strength: float,
    shift: float,
) -> Tuple[SampleOutputs, np.ndarray]:
    from affordance_wam.wan import expert_blocks as experts
    from affordance_wam.wan.dual_self_attn import run_dual_self_attn_loop
    from affordance_wam.wan_vendor.shared.diffusion import FlowMatchScheduler

    if refinement_steps < 0:
        raise ValueError("action_refinement_steps must be >= 0")
    device, dtype = model.device, model.dtype
    cfg = model.config
    rgb_pixels, history = prepare_world_condition(model, data_batch)
    batch, z, _t, h, w = history.shape
    ncond = int(cfg.num_cond_latents)

    context = model.encode_text(data_batch.get("ai_caption"), batch)
    null_context = (
        model.encode_text(None, batch) if abs(float(guidance) - 1.0) > 1e-6 else None
    )
    context_video = model.dit.embed_text(context)
    proprio = data_batch.get("proprio_current")
    if proprio is not None:
        proprio = proprio.to(device=device, dtype=dtype)

    world_scheduler = FlowMatchScheduler("Wan")
    world_scheduler.set_timesteps(world_steps, training=False, shift=float(shift))
    action_scheduler = _action_grid(model, action_steps)
    world_generator = torch.Generator(device=device).manual_seed(int(seed))
    action_generator = torch.Generator(device=device).manual_seed(int(seed) + 1)
    world = torch.randn(
        batch,
        z,
        len(model.slot_temporal_ids),
        h,
        w,
        generator=world_generator,
        device=device,
        dtype=torch.float32,
    )
    world[:, :, :ncond] = history
    action = torch.randn(
        batch,
        cfg.action_horizon,
        cfg.action_dim,
        generator=action_generator,
        device=device,
        dtype=torch.float32,
    )

    world_done = action_done = 0
    zero_t = torch.zeros(batch, device=device)

    def joint_velocity(raw_context, embedded_context, world_t, action_t):
        world_state = experts.video_prepare(
            model,
            world.to(dtype),
            world_t.to(dtype),
            embedded_context,
        )
        action_state = model.action_backbone.prepare(
            action.to(dtype), action_t.to(dtype), raw_context, proprio=proprio
        )
        world_hidden, action_hidden = run_dual_self_attn_loop(
            model.dit, model.action_backbone, world_state, action_state
        )
        return (
            experts.video_finalize(model, world_hidden, world_state).float(),
            model.action_backbone.decode(action_hidden).float(),
        )

    null_context_video = (
        model.dit.embed_text(null_context) if null_context is not None else None
    )
    for progress in _joint_events(world_steps, action_steps):
        update_world = progress * world_steps == world_done
        update_action = progress * action_steps == action_done
        world_t = (
            world_scheduler.timesteps[world_done].to(device).repeat(batch)
            if world_done < world_steps
            else zero_t
        )
        action_t = (
            action_scheduler.timesteps[action_done].to(device).repeat(batch)
            if action_done < action_steps
            else zero_t
        )
        world_velocity, action_velocity = joint_velocity(
            context, context_video, world_t, action_t
        )
        if null_context is not None:
            null_world_velocity, null_action_velocity = joint_velocity(
                null_context, null_context_video, world_t, action_t
            )
            world_velocity = null_world_velocity + float(guidance) * (
                world_velocity - null_world_velocity
            )
            action_velocity = null_action_velocity + float(guidance) * (
                action_velocity - null_action_velocity
            )

        if update_world:
            world_index = world_done
            world = world_scheduler.step(
                world_velocity,
                world_scheduler.timesteps[world_index],
                world,
            )
            world[:, :, :ncond] = history
            world_done += 1
        if update_action:
            action_index = action_done
            sigma = action_scheduler.sigmas[action_index].to(device)
            action = action_scheduler.flow_step(
                action_velocity,
                sigma,
                _next_sigma(action_scheduler.sigmas, action_index, device),
                action,
            )
            action_done += 1
    if world_done != world_steps or action_done != action_steps:
        raise RuntimeError(
            f"joint scheduler incomplete: world={world_done}/{world_steps}, "
            f"action={action_done}/{action_steps}"
        )

    if refinement_steps:
        action = _action_from_frozen_world(
            model,
            world,
            context,
            proprio,
            seed=int(seed) + 1,
            num_steps=refinement_steps,
            initial_action=action,
            strength=refinement_strength,
        )

    rgb_gt, scalar_gt, overlay_gt = capture_world_ground_truth(data_batch)
    outputs = decode_world_outputs(
        model,
        world,
        history,
        rgb_gt=rgb_gt,
        scalar_gt=scalar_gt,
        overlay_gt=overlay_gt,
    )
    return outputs, action.detach().cpu().numpy()


__all__ = ["run_wan_action_inference"]
