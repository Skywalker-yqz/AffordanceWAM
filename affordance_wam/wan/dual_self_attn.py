"""Dual-expert joint self-attention for the Wan world/action model.

The world and action streams keep independent residual widths, AdaLN, Q/K/V
projections, cross-attention and FFNs.  At every layer only their projected
Q/K/V tensors are concatenated and sent through one masked self-attention.

The attention mask is directional::

    query \\ key       world   action
    world                yes       no
    action               yes      yes

Consequently the world model is exactly usable on action-free human batches,
and robot actions cannot leak into the future RGB/Afd prediction.  The action
expert still consumes the complete, progressively denoised world state.
"""

from __future__ import annotations

from typing import Tuple

import torch

from affordance_wam.wan.expert_blocks import (
    ActionDiT,
    ActionState,
    VideoState,
    block_post_attn,
    block_pre_attn,
    mixed_attention,
    _stable_training_context_mask,
)
from affordance_wam.wan.slot_latent import WorldTokenLayout


def action_sees_world_mask(
    world: int | WorldTokenLayout, s_action: int, device: torch.device
) -> torch.Tensor:
    """Return the one-way ``[world | action]`` visibility mask.

    ``True`` means visible, matching ``scaled_dot_product_attention``'s bool
    mask convention.  Empty action sequences are supported for shape-level
    tests, although action-free training bypasses the action expert entirely.
    """

    if isinstance(world, WorldTokenLayout):
        s_world = int(world.world_token_count)
        s_core = int(world.core_token_count)
    else:
        s_world = int(world)
        s_core = s_world
    if s_world < 1:
        raise ValueError(f"s_world must be positive, got {s_world}")
    if s_action < 0:
        raise ValueError(f"s_action must be non-negative, got {s_action}")
    total = int(s_world) + int(s_action)
    mask = torch.zeros(total, total, dtype=torch.bool, device=device)
    # P1: core == complete World. P2: RGB+scalar core cannot read explicit
    # overlay, while overlay is an auxiliary sink that reads core + itself.
    mask[:s_core, :s_core] = True
    if s_core < s_world:
        mask[s_core:s_world, :s_world] = True
    if s_action:
        mask[s_world:, :s_core] = True
        mask[s_world:, s_world:] = True
    return mask


def run_dual_self_attn_loop(
    dit,
    action_dit: ActionDiT,
    world_state: VideoState,
    action_state: ActionState,
    *,
    use_gradient_checkpointing: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the layer-paired dual experts and return both final hidden states."""

    if len(dit.blocks) != action_dit.num_layers:
        raise ValueError(
            "dual self-attention requires one ActionDiT block per world block: "
            f"world={len(dit.blocks)} action={action_dit.num_layers}"
        )
    num_heads = int(dit.blocks[0].self_attn.num_heads)
    s_world = int(world_state.x.shape[1])
    s_action = int(action_state.x.shape[1])
    mask = action_sees_world_mask(world_state.layout, s_action, world_state.x.device)

    world_x = world_state.x
    action_x = action_state.x
    world_context_mask = _stable_training_context_mask(world_x, world_state.context)
    action_context_mask = _stable_training_context_mask(action_x, action_state.context)

    def layer_step(layer_id: int, wx: torch.Tensor, ax: torch.Tensor):
        world_block = dit.blocks[layer_id]
        action_block = action_dit.blocks[layer_id]
        q_w, k_w, v_w, post_w = block_pre_attn(
            world_block, wx, world_state.t_mod, world_state.freqs
        )
        q_a, k_a, v_a, post_a = block_pre_attn(
            action_block, ax, action_state.t_mod, action_state.freqs
        )
        out = mixed_attention(
            torch.cat([q_w, q_a], dim=1),
            torch.cat([k_w, k_a], dim=1),
            torch.cat([v_w, v_a], dim=1),
            num_heads,
            mask,
        )
        wx = block_post_attn(
            world_block,
            out[:, :s_world],
            post_w,
            world_state.context,
            context_mask=world_context_mask,
        )
        ax = block_post_attn(
            action_block,
            out[:, s_world:],
            post_a,
            action_state.context,
            context_mask=action_context_mask,
        )
        return wx, ax

    for layer_id in range(action_dit.num_layers):
        if use_gradient_checkpointing and torch.is_grad_enabled():
            world_x, action_x = torch.utils.checkpoint.checkpoint(
                layer_step, layer_id, world_x, action_x, use_reentrant=False
            )
        else:
            world_x, action_x = layer_step(layer_id, world_x, action_x)
    return world_x, action_x


__all__ = ["action_sees_world_mask", "run_dual_self_attn_loop"]
