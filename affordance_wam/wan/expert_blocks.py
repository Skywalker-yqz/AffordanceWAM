"""World and Action Transformer blocks, token preparation, and frozen World caches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from affordance_wam.action.components import StateEncoder
from affordance_wam.wan.slot_latent import (
    WorldTokenLayout,
    add_world_view_embeddings,
    build_world_token_rope_freqs,
)
from affordance_wam.wan_vendor.models.dit import (
    CrossAttention,
    GateModule,
    RMSNorm,
    modulate,
    precompute_freqs_cis,
    rope_apply,
    sinusoidal_embedding_1d,
)


class _ScaleGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return value

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Apply optional gradient rescaling in FP32.
        gradient = grad_output.float() * ctx.scale
        return gradient.to(dtype=grad_output.dtype), None


def _scale_gradient(value: torch.Tensor, scale: float) -> torch.Tensor:
    """Identity forward with an explicit backward multiplier."""

    return _ScaleGradient.apply(value, float(scale))


# ======================================================================
# Block split helpers — replicate ``DiTBlock.forward`` in two halves so the
# self-attention can be mixed across modalities. They read the block's
# submodules by name (``norm1`` / ``modulation`` / ``self_attn`` / ``cross_attn``
# / ``norm2`` / ``norm3`` / ``ffn`` / ``gate``); both the vendored video
# ``DiTBlock`` and :class:`ActionDiTBlock` expose that exact interface.
# ======================================================================
def _chunk_modulation(modulation: torch.Tensor, t_mod: torch.Tensor):
    has_seq = t_mod.dim() == 4
    chunk_dim = 2 if has_seq else 1
    parts = (modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(
        6, dim=chunk_dim
    )
    if has_seq:
        parts = tuple(p.squeeze(2) for p in parts)
    return parts  # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp


def block_pre_attn(block, x: torch.Tensor, t_mod: torch.Tensor, freqs: torch.Tensor):
    """First half of a DiTBlock: modulate → Q/K/V → RoPE. Returns
    ``(q, k, v, post_state)`` with q/k/v flattened ``[B, S, attn_dim]``."""
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = _chunk_modulation(
        block.modulation, t_mod
    )
    input_x = modulate(block.norm1(x), shift_msa, scale_msa)
    sa = block.self_attn
    q = sa.norm_q(sa.q(input_x))
    k = sa.norm_k(sa.k(input_x))
    v = sa.v(input_x)
    q = rope_apply(q, freqs, sa.num_heads)
    k = rope_apply(k, freqs, sa.num_heads)
    return q, k, v, (x, gate_msa, shift_mlp, scale_mlp, gate_mlp)


def block_post_attn(
    block,
    attn_out: torch.Tensor,
    post_state,
    context: torch.Tensor,
    *,
    context_mask: Optional[torch.Tensor] = None,
):
    """Second half of a DiTBlock: out-proj + gate + cross-attn + FFN."""
    x, gate_msa, shift_mlp, scale_mlp, gate_mlp = post_state
    x = block.gate(x, gate_msa, block.self_attn.o(attn_out))
    x = x + block.cross_attn(block.norm3(x), context, ctx_mask=context_mask)
    input_x = modulate(block.norm2(x), shift_mlp, scale_mlp)
    x = block.gate(x, gate_mlp, block.ffn(input_x))
    return x


def _stable_training_context_mask(
    query: torch.Tensor, context: torch.Tensor
) -> Optional[torch.Tensor]:
    """Preserve full text visibility with a stable low-precision backward.

    Some fused SDPA implementations select a less stable no-mask BF16 backward
    for cross-attention. A full boolean mask is mathematically identical,
    small for text cross-attention, and reused across all layers. Inference
    runs under ``torch.no_grad()`` and retains the fast path.
    """

    if not torch.is_grad_enabled():
        return None
    return torch.ones(
        (int(query.shape[1]), int(context.shape[1])),
        dtype=torch.bool,
        device=query.device,
    )


def mixed_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    mask: Optional[torch.Tensor],
):
    """SDPA over flattened-head ``[B, S, attn_dim]`` tensors; ``mask`` is a
    ``[S_q, S_k]`` boolean (True = attend) broadcast over batch & heads."""
    output_dtype = q.dtype
    q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
    stable_masked_backward = (
        mask is not None
        and torch.is_grad_enabled()
        and output_dtype in (torch.float16, torch.bfloat16)
        and any(tensor.requires_grad for tensor in (q, k, v))
    )
    if stable_masked_backward:
        q, k, v = q.float(), k.float(), v.float()
    with torch.autocast(device_type=q.device.type, enabled=False):
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    if stable_masked_backward:
        out = out.to(output_dtype)
    return rearrange(out, "b n s d -> b s (n d)", n=num_heads)


def world_visibility_mask(
    layout: WorldTokenLayout, device: torch.device
) -> torch.Tensor:
    """Core is isolated from overlay; overlay is a directed auxiliary sink."""

    total = int(layout.world_token_count)
    core = int(layout.core_token_count)
    mask = torch.zeros(total, total, dtype=torch.bool, device=device)
    mask[:core, :core] = True
    if core < total:
        mask[core:, :total] = True
    return mask


# ======================================================================
# ActionDiT — a 1D transformer mirroring the video DiTBlock interface.
# ======================================================================
class ActionSelfAttention(nn.Module):
    """Self-attention whose Q/K/V live in the **video attention space**
    (``attn_dim = video_num_heads * video_head_dim``) so action tokens can be
    concatenated with video tokens for one joint attention. 1D RoPE on Q/K."""

    def __init__(
        self, hidden_dim: int, attn_dim: int, num_heads: int, eps: float = 1e-6
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self.q = nn.Linear(hidden_dim, attn_dim)
        self.k = nn.Linear(hidden_dim, attn_dim)
        self.v = nn.Linear(hidden_dim, attn_dim)
        self.o = nn.Linear(attn_dim, hidden_dim)
        self.norm_q = RMSNorm(attn_dim, eps=eps)
        self.norm_k = RMSNorm(attn_dim, eps=eps)


class ActionDiTBlock(nn.Module):
    """DiTBlock-isomorphic action layer (so the MoT split helpers apply as-is)."""

    def __init__(
        self,
        hidden_dim: int,
        attn_dim: int,
        num_heads_self: int,
        num_heads_cross: int,
        ffn_dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.self_attn = ActionSelfAttention(hidden_dim, attn_dim, num_heads_self, eps)
        self.cross_attn = CrossAttention(
            hidden_dim, num_heads_cross, eps, has_image_input=False
        )
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim**0.5)
        self.gate = GateModule()
        self.num_heads = num_heads_self


@dataclass
class ActionState:
    x: torch.Tensor  # [B, T, hidden_dim]
    t_mod: torch.Tensor  # [B, 6, hidden_dim]
    freqs: torch.Tensor  # [T, 1, head_dim/2] complex
    context: torch.Tensor  # [B, L_text, hidden_dim]
    time_embed: torch.Tensor


class ActionDiT(nn.Module):
    """Action Transformer paired with the World Expert at each layer."""

    def __init__(
        self,
        *,
        action_dim: int,
        hidden_dim: int,
        attn_dim: int,
        num_heads_self: int,
        num_heads_cross: int,
        ffn_dim: int,
        num_layers: int,
        text_dim: int = 4096,
        freq_dim: int = 256,
        max_action_len: int = 512,
        eps: float = 1e-6,
        use_proprio: bool = False,
        state_dim: int = 7,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads_self
        self.head_dim = attn_dim // num_heads_self
        self.freq_dim = freq_dim
        self.num_layers = num_layers
        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        # Leak-free CURRENT-state proprio → one context token (cross-attn), like
        # the shared path's state token but in the ActionDiT's own hidden space.
        self.proprio_encoder = (
            StateEncoder(state_dim, hidden_dim) if use_proprio else None
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6)
        )
        self.blocks = nn.ModuleList(
            [
                ActionDiTBlock(
                    hidden_dim, attn_dim, num_heads_self, num_heads_cross, ffn_dim, eps
                )
                for _ in range(num_layers)
            ]
        )
        self.action_decoder = nn.Linear(hidden_dim, action_dim)
        # 1D RoPE in the (video) attention head space. Kept as a PLAIN attribute
        # (not a buffer/param) — exactly like the vendored ``WanModel.freqs`` —
        # so ``model.to(dtype=bf16)`` never casts this complex tensor to real
        # (which would silently discard the imaginary part and break RoPE).
        self.action_freqs = precompute_freqs_cis(self.head_dim, max_action_len)

    def prepare(
        self,
        noisy_actions: torch.Tensor,
        action_timestep: torch.Tensor,
        context_raw: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
    ) -> ActionState:
        x = self.action_encoder(noisy_actions)
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, action_timestep).to(x.dtype)
        )
        t_mod = self.time_projection(t).unflatten(
            1, (6, self.hidden_dim)
        )  # [B, 6, hidden]
        T = x.shape[1]
        freqs = self.action_freqs[:T].view(T, 1, -1).to(x.device)
        context = self.embed_text(context_raw.to(x.dtype))
        # CURRENT-state proprio → extra cross-attn context token (leak-free).
        if self.proprio_encoder is not None and proprio is not None:
            ptok = self.proprio_encoder(proprio.to(x.dtype))  # [B, 1, hidden]
            context = torch.cat([context, ptok], dim=1)
        return ActionState(x=x, t_mod=t_mod, freqs=freqs, context=context, time_embed=t)

    def embed_text(self, context: torch.Tensor) -> torch.Tensor:
        return self.text_embedding(context)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.action_decoder(x)


# ======================================================================
# Video prepare / finalize (WanBase-style explicit loop entry/exit).
# ======================================================================
@dataclass
class VideoState:
    x: torch.Tensor
    t_mod: torch.Tensor
    freqs: torch.Tensor
    context: torch.Tensor
    time_embed: torch.Tensor
    layout: WorldTokenLayout


def _stable_video_time_conditioning(
    dit: nn.Module,
    time_sinusoid: torch.Tensor,
    output_dtype: torch.dtype,
    parameter_gradient_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate Wan's shared timestep MLP in FP32 during BF16 autograd."""

    low_precision = output_dtype in (torch.bfloat16, torch.float16)
    if not (torch.is_grad_enabled() and low_precision):
        time_embed = dit.time_embedding(time_sinusoid.to(dtype=output_dtype))
        return time_embed, dit.time_projection(time_embed)

    embedding_in = dit.time_embedding[0]
    embedding_out = dit.time_embedding[2]
    projection = dit.time_projection[1]
    with torch.autocast(device_type=time_sinusoid.device.type, enabled=False):
        value = F.linear(
            time_sinusoid.float(),
            embedding_in.weight.float(),
            embedding_in.bias.float() if embedding_in.bias is not None else None,
        )
        value = F.silu(value)
        time_embed32 = F.linear(
            value,
            embedding_out.weight.float(),
            embedding_out.bias.float() if embedding_out.bias is not None else None,
        )
        # Place the boundary on embedding_out itself.  Applying it only to the
        # returned time_embed would miss the internal time_projection branch.
        time_embed32 = _scale_gradient(time_embed32, float(parameter_gradient_scale))
        t_mod32 = F.linear(
            F.silu(time_embed32),
            projection.weight.float(),
            projection.bias.float() if projection.bias is not None else None,
        )
    return time_embed32.to(dtype=output_dtype), t_mod32.to(dtype=output_dtype)


def _heterogeneous_time_modulation(model, timestep, layout: WorldTokenLayout):
    """TI2V per-token modulation with only RGB-history tokens pinned to t=0."""

    dit = model.dit
    batch = int(timestep.shape[0])
    seq = int(layout.world_token_count)
    token_timestep = timestep.reshape(batch, 1).expand(batch, seq).clone()
    clean_tokens = int(layout.num_cond * layout.rgb_tokens_per_slot)
    token_timestep[:, :clean_tokens] = 0
    time_sinusoid = sinusoidal_embedding_1d(dit.freq_dim, token_timestep.reshape(-1))
    normalize_time_gradient = bool(
        getattr(model.config, "time_conditioning_token_grad_normalization", 1)
    )
    token_scale = 1.0 / float(seq) if normalize_time_gradient else 1.0
    time_embed, t_mod = _stable_video_time_conditioning(
        dit,
        time_sinusoid.to(device=model.device),
        model.dtype,
        parameter_gradient_scale=token_scale,
    )
    # One timestep condition is replicated over hundreds of heterogeneous
    # tokens.  Normalize only its parameter-learning gradient by that fan-out;
    # the forward values and gradients into every DiT block are unchanged.
    # This prevents pixel auxiliary directions from overflowing the shared
    # BF16 timestep MLP before the optimizer's global gradient clip can run.
    if normalize_time_gradient:
        t_mod = _scale_gradient(t_mod, token_scale)
    return (
        time_embed.reshape(batch, seq, dit.dim),
        t_mod.reshape(batch, seq, 6, dit.dim),
    )


def video_prepare(model, latents, timestep, context_embedded) -> VideoState:
    """Patchify every geometry-aligned World slot through Wan's shared stem."""

    dit = model.dit
    layout = model.world_layout_for(latents)
    embedded = dit.patch_embedding(latents)
    x = rearrange(embedded, "b c f h w -> b (f h w) c").contiguous()
    if int(x.shape[1]) != layout.world_token_count:
        raise RuntimeError(
            f"heterogeneous patchify produced {x.shape[1]} tokens; "
            f"layout requires {layout.world_token_count}"
        )
    time_embed, t_mod = _heterogeneous_time_modulation(model, timestep, layout)
    freqs = build_world_token_rope_freqs(dit, layout, x.device)
    x = add_world_view_embeddings(x, layout, model.view_embedder)
    return VideoState(
        x=x,
        t_mod=t_mod,
        freqs=freqs,
        context=context_embedded,
        time_embed=time_embed,
        layout=layout,
    )


def _unpatchify_group(
    x: torch.Tensor,
    *,
    slots: int,
    grid: Tuple[int, int],
    patch_size: Tuple[int, int, int],
) -> torch.Tensor:
    return rearrange(
        x,
        "b (f h w) (pt ph pw c) -> b c (f pt) (h ph) (w pw)",
        f=int(slots),
        h=int(grid[0]),
        w=int(grid[1]),
        pt=int(patch_size[0]),
        ph=int(patch_size[1]),
        pw=int(patch_size[2]),
    )


def video_finalize(model, x: torch.Tensor, state: VideoState) -> torch.Tensor:
    """Apply Wan's shared head and restore the configured latent slots."""

    layout = state.layout
    dit = model.dit
    tokens = dit.head(x, state.time_embed)
    result = _unpatchify_group(
        tokens,
        slots=layout.latent_slots,
        grid=layout.grid,
        patch_size=tuple(dit.patch_size),
    ).contiguous()
    if tuple(result.shape[-3:]) != (
        layout.latent_slots,
        layout.latent_h,
        layout.latent_w,
    ):
        raise RuntimeError(
            f"World finalize returned {tuple(result.shape)} for {layout}"
        )
    return result


def run_world_loop(
    dit,
    state: VideoState,
    *,
    use_gradient_checkpointing: bool = False,
) -> torch.Tensor:
    """Run a World-only stream through the same split-block SDPA path.

    The dual-expert modes require SDPA for their cross-stream visibility masks.
    Keeping action-free Human batches on this identical block implementation
    avoids silently switching attention kernels between data routes.
    """

    num_heads = int(dit.blocks[0].self_attn.num_heads)
    x = state.x
    # Retain full visibility through an all-True boolean mask so World-only and
    # dual-expert routes use the same stable SDPA implementation. Reuse its
    # ~S^2 bytes across all layers.
    world_mask = world_visibility_mask(state.layout, x.device)
    context_mask = _stable_training_context_mask(x, state.context)

    def layer_step(layer_id: int, hidden: torch.Tensor) -> torch.Tensor:
        block = dit.blocks[layer_id]
        q, k, v, post = block_pre_attn(block, hidden, state.t_mod, state.freqs)
        attended = mixed_attention(q, k, v, num_heads, world_mask)
        return block_post_attn(
            block,
            attended,
            post,
            state.context,
            context_mask=context_mask,
        )

    for layer_id in range(len(dit.blocks)):
        if use_gradient_checkpointing and torch.is_grad_enabled():
            x = torch.utils.checkpoint.checkpoint(
                layer_step, layer_id, x, use_reentrant=False
            )
        else:
            x = layer_step(layer_id, x)
    return x


# ======================================================================
# MoT training loop (three teacher-forcing branches in one forward).
# ======================================================================


# ======================================================================
# Build + loss + inference entry points used by WanActionAffordanceModel.
# ======================================================================
def build_action_stack(dit, config) -> ActionDiT:
    """Construct the ActionDiT paired layer-for-layer with the video DiT."""
    video_num_heads = int(dit.blocks[0].self_attn.num_heads)
    attn_dim = int(dit.dim)  # video Q/K/V width
    num_layers = len(dit.blocks)
    hidden = int(config.action_dit_dim)
    num_heads_cross = int(config.action_cross_num_heads)
    if hidden % num_heads_cross:
        raise ValueError(
            f"action_dit_dim={hidden} must be divisible by "
            f"action_cross_num_heads={num_heads_cross}"
        )
    action_dit = ActionDiT(
        action_dim=config.action_dim,
        hidden_dim=hidden,
        attn_dim=attn_dim,
        num_heads_self=video_num_heads,
        num_heads_cross=num_heads_cross,
        ffn_dim=int(config.action_ffn_dim),
        num_layers=num_layers,
        text_dim=int(dit.text_embedding[0].in_features),
        use_proprio=bool(getattr(config, "use_proprio", False)),
        state_dim=int(getattr(config, "state_dim", 7)),
    )
    return action_dit


# ======================================================================
# Two-stage inference (Stage 2 reuses a per-layer frozen-video K/V cache).
# ======================================================================
@torch.no_grad()
def prefill_video_cache(model, vstate: VideoState) -> List[Dict[str, torch.Tensor]]:
    """Run the (clean) video through every layer alone, caching per-layer K/V
    for the action stream to attend to in Stage 2."""
    dit = model.dit
    num_heads = dit.blocks[0].self_attn.num_heads
    x = vstate.x
    cache: List[Dict[str, torch.Tensor]] = []
    world_mask = world_visibility_mask(vstate.layout, x.device)
    core = int(vstate.layout.core_token_count)
    for layer_id in range(len(dit.blocks)):
        block = dit.blocks[layer_id]
        q, k, v, post = block_pre_attn(block, x, vstate.t_mod, vstate.freqs)
        cache.append({"k": k[:, :core], "v": v[:, :core]})
        out = mixed_attention(q, k, v, num_heads, world_mask)
        x = block_post_attn(block, out, post, vstate.context)
    return cache


def run_action_with_video_cache(
    action_dit: ActionDiT, astate: ActionState, video_kv_cache
) -> torch.Tensor:
    """Stage-2 action denoise step: action attends to [frozen video K/V | action]."""
    num_heads = action_dit.num_heads
    s_action = astate.x.shape[1]
    if len(video_kv_cache) != action_dit.num_layers:
        raise ValueError("video K/V cache layer count does not match ActionDiT")
    video_seq_len = int(video_kv_cache[0]["k"].shape[1])
    if any(
        int(item["k"].shape[1]) != video_seq_len
        or int(item["v"].shape[1]) != video_seq_len
        for item in video_kv_cache
    ):
        raise ValueError("video K/V cache sequence lengths are inconsistent")
    mask = torch.ones(
        s_action, video_seq_len + s_action, dtype=torch.bool, device=astate.x.device
    )
    x = astate.x
    context_mask = _stable_training_context_mask(x, astate.context)
    for layer_id in range(action_dit.num_layers):
        block = action_dit.blocks[layer_id]
        q_a, k_a, v_a, post = block_pre_attn(block, x, astate.t_mod, astate.freqs)
        cache = video_kv_cache[layer_id]
        k = torch.cat([cache["k"], k_a], dim=1)
        v = torch.cat([cache["v"], v_a], dim=1)
        out = mixed_attention(q_a, k, v, num_heads, mask)
        x = block_post_attn(
            block,
            out,
            post,
            astate.context,
            context_mask=context_mask,
        )
    return action_dit.decode(x)
