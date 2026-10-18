"""World-token layouts for Wan2.2-TI2V.

RGB uses the frozen Wan causal VAE and contributes five latent slots.  The
future-only scalar affordance codec contributes three slots with exactly the
same ``[C,H,W]`` geometry.  All eight slots therefore reuse Wan's native 2x2
patch stem/head; modality identity is supplied only by a two-entry view
embedding. The ``rgb_scalar_overlay`` layout additionally packs
three RGB-space heatmap slots as a directed auxiliary World stream: RGB and
affordance tokens never read heatmap tokens, and Action conditions only on RGB
and affordance. ``rgb_scalar`` selects the RGB and Scalar Affordance configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import torch

from affordance_wam.base_model import (
    NUM_COND_LATENTS,
    NUM_OVERLAY_FUTURE_LATENTS,
    NUM_RGB_FUTURE_LATENTS,
    NUM_SCALAR_AFD_FUTURE_LATENTS,
    WAN_SLOT_T as SLOT_T,
)


RGB_PATCH_SIZE = (1, 2, 2)
WORLD_MODE_RGB_SCALAR = "rgb_scalar"
WORLD_MODE_RGB_SCALAR_OVERLAY = "rgb_scalar_overlay"
WORLD_MODES = (WORLD_MODE_RGB_SCALAR, WORLD_MODE_RGB_SCALAR_OVERLAY)
WORLD_SCHEMA_RGB_SCALAR = "rgb5-scalar3-shared-io-v1"
WORLD_SCHEMA_RGB_SCALAR_OVERLAY = "rgb5-scalar3-overlay3-directed-io-v1"
WORLD_SCHEMA_VERSION = WORLD_SCHEMA_RGB_SCALAR_OVERLAY


@dataclass(frozen=True)
class WorldTokenLayout:
    """Homogeneous slot geometry with a directed RGB-space heatmap suffix."""

    latent_h: int
    latent_w: int
    num_cond: int = NUM_COND_LATENTS
    num_rgb_future: int = NUM_RGB_FUTURE_LATENTS
    num_scalar_future: int = NUM_SCALAR_AFD_FUTURE_LATENTS
    num_overlay_future: int = 0
    patch_size: Tuple[int, int, int] = RGB_PATCH_SIZE

    def __post_init__(self) -> None:
        if (self.num_cond, self.num_rgb_future, self.num_scalar_future) != (2, 3, 3):
            raise ValueError("World layout requires RGB 2+3 and scalar future 3 slots")
        if self.num_overlay_future not in (0, NUM_OVERLAY_FUTURE_LATENTS):
            raise ValueError("World layout overlay future slots must be 0 or 3")
        if self.latent_h < 1 or self.latent_w < 1:
            raise ValueError("latent_h/latent_w must be positive")
        if (
            len(self.patch_size) != 3
            or self.patch_size[0] != 1
            or min(self.patch_size) < 1
        ):
            raise ValueError("patch_size must be a positive (1,H,W) tuple")
        if self.latent_h % self.patch_size[1] or self.latent_w % self.patch_size[2]:
            raise ValueError(
                f"latent grid {(self.latent_h, self.latent_w)} is not divisible by "
                f"patch_size={self.patch_size}"
            )

    @property
    def rgb_slots(self) -> int:
        return self.num_cond + self.num_rgb_future

    @property
    def scalar_slots(self) -> int:
        return self.num_scalar_future

    @property
    def latent_slots(self) -> int:
        return self.rgb_slots + self.scalar_slots + self.overlay_slots

    @property
    def grid(self) -> Tuple[int, int]:
        return (
            self.latent_h // self.patch_size[1],
            self.latent_w // self.patch_size[2],
        )

    @property
    def rgb_grid(self) -> Tuple[int, int]:
        return self.grid

    @property
    def afd_grid(self) -> Tuple[int, int]:
        return self.grid

    @property
    def rgb_patch_size(self) -> Tuple[int, int, int]:
        return self.patch_size

    @property
    def afd_patch_size(self) -> Tuple[int, int, int]:
        return self.patch_size

    @property
    def tokens_per_slot(self) -> int:
        return self.grid[0] * self.grid[1]

    @property
    def rgb_tokens_per_slot(self) -> int:
        return self.tokens_per_slot

    @property
    def afd_tokens_per_slot(self) -> int:
        return self.tokens_per_slot

    @property
    def rgb_token_count(self) -> int:
        return self.rgb_slots * self.tokens_per_slot

    @property
    def scalar_token_count(self) -> int:
        return self.scalar_slots * self.tokens_per_slot

    @property
    def core_token_count(self) -> int:
        return (self.rgb_slots + self.scalar_slots) * self.tokens_per_slot

    @property
    def overlay_token_count(self) -> int:
        return self.overlay_slots * self.tokens_per_slot

    @property
    def overlay_slots(self) -> int:
        return self.num_overlay_future

    @property
    def world_token_count(self) -> int:
        return self.latent_slots * self.tokens_per_slot

    @property
    def rgb_token_slice(self) -> slice:
        return slice(0, self.rgb_token_count)

    @property
    def scalar_token_slice(self) -> slice:
        return slice(self.rgb_token_count, self.core_token_count)

    @property
    def core_token_slice(self) -> slice:
        return slice(0, self.core_token_count)

    @property
    def overlay_token_slice(self) -> slice:
        return slice(self.core_token_count, self.world_token_count)

    @property
    def world_schema(self) -> str:
        prefix = (
            WORLD_SCHEMA_RGB_SCALAR_OVERLAY
            if self.overlay_slots
            else WORLD_SCHEMA_RGB_SCALAR
        )
        return f"{prefix}-{self.tokens_per_slot}tok"

    @property
    def world_mode(self) -> str:
        return (
            WORLD_MODE_RGB_SCALAR_OVERLAY
            if self.overlay_slots
            else WORLD_MODE_RGB_SCALAR
        )

    def token_coordinates(
        self,
    ) -> Tuple[List[int], List[float], List[float], List[int]]:
        temporal_ids = default_slot_temporal_ids(
            self.num_cond,
            self.num_rgb_future,
            self.num_scalar_future,
            self.num_overlay_future,
        )
        view_ids = default_slot_view_ids(
            self.num_cond,
            self.num_rgb_future,
            self.num_scalar_future,
            self.num_overlay_future,
        )
        tids: List[int] = []
        ys: List[float] = []
        xs: List[float] = []
        views: List[int] = []
        for slot, temporal_id in enumerate(temporal_ids):
            for y in range(self.grid[0]):
                for x in range(self.grid[1]):
                    tids.append(temporal_id)
                    ys.append(float(y))
                    xs.append(float(x))
                    views.append(view_ids[slot])
        return tids, ys, xs, views


def build_world_token_layout(
    latent_h: int,
    latent_w: int,
    *,
    patch_size: Tuple[int, int, int] = RGB_PATCH_SIZE,
    world_mode: str = WORLD_MODE_RGB_SCALAR_OVERLAY,
) -> WorldTokenLayout:
    if world_mode not in WORLD_MODES:
        raise ValueError(f"unknown world_mode {world_mode!r}; expected {WORLD_MODES}")
    return WorldTokenLayout(
        latent_h=int(latent_h),
        latent_w=int(latent_w),
        num_overlay_future=(
            NUM_OVERLAY_FUTURE_LATENTS
            if world_mode == WORLD_MODE_RGB_SCALAR_OVERLAY
            else 0
        ),
        patch_size=tuple(patch_size),
    )


def default_slot_temporal_ids(
    num_cond: int = NUM_COND_LATENTS,
    num_rgb_future: int = NUM_RGB_FUTURE_LATENTS,
    num_scalar_afd_future: int = NUM_SCALAR_AFD_FUTURE_LATENTS,
    num_overlay_future: int = 0,
) -> List[int]:
    """Return physical-time IDs; all future modalities reuse ``2,3,4``."""

    ids = list(range(num_cond + num_rgb_future)) + list(
        range(num_cond, num_cond + num_scalar_afd_future)
    )
    ids += list(range(num_cond, num_cond + num_overlay_future))
    return ids


def default_slot_view_ids(
    num_cond: int = NUM_COND_LATENTS,
    num_rgb_future: int = NUM_RGB_FUTURE_LATENTS,
    num_scalar_afd_future: int = NUM_SCALAR_AFD_FUTURE_LATENTS,
    num_overlay_future: int = 0,
) -> List[int]:
    """Per-slot modality identity: RGB=0, scalar Afd=1, overlay=2."""

    return (
        [0] * (num_cond + num_rgb_future)
        + [1] * num_scalar_afd_future
        + [2] * num_overlay_future
    )


def pack_world_latent(
    rgb_latent: torch.Tensor,
    scalar_afd_future_latent: torch.Tensor,
    overlay_future_latent: Optional[torch.Tensor] = None,
    *,
    num_cond: int = NUM_COND_LATENTS,
    num_rgb_future: int = NUM_RGB_FUTURE_LATENTS,
    num_scalar_afd_future: int = NUM_SCALAR_AFD_FUTURE_LATENTS,
    num_overlay_future: int = 0,
) -> torch.Tensor:
    """Pack ``RGB5 + scalar3 [+ overlay3]`` after geometry validation."""

    rgb_slots = num_cond + num_rgb_future
    if rgb_latent.ndim != 5 or scalar_afd_future_latent.ndim != 5:
        raise ValueError("RGB/scalar latents must be [B,C,T,H,W]")
    if rgb_latent.shape[2] < rgb_slots:
        raise ValueError(
            f"rgb_latent has {rgb_latent.shape[2]} slots; need {rgb_slots}"
        )
    if scalar_afd_future_latent.shape[2] != num_scalar_afd_future:
        raise ValueError(
            f"scalar future latent has {scalar_afd_future_latent.shape[2]} slots; "
            f"need exactly {num_scalar_afd_future}"
        )
    if (
        rgb_latent.shape[:2] != scalar_afd_future_latent.shape[:2]
        or rgb_latent.shape[3:] != scalar_afd_future_latent.shape[3:]
    ):
        raise ValueError(
            "RGB/scalar latent geometry mismatch: "
            f"{tuple(rgb_latent.shape)} vs {tuple(scalar_afd_future_latent.shape)}"
        )
    parts = [rgb_latent[:, :, :rgb_slots], scalar_afd_future_latent]
    if num_overlay_future:
        if overlay_future_latent is None:
            raise ValueError("overlay World mode requires overlay future latent")
        if (
            overlay_future_latent.ndim != 5
            or overlay_future_latent.shape[2] != num_overlay_future
        ):
            raise ValueError(
                f"overlay future latent must be [B,C,{num_overlay_future},H,W], "
                f"got {tuple(overlay_future_latent.shape)}"
            )
        if (
            rgb_latent.shape[:2] != overlay_future_latent.shape[:2]
            or rgb_latent.shape[3:] != overlay_future_latent.shape[3:]
        ):
            raise ValueError("RGB/overlay latent geometry mismatch")
        parts.append(overlay_future_latent)
    elif overlay_future_latent is not None:
        raise ValueError("rgb_scalar mode must not receive overlay future latent")
    return torch.cat(parts, dim=2).contiguous()


def split_world_velocity(
    value: torch.Tensor,
    *,
    num_cond: int = NUM_COND_LATENTS,
    num_rgb_future: int = NUM_RGB_FUTURE_LATENTS,
    num_scalar_afd_future: int = NUM_SCALAR_AFD_FUTURE_LATENTS,
    num_overlay_future: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Split supervised RGB, scalar and optional overlay future blocks."""

    rgb_end = num_cond + num_rgb_future
    scalar_end = rgb_end + num_scalar_afd_future
    expected = scalar_end + num_overlay_future
    if value.ndim != 5 or value.shape[2] != expected:
        raise ValueError(
            f"World tensor must have {expected} slots, got {tuple(value.shape)}"
        )
    overlay = value[:, :, scalar_end:expected] if num_overlay_future else None
    return value[:, :, num_cond:rgb_end], value[:, :, rgb_end:scalar_end], overlay


def build_world_token_rope_freqs(
    dit, layout: WorldTokenLayout, device: torch.device
) -> torch.Tensor:
    temporal_ids, y_positions, x_positions, _ = layout.token_coordinates()
    tids = torch.as_tensor(temporal_ids, dtype=torch.long)
    ys = torch.as_tensor(y_positions, dtype=torch.long)
    xs = torch.as_tensor(x_positions, dtype=torch.long)
    return (
        torch.cat([dit.freqs[0][tids], dit.freqs[1][ys], dit.freqs[2][xs]], dim=-1)
        .view(layout.world_token_count, 1, -1)
        .to(device)
    )


def add_world_view_embeddings(
    x: torch.Tensor, layout: WorldTokenLayout, view_embedder
) -> torch.Tensor:
    if view_embedder is None:
        return x
    _, _, _, view_ids = layout.token_coordinates()
    ids = torch.as_tensor(view_ids, dtype=torch.long, device=x.device)
    return x + view_embedder(ids).to(dtype=x.dtype).unsqueeze(0)
