"""World prediction outputs and video visualizations."""

from __future__ import annotations

import contextlib
import dataclasses
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from einops import rearrange

from affordance_wam.data.video_io import write_h264_mp4


@dataclasses.dataclass
class SampleOutputs:
    """Outputs of one ``run_inference`` call.

    Tensors are kept on the model's device (GPU at inference time).
    Pixel tensors are float in ``[-1, 1]``; convert via
    ``((x.clamp(-1, 1) + 1) * 127.5).round().byte()`` for display.
    """

    rgb_pixel: torch.Tensor  # [B, 3, 17, H_pix, W_pix] float
    aff_future_pixel: Optional[torch.Tensor]  # optional inference overlay
    samples_latent: torch.Tensor  # [B, Z, world_slots, H_lat, W_lat]
    rgb_history_latent: torch.Tensor  # [B, Z, 2, H_lat, W_lat]
    # Optional reference tensors for prediction visualizations.
    rgb_gt: Optional[torch.Tensor] = None  # [B, 3, 17, H_pix, W_pix] uint8
    aff_gt: Optional[torch.Tensor] = None  # [B, 3, 17, H_pix, W_pix] uint8
    # Wan canonical three-view world output. ``aff_future_pixel``/``aff_gt``
    # describe the RGB-space heatmap branch.
    # Scalar Afd is a genuine one-channel probability field in [0, 1].
    scalar_affordance_future_pixel: Optional[torch.Tensor] = None  # [B,1,12,H,W]
    scalar_affordance_gt: Optional[torch.Tensor] = None  # [B,1,17,H,W]

    @property
    def overlay_future_pixel(self) -> Optional[torch.Tensor]:
        """Name for the ``aff_future_pixel`` field."""

        return self.aff_future_pixel

    @property
    def overlay_gt(self) -> Optional[torch.Tensor]:
        """Name for the ``aff_gt`` field."""

        return self.aff_gt


# ---------------------------------------------------------------------------
# run_inference
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# save_side_by_side_mp4
# ---------------------------------------------------------------------------


def _to_uint8(pixel: torch.Tensor) -> torch.Tensor:
    """``[-1, 1]`` float → ``[0, 255]`` uint8 (clamped)."""
    return ((pixel.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def save_side_by_side_mp4(
    out_path: Path | str,
    rgb_pixel: torch.Tensor,  # [3, 17, H, W] float in [-1, 1]
    aff_future_pixel: torch.Tensor,  # [3, 12, H, W] float in [-1, 1]
    *,
    fps: int = 10,
    crf: int = 23,
) -> Path:
    """Encode ``RGB ‖ AFF`` side-by-side mp4 at ``fps`` (10 FPS by default).

    The AFF timeline is 12 frames (future only); we left-pad it with the
    first 5 RGB cond frames so the two timelines align horizontally and
    the resulting 17-frame video can be inspected at a glance.
    """
    assert (
        rgb_pixel.dim() == 4 and rgb_pixel.shape[0] == 3
    ), f"rgb_pixel must be [3, 17, H, W], got {tuple(rgb_pixel.shape)}"
    assert (
        aff_future_pixel.dim() == 4 and aff_future_pixel.shape[0] == 3
    ), f"aff_future_pixel must be [3, 12, H, W], got {tuple(aff_future_pixel.shape)}"
    assert (
        rgb_pixel.shape[1] == 17
    ), f"rgb_pixel temporal dim must be 17, got {rgb_pixel.shape[1]}"
    assert (
        aff_future_pixel.shape[1] == 12
    ), f"aff_future_pixel temporal dim must be 12, got {aff_future_pixel.shape[1]}"

    rgb_uint8 = _to_uint8(rgb_pixel.cpu())  # [3, 17, H, W]
    aff_uint8 = _to_uint8(aff_future_pixel.cpu())  # [3, 12, H, W]

    # Pad AFF with the first 5 RGB cond frames so timelines align.
    aff_padded = torch.cat([rgb_uint8[:, :5], aff_uint8], dim=1)  # [3, 17, H, W]

    # Side-by-side along width.
    sbs = torch.cat([rgb_uint8, aff_padded], dim=3)  # [3, 17, H, 2W]
    sbs_thwc: np.ndarray = rearrange(sbs.numpy(), "c t h w -> t h w c")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_h264_mp4(out_path, sbs_thwc, fps=fps, crf=crf)
    return out_path


def _draw_border(frames: torch.Tensor, width: int = 3, rgb=(255, 0, 0)) -> torch.Tensor:
    """Burn an ``rgb`` border ``width`` px thick around every frame.

    ``frames``: ``[3, T, H, W]`` uint8 (modified out-of-place; a clone is
    returned). Used to flag visualisation-only padding frames so a reviewer
    never mistakes them for model output.
    """
    f = frames.clone()
    r, g, b = rgb
    for c, val in enumerate((r, g, b)):
        f[c, :, :width, :] = val
        f[c, :, -width:, :] = val
        f[c, :, :, :width] = val
        f[c, :, :, -width:] = val
    return f


def save_comparison_grid(
    out_path: Path | str,
    rgb_gt: torch.Tensor,  # [3, 17, H, W] uint8 [0, 255]
    rgb_pred: torch.Tensor,  # [3, 17, H, W] float [-1, 1]
    aff_gt: torch.Tensor,  # [3, 17, H, W] uint8 [0, 255]
    aff_pred_future: torch.Tensor,  # [3, 12, H, W] float [-1, 1]
    *,
    fps: int = 2,
    crf: int = 23,
) -> Path:
    """Encode a 4-column ``RGB GT | RGB pred | AFF GT | AFF pred`` grid.

    All columns span 17 frames. ``aff_pred_future`` (12 future frames) is
    left-padded with the *first 5 GT affordance* frames so its timeline
    aligns with the others; those 5 pad frames carry a red border to mark
    them as visualization padding.

    Default ``fps=2`` (5× slow vs the native 10 fps) so the heatmap
    evolution over the short 17-frame clip is easy to inspect frame by
    frame.
    """
    rgb_gt_u = rgb_gt.detach().cpu().to(torch.uint8)
    aff_gt_u = aff_gt.detach().cpu().to(torch.uint8)
    rgb_pred_u = _to_uint8(rgb_pred.detach().cpu())
    aff_pred_u = _to_uint8(aff_pred_future.detach().cpu())  # [3, 12, H, W]

    for name, t, expect_t in (
        ("rgb_gt", rgb_gt_u, 17),
        ("rgb_pred", rgb_pred_u, 17),
        ("aff_gt", aff_gt_u, 17),
        ("aff_pred_future", aff_pred_u, 12),
    ):
        assert (
            t.dim() == 4 and t.shape[0] == 3
        ), f"{name} must be [3, T, H, W], got {tuple(t.shape)}"
        assert (
            t.shape[1] == expect_t
        ), f"{name} temporal dim must be {expect_t}, got {t.shape[1]}"

    # Left-pad AFF pred with GT affordance history (first 5 frames), bordered.
    pad = _draw_border(aff_gt_u[:, :5], width=3, rgb=(255, 0, 0))
    aff_pred_padded = torch.cat([pad, aff_pred_u], dim=1)  # [3, 17, H, W]

    grid = torch.cat(
        [rgb_gt_u, rgb_pred_u, aff_gt_u, aff_pred_padded], dim=3
    )  # [3, 17, H, 4W]
    grid_thwc: np.ndarray = rearrange(grid.numpy(), "c t h w -> t h w c")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_h264_mp4(out_path, grid_thwc, fps=fps, crf=crf)
    return out_path
