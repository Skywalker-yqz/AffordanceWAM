import pytest
import torch
from torch import nn
from types import SimpleNamespace

from affordance_wam.base_model import BaseModelSpec
from affordance_wam.wan.action_model import (
    WanActionAffordanceConfig,
    WanActionAffordanceModel,
)
import affordance_wam.wan_vendor.models.dit as vendor_dit


def _mock_world_output():
    return SimpleNamespace(
        aff_future_pixel=None,
        scalar_affordance_future_pixel=None,
    )


class _TinyVAE(nn.Module):
    def batch_encode(self, pixels, device=None):
        groups = [pixels[:, :, 0:1]]
        groups.extend(
            pixels[:, :, start : start + 4] for start in range(1, pixels.shape[2], 4)
        )
        latent = torch.cat([group.mean(2, keepdim=True) for group in groups], dim=2)
        latent = latent.mean(1, keepdim=True)
        return torch.nn.functional.avg_pool3d(latent, (1, 2, 2)).repeat(1, 4, 1, 1, 1)

    def decode(self, latent, device=None, tiled=True):
        grouped = latent.float().mean(1, keepdim=True)
        b, _c, t, h, w = grouped.shape
        spatial = (
            torch.nn.functional.interpolate(
                grouped.permute(0, 2, 1, 3, 4).reshape(b * t, 1, h, w),
                scale_factor=2,
                mode="nearest",
            )
            .reshape(b, t, 1, h * 2, w * 2)
            .permute(0, 2, 1, 3, 4)
        )
        lengths = [1] + [4] * (t - 1)
        pixels = torch.cat(
            [
                spatial[:, :, i : i + 1].repeat(1, 1, length, 1, 1)
                for i, length in enumerate(lengths)
            ],
            dim=2,
        )
        return pixels.repeat(1, 3, 1, 1, 1).clamp(-1, 1)


_SPEC = BaseModelSpec(
    key="tiny",
    family="wan",
    stack="wan",
    model_subdir="",
    vae_z_dim=4,
    spatial_compression=2,
    temporal_compression=4,
    causal_temporal=True,
    text_encoder="none",
    text_dim=8,
    wan_variant="ti2v",
    dit_dim=24,
)


def _model(mode="dual_self_attn", world_mode="rgb_scalar_overlay"):
    # CPU tests must not enter an installed CUDA FlashAttention kernel.
    vendor_dit.FLASH_ATTN_2_AVAILABLE = False
    vendor_dit.FLASH_ATTN_3_AVAILABLE = False
    vendor_dit.SAGE_ATTN_AVAILABLE = False
    dit = vendor_dit.WanModel(
        dim=24,
        in_dim=4,
        ffn_dim=48,
        out_dim=4,
        text_dim=8,
        freq_dim=8,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=4,
        num_layers=2,
        has_image_input=False,
        seperated_timestep=True,
        require_vae_embedding=False,
        require_clip_embedding=False,
        fuse_vae_embedding_in_latents=True,
    )
    config = WanActionAffordanceConfig(
        base_model="tiny",
        action_mode=mode,
        world_mode=world_mode,
        action_dit_dim=16,
        action_ffn_dim=32,
        action_cross_num_heads=4,
        text_max_len=4,
        use_gradient_checkpointing=False,
        scalar_warmup_start_iter=0,
        scalar_warmup_end_iter=1,
        overlay_warmup_start_iter=0,
        overlay_warmup_end_iter=1,
        pretrain_world_only_updates=0,
        action_warmup_start=0,
        action_warmup_end=1,
        fixed_world_prob_start=0.0,
        fixed_world_prob_end=0.0,
    )
    return WanActionAffordanceModel(spec=_SPEC, dit=dit, vae=_TinyVAE(), config=config)
