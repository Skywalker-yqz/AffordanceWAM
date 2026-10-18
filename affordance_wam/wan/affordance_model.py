"""World flow matching for RGB, Scalar Affordance, and Affordance Heatmaps.

The frozen causal video VAE encodes RGB and RGB-space heatmaps. A trainable
Scalar Affordance codec maps twelve future fields to three aligned slots.
Latent velocity losses train World prediction; clean field reconstruction
supervises the scalar codec. All streams share the Wan patch embedding/head.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from affordance_wam.base_model import (
    NUM_COND_LATENTS,
    NUM_OVERLAY_FUTURE_LATENTS,
    NUM_RGB_FUTURE_LATENTS,
    NUM_SCALAR_AFD_FUTURE_LATENTS,
    BaseModelSpec,
    get_spec,
)
from affordance_wam.data.wan_adapt import wan_batch_to_world_pixels
from affordance_wam.losses import compute_lambda_warmup
from affordance_wam.wan.overlay_renderer import (
    SUPPORTED_OVERLAY_COLORMAPS,
    render_overlay_tensor,
)
from affordance_wam.wan.scalar_affordance import ScalarAffordanceCodec
from affordance_wam.wan.slot_latent import (
    WorldTokenLayout,
    WORLD_MODE_RGB_SCALAR,
    WORLD_MODE_RGB_SCALAR_OVERLAY,
    WORLD_MODES,
    build_world_token_layout,
    default_slot_temporal_ids,
    default_slot_view_ids,
    pack_world_latent,
    split_world_velocity,
)


@dataclass
class WanAffordanceConfig:
    """World training and inference-independent hyperparameters."""

    base_model: str
    world_mode: str = WORLD_MODE_RGB_SCALAR_OVERLAY
    lambda_scalar_afd_final: float = 0.2
    lambda_scalar_reconstruction: float = 0.1
    lambda_overlay_final: float = 0.5
    scalar_warmup_start_iter: int = 1000
    scalar_warmup_end_iter: int = 5000
    overlay_warmup_start_iter: int = 3000
    overlay_warmup_end_iter: int = 7000
    overlay_target_alpha: float = 0.6
    overlay_target_colormap: str = "jet"
    num_cond_latents: int = NUM_COND_LATENTS
    num_rgb_future_latents: int = NUM_RGB_FUTURE_LATENTS
    num_scalar_afd_future_latents: int = NUM_SCALAR_AFD_FUTURE_LATENTS
    flow_shift: float = 5.0
    text_max_len: int = 512
    use_gradient_checkpointing: bool = True
    time_conditioning_token_grad_normalization: int = 0
    freeze_world_time_conditioning: int = 0
    freeze_world_text_embedding: int = 0

    def __post_init__(self) -> None:
        if self.world_mode not in WORLD_MODES:
            raise ValueError(
                f"unknown world_mode {self.world_mode!r}; expected {WORLD_MODES}"
            )
        if self.scalar_warmup_start_iter < 0:
            raise ValueError("scalar_warmup_start_iter must be non-negative")
        if self.scalar_warmup_end_iter <= self.scalar_warmup_start_iter:
            raise ValueError(
                "scalar_warmup_end_iter must exceed scalar_warmup_start_iter"
            )
        if self.overlay_warmup_start_iter < 0:
            raise ValueError("overlay_warmup_start_iter must be non-negative")
        if self.overlay_warmup_end_iter <= self.overlay_warmup_start_iter:
            raise ValueError(
                "overlay_warmup_end_iter must exceed overlay_warmup_start_iter"
            )
        for name in (
            "lambda_scalar_afd_final",
            "lambda_scalar_reconstruction",
            "lambda_overlay_final",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            not math.isfinite(float(self.overlay_target_alpha))
            or not 0.0 <= float(self.overlay_target_alpha) <= 1.0
        ):
            raise ValueError("overlay_target_alpha must be finite and in [0,1]")
        if self.overlay_target_colormap not in SUPPORTED_OVERLAY_COLORMAPS:
            raise ValueError(
                f"unsupported overlay_target_colormap "
                f"{self.overlay_target_colormap!r}"
            )
        if (
            self.num_cond_latents,
            self.num_rgb_future_latents,
            self.num_scalar_afd_future_latents,
        ) != (2, 3, 3):
            raise ValueError(
                "17-frame World contract requires RGB 2+3 and scalar 3 slots"
            )
        if not math.isfinite(float(self.flow_shift)) or float(self.flow_shift) <= 0:
            raise ValueError("flow_shift must be finite and positive")
        if int(self.text_max_len) < 1:
            raise ValueError("text_max_len must be positive")
        for name in (
            "time_conditioning_token_grad_normalization",
            "freeze_world_time_conditioning",
            "freeze_world_text_embedding",
        ):
            if int(getattr(self, name)) not in (0, 1):
                raise ValueError(f"{name} must be 0 or 1")


class WanAffordanceModel(nn.Module):
    """RGB, Scalar Affordance, and Affordance Heatmap World Expert."""

    def __init__(
        self,
        *,
        spec: BaseModelSpec,
        dit: nn.Module,
        vae: nn.Module,
        config: WanAffordanceConfig,
        text_encoder: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
    ) -> None:
        super().__init__()
        if not spec.is_wan or spec.wan_variant != "ti2v":
            raise ValueError("unified World requires Wan2.2-TI2V")
        if config.base_model != spec.key:
            raise ValueError(
                f"config.base_model={config.base_model!r} does not match {spec.key!r}"
            )
        if tuple(dit.patch_size) != (1, 2, 2):
            raise ValueError(
                f"World requires native Wan patch_size (1,2,2), got {dit.patch_size}"
            )
        self.spec = spec
        self.config = config
        self.variant = spec.wan_variant
        self.fuse_vae_embedding = bool(
            getattr(dit, "fuse_vae_embedding_in_latents", False)
        )
        self.dit = dit
        if bool(config.freeze_world_time_conditioning):
            self.dit.time_embedding.requires_grad_(False)
            self.dit.time_projection.requires_grad_(False)
        if bool(config.freeze_world_text_embedding):
            self.dit.text_embedding.requires_grad_(False)

        self._frozen: Dict[str, Any] = {
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }
        self._freeze(vae)
        self._freeze(text_encoder)

        num_overlay_future = (
            NUM_OVERLAY_FUTURE_LATENTS
            if config.world_mode == WORLD_MODE_RGB_SCALAR_OVERLAY
            else 0
        )
        self.slot_temporal_ids: List[int] = default_slot_temporal_ids(
            config.num_cond_latents,
            config.num_rgb_future_latents,
            config.num_scalar_afd_future_latents,
            num_overlay_future,
        )
        self.slot_view_ids: List[int] = default_slot_view_ids(
            config.num_cond_latents,
            config.num_rgb_future_latents,
            config.num_scalar_afd_future_latents,
            num_overlay_future,
        )
        self.view_embedder = nn.Embedding(3 if num_overlay_future else 2, dit.dim)
        nn.init.zeros_(self.view_embedder.weight)
        self.scalar_afd_codec = ScalarAffordanceCodec(
            latent_channels=spec.vae_z_dim,
            spatial_compression=spec.spatial_compression,
            temporal_compression=spec.temporal_compression,
        )
        self.view_embedder.to(device=self.device, dtype=self.dtype)
        self.scalar_afd_codec.to(device=self.device, dtype=self.dtype)
        self.nominal_world_layout = build_world_token_layout(
            spec.latent_h,
            spec.latent_w,
            patch_size=tuple(dit.patch_size),
            world_mode=config.world_mode,
        )
        self._iteration = 0

    @staticmethod
    def _freeze(module: Optional[nn.Module]) -> None:
        if module is None:
            return
        module.eval()
        module.requires_grad_(False)

    @property
    def vae(self) -> nn.Module:
        return self._frozen["vae"]

    @property
    def text_encoder(self) -> Optional[nn.Module]:
        return self._frozen["text_encoder"]

    @property
    def tokenizer(self):
        return self._frozen["tokenizer"]

    @property
    def device(self) -> torch.device:
        return next(self.dit.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.dit.parameters()).dtype

    def world_layout_for(self, latents: torch.Tensor) -> WorldTokenLayout:
        if latents.ndim != 5:
            raise ValueError(
                f"World latents must be [B,C,T,H,W], got {tuple(latents.shape)}"
            )
        layout = build_world_token_layout(
            int(latents.shape[-2]),
            int(latents.shape[-1]),
            patch_size=tuple(self.dit.patch_size),
            world_mode=self.config.world_mode,
        )
        if int(latents.shape[2]) != layout.latent_slots:
            raise ValueError(
                f"World latent has {latents.shape[2]} slots; requires {layout.latent_slots}"
            )
        return layout

    @torch.no_grad()
    def encode_video(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.vae.batch_encode(pixels, device=pixels.device).float()

    @torch.no_grad()
    def encode_text(
        self, captions: Optional[Sequence[str]], batch_size: int
    ) -> torch.Tensor:
        if self.text_encoder is None or self.tokenizer is None:
            return torch.zeros(
                batch_size,
                self.config.text_max_len,
                self.spec.text_dim,
                device=self.device,
                dtype=self.dtype,
            )
        from affordance_wam.wan_vendor.encode import encode_text as _encode_text

        if captions is None:
            captions = [""] * batch_size
        if isinstance(captions, str):
            captions = [captions]
        context, _ = _encode_text(
            list(captions),
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            device=self.device,
            max_length=self.config.text_max_len,
        )
        return context.to(self.dtype)

    def _affordance_presence_mask(
        self, batch: Dict[str, Any], batch_size: int
    ) -> torch.Tensor:
        value = batch.get("has_affordance")
        if value is None:
            return torch.ones(batch_size, device=self.device, dtype=torch.float32)
        mask = torch.as_tensor(value, device=self.device, dtype=torch.float32).reshape(
            -1
        )
        if mask.numel() != batch_size or not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError("has_affordance must be a B-element binary mask")
        return mask

    def _scalar_reconstruction_loss(
        self, scalar_future: torch.Tensor, has_affordance: torch.Tensor
    ) -> torch.Tensor:
        target = self.scalar_afd_codec._pack_future_groups(scalar_future.float())
        pred = self.scalar_afd_codec.decode_groups(
            self.scalar_afd_codec.encode_future(scalar_future)
        )
        per_sample = (pred.float() - target).square().flatten(1).mean(dim=1)
        valid = has_affordance > 0
        if not bool(valid.any()):
            return pred.sum() * 0.0
        return (per_sample[valid] * has_affordance[valid]).sum() / has_affordance[
            valid
        ].sum()

    def _encode_world(self, batch: Dict[str, torch.Tensor]):
        """Encode RGB17, scalar-Afd12 and optional future overlay12."""

        rgb_pixels, scalar_pixels = wan_batch_to_world_pixels(
            batch, dtype=self.dtype, device=self.device
        )
        expected_frames = (
            1
            + (self.config.num_cond_latents + self.config.num_rgb_future_latents - 1)
            * self.spec.temporal_compression
        )
        if (
            rgb_pixels.shape[2] != expected_frames
            or scalar_pixels.shape[2] != expected_frames
        ):
            raise ValueError(
                f"World expects exactly {expected_frames} RGB/scalar frames, got "
                f"{rgb_pixels.shape[2]}/{scalar_pixels.shape[2]}"
            )
        history_frames = (
            1 + (self.config.num_cond_latents - 1) * self.spec.temporal_compression
        )
        scalar_future = scalar_pixels[:, :, history_frames:]
        expected_future = (
            self.config.num_scalar_afd_future_latents * self.spec.temporal_compression
        )
        if scalar_future.shape[2] != expected_future:
            raise ValueError(
                f"scalar future must contain {expected_future} frames, got {scalar_future.shape[2]}"
            )
        rgb_latent = self.encode_video(rgb_pixels)
        scalar_live = self.scalar_afd_codec.encode_future(scalar_future.to(self.dtype))
        overlay_future_latent = None
        if self.config.world_mode == WORLD_MODE_RGB_SCALAR_OVERLAY:
            rgb_unit = (rgb_pixels.float().clamp(-1.0, 1.0) + 1.0) * 0.5
            overlay_future = render_overlay_tensor(
                rgb_unit[:, :, history_frames:],
                scalar_future.float(),
                alpha=float(self.config.overlay_target_alpha),
                colormap_name=str(self.config.overlay_target_colormap),
            )
            # Encode a complete 17-frame causal stream so overlay slots 2/3/4
            # have exactly the same temporal receptive field as RGB slots
            # 2/3/4.  The five history frames remain plain RGB and are not
            # duplicated into the learned state.
            overlay_full = torch.cat(
                [rgb_unit[:, :, :history_frames], overlay_future], dim=2
            )
            overlay_all_latent = self.encode_video(
                overlay_full.mul(2.0).sub(1.0).to(self.dtype)
            )
            overlay_future_latent = overlay_all_latent[
                :,
                :,
                self.config.num_cond_latents : (
                    self.config.num_cond_latents + NUM_OVERLAY_FUTURE_LATENTS
                ),
            ]
            if overlay_future_latent.shape[2] != NUM_OVERLAY_FUTURE_LATENTS:
                raise ValueError("overlay VAE must produce exactly three future slots")
        has_affordance = self._affordance_presence_mask(batch, rgb_pixels.shape[0])
        reconstruction = self._scalar_reconstruction_loss(
            scalar_future.to(self.dtype), has_affordance
        )
        # Scalar latents connect flow learning to the trainable affordance encoder.
        world = pack_world_latent(
            rgb_latent,
            scalar_live.float(),
            None if overlay_future_latent is None else overlay_future_latent.float(),
            num_cond=self.config.num_cond_latents,
            num_rgb_future=self.config.num_rgb_future_latents,
            num_scalar_afd_future=self.config.num_scalar_afd_future_latents,
            num_overlay_future=(
                NUM_OVERLAY_FUTURE_LATENTS
                if self.config.world_mode == WORLD_MODE_RGB_SCALAR_OVERLAY
                else 0
            ),
        )
        return world, reconstruction

    def _sample_world_flow(self, x0: torch.Tensor):
        batch = x0.shape[0]
        sigma = torch.rand(batch, device=x0.device, dtype=torch.float32)
        timestep = sigma * 1000.0
        weight = torch.ones_like(sigma)
        noise = torch.randn_like(x0)
        xt = x0 + sigma.view(batch, 1, 1, 1, 1) * (noise - x0)
        target = (noise - x0).detach()
        xt[:, :, : self.config.num_cond_latents] = x0[
            :, :, : self.config.num_cond_latents
        ]
        return sigma, timestep, weight, xt, target

    def _forward_world(
        self, xt: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        from affordance_wam.wan import expert_blocks as experts

        state = experts.video_prepare(
            self,
            xt.to(self.dtype),
            timestep.to(self.dtype),
            self.dit.embed_text(context),
        )
        hidden = experts.run_world_loop(
            self.dit,
            state=state,
            use_gradient_checkpointing=bool(
                self.config.use_gradient_checkpointing and self.training
            ),
        )
        return experts.video_finalize(self, hidden, state)

    def _weighted_mse(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is not None:
            mask = mask.to(pred.device, torch.float32).reshape(pred.shape[0])
            valid = mask > 0
            if not bool(valid.any()):
                return pred.sum() * 0.0
            per_sample = (
                (pred[valid].float() - target[valid].float())
                .square()
                .flatten(1)
                .mean(1)
            )
            return (per_sample * weight[valid] * mask[valid]).sum() / mask[valid].sum()
        per_sample = (pred.float() - target.float()).square().flatten(1).mean(1)
        return (per_sample * weight).mean()

    def _auxiliary_schedule_iteration(self, iteration: int) -> int:
        return int(iteration)

    def _auxiliary_warmup_fraction(self, iteration: int) -> float:
        return float(
            compute_lambda_warmup(
                self._auxiliary_schedule_iteration(iteration),
                1.0,
                self.config.scalar_warmup_start_iter,
                self.config.scalar_warmup_end_iter,
            )
        )

    def _overlay_warmup_fraction(self, iteration: int) -> float:
        if self.config.world_mode != WORLD_MODE_RGB_SCALAR_OVERLAY:
            return 0.0
        return float(
            compute_lambda_warmup(
                self._auxiliary_schedule_iteration(iteration),
                1.0,
                self.config.overlay_warmup_start_iter,
                self.config.overlay_warmup_end_iter,
            )
        )

    def _world_losses(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        _x0: torch.Tensor,
        _xt: torch.Tensor,
        _sigma: torch.Tensor,
        weight: torch.Tensor,
        has_affordance: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        num_overlay_future = int(self.nominal_world_layout.overlay_slots)
        pred_rgb, pred_scalar, pred_overlay = split_world_velocity(
            prediction,
            num_cond=self.config.num_cond_latents,
            num_rgb_future=self.config.num_rgb_future_latents,
            num_scalar_afd_future=self.config.num_scalar_afd_future_latents,
            num_overlay_future=num_overlay_future,
        )
        target_rgb, target_scalar, target_overlay = split_world_velocity(
            target,
            num_cond=self.config.num_cond_latents,
            num_rgb_future=self.config.num_rgb_future_latents,
            num_scalar_afd_future=self.config.num_scalar_afd_future_latents,
            num_overlay_future=num_overlay_future,
        )
        rgb = self._weighted_mse(pred_rgb, target_rgb, weight)
        scalar = self._weighted_mse(pred_scalar, target_scalar, weight, has_affordance)
        overlay = (
            self._weighted_mse(pred_overlay, target_overlay, weight, has_affordance)
            if pred_overlay is not None and target_overlay is not None
            else rgb.detach().new_zeros((), dtype=torch.float32)
        )
        return {
            "L_rgb": rgb,
            "L_scalar_afd": scalar,
            "L_overlay": overlay,
        }

    def compute_loss(
        self, batch: Dict[str, torch.Tensor], iteration: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        if iteration is None:
            iteration = self._iteration
            self._iteration += 1
        x0, reconstruction = self._encode_world(batch)
        batch_size = x0.shape[0]
        context = self.encode_text(batch.get("ai_caption"), batch_size)
        sigma, timestep, weight, xt, target = self._sample_world_flow(x0)
        prediction = self._forward_world(xt, timestep, context)
        has_affordance = self._affordance_presence_mask(batch, batch_size)
        losses = self._world_losses(
            prediction, target, x0, xt, sigma, weight, has_affordance
        )
        warm = self._auxiliary_warmup_fraction(iteration)
        overlay_warm = self._overlay_warmup_fraction(iteration)
        loss = losses["L_rgb"]
        if warm > 0 and self.config.lambda_scalar_afd_final > 0:
            loss = (
                loss
                + warm * self.config.lambda_scalar_afd_final * losses["L_scalar_afd"]
            )
        if self.config.lambda_scalar_reconstruction > 0:
            loss = loss + self.config.lambda_scalar_reconstruction * reconstruction
        if overlay_warm > 0 and self.config.lambda_overlay_final > 0:
            loss = (
                loss
                + overlay_warm * self.config.lambda_overlay_final * losses["L_overlay"]
            )
        return {
            "loss": loss,
            "L_obs": losses["L_rgb"].detach(),
            "L_aff": losses["L_scalar_afd"].detach(),
            "L_rgb": losses["L_rgb"].detach(),
            "L_scalar_afd": losses["L_scalar_afd"].detach(),
            "L_overlay": losses["L_overlay"].detach(),
            "L_scalar_recon": reconstruction.detach(),
            "lambda_aff": torch.as_tensor(
                warm * self.config.lambda_scalar_afd_final, device=self.device
            ),
            "lambda_overlay": torch.as_tensor(
                overlay_warm * self.config.lambda_overlay_final,
                device=self.device,
            ),
        }

    def forward(
        self, batch: Dict[str, torch.Tensor], iteration: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        return self.compute_loss(batch, iteration)


def _filter_dit_vae_only(model_configs):
    return [
        config
        for config in model_configs
        if not any(
            token in str(config.path).lower()
            for token in ("t5", "text_encoder", "clip")
        )
    ]


def build_wan_affordance_model(
    spec_or_key,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    config: Optional[WanAffordanceConfig] = None,
    load_text_encoder: bool = True,
) -> WanAffordanceModel:
    spec = (
        spec_or_key if isinstance(spec_or_key, BaseModelSpec) else get_spec(spec_or_key)
    )
    if config is None:
        config = WanAffordanceConfig(base_model=spec.key)
    from affordance_wam.wan_vendor.loader import load_wan_components
    from affordance_wam.wan_vendor.pipeline_builder import discover_model_files

    model_configs, tokenizer_config = discover_model_files(spec.model_dir)
    if not load_text_encoder:
        model_configs = _filter_dit_vae_only(model_configs)
        tokenizer_config = None
    holder = load_wan_components(
        model_configs,
        tokenizer_config if load_text_encoder else None,
        device=device,
        torch_dtype=dtype,
    )
    if holder.dit is None or holder.vae is None:
        raise RuntimeError(f"missing DiT/VAE from {spec.model_dir}")
    return WanAffordanceModel(
        spec=spec,
        dit=holder.dit,
        vae=holder.vae,
        config=config,
        text_encoder=holder.text_encoder if load_text_encoder else None,
        tokenizer=holder.tokenizer if load_text_encoder else None,
    )


__all__ = ["WanAffordanceConfig", "WanAffordanceModel", "build_wan_affordance_model"]
