"""Affordance-aware joint World/Action flow matching and two-stage training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import torch
from affordance_wam.base_model import BaseModelSpec, get_spec
from affordance_wam.losses import compute_lambda_warmup
from affordance_wam.wan.affordance_model import WanAffordanceConfig, WanAffordanceModel


@dataclass
class WanActionAffordanceConfig(WanAffordanceConfig):
    """Model, curriculum, and sampling configuration."""

    action_mode: str = "dual_self_attn"
    action_dim: int = 7
    action_horizon: int = 12
    use_proprio: bool = True
    state_dim: int = 7
    action_flow_shift: float = 5.0
    lambda_action: float = 1.0
    pretrain_world_only_updates: int = 7000
    action_warmup_start: int = 0
    action_warmup_end: int = 2000
    action_clamp: float = 0.0
    action_dit_dim: int = 1024
    action_ffn_dim: int = 4096
    action_cross_num_heads: int = 8
    # Linear curricula over Action pre-training optimizer updates.
    curriculum_updates: int = 50000
    fixed_world_prob_start: float = 0.8
    fixed_world_prob_end: float = 0.4
    joint_sync_prob_start: float = 0.5
    joint_sync_prob_end: float = 1.0
    predicted_world_prob_start: float = 0.0
    predicted_world_prob_end: float = 1.0
    predicted_world_denoise_steps: int = 35
    predicted_world_guidance: float = 1.0
    training_seed: int = 0
    posttrain_noisy_world_fraction: float = 0.75
    posttrain_world_denoise_steps: int = 35
    posttrain_world_guidance: float = 1.0
    inference_world_denoise_steps: int = 35
    inference_action_denoise_steps: int = 35
    action_refinement_steps: int = 8
    action_refinement_strength: float = 0.25

    def __post_init__(self):
        super().__post_init__()
        if self.action_mode != "dual_self_attn":
            raise ValueError("action_mode must be dual_self_attn")
        for name in (
            "action_dim",
            "action_horizon",
            "state_dim",
            "action_dit_dim",
            "action_ffn_dim",
            "action_cross_num_heads",
            "curriculum_updates",
            "predicted_world_denoise_steps",
            "posttrain_world_denoise_steps",
            "inference_world_denoise_steps",
            "inference_action_denoise_steps",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.action_dit_dim % self.action_cross_num_heads:
            raise ValueError("action_cross_num_heads must divide action_dit_dim")
        for name in (
            "fixed_world_prob_start",
            "fixed_world_prob_end",
            "joint_sync_prob_start",
            "joint_sync_prob_end",
            "predicted_world_prob_start",
            "predicted_world_prob_end",
            "posttrain_noisy_world_fraction",
        ):
            if not 0 <= float(getattr(self, name)) <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        if self.fixed_world_prob_start < self.fixed_world_prob_end:
            raise ValueError(
                "fixed-World probability must decrease over the curriculum"
            )
        if (
            self.joint_sync_prob_start > self.joint_sync_prob_end
            or self.predicted_world_prob_start > self.predicted_world_prob_end
        ):
            raise ValueError(
                "synchronized and predicted-World probabilities must increase"
            )
        if (
            self.action_warmup_start < 0
            or self.action_warmup_end <= self.action_warmup_start
        ):
            raise ValueError("invalid action warmup interval")
        if self.pretrain_world_only_updates < 0 or self.lambda_action < 0:
            raise ValueError(
                "warmup updates and action loss weight must be non-negative"
            )
        if (
            self.action_refinement_steps < 0
            or not 0 < self.action_refinement_strength <= 1
        ):
            raise ValueError("invalid action refinement configuration")


class WanActionAffordanceModel(WanAffordanceModel):
    """World and Action Experts coupled by masked joint self-attention."""

    def __init__(
        self,
        *,
        spec: BaseModelSpec,
        dit,
        vae,
        config: WanActionAffordanceConfig,
        text_encoder=None,
        tokenizer=None,
    ):
        super().__init__(
            spec=spec,
            dit=dit,
            vae=vae,
            config=config,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        from affordance_wam.wan.expert_blocks import build_action_stack

        self.action_mode = config.action_mode
        self.action_backbone = build_action_stack(dit, config).to(
            device=self.device, dtype=self.dtype
        )
        self.training_stage = "pretrain"

    def set_training_stage(self, stage: str) -> None:
        """Configure codec calibration, joint pre-training, or Action post-training."""
        if stage not in ("codec_calibration", "pretrain", "posttrain"):
            raise ValueError(f"unknown training stage {stage!r}")
        self.training_stage = stage
        self.dit.requires_grad_(stage == "pretrain")
        self.view_embedder.requires_grad_(stage == "pretrain")
        self.scalar_afd_codec.requires_grad_(stage != "posttrain")
        self.action_backbone.requires_grad_(stage != "codec_calibration")
        if stage == "posttrain":
            self.dit.eval()
            self.view_embedder.eval()
            self.scalar_afd_codec.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.training_stage == "posttrain":
            self.dit.eval()
            self.view_embedder.eval()
            self.scalar_afd_codec.eval()
        return self

    def _optimizer_update_index(self, iteration: int) -> int:
        return int(iteration)

    def _action_schedule_iteration(self, iteration: int) -> int:
        return max(0, int(iteration) - self.config.pretrain_world_only_updates)

    def _action_lambda(self, iteration: int) -> float:
        if self.training_stage == "posttrain":
            return float(self.config.lambda_action)
        if iteration < self.config.pretrain_world_only_updates:
            return 0.0
        return float(
            compute_lambda_warmup(
                self._action_schedule_iteration(iteration),
                self.config.lambda_action,
                self.config.action_warmup_start,
                self.config.action_warmup_end,
            )
        )

    @staticmethod
    def _unit_interval_draw(seed: int, iteration: int, salt: int) -> float:
        # Identical route selection on every distributed rank.
        import random

        return random.Random((int(seed) << 32) ^ int(iteration) ^ int(salt)).random()

    def _curriculum(self, iteration: int, start: float, end: float) -> float:
        progress = min(
            1.0,
            self._action_schedule_iteration(iteration) / self.config.curriculum_updates,
        )
        return float(start + progress * (end - start))

    def _self_attn_refinement_active(self, iteration: int) -> bool:
        probability = self._curriculum(
            iteration,
            self.config.fixed_world_prob_start,
            self.config.fixed_world_prob_end,
        )
        return (
            self._unit_interval_draw(self.config.training_seed, iteration, 0x51A7)
            < probability
        )

    def _predicted_condition_probability(self, iteration: int) -> float:
        return self._curriculum(
            iteration,
            self.config.predicted_world_prob_start,
            self.config.predicted_world_prob_end,
        )

    def _predicted_condition_active(self, iteration: int, probability=None) -> bool:
        if probability is None:
            probability = self._predicted_condition_probability(iteration)
        return (
            self._unit_interval_draw(self.config.training_seed, iteration, 0xC04D)
            < probability
        )

    def _sample_flow(self, x0: torch.Tensor):
        # Continuous uniform flow times, Eq. (1) and Appendix D.
        b = x0.shape[0]
        sigma = torch.rand(b, device=x0.device, dtype=torch.float32)
        timestep = sigma * 1000.0
        noise = torch.randn_like(x0)
        view = (b,) + (1,) * (x0.ndim - 1)
        xt = x0 + sigma.view(view) * (noise - x0)
        return None, sigma, timestep, torch.ones_like(sigma), noise, xt, noise - x0

    def compute_loss(
        self, batch: Dict[str, Any], iteration: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        if iteration is None:
            iteration = self._iteration
            self._iteration += 1
        if iteration < 0:
            raise ValueError("iteration must be non-negative")
        if self.training_stage == "codec_calibration":
            return self._compute_loss_codec_calibration(batch, iteration)
        if self.training_stage == "posttrain":
            return self._compute_loss_posttrain(batch, iteration)
        if iteration < self.config.pretrain_world_only_updates:
            return self._compute_loss_pretrain_world_only(batch, iteration)
        return self._compute_loss_dual_self_attn(batch, iteration)

    def _posttrain_world_condition(self, predicted_world: torch.Tensor):
        """Mix noisy and completed predicted Worlds within each batch (75%/25%)."""
        batch = predicted_world.shape[0]
        desired = batch * self.config.posttrain_noisy_world_fraction
        count = int(desired)
        if torch.rand((), device=predicted_world.device) < desired - count:
            count += 1
        noisy_rows = torch.randperm(batch, device=predicted_world.device)[:count]
        sigma = torch.zeros(batch, device=predicted_world.device, dtype=torch.float32)
        sigma[noisy_rows] = torch.rand(count, device=predicted_world.device)
        world = predicted_world.float().detach()
        condition = world + sigma[:, None, None, None, None] * (
            torch.randn_like(world) - world
        )
        condition[:, :, : self.config.num_cond_latents] = world[
            :, :, : self.config.num_cond_latents
        ]
        return condition, sigma * 1000.0, sigma > 0

    def _compute_loss_posttrain(self, batch: Dict[str, Any], iteration: int):
        """Train actions against detached noisy/completed predicted World features."""
        from affordance_wam.wan import expert_blocks as experts
        from affordance_wam.wan.wan_sampler import sample_world_latents

        action = torch.as_tensor(
            batch["action"], device=self.device, dtype=torch.float32
        )
        b = action.shape[0]
        if tuple(action.shape) != (
            b,
            self.config.action_horizon,
            self.config.action_dim,
        ):
            raise ValueError("action must have shape [B,action_horizon,action_dim]")
        proprio = batch.get("proprio_current")
        if self.config.use_proprio and proprio is None:
            raise KeyError("posttrain requires proprio_current")
        if proprio is not None:
            proprio = proprio.to(self.device, torch.float32)
        mask = self._action_loss_mask(
            batch, self._presence_mask(batch, "has_action", b, default=1.0)
        )
        if not bool((mask.reshape(b, -1).sum(1) > 0).all()):
            raise ValueError("each robot sample requires valid action targets")
        context = self.encode_text(batch.get("ai_caption"), b)
        with torch.no_grad():
            predicted = batch.get("predicted_world_latent")
            if predicted is None:
                predicted, _, _ = sample_world_latents(
                    self,
                    batch,
                    guidance=self.config.posttrain_world_guidance,
                    seed=self.config.training_seed + iteration + 17,
                    num_steps=self.config.posttrain_world_denoise_steps,
                    shift=self.config.flow_shift,
                )
            condition, world_time, noisy_rows = self._posttrain_world_condition(
                predicted.to(self.device)
            )
            state = experts.video_prepare(
                self,
                condition.to(self.dtype),
                world_time.to(self.dtype),
                self.dit.embed_text(context),
            )
            cache = experts.prefill_video_cache(self, state)
        _, _, action_time, weight, _, noisy_action, target = self._sample_flow(action)
        action_state = self.action_backbone.prepare(
            noisy_action.to(self.dtype),
            action_time.to(self.dtype),
            context,
            proprio=proprio,
        )
        prediction = experts.run_action_with_video_cache(
            self.action_backbone, action_state, cache
        )
        loss = self._weighted_mse(prediction.float(), target, weight, mask_b=mask)
        return {
            "loss": self.config.lambda_action * loss,
            "L_action": loss.detach(),
            "posttrain_noisy_world_fraction": noisy_rows.float().mean(),
        }

    def _presence_mask(
        self,
        batch: Dict[str, Any],
        key: str,
        batch_size: int,
        *,
        default: float,
    ) -> torch.Tensor:
        value = batch.get(key)
        if value is None:
            return torch.full(
                (batch_size,),
                float(default),
                device=self.device,
                dtype=torch.float32,
            )
        mask = torch.as_tensor(value, device=self.device, dtype=torch.float32)
        if mask.numel() != batch_size:
            raise ValueError(
                f"{key} must contain {batch_size} values, got {tuple(mask.shape)}"
            )
        mask = mask.reshape(batch_size)
        if not bool(torch.isfinite(mask).all()):
            raise ValueError(f"{key} must be finite")
        if not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError(f"{key} must be a binary 0/1 presence mask")
        return mask

    def _weighted_mse(
        self, pred, target, weight_b, mask_b: Optional[torch.Tensor] = None
    ):
        """Per-sample flow-weighted MSE; ``mask_b`` (``[B]`` 0/1) drops samples
        from the mean (used for explicit modality-presence masks)."""
        if mask_b is not None and mask_b.ndim == 2:
            if pred.ndim < 3 or tuple(mask_b.shape) != tuple(pred.shape[:2]):
                raise ValueError(
                    f"step mask {tuple(mask_b.shape)} incompatible with prediction {tuple(pred.shape)}"
                )
            mask = mask_b.to(device=pred.device, dtype=torch.float32)
            valid = mask > 0
            if not bool(valid.any()):
                return pred.detach().new_zeros((), dtype=torch.float32)
            # Select before subtract/square. Multiplying a masked NaN by zero
            # is still NaN and can poison otherwise valid samples in BF16.
            se = (pred[valid].float() - target[valid].float()).square()
            per_step = se.flatten(1).mean(dim=1)
            selected_weight = weight_b[:, None].expand_as(mask)[valid]
            selected_mask = mask[valid]
            return (
                per_step * selected_weight * selected_mask
            ).sum() / selected_mask.sum()
        if mask_b is not None:
            mask = mask_b.to(device=pred.device, dtype=torch.float32).reshape(
                pred.shape[0]
            )
            valid = mask > 0
            if not bool(valid.any()):
                return pred.detach().new_zeros((), dtype=torch.float32)
            se = (pred[valid].float() - target[valid].float()).square()
            per_sample = se.flatten(1).mean(dim=1)
            selected_mask = mask[valid]
            return (
                per_sample * weight_b[valid] * selected_mask
            ).sum() / selected_mask.sum()
        se = (pred.float() - target.float()).square()
        per_sample = se.flatten(1).mean(dim=1)  # [B]
        return (per_sample * weight_b).mean()

    def _action_loss_mask(self, batch, has_action: torch.Tensor) -> torch.Tensor:
        mask = batch.get("action_step_mask")
        if mask is None:
            return has_action
        mask = torch.as_tensor(mask, device=self.device, dtype=torch.float32)
        expected = (has_action.shape[0], int(self.config.action_horizon))
        if tuple(mask.shape) != expected:
            raise ValueError(
                f"action_step_mask must be {expected}, got {tuple(mask.shape)}"
            )
        if not bool(torch.isfinite(mask).all()):
            raise ValueError("action_step_mask must be finite")
        if not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError("action_step_mask must be binary 0/1")
        return mask * has_action[:, None]

    def _compute_loss_codec_calibration(
        self, batch: Dict[str, Any], iteration: int
    ) -> Dict[str, torch.Tensor]:
        """Stage-0: independently calibrate only the lightweight scalar codec."""

        from affordance_wam.data.wan_adapt import wan_scalar_affordance_from_batch

        scalar_pixels = wan_scalar_affordance_from_batch(
            batch, dtype=self.dtype, device=self.device
        )
        batch_size = scalar_pixels.shape[0]
        has_aff = self._presence_mask(batch, "has_affordance", batch_size, default=1.0)
        history_frames = (
            1 + (self.config.num_cond_latents - 1) * self.spec.temporal_compression
        )
        scalar_recon = self._scalar_reconstruction_loss(
            scalar_pixels[:, :, history_frames:], has_aff
        )
        loss = scalar_recon
        zero = scalar_recon.detach().new_zeros((), dtype=torch.float32)
        world_losses = {
            "L_rgb": zero,
            "L_scalar_afd": zero,
        }
        result = self._format_action_loss(
            loss, world_losses, scalar_recon, zero, warm=0.0
        )
        result["lambda_action"] = zero
        result["pretrain_world_only"] = torch.ones((), device=self.device)
        result["self_attn_refinement"] = zero
        result["codec_calibration"] = torch.ones((), device=self.device)
        return result

    def _compute_loss_pretrain_world_only(
        self, batch: Dict[str, Any], iteration: int
    ) -> Dict[str, torch.Tensor]:
        """World-only warmup for joint pre-training.

        This path deliberately does not read action/proprio fields and never
        calls the Action Expert, even for a Robot batch that carries them.
        """

        x0, scalar_recon = self._encode_world(batch)
        batch_size = x0.shape[0]
        context = self.encode_text(batch.get("ai_caption"), batch_size)
        has_aff = self._presence_mask(batch, "has_affordance", batch_size, default=1.0)
        sigma, timestep, weight, noisy_world, target = self._sample_world_flow(x0)
        prediction = self._forward_world(noisy_world, timestep, context)
        world_losses = self._world_losses(
            prediction,
            target,
            x0,
            noisy_world,
            sigma,
            weight,
            has_aff,
        )
        zero = prediction.detach().new_zeros((), dtype=torch.float32)
        loss, warm, lambda_action = self._compose_loss(
            world_losses, scalar_recon, zero, iteration
        )
        if float(lambda_action) != 0:
            raise RuntimeError(
                "World-only pre-training produced a non-zero Action weight"
            )
        result = self._format_action_loss(loss, world_losses, scalar_recon, zero, warm)
        result["lambda_action"] = torch.zeros((), device=self.device)
        result["pretrain_world_only"] = torch.ones((), device=self.device)
        result["self_attn_refinement"] = torch.zeros((), device=self.device)
        return result

    def _encode_slot_and_action(self, batch):
        """Common front-end for homogeneous human/robot modality batches."""
        device = self.device
        x0, scalar_recon = self._encode_world(batch)
        B = x0.shape[0]
        context = self.encode_text(batch.get("ai_caption"), B)
        action0 = batch.get("action")
        if action0 is not None:
            action0 = torch.as_tensor(action0, device=device, dtype=torch.float32)
            expected = (B, int(self.config.action_horizon), int(self.config.action_dim))
            if tuple(action0.shape) != expected:
                raise ValueError(
                    f"action must be {expected}, got {tuple(action0.shape)}"
                )
        if action0 is not None and float(self.config.action_clamp) > 0:
            k = float(self.config.action_clamp)
            action0 = action0.clamp(-k, k)
        has_aff = self._presence_mask(batch, "has_affordance", B, default=1.0)
        proprio_cur = batch.get("proprio_current")
        if proprio_cur is not None:
            proprio_cur = proprio_cur.to(
                device=device, dtype=torch.float32
            )  # [B, state_dim] current state
            if proprio_cur.shape != (B, int(self.config.state_dim)):
                raise ValueError(
                    f"proprio_current must be [B,{self.config.state_dim}], got {tuple(proprio_cur.shape)}"
                )
        if (
            action0 is not None
            and bool(self.config.use_proprio)
            and proprio_cur is None
        ):
            raise KeyError(
                "robot action batch requires proprio_current when use_proprio=True"
            )
        has_action = self._presence_mask(
            batch,
            "has_action",
            B,
            default=1.0 if action0 is not None else 0.0,
        )
        if action0 is None and bool((has_action != 0).any()):
            raise ValueError("has_action=1 requires a physical action tensor")
        if action0 is not None and not bool((has_action == 1).all()):
            raise ValueError(
                "homogeneous Robot batches with an action tensor require "
                "has_action=1 for every sample; Human batches must omit action"
            )
        return (
            x0,
            action0,
            context,
            has_aff,
            has_action,
            proprio_cur,
            scalar_recon,
        )

    def _action_loss_against_frozen_world(
        self,
        *,
        fixed_world: torch.Tensor,
        action0: torch.Tensor,
        context: torch.Tensor,
        proprio_cur: Optional[torch.Tensor],
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Action loss with a stop-gradient World feature cache."""

        from affordance_wam.wan import expert_blocks as experts

        device, dtype = self.device, self.dtype
        batch_size = action0.shape[0]
        if fixed_world.ndim != 5:
            raise ValueError(
                f"fixed refinement world must be [B,C,T,H,W], got {tuple(fixed_world.shape)}"
            )
        if fixed_world.shape[0] != batch_size:
            raise ValueError(
                f"fixed refinement world batch={fixed_world.shape[0]} != action batch={batch_size}"
            )
        if fixed_world.shape[2] != len(self.slot_temporal_ids):
            raise ValueError(
                f"fixed refinement world has {fixed_world.shape[2]} slots; "
                f"expected {len(self.slot_temporal_ids)}"
            )

        _ids, _sigma, timestep, weight, _noise, noisy_action, target = (
            self._sample_flow(action0)
        )
        clean_timestep = torch.zeros(batch_size, device=device, dtype=dtype)
        # Both the latent and every per-layer World K/V are outside autograd.
        # L_action can therefore update only the Action Expert.
        with torch.no_grad():
            frozen = fixed_world.detach().to(device=device, dtype=dtype)
            context_video = self.dit.embed_text(context)
            world_state = experts.video_prepare(
                self,
                frozen,
                clean_timestep,
                context_video,
            )
            video_cache = experts.prefill_video_cache(self, world_state)
        action_state = self.action_backbone.prepare(
            noisy_action.to(dtype),
            timestep.to(dtype),
            context,
            proprio=proprio_cur,
        )
        prediction = experts.run_action_with_video_cache(
            self.action_backbone,
            action_state,
            video_cache,
        ).float()
        return self._weighted_mse(prediction, target, weight, mask_b=action_mask)

    def _compose_loss(self, world_losses, scalar_recon, l_action, iteration):
        cfg = self.config
        warm = self._auxiliary_warmup_fraction(iteration)
        overlay_warm = (
            self._overlay_warmup_fraction(iteration)
            if hasattr(self, "_overlay_warmup_fraction")
            else 0.0
        )
        world_losses["_overlay_warm"] = float(overlay_warm)
        lam_action = self._action_lambda(iteration)
        # Do not retain zero-weight branches in the autograd graph. A finite
        # scalar loss can still have Inf gradients, and downstream ``0 * Inf``
        # is NaN under low-precision training.
        loss = world_losses["L_rgb"]
        if warm > 0 and float(cfg.lambda_scalar_afd_final) > 0:
            loss = (
                loss
                + warm
                * float(cfg.lambda_scalar_afd_final)
                * world_losses["L_scalar_afd"]
            )
        if float(cfg.lambda_scalar_reconstruction) > 0:
            loss = loss + float(cfg.lambda_scalar_reconstruction) * scalar_recon
        overlay_weight = float(getattr(cfg, "lambda_overlay_final", 0.0))
        if overlay_warm > 0 and overlay_weight > 0:
            loss = loss + overlay_warm * overlay_weight * world_losses["L_overlay"]
        if lam_action > 0:
            loss = loss + lam_action * l_action
        return loss, warm, lam_action

    def _format_action_loss(self, loss, world_losses, scalar_recon, l_action, warm):
        device = loss.device
        zero = loss.detach().new_zeros((), dtype=torch.float32)
        l_overlay = world_losses.get("L_overlay", zero)
        overlay_warm = float(world_losses.get("_overlay_warm", 0.0))
        return {
            "loss": loss,
            "L_obs": world_losses["L_rgb"].detach(),
            "L_aff": world_losses["L_scalar_afd"].detach(),
            "L_rgb": world_losses["L_rgb"].detach(),
            "L_scalar_afd": world_losses["L_scalar_afd"].detach(),
            "L_overlay": l_overlay.detach(),
            "L_scalar_recon": scalar_recon.detach(),
            "L_action": l_action.detach(),
            "lambda_aff": torch.as_tensor(
                float(warm * self.config.lambda_scalar_afd_final), device=device
            ),
            "lambda_overlay": torch.as_tensor(
                float(overlay_warm * getattr(self.config, "lambda_overlay_final", 0.0)),
                device=device,
            ),
        }

    def _compute_loss_dual_self_attn(self, batch, iteration):
        """True two-expert joint attention with one-way world->action visibility."""

        from affordance_wam.wan import expert_blocks as experts
        from affordance_wam.wan.dual_self_attn import run_dual_self_attn_loop

        device, dtype = self.device, self.dtype
        (
            x0,
            action0,
            context,
            has_aff,
            has_action,
            proprio_cur,
            scalar_recon,
        ) = self._encode_slot_and_action(batch)
        b = x0.shape[0]
        v_sigma, v_timestep, v_weight, xt, v_target = self._sample_world_flow(x0)
        lambda_action = self._action_lambda(iteration)

        # Action-free Human V+L+Afd: do not instantiate fake action tokens and
        # do not execute/update the Action Expert. The exact zero-weight action
        # warm-up endpoint takes the same path: a masked joint kernel can still
        # materialize zero Action gradients, so bypass the branch completely.
        if (
            action0 is None
            or not bool((has_action > 0).any())
            or float(lambda_action) <= 0
        ):
            v_pred = self._forward_world(xt, v_timestep, context)
            world_losses = self._world_losses(
                v_pred,
                v_target,
                x0,
                xt,
                v_sigma,
                v_weight,
                has_aff,
            )
            zero = v_pred.detach().new_zeros((), dtype=torch.float32)
            loss, warm, _ = self._compose_loss(
                world_losses, scalar_recon, zero, iteration
            )
            result = self._format_action_loss(
                loss, world_losses, scalar_recon, zero, warm
            )
            result["lambda_action"] = torch.as_tensor(
                float(lambda_action), device=device
            )
            result["pretrain_world_only"] = torch.zeros((), device=device)
            result["self_attn_refinement"] = torch.zeros((), device=device)
            return result

        action_mask = self._action_loss_mask(batch, has_action)
        valid_rows = (
            (action_mask > 0).any(dim=1) if action_mask.ndim == 2 else action_mask > 0
        )
        if not bool(valid_rows.all()):
            raise ValueError(
                "Robot pretrain batch has a sample with no valid action target"
            )

        if self._self_attn_refinement_active(iteration):
            # The noisy World branch remains a complete L_W training target.
            # Only the Action branch switches to a fixed final-World cache.
            v_pred = self._forward_world(xt, v_timestep, context)
            world_losses = self._world_losses(
                v_pred,
                v_target,
                x0,
                xt,
                v_sigma,
                v_weight,
                has_aff,
            )
            pred_probability = self._predicted_condition_probability(iteration)
            use_predicted = self._predicted_condition_active(
                iteration, pred_probability
            )
            if use_predicted:
                from affordance_wam.wan.wan_sampler import sample_world_latents

                fixed_world, _history, _pixels = sample_world_latents(
                    self,
                    batch,
                    guidance=float(self.config.predicted_world_guidance),
                    seed=(int(self.config.training_seed) + int(iteration) + 104729)
                    & ((1 << 63) - 1),
                    num_steps=int(self.config.predicted_world_denoise_steps),
                    shift=float(self.config.flow_shift),
                )
            else:
                fixed_world = x0.detach()
            l_action = self._action_loss_against_frozen_world(
                fixed_world=fixed_world,
                action0=action0,
                context=context,
                proprio_cur=proprio_cur,
                action_mask=action_mask,
            )
            loss, warm, _ = self._compose_loss(
                world_losses, scalar_recon, l_action, iteration
            )
            result = self._format_action_loss(
                loss, world_losses, scalar_recon, l_action, warm
            )
            result["lambda_action"] = torch.as_tensor(
                float(lambda_action), device=device
            )
            result["pretrain_world_only"] = torch.zeros((), device=device)
            result["self_attn_refinement"] = torch.ones((), device=device)
            result["self_refinement_pred_condition_prob"] = torch.as_tensor(
                pred_probability, device=device
            )
            result["self_refinement_pred_condition_fraction"] = torch.as_tensor(
                float(use_predicted), device=device
            )
            result["action_training_active"] = torch.ones((), device=device)
            result["predicted_condition_training_active"] = torch.as_tensor(
                float(use_predicted), device=device
            )
            return result

        # Half of robot steps lie exactly on the lockstep deployment path;
        # remaining steps fill the off-diagonal noise plane for robustness.
        if torch.rand((), device=device).item() < self._curriculum(
            iteration,
            self.config.joint_sync_prob_start,
            self.config.joint_sync_prob_end,
        ):
            noise = torch.randn_like(action0)
            a_xt = action0 + v_sigma.view(b, 1, 1) * (noise - action0)
            a_target = noise - action0
            a_timestep = v_timestep
            a_weight = v_weight
        else:
            _i, _s, a_timestep, a_weight, _n, a_xt, a_target = self._sample_flow(
                action0
            )

        context_v = self.dit.embed_text(context)
        world_state = experts.video_prepare(
            self,
            xt.to(dtype),
            v_timestep.to(dtype),
            context_v,
        )
        action_state = self.action_backbone.prepare(
            a_xt.to(dtype), a_timestep.to(dtype), context, proprio=proprio_cur
        )
        world_hidden, action_hidden = run_dual_self_attn_loop(
            self.dit,
            self.action_backbone,
            world_state,
            action_state,
            use_gradient_checkpointing=bool(
                self.config.use_gradient_checkpointing and self.training
            ),
        )
        v_pred = experts.video_finalize(self, world_hidden, world_state)
        a_pred = self.action_backbone.decode(action_hidden).float()
        world_losses = self._world_losses(
            v_pred,
            v_target,
            x0,
            xt,
            v_sigma,
            v_weight,
            has_aff,
        )
        l_action = self._weighted_mse(a_pred, a_target, a_weight, mask_b=action_mask)
        loss, warm, _ = self._compose_loss(
            world_losses, scalar_recon, l_action, iteration
        )
        result = self._format_action_loss(
            loss, world_losses, scalar_recon, l_action, warm
        )
        result["lambda_action"] = torch.as_tensor(float(lambda_action), device=device)
        result["pretrain_world_only"] = torch.zeros((), device=device)
        result["self_attn_refinement"] = torch.zeros((), device=device)
        result["action_training_active"] = torch.ones((), device=device)
        return result

    def forward(self, batch, iteration=None):
        return self.compute_loss(batch, iteration)


def build_wan_action_affordance_model(
    spec_or_key,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    config: Optional[WanActionAffordanceConfig] = None,
    load_text_encoder: bool = True,
) -> WanActionAffordanceModel:
    """Load vendored Wan components and assemble a :class:`WanActionAffordanceModel`."""
    spec = (
        spec_or_key if isinstance(spec_or_key, BaseModelSpec) else get_spec(spec_or_key)
    )
    if spec.wan_variant != "ti2v":
        raise ValueError(
            "AffordanceWAM uses "
            f"Wan2.2-TI2V; got {spec.key!r} (variant={spec.wan_variant!r})"
        )
    if config is None:
        config = WanActionAffordanceConfig(base_model=spec.key)
    if config.base_model != spec.key:
        raise ValueError(
            f"config.base_model={config.base_model!r} does not match spec={spec.key!r}"
        )

    from affordance_wam.wan.affordance_model import _filter_dit_vae_only
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
    return WanActionAffordanceModel(
        spec=spec,
        dit=holder.dit,
        vae=holder.vae,
        config=config,
        text_encoder=holder.text_encoder if load_text_encoder else None,
        tokenizer=holder.tokenizer if load_text_encoder else None,
    )
