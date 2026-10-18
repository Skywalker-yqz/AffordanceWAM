"""Wan2.2-TI2V-5B assets and AffordanceWAM input geometry."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

NUM_PIXEL_FRAMES: int = 17
PIXEL_HISTORY_LEN: int = 5
PIXEL_FUTURE_LEN: int = 12

NUM_COND_LATENTS: int = 2
NUM_RGB_FUTURE_LATENTS: int = 3
NUM_SCALAR_AFD_FUTURE_LATENTS: int = 3
NUM_OVERLAY_FUTURE_LATENTS: int = 3
NUM_AFF_FUTURE_LATENTS: int = 3
RGB_SLOT_END: int = NUM_COND_LATENTS + NUM_RGB_FUTURE_LATENTS  # 5
SCALAR_AFD_SLOT_END: int = RGB_SLOT_END + NUM_SCALAR_AFD_FUTURE_LATENTS  # 8
WAN_SLOT_T: int = SCALAR_AFD_SLOT_END  # 8
OVERLAY_SLOT_END: int = SCALAR_AFD_SLOT_END + NUM_OVERLAY_FUTURE_LATENTS  # 11
SLOT_T: int = NUM_COND_LATENTS + NUM_RGB_FUTURE_LATENTS + NUM_AFF_FUTURE_LATENTS  # 8

DEFAULT_HEIGHT: int = 256
DEFAULT_WIDTH: int = 320

_MODELS_ROOT_DEFAULT = str(Path(__file__).resolve().parent.parent / "models")


def models_root() -> str:
    """Root dir holding the per-model weight folders (``$AFF_MODELS_ROOT`` override)."""
    return os.environ.get("AFF_MODELS_ROOT", _MODELS_ROOT_DEFAULT)


@dataclass(frozen=True)
class BaseModelSpec:
    """Backbone dimensions and local model asset directory."""

    key: str
    family: str
    stack: str
    model_subdir: str  # folder name under models_root()
    vae_z_dim: int  # latent channel count
    spatial_compression: int  # VAE spatial downsample factor
    temporal_compression: int  # VAE temporal downsample factor
    causal_temporal: bool
    text_encoder: str
    text_dim: int  # raw cross-attn context width
    wan_variant: Optional[str] = None  # "ti2v" | "vace"
    dit_dim: Optional[int] = None

    @property
    def model_dir(self) -> str:
        """Absolute weight dir. Per-model env override:
        ``$AFF_MODEL_DIR_<KEY-UPPER>`` (e.g. ``AFF_MODEL_DIR_WAN2_2_TI2V_5B``)."""
        env_key = f"AFF_MODEL_DIR_{self.key.upper()}"
        override = os.environ.get(env_key)
        if override:
            return override
        return str(Path(models_root()) / self.model_subdir)

    @property
    def latent_frames(self) -> int:
        n = (NUM_PIXEL_FRAMES - 1) // self.temporal_compression
        return n + (1 if self.causal_temporal else 0)

    @property
    def latent_h(self) -> int:
        return DEFAULT_HEIGHT // self.spatial_compression

    @property
    def latent_w(self) -> int:
        return DEFAULT_WIDTH // self.spatial_compression

    @property
    def is_wan(self) -> bool:
        return self.family == "wan"


BASE_MODELS: Dict[str, BaseModelSpec] = {
    "wan2_2_ti2v_5b": BaseModelSpec(
        key="wan2_2_ti2v_5b",
        family="wan",
        stack="wan",
        model_subdir="Wan2.2-TI2V-5B",
        vae_z_dim=48,
        spatial_compression=16,
        temporal_compression=4,
        causal_temporal=True,
        text_encoder="umt5-xxl",
        text_dim=4096,
        wan_variant="ti2v",
        dit_dim=3072,
    ),
}

CHOICES: List[str] = list(BASE_MODELS.keys())


class BaseModelError(ValueError):
    """Raised on an unknown ``--base_model`` or a failed asset/consistency check."""


def get_spec(base_model: str) -> BaseModelSpec:
    """Return the :class:`BaseModelSpec` for ``base_model`` (no path checks)."""
    if base_model not in BASE_MODELS:
        raise BaseModelError(
            f"unknown base_model {base_model!r}; choices are {CHOICES}"
        )
    return BASE_MODELS[base_model]


def validate(base_model: str, *, check_paths: bool = True) -> BaseModelSpec:
    """Resolve + validate ``base_model``.

    With ``check_paths`` (default) the model weight dir must exist; for Wan
    models we additionally require at least one ``*.safetensors`` weight file
    so a typo / wrong override surfaces immediately rather than deep inside the
    backbone loader. Raises :class:`BaseModelError` with an actionable message.
    """
    spec = get_spec(base_model)
    if not check_paths:
        return spec

    mdir = Path(spec.model_dir)
    if not mdir.is_dir():
        raise BaseModelError(
            f"base_model={base_model!r}: model_dir does not exist: {mdir}\n"
            f"  set $AFF_MODELS_ROOT or $AFF_MODEL_DIR_{base_model.upper()} to override."
        )
    if spec.is_wan:
        weights = list(mdir.glob("*.safetensors"))
        if not weights:
            raise BaseModelError(
                f"base_model={base_model!r}: no *.safetensors weights under {mdir}; "
                "this does not look like a Wan model dir."
            )
    return spec


def add_argument(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    """Register the shared ``--base_model`` flag on an argparse parser."""
    parser.add_argument(
        "--base_model",
        required=required,
        choices=CHOICES,
        help="Which base model to train/eval/infer the AffordanceWAM objective on.",
    )


def _env_pairs(spec: BaseModelSpec) -> Dict[str, str]:
    """Return ``KEY=VALUE`` pairs for shell launchers (``--print-env``)."""
    return {
        "AFF_BASE_MODEL": spec.key,
        "AFF_BASE_FAMILY": spec.family,
        "AFF_TRAIN_STACK": spec.stack,
        "AFF_MODEL_DIR": spec.model_dir,
        "AFF_VAE_Z_DIM": str(spec.vae_z_dim),
        "AFF_TEXT_ENCODER": spec.text_encoder,
        "AFF_TEXT_DIM": str(spec.text_dim),
        "AFF_WAN_VARIANT": spec.wan_variant or "",
        "AFF_VIDEO_H": str(DEFAULT_HEIGHT),
        "AFF_VIDEO_W": str(DEFAULT_WIDTH),
        "AFF_NUM_FRAMES": str(NUM_PIXEL_FRAMES),
        "AFF_LATENT_T": str(spec.latent_frames),
        "AFF_SLOT_T": str(
            OVERLAY_SLOT_END
            if spec.key == "wan2_2_ti2v_5b"
            else WAN_SLOT_T if spec.is_wan else SLOT_T
        ),
    }


def _main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="affordance_wam.base_model",
        description="Resolve / validate the AffordanceWAM base model selector.",
    )
    add_argument(p)
    p.add_argument("--field", help="print a single BaseModelSpec field and exit")
    p.add_argument(
        "--print-env",
        action="store_true",
        help="print shell KEY=VALUE asset pairs",
    )
    p.add_argument(
        "--no-check-paths",
        action="store_true",
        help="skip on-disk weight existence checks",
    )
    args = p.parse_args(argv)

    try:
        spec = validate(args.base_model, check_paths=not args.no_check_paths)
    except BaseModelError as e:
        print(f"ERROR: {e}", flush=True)
        return 2

    if args.field:
        if args.field == "model_dir":
            print(spec.model_dir)
        elif args.field == "latent_frames":
            print(spec.latent_frames)
        elif hasattr(spec, args.field):
            print(getattr(spec, args.field))
        else:
            print(f"ERROR: unknown field {args.field!r}", flush=True)
            return 2
        return 0

    if args.print_env:
        for k, v in _env_pairs(spec).items():
            print(f"{k}={v}")
        return 0

    for k, v in asdict(spec).items():
        print(f"{k}={v}")
    print(f"model_dir={spec.model_dir}")
    print(f"latent_frames={spec.latent_frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "BASE_MODELS",
    "CHOICES",
    "BaseModelError",
    "BaseModelSpec",
    "NUM_PIXEL_FRAMES",
    "PIXEL_HISTORY_LEN",
    "PIXEL_FUTURE_LEN",
    "NUM_COND_LATENTS",
    "NUM_RGB_FUTURE_LATENTS",
    "NUM_SCALAR_AFD_FUTURE_LATENTS",
    "NUM_AFF_FUTURE_LATENTS",
    "SLOT_T",
    "WAN_SLOT_T",
    "RGB_SLOT_END",
    "SCALAR_AFD_SLOT_END",
    "DEFAULT_HEIGHT",
    "DEFAULT_WIDTH",
    "add_argument",
    "get_spec",
    "models_root",
    "validate",
]
