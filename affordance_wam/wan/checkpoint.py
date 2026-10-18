"""Model and metadata loading for AffordanceWAM DeepSpeed checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
import torch
from affordance_wam.wan.action_model import (
    WanActionAffordanceConfig,
    build_wan_action_affordance_model,
)


def resolve_checkpoint(path):
    path = Path(path).resolve()
    if (path / "checkpoints").is_dir():
        path = path / "checkpoints"
    if (path / "latest").is_file():
        tag = (path / "latest").read_text().strip()
        if Path(tag).name != tag:
            raise ValueError("checkpoint latest must contain a tag directory name")
        path = path / tag
    if not (path / "metadata.json").is_file():
        raise FileNotFoundError(
            f"missing checkpoint metadata: {path / 'metadata.json'}"
        )
    return path


def read_checkpoint(path):
    path = resolve_checkpoint(path)
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata.get("schema_version") != 1:
        raise ValueError("unsupported AffordanceWAM checkpoint schema")
    WanActionAffordanceConfig(**metadata["model_config"])
    return path, metadata


def load_model_parameters(model, path):
    """Load replicated model parameters from a ZeRO-1/2 checkpoint."""
    path, metadata = read_checkpoint(path)
    file = path / "mp_rank_00_model_states.pt"
    state = torch.load(file, map_location="cpu", weights_only=True)
    model.load_state_dict(state["module"], strict=True)
    return metadata


def load_wan_action_checkpoint(path, *, device="cuda", dtype=torch.bfloat16):
    path, metadata = read_checkpoint(path)
    config = WanActionAffordanceConfig(**metadata["model_config"])
    model = build_wan_action_affordance_model(
        config.base_model, device=device, dtype=dtype, config=config
    )
    load_model_parameters(model, path)
    model.eval()
    return model, metadata
