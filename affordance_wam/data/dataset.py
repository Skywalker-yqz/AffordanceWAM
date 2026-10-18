"""Dataset interface for temporally aligned human and robot video windows.

Each JSONL record contains ``sample`` (an NPZ path relative to the manifest),
``instruction``, and an optional ``source``. NPZ arrays use the following layout:
``video``: uint8 [17,H,W,3] for training, [5,H,W,3] for deployment;
``affordance``: float32 [17,H,W] in [0,1] for Stage I;
``action``: float32 [12,7] and ``proprio_current``: float32 [7] for robot data.
Robot arrays are normalized using supplied statistics or are already normalized.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from affordance_wam.data.normalization import normalize

DATA_FORM_VL_ACTION_AFFORDANCE = "vl_action_affordance"
DATA_FORM_VL_AFFORDANCE = "vl_affordance"
DATA_FORM_VL_ACTION = "vl_action"
DATA_FORMS = (
    DATA_FORM_VL_ACTION_AFFORDANCE,
    DATA_FORM_VL_AFFORDANCE,
    DATA_FORM_VL_ACTION,
)


class AffordanceWAMDataset(Dataset):
    """Read aligned windows in a homogeneous supervision form."""

    def __init__(
        self,
        manifest_path,
        *,
        data_form="vl_action_affordance",
        normalization_stats=None,
        video_size=(256, 320),
        inference=False,
    ):
        if data_form not in DATA_FORMS:
            raise ValueError(f"unknown data_form {data_form!r}")
        self.path = Path(manifest_path).resolve()
        self.rows = [
            json.loads(line)
            for line in self.path.read_text().splitlines()
            if line.strip()
        ]
        if not self.rows:
            raise ValueError(f"empty manifest: {self.path}")
        self.data_form, self.video_size, self.inference = (
            data_form,
            tuple(video_size),
            bool(inference),
        )
        self.stats = (
            normalization_stats
            if isinstance(normalization_stats, dict)
            else (
                json.loads(Path(normalization_stats).read_text())
                if normalization_stats
                else None
            )
        )

    def __len__(self):
        return len(self.rows)

    def _normalize(self, value, name, source):
        if self.stats is None:
            return value
        stats = self.stats.get("sources", {}).get(source, self.stats)
        return normalize(value, stats[name], mode=stats.get("mode", "q99")).astype(
            np.float32
        )

    def __getitem__(self, index):
        row = self.rows[index]
        source = str(row.get("source", "default"))
        sample_path = self.path.parent / row["sample"]
        with np.load(sample_path, allow_pickle=False) as data:
            video = np.asarray(data["video"])
            if video.dtype != np.uint8 or video.ndim != 4 or video.shape[-1] != 3:
                raise ValueError(f"{sample_path}: video must be uint8 [T,H,W,3]")
            allowed_frames = (5, 17) if self.inference else (17,)
            if video.shape[0] not in allowed_frames:
                raise ValueError(
                    f"video requires {allowed_frames} frames; got {video.shape[0]}"
                )
            tensor = torch.from_numpy(video.copy()).permute(0, 3, 1, 2)
            if tuple(tensor.shape[-2:]) != self.video_size:
                tensor = (
                    F.interpolate(
                        tensor.float(),
                        self.video_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                    .round()
                    .clamp(0, 255)
                    .byte()
                )
            has_action = self.data_form != DATA_FORM_VL_AFFORDANCE
            has_affordance = (
                self.data_form != DATA_FORM_VL_ACTION and not self.inference
            )
            result = {
                "video": tensor.permute(1, 0, 2, 3).contiguous(),
                "ai_caption": str(row["instruction"]),
                "source": source,
                "fps": 10,
                "has_action": float(has_action),
                "has_affordance": float(has_affordance),
                "__key__": str(row.get("key", index)),
            }
            if has_affordance:
                scalar = torch.from_numpy(
                    np.asarray(data["affordance"], dtype=np.float32).copy()
                )
                if (
                    tuple(scalar.shape) != tuple(video.shape[:3])
                    or not torch.isfinite(scalar).all()
                    or scalar.min() < 0
                    or scalar.max() > 1
                ):
                    raise ValueError(
                        "affordance must be finite [T,H,W] in [0,1], aligned with video"
                    )
                scalar = F.interpolate(
                    scalar[:, None],
                    self.video_size,
                    mode="bilinear",
                    align_corners=False,
                )
                result["affordance"] = scalar.permute(1, 0, 2, 3).contiguous()
            if has_action:
                state = np.asarray(data["proprio_current"], dtype=np.float32)
                if state.shape != (7,) or not np.isfinite(state).all():
                    raise ValueError("proprio_current must be finite [7]")
                result["proprio_current"] = torch.from_numpy(
                    self._normalize(state, "proprio", source).copy()
                )
                if not self.inference:
                    action = np.asarray(data["action"], dtype=np.float32)
                    if action.shape != (12, 7) or not np.isfinite(action).all():
                        raise ValueError("action must be finite [12,7]")
                    result["action"] = torch.from_numpy(
                        self._normalize(action, "action", source).copy()
                    )
                    if "action_step_mask" in data:
                        mask = torch.from_numpy(
                            np.asarray(
                                data["action_step_mask"], dtype=np.float32
                            ).copy()
                        )
                        if mask.shape != (12,) or not ((mask == 0) | (mask == 1)).all():
                            raise ValueError("action_step_mask must be binary [12]")
                        result["action_step_mask"] = mask
        return result
