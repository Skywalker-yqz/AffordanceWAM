"""Generate a future World and a continuous action chunk from RGB history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from affordance_wam.data.dataset import AffordanceWAMDataset
from affordance_wam.data.normalization import denormalize
from affordance_wam.inference.sampler import save_side_by_side_mp4
from affordance_wam.wan.action_sampler import run_wan_action_inference
from affordance_wam.wan.checkpoint import load_wan_action_checkpoint


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--normalization_stats")
    parser.add_argument("--output_dir", default="outputs/inference")
    parser.add_argument("--world_steps", type=int, default=35)
    parser.add_argument("--action_steps", type=int, default=35)
    parser.add_argument("--refinement_steps", type=int, default=8)
    parser.add_argument("--refinement_strength", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    model, metadata = load_wan_action_checkpoint(
        args.checkpoint,
        device=args.device,
        dtype=torch.bfloat16 if str(args.device).startswith("cuda") else torch.float32,
    )
    stats = (
        json.loads(Path(args.normalization_stats).read_text())
        if args.normalization_stats
        else metadata.get("normalization_stats")
    )
    dataset = AffordanceWAMDataset(
        args.manifest, data_form="vl_action", normalization_stats=stats, inference=True
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, batch in enumerate(loader):
        if args.limit is not None and index >= args.limit:
            break
        outputs, normalized_actions = run_wan_action_inference(
            model,
            batch,
            seed=args.seed + index,
            num_steps=args.world_steps,
            action_num_steps=args.action_steps,
            action_refinement_steps=args.refinement_steps,
            action_refinement_strength=args.refinement_strength,
        )
        actions = normalized_actions
        if stats is not None:
            source_stats = stats.get("sources", {}).get(batch["source"][0], stats)
            actions = denormalize(
                actions, source_stats["action"], mode=source_stats.get("mode", "q99")
            )
        np.savez(
            output_dir / f"sample_{index:06d}.npz",
            action=actions[0],
            normalized_action=normalized_actions[0],
            scalar_affordance=outputs.scalar_affordance_future_pixel[0]
            .float()
            .cpu()
            .numpy(),
        )
        save_side_by_side_mp4(
            output_dir / f"sample_{index:06d}.mp4",
            outputs.rgb_pixel[0],
            outputs.aff_future_pixel[0],
            fps=10,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
