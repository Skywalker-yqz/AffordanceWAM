"""Train AffordanceWAM with torchrun and DeepSpeed ZeRO-2."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from affordance_wam.data.dataset import AffordanceWAMDataset
from affordance_wam.runtime import (
    DeterministicDistributedBatchSampler,
    HomogeneousTaskRouter,
    TaskStream,
    optimizer_groups,
    _step_with_bf16_overflow_guard,
)
from affordance_wam.wan.action_model import (
    WanActionAffordanceConfig,
    build_wan_action_affordance_model,
)
from affordance_wam.wan.checkpoint import load_model_parameters, read_checkpoint


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config", help="stage configuration (defaults to configs/<stage>.json)"
    )
    p.add_argument("--stage", choices=["pretrain", "posttrain"])
    p.add_argument("--robot_affordance_manifest")
    p.add_argument("--human_affordance_manifest")
    p.add_argument("--vla_manifest")
    p.add_argument("--normalization_stats")
    p.add_argument("--output_dir")
    p.add_argument("--init_from")
    p.add_argument("--resume")
    p.add_argument("--batch_size", type=int)
    p.add_argument("--gradient_accumulation_steps", type=int)
    p.add_argument("--num_workers", type=int)
    p.add_argument("--max_steps", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument(
        "--dryrun",
        action="store_true",
        help="validate and print the resolved configuration",
    )
    args = p.parse_args(argv)
    config_path = (
        Path(args.config)
        if args.config
        else Path(__file__).resolve().parent.parent
        / "configs"
        / ("posttrain.json" if args.stage == "posttrain" else "pretrain.json")
    )
    document = json.loads(config_path.read_text())
    config = {
        key: value for key, value in document.items() if key not in ("data", "training")
    }
    config.update(document.get("data", {}))
    config.update(document.get("training", {}))
    for key, value in vars(args).items():
        if key not in ("config", "dryrun") and value is not None:
            config[key] = value
    stage = config.get("stage", "pretrain")
    if args.config and args.stage and document.get("stage") != args.stage:
        raise ValueError("--stage must match the selected configuration")
    defaults = {
        "stage": stage,
        "batch_size": 4,
        "gradient_accumulation_steps": 1,
        "num_workers": 4,
        "max_steps": 10000 if stage == "posttrain" else 100000,
        "lr": 1e-5 if stage == "posttrain" else 1e-4,
        "weight_decay": 0.01,
        "betas": [0.9, 0.95],
        "grad_clip": 1.0,
        "seed": 0,
        "save_steps": 1000,
        "log_steps": 10,
        "codec_calibration_updates": 1000,
        "output_dir": f"outputs/{stage}",
        "human_ratio": 0.5,
        "normalization_stats": None,
        "init_from": None,
        "resume": None,
    }
    config = {**defaults, **config}
    for key in (
        "batch_size",
        "gradient_accumulation_steps",
        "max_steps",
        "save_steps",
        "log_steps",
    ):
        if config[key] < 1:
            raise ValueError(f"{key} must be positive")
    if (
        config["num_workers"] < 0
        or config["lr"] <= 0
        or not 0 <= config["human_ratio"] <= 1
    ):
        raise ValueError("invalid worker count, learning rate, or human ratio")
    if config["init_from"] and config["resume"]:
        raise ValueError("select either init_from or resume")
    config["model"] = asdict(
        WanActionAffordanceConfig(
            **config.get("model", {"base_model": "wan2_2_ti2v_5b"})
        )
    )
    return args, config


def task_specs(config):
    if config["stage"] == "posttrain":
        return {"robot": (config.get("vla_manifest"), "vl_action", 1.0)}
    ratio = config["human_ratio"]
    return {
        name: value
        for name, value in {
            "human": (config.get("human_affordance_manifest"), "vl_affordance", ratio),
            "robot": (
                config.get("robot_affordance_manifest"),
                "vl_action_affordance",
                1 - ratio,
            ),
        }.items()
        if value[2] > 0
    }


def build_router(config, rank, world, start_micro):
    specs = task_specs(config)
    weights = {name: spec[2] for name, spec in specs.items()}
    selector = HomogeneousTaskRouter(
        {name: None for name in specs}, weights, config["seed"]
    )
    counts = Counter(selector.task_at(i) for i in range(start_micro))
    streams = {}
    for i, (name, (manifest, form, _)) in enumerate(specs.items()):
        if not manifest:
            raise ValueError(f"missing {name} manifest")
        dataset = AffordanceWAMDataset(
            manifest, data_form=form, normalization_stats=config["normalization_stats"]
        )
        sampler = DeterministicDistributedBatchSampler(
            len(dataset),
            config["batch_size"],
            rank,
            world,
            config["seed"] + i * 1009,
            counts[name],
        )
        workers = config["num_workers"]
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
            timeout=120 if workers else 0,
            **(
                {"multiprocessing_context": "spawn", "prefetch_factor": 2}
                if workers
                else {}
            ),
        )
        streams[name] = TaskStream(name, loader, iter(loader))
    return HomogeneousTaskRouter(streams, weights, config["seed"])


def main(argv=None):
    args, config = parse_args(argv)
    if args.dryrun:
        print(json.dumps(config, indent=2))
        return 0
    for name, (path, _, _) in task_specs(config).items():
        if not path or not Path(path).is_file():
            raise FileNotFoundError(f"set a valid {name} training manifest")
    import deepspeed
    import torch.distributed as dist

    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}")
    if not dist.is_initialized():
        dist.init_process_group("nccl", init_method="env://", device_id=device)
    deepspeed.init_distributed(dist_backend="nccl", dist_init_required=False)
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(config["seed"] + rank)
    random.seed(config["seed"] + rank)
    model_config = WanActionAffordanceConfig(**config["model"])
    model_config.training_seed = config["seed"]
    model = build_wan_action_affordance_model(
        model_config.base_model,
        device=str(device),
        dtype=torch.bfloat16,
        config=model_config,
    )
    if config["init_from"]:
        load_model_parameters(model, config["init_from"])
    model.set_training_stage(config["stage"])
    ds_config = {
        "train_micro_batch_size_per_gpu": config["batch_size"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "bf16": {"enabled": True},
        "gradient_clipping": config["grad_clip"],
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "ignore_unused_parameters": True,
            "reduce_bucket_size": 5000000,
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": config["lr"],
                "betas": config["betas"],
                "eps": 1e-8,
                "weight_decay": config["weight_decay"],
                "torch_adam": True,
            },
        },
        "steps_per_print": 10000000,
    }
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=optimizer_groups(model, config["weight_decay"]),
        config=ds_config,
        dist_init_required=False,
    )
    normalization_stats = (
        json.loads(Path(config["normalization_stats"]).read_text())
        if config["normalization_stats"]
        else None
    )
    output_dir = Path(config["output_dir"])
    start_step = 0
    if config["resume"]:
        checkpoint, metadata = read_checkpoint(config["resume"])
        if metadata["stage"] != config["stage"] or metadata["model_config"] != asdict(
            model_config
        ):
            raise ValueError("resume configuration differs from the checkpoint")
        loaded, state = engine.load_checkpoint(
            str(checkpoint.parent), tag=checkpoint.name, load_module_strict=True
        )
        if not loaded:
            raise RuntimeError(f"could not resume {checkpoint}")
        for key in (
            "batch_size",
            "gradient_accumulation_steps",
            "seed",
            "human_ratio",
            "robot_affordance_manifest",
            "human_affordance_manifest",
            "vla_manifest",
            "codec_calibration_updates",
        ):
            if metadata["training_config"].get(key) != config.get(key):
                raise ValueError(f"resume {key} differs from the checkpoint")
        if metadata.get("normalization_stats") != normalization_stats:
            raise ValueError(
                "resume normalization statistics differ from the checkpoint"
            )
        start_step = int(state["step"])
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    gas = config["gradient_accumulation_steps"]
    router = build_router(config, rank, world, start_step * gas)
    engine.train()
    consecutive_overflows = 0
    for step in range(start_step, config["max_steps"]):
        calibrate = (
            config["stage"] == "pretrain" and step < config["codec_calibration_updates"]
        )
        model.training_stage = "codec_calibration" if calibrate else config["stage"]
        stage_step = (
            max(0, step - config["codec_calibration_updates"])
            if config["stage"] == "pretrain"
            else step
        )
        for micro in range(gas):
            # Per-microbatch random streams reproduce noise and dropout on resume.
            micro_seed = config["seed"] + rank + (step * gas + micro) * 1000003
            torch.manual_seed(micro_seed)
            random.seed(micro_seed)
            route, batch = router.next(step * gas + micro)
            result = engine(batch, stage_step)
            loss = result["loss"]
            finite = torch.isfinite(loss).all().int()
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                engine.zero_grad()
                raise FloatingPointError(
                    f"nonfinite loss at step={step}, route={route}"
                )
            engine.backward(loss)
            overflow = _step_with_bf16_overflow_guard(
                engine, enabled=True, iteration=step, rank=rank
            )
            if micro == gas - 1:
                consecutive_overflows = consecutive_overflows + 1 if overflow else 0
                if consecutive_overflows > 8:
                    raise FloatingPointError("repeated nonfinite gradient updates")
        if rank == 0 and (
            step % config["log_steps"] == 0 or step + 1 == config["max_steps"]
        ):
            metrics = {
                key: float(value.detach())
                for key, value in result.items()
                if torch.is_tensor(value) and value.numel() == 1
            }
            print(json.dumps({"step": step + 1, "route": route, **metrics}), flush=True)
        if (step + 1) % config["save_steps"] == 0 or step + 1 == config["max_steps"]:
            metadata = {
                "schema_version": 1,
                "step": step + 1,
                "stage": config["stage"],
                "model_config": asdict(model_config),
                "training_config": config,
                "normalization_stats": normalization_stats,
            }
            tag = f"step_{step + 1:08d}"
            engine.save_checkpoint(
                str(output_dir / "checkpoints"), tag=tag, client_state=metadata
            )
            if rank == 0:
                (output_dir / "checkpoints" / tag / "metadata.json").write_text(
                    json.dumps(metadata, indent=2) + "\n"
                )
            dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
