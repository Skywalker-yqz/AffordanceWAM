#!/usr/bin/env python3
"""Check dependencies, video I/O, attention, and optional distributed training."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
from pathlib import Path
import sys
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--training", action="store_true", help="also import DeepSpeed")
    parser.add_argument(
        "--distributed", action="store_true",
        help="run one ZeRO-2 update; launch with torchrun and --device cuda",
    )
    args = parser.parse_args()

    import numpy as np
    import torch
    import torch.nn.functional as F

    for module, package in (
        ("torch", "torch"), ("torchvision", "torchvision"),
        ("transformers", "transformers"), ("sentencepiece", "sentencepiece"),
        ("safetensors", "safetensors"), ("einops", "einops"),
        ("omegaconf", "omegaconf"), ("pandas", "pandas"),
        ("av", "av"),
        ("imageio", "imageio"), ("imageio_ffmpeg", "imageio-ffmpeg"),
        ("cv2", "opencv-python-headless"),
    ):
        importlib.import_module(module)
        print(f"[ok] {package} {importlib.metadata.version(package)}", flush=True)

    from transformers import AutoTokenizer, T5TokenizerFast  # noqa: F401
    from affordance_wam.wan.action_model import WanActionAffordanceModel  # noqa: F401
    from affordance_wam.wan_vendor.models.dit import flash_attention

    print("[ok] Wan2.2 backbone, joint World-Action model, and text tokenizer imports")
    if args.training or args.distributed:
        import deepspeed
        print(f"[ok] DeepSpeed {deepspeed.__version__}")

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Check nvidia-smi and the installed PyTorch wheel.")
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(rank)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The training configuration requires a GPU with BF16 support.")
        print(f"[ok] {torch.cuda.get_device_name(rank)}; CUDA {torch.version.cuda}; BF16")

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    torch.manual_seed(0)
    q = torch.randn(2, 16, 64, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(2, 12, 64, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(2, 12, 64, device=device, dtype=dtype, requires_grad=True)
    mask = torch.ones(16, 12, device=device, dtype=torch.bool)
    mask[:, -3:] = False
    result = flash_attention(q, k, v, num_heads=4, compatibility_mode=True, attn_mask=mask)
    reference = F.scaled_dot_product_attention(
        q.float().reshape(2, 16, 4, 16).transpose(1, 2),
        k.float().reshape(2, 12, 4, 16).transpose(1, 2),
        v.float().reshape(2, 12, 4, 16).transpose(1, 2),
        attn_mask=mask,
    ).transpose(1, 2).reshape(2, 16, 64).to(dtype)
    torch.testing.assert_close(result, reference, atol=0.02, rtol=0.02)
    result.float().square().mean().backward()
    for value in (result, q.grad, k.grad, v.grad):
        if value is None or not torch.isfinite(value).all():
            raise RuntimeError("Attention produced non-finite values or gradients.")
    print(f"[ok] masked SDPA forward/backward ({args.device}, {dtype})")

    import av
    import imageio.v2 as imageio

    with tempfile.TemporaryDirectory(prefix="affordancewam-check-") as temp:
        video_path = Path(temp) / "frames.mp4"
        frames = np.zeros((3, 32, 32, 3), dtype=np.uint8)
        frames[1] = 128
        frames[2] = 255
        imageio.mimwrite(str(video_path), frames, fps=10, codec="libx264")
        with av.open(str(video_path)) as container:
            decoded = list(container.decode(video=0))
        if len(decoded) != len(frames):
            raise RuntimeError("Video decoding returned an unexpected frame count.")
    print("[ok] FFmpeg video encoding and PyAV decoding")

    if args.distributed:
        if args.device != "cuda" or "LOCAL_RANK" not in os.environ:
            raise RuntimeError("Use torchrun --standalone --nproc_per_node=1 env/check.py --device cuda --distributed")
        import deepspeed
        import torch.distributed as dist

        deepspeed.init_distributed(dist_backend="nccl")
        model = torch.nn.Linear(16, 8).to(device=device, dtype=dtype)
        before = model.weight.detach().float().clone()
        config = {
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 1,
            "bf16": {"enabled": True},
            "zero_optimization": {"stage": 2},
            "optimizer": {
                "type": "AdamW",
                "params": {"lr": 0.01, "torch_adam": True},
            },
            "gradient_clipping": 1.0,
            "steps_per_print": 1000000,
        }
        engine, _, _, _ = deepspeed.initialize(
            model=model, model_parameters=model.parameters(), config=config,
        )
        loss = engine(torch.randn(2, 16, device=device, dtype=dtype)).float().square().mean()
        engine.backward(loss)
        engine.step()
        after = engine.module.weight.detach().float()
        if not torch.isfinite(after).all() or torch.equal(before, after):
            raise RuntimeError("The distributed AdamW update did not produce finite, changed weights.")
        dist.barrier()
        dist.destroy_process_group()
        print("[ok] NCCL communication and DeepSpeed ZeRO-2 BF16 AdamW update")

    print("Environment checks passed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[error] {type(error).__name__}: {error}", file=sys.stderr)
        raise
