"""Discover local Wan model weights and tokenizer assets."""
import glob as _glob
import logging
import os
import re
from collections import defaultdict
logger = logging.getLogger(__name__)

def discover_model_files(model_dir: str):
    """Auto-discover model files and tokenizer config from a model directory.

    Groups sharded safetensors by prefix, collects standalone safetensors
    and .pth files, and detects the tokenizer.

    Returns:
        (model_configs, tokenizer_config) — lists of ``ModelConfig`` objects.
    """
    from affordance_wam.wan_vendor.shared.core.loader import ModelConfig

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"model_path does not exist: {model_dir}")

    safetensors = sorted(_glob.glob(os.path.join(model_dir, "*.safetensors")))
    pth_files = sorted(_glob.glob(os.path.join(model_dir, "*.pth")))
    if not safetensors and not pth_files:
        raise FileNotFoundError(f"No *.safetensors or *.pth files found in {model_dir}")

    shard_groups = defaultdict(list)
    standalone = []
    for f in safetensors:
        basename = os.path.basename(f)
        m = re.match(r"^(.+)-\d{5}-of-\d{5}\.safetensors$", basename)
        if m:
            shard_groups[m.group(1)].append(f)
        else:
            standalone.append(f)

    model_paths = []
    for prefix in sorted(shard_groups):
        shards = sorted(shard_groups[prefix])
        model_paths.append(shards)
        logger.info("Grouped %d shards as one model: %s-*", len(shards), prefix)
    for f in standalone:
        model_paths.append(f)
    for f in pth_files:
        model_paths.append(f)
    logger.info("Auto-discovered %d model entries from %s", len(model_paths), model_dir)

    model_configs = [ModelConfig(p) for p in model_paths]

    tokenizer_dir = os.path.join(model_dir, "google", "umt5-xxl")
    if os.path.isdir(tokenizer_dir):
        tokenizer_config = ModelConfig(tokenizer_dir)
        logger.info("Auto-detected tokenizer at %s", tokenizer_dir)
    else:
        raise FileNotFoundError(f"Missing tokenizer directory: {tokenizer_dir}")

    return model_configs, tokenizer_config
