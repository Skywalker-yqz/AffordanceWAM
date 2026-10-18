"""Distributed sampling, optimizer groups, and BF16 gradient checks."""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Mapping
import torch


class DeterministicDistributedBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Resume-exact, rank-disjoint affine permutation without a giant randperm.

    Epochs contain only complete global batches. The rotating affine
    permutation changes which rows form the small dropped tail, while no batch
    can straddle two independently permuted epochs and duplicate a row across
    ranks.
    """

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        rank: int,
        world: int,
        seed: int,
        start_occurrence: int,
    ):
        if dataset_size < 1:
            raise ValueError("dataset must not be empty")
        self.n = int(dataset_size)
        self.batch_size = int(batch_size)
        self.rank = int(rank)
        self.world = int(world)
        self.seed = int(seed)
        self.start_occurrence = int(start_occurrence)
        if self.batch_size < 1 or self.world < 1 or not 0 <= self.rank < self.world:
            raise ValueError("invalid batch_size/world/rank")
        self.global_batch = self.batch_size * self.world
        self.batches_per_epoch = self.n // self.global_batch
        if self.batches_per_epoch < 1:
            raise ValueError(
                f"dataset_size={self.n} is smaller than global_batch={self.global_batch}"
            )
        self._affine_cache: Dict[int, tuple[int, int]] = {}

    def _affine(self, epoch: int) -> tuple[int, int]:
        if epoch not in self._affine_cache:
            rng = random.Random(self.seed + epoch * 1_000_003)
            candidate = rng.randrange(1, max(2, self.n))
            while math.gcd(candidate, self.n) != 1:
                candidate = (candidate + 1) % self.n or 1
            self._affine_cache[epoch] = candidate, rng.randrange(self.n)
        return self._affine_cache[epoch]

    def __iter__(self) -> Iterator[list[int]]:
        occurrence = self.start_occurrence
        while True:
            epoch, batch_in_epoch = divmod(occurrence, self.batches_per_epoch)
            batch_position = batch_in_epoch * self.global_batch
            multiplier, offset = self._affine(epoch)
            indices = []
            for local_index in range(self.batch_size):
                position = batch_position + self.rank * self.batch_size + local_index
                indices.append((multiplier * position + offset) % self.n)
            yield indices
            occurrence += 1

    def __len__(self) -> int:
        return 2**31 - 1


@dataclass
class TaskStream:
    name: str
    loader: Any
    iterator: Iterator[Dict[str, Any]]

    def next(self) -> Dict[str, Any]:
        return next(self.iterator)


class HomogeneousTaskRouter:
    """Choose one globally identical modality form for every micro-iteration."""

    def __init__(
        self, streams: Mapping[str, TaskStream], weights: Mapping[str, float], seed: int
    ):
        self.streams = dict(streams)
        total = float(sum(weights.values()))
        self.weights = {name: float(weights[name]) / total for name in self.streams}
        self.seed = int(seed)
        self.names = sorted(self.streams)

    def task_at(self, step: int) -> str:
        draw = random.Random(self.seed ^ (int(step) * 0x9E3779B1)).random()
        cumulative = 0.0
        for name in self.names:
            cumulative += self.weights[name]
            if draw < cumulative:
                return name
        return self.names[-1]

    def next(self, step: int) -> tuple[str, Dict[str, Any]]:
        name = self.task_at(step)
        return name, self.streams[name].next()


def optimizer_groups(model: torch.nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lower = name.lower()
        if (
            parameter.ndim < 2
            or lower.endswith("bias")
            or "norm" in lower
            or "embedding" in lower
        ):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _check_bf16_grad_overflow(engine, *, enabled: bool) -> bool:
    """Run DeepSpeed ZeRO's cross-rank Inf/NaN check for BF16 gradients.

    Some DeepSpeed releases do not perform this check automatically for BF16.
    Setting the ZeRO optimizer's synchronized ``overflow`` flag before
    ``engine.step`` makes the existing safe-skip path clear gradients without
    touching FP32 master parameters.
    """

    if not enabled or not engine.is_gradient_accumulation_boundary():
        return False
    optimizer = engine.optimizer
    check = getattr(optimizer, "check_overflow", None)
    if not callable(check) or not hasattr(optimizer, "overflow"):
        raise RuntimeError(
            "bf16_grad_overflow_check=1 requires DeepSpeed ZeRO optimizer.check_overflow"
        )
    check()
    return bool(optimizer.overflow)


def _local_nonfinite_gradient_records(engine, limit: int = 12) -> list[dict[str, Any]]:
    """Map ZeRO-1/2 partitioned non-finite gradients back to module names.

    The scan runs only after the synchronized overflow flag is true, so it adds
    no kernels to clean updates. ``averaged_gradients`` and
    ``params_in_partition`` are the exact tensors inspected by DeepSpeed's
    Stage 1/2 overflow check.
    """

    optimizer = engine.optimizer
    gradients = getattr(optimizer, "averaged_gradients", None)
    partitions = getattr(optimizer, "params_in_partition", None)
    if gradients is None or partitions is None or not gradients:
        return [{"name": "<unavailable ZeRO partition metadata>"}]
    module_names = {
        id(parameter): name for name, parameter in engine.module.named_parameters()
    }
    zero_names = getattr(optimizer, "param_names", None)
    records: list[dict[str, Any]] = []
    for group_index, parameters in enumerate(partitions):
        group_gradients = (
            gradients.get(group_index)
            if isinstance(gradients, Mapping)
            else gradients[group_index]
        )
        if group_gradients is None:
            continue
        for parameter, gradient in zip(parameters, group_gradients):
            if gradient is None:
                continue
            finite = torch.isfinite(gradient)
            if bool(finite.all()):
                continue
            records.append(
                {
                    "name": (
                        zero_names.get(parameter)
                        if isinstance(zero_names, Mapping) and parameter in zero_names
                        else module_names.get(
                            id(parameter), f"<unnamed:{tuple(parameter.shape)}>"
                        )
                    ),
                    "shape": tuple(parameter.shape),
                    "fragment_numel": int(gradient.numel()),
                    "nan": int(torch.isnan(gradient).sum()),
                    "inf": int(torch.isinf(gradient).sum()),
                }
            )
            if len(records) >= limit:
                return records
    return records


def _report_nonfinite_gradients(engine, *, iteration: int, rank: int) -> None:
    records = _local_nonfinite_gradient_records(engine)
    if records:
        print(
            f"[numeric] iteration={iteration} rank={rank} non-finite ZeRO gradient fragments: "
            f"{records}",
            flush=True,
        )


def _step_with_bf16_overflow_guard(
    engine,
    *,
    enabled: bool,
    iteration: int,
    rank: int,
) -> bool:
    """Check/report partitioned BF16 gradients, then perform one ZeRO step.

    The ordering is intentional: ZeRO-1/2 clears ``averaged_gradients`` inside
    ``step`` when it safely skips an overflowed update.  Keeping the three
    operations together prevents diagnostics from silently moving after that
    destructive boundary.
    """

    overflow = _check_bf16_grad_overflow(engine, enabled=enabled)
    if overflow:
        _report_nonfinite_gradients(engine, iteration=iteration, rank=rank)
    engine.step()
    return overflow
