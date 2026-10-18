from itertools import islice
import random
from affordance_wam.runtime import (
    DeterministicDistributedBatchSampler,
    HomogeneousTaskRouter,
)
from scripts.train import parse_args


def test_distributed_sampler_is_disjoint_and_resumes_same_permutation():
    args = dict(dataset_size=43, batch_size=3, world=4, seed=9)
    streams = [
        iter(
            DeterministicDistributedBatchSampler(rank=rank, start_occurrence=0, **args)
        )
        for rank in range(4)
    ]
    all_batches = [[next(stream) for stream in streams] for _ in range(10)]
    for rank_batches in all_batches:
        combined = [item for part in rank_batches for item in part]
        assert len(set(combined)) == 12
    resumed = iter(
        DeterministicDistributedBatchSampler(rank=2, start_occurrence=6, **args)
    )
    assert list(islice(resumed, 4)) == [step[2] for step in all_batches[6:]]


def test_modality_routing_is_rank_independent():
    args = ({"human": None, "robot": None}, {"human": 0.5, "robot": 0.5}, 7)
    first, second = HomogeneousTaskRouter(*args), HomogeneousTaskRouter(*args)
    random.seed(38)
    for step in range(100):
        random.random()
        assert first.task_at(step) == second.task_at(step)


def test_stage_selection_resolves_published_posttraining_parameters():
    _, config = parse_args(["--stage", "posttrain", "--dryrun"])
    assert config["stage"] == "posttrain"
    assert config["lr"] == 1e-5
    assert config["max_steps"] == 10000
    assert config["model"]["posttrain_noisy_world_fraction"] == 0.75
    assert config["model"]["inference_world_denoise_steps"] == 35
    assert config["model"]["inference_action_denoise_steps"] == 35
