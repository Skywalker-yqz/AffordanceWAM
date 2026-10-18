from types import SimpleNamespace

import pytest

from affordance_wam.wan.wan_sampler import resolve_world_shift


def test_world_shift_defaults_to_checkpoint_config():
    model = SimpleNamespace(config=SimpleNamespace(flow_shift=7.25))

    assert resolve_world_shift(model, None) == 7.25


def test_world_shift_explicit_override_wins():
    model = SimpleNamespace(config=SimpleNamespace(flow_shift=7.25))

    assert resolve_world_shift(model, 3.5) == 3.5


def test_world_shift_rejects_non_positive_override():
    model = SimpleNamespace(config=SimpleNamespace(flow_shift=7.25))

    with pytest.raises(ValueError, match="shift must be positive"):
        resolve_world_shift(model, 0.0)
