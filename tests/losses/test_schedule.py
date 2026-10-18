"""Unit tests for ``affordance_wam.losses.schedule.compute_lambda_warmup``."""

from __future__ import annotations

import pytest

from affordance_wam.losses import compute_lambda_warmup


FINAL = 0.2
START = 1000
END = 5000


def test_zero_before_start():
    assert compute_lambda_warmup(0, FINAL, START, END) == 0.0
    assert compute_lambda_warmup(500, FINAL, START, END) == 0.0
    assert compute_lambda_warmup(999, FINAL, START, END) == 0.0


def test_final_after_end():
    assert compute_lambda_warmup(END, FINAL, START, END) == pytest.approx(FINAL)
    assert compute_lambda_warmup(10_000, FINAL, START, END) == pytest.approx(FINAL)


def test_at_start_boundary_returns_zero():
    """At iter == warmup_start, the linear segment evaluates to 0.0."""
    assert compute_lambda_warmup(START, FINAL, START, END) == 0.0


def test_at_end_boundary_returns_final():
    """At iter == warmup_end, the schedule has fully saturated."""
    assert compute_lambda_warmup(END, FINAL, START, END) == pytest.approx(FINAL)


def test_midpoint():
    """Halfway through the warm-up window → half of final_value."""
    mid = (START + END) // 2
    assert compute_lambda_warmup(mid, FINAL, START, END) == pytest.approx(FINAL / 2)


def test_quarter_point():
    """Quarter-way through → 0.25 · final_value."""
    quarter = START + (END - START) // 4
    assert compute_lambda_warmup(quarter, FINAL, START, END) == pytest.approx(FINAL / 4)


def test_invalid_window_raises():
    with pytest.raises(ValueError):
        compute_lambda_warmup(0, FINAL, START, START)  # end == start
    with pytest.raises(ValueError):
        compute_lambda_warmup(0, FINAL, END, START)  # end < start


def test_returns_python_float():
    out = compute_lambda_warmup(2500, FINAL, START, END)
    assert type(out) is float
