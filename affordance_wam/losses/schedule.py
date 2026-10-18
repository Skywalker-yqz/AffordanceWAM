"""Linear weighting schedules for the World and Action objectives."""

from __future__ import annotations


def compute_lambda_warmup(
    iteration: int,
    final_value: float,
    warmup_start: int,
    warmup_end: int,
) -> float:
    """Linear warm-up of an auxiliary-loss coefficient.

    Args:
        iteration: training step counter (0-indexed).
        final_value: post-warm-up coefficient. Returned verbatim once
            ``iteration >= warmup_end``.
        warmup_start: first iteration at which the coefficient may be
            non-zero. At exactly ``iteration == warmup_start`` the function
            still returns 0 (linear-segment left endpoint).
        warmup_end: first iteration at which the coefficient saturates to
            ``final_value``.

    Returns:
        The scalar objective coefficient.
    """
    if warmup_end <= warmup_start:
        raise ValueError(
            f"warmup_end ({warmup_end}) must be > warmup_start ({warmup_start})"
        )
    if iteration < warmup_start:
        return 0.0
    if iteration >= warmup_end:
        return float(final_value)
    progress = (iteration - warmup_start) / (warmup_end - warmup_start)
    return float(final_value) * progress


__all__ = ["compute_lambda_warmup"]
