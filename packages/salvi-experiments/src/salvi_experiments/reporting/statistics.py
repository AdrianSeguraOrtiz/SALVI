"""Deterministic aggregate summaries for benchmark reports."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from salvi_experiments.configuration import UncertaintyConfiguration


def summarize_metric(
    values: Sequence[float],
    uncertainty: UncertaintyConfiguration,
    *,
    seed_offset: int = 0,
) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot summarize an empty metric")
    mean = float(array.mean())
    result: dict[str, float | int] = {
        "mean": mean,
        "standard_deviation": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "count": int(array.size),
    }
    if uncertainty.bootstrap_samples == 0:
        result["ci_lower"] = mean
        result["ci_upper"] = mean
        return result
    random = np.random.default_rng(uncertainty.seed + seed_offset)
    samples = random.choice(
        array,
        size=(uncertainty.bootstrap_samples, array.size),
        replace=True,
    ).mean(axis=1)
    alpha = (1.0 - uncertainty.confidence_level) / 2.0
    lower, upper = np.quantile(samples, (alpha, 1.0 - alpha))
    result["ci_lower"] = float(lower)
    result["ci_upper"] = float(upper)
    return result


__all__ = ["summarize_metric"]
