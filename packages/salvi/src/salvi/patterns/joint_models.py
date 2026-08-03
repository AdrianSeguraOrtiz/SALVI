"""Reusable row-effect calculations for joint numeric pattern models."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from salvi.application.context import RunContext
from salvi.patterns.math import NUMERIC_TOLERANCE, nanmedian_2d


def additive_row_effects(
    context: RunContext,
    matrix: npt.NDArray[np.float64],
    source: npt.NDArray[np.bool_],
    beta: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    usable = np.isfinite(matrix)
    original = np.count_nonzero(source & usable, axis=1)
    required = context.evaluation_support_policy.required_observations(
        matrix.shape[1]
    )
    sufficient = original >= required
    residuals = np.where(usable, matrix - beta[np.newaxis, :], np.nan)
    alpha = nanmedian_2d(residuals, axis=1)
    alpha[~sufficient] = np.nan
    return alpha


def robust_column_scales(
    context: RunContext,
    columns: Sequence[int],
) -> npt.NDArray[np.float64]:
    """Return positive per-column scales used only to normalize residuals."""

    scales = np.empty(len(columns), dtype=np.float64)
    for position, column_index in enumerate(columns):
        statistics = context.dataset.numeric_statistics[
            context.dataset.numeric_positions[column_index]
        ]
        if statistics.robust_range > NUMERIC_TOLERANCE:
            scales[position] = statistics.robust_range
        else:
            median = statistics.median or 0.0
            scales[position] = max(abs(median), 1.0)
    return scales


def normalized_absolute_residuals(
    residuals: npt.NDArray[np.float64],
    scales: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Map numeric residuals to the common pattern-error scale without changing a model."""

    if residuals.ndim != 2:
        raise ValueError("numeric pattern residuals must be a matrix")
    if scales.shape != (residuals.shape[1],):
        raise ValueError("numeric pattern residual scales must align with columns")
    return np.minimum(1.0, np.abs(residuals) / scales[np.newaxis, :])


def multiplicative_row_effects(
    context: RunContext,
    matrix: npt.NDArray[np.float64],
    source: npt.NDArray[np.bool_],
    beta: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    usable_beta = np.abs(beta) > NUMERIC_TOLERANCE
    opportunities = int(np.count_nonzero(usable_beta))
    if opportunities < 2:
        return np.full(matrix.shape[0], np.nan, dtype=np.float64)
    usable = np.isfinite(matrix) & usable_beta[np.newaxis, :]
    original = np.count_nonzero(source & usable, axis=1)
    required = context.evaluation_support_policy.required_observations(opportunities)
    sufficient = original >= required
    ratios = np.full_like(matrix, np.nan)
    np.divide(
        matrix,
        beta[np.newaxis, :],
        out=ratios,
        where=usable,
    )
    alpha = nanmedian_2d(ratios, axis=1)
    alpha[~sufficient] = np.nan
    return alpha


__all__ = [
    "additive_row_effects",
    "multiplicative_row_effects",
    "normalized_absolute_residuals",
    "robust_column_scales",
]
