"""Reusable row-effect calculations for joint numeric pattern models."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from salvi.application.context import RunContext
from salvi.patterns.math import NUMERIC_TOLERANCE


def additive_row_effects(
    context: RunContext,
    matrix: npt.NDArray[np.float64],
    source: npt.NDArray[np.bool_],
    beta: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    usable = np.isfinite(matrix)
    original = np.count_nonzero(source & usable, axis=1)
    sufficient = np.fromiter(
        (
            count >= 2
            and context.evaluation_support_policy.is_sufficient(
                int(count),
                matrix.shape[1],
            )
            for count in original
        ),
        dtype=np.bool_,
        count=matrix.shape[0],
    )
    residuals = np.where(usable, matrix - beta[np.newaxis, :], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        alpha = np.nanmedian(residuals, axis=1)
    alpha[~sufficient] = np.nan
    return alpha


def multiplicative_column_scales(
    context: RunContext,
    columns: Sequence[int],
) -> npt.NDArray[np.float64]:
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
    sufficient = np.fromiter(
        (
            count >= 2
            and context.evaluation_support_policy.is_sufficient(
                int(count),
                opportunities,
            )
            for count in original
        ),
        dtype=np.bool_,
        count=matrix.shape[0],
    )
    ratios = np.full_like(matrix, np.nan)
    np.divide(
        matrix,
        beta[np.newaxis, :],
        out=ratios,
        where=usable,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        alpha = np.nanmedian(ratios, axis=1)
    alpha[~sufficient] = np.nan
    return alpha


__all__ = [
    "additive_row_effects",
    "multiplicative_column_scales",
    "multiplicative_row_effects",
]
