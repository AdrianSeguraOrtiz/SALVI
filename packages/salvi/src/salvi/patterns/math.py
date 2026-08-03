"""Small numerical helpers shared by pattern and objective implementations."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

NUMERIC_TOLERANCE = 1e-12


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def finite_median(values: npt.NDArray[np.float64]) -> float | None:
    finite = values[np.isfinite(values)]
    return None if finite.size == 0 else float(np.median(finite))


def nanmedian_2d(
    values: npt.NDArray[np.float64],
    *,
    axis: int,
) -> npt.NDArray[np.float64]:
    """Compute a NaN-aware median without NumPy's masked-array slow path."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("nanmedian_2d requires a two-dimensional array")
    if axis not in (0, 1):
        raise ValueError("nanmedian_2d axis must be 0 or 1")
    observed = ~np.isnan(matrix)
    if np.all(observed):
        return np.asarray(np.median(matrix, axis=axis), dtype=np.float64)
    counts = np.count_nonzero(observed, axis=axis)
    ordered = np.sort(np.where(observed, matrix, np.inf), axis=axis)
    lower = np.maximum(counts - 1, 0) // 2
    upper = counts // 2
    if axis == 0:
        lower_values = np.take_along_axis(ordered, lower[np.newaxis, :], axis=0)[0]
        upper_values = np.take_along_axis(ordered, upper[np.newaxis, :], axis=0)[0]
    else:
        lower_values = np.take_along_axis(ordered, lower[:, np.newaxis], axis=1)[:, 0]
        upper_values = np.take_along_axis(ordered, upper[:, np.newaxis], axis=1)[:, 0]
    with np.errstate(invalid="ignore", over="ignore"):
        result = np.asarray((lower_values + upper_values) / 2.0, dtype=np.float64)
    result[counts == 0] = np.nan
    return result


def diagnostics(
    **values: float | int | str | bool | None,
) -> tuple[tuple[str, float | int | str | bool | None], ...]:
    return tuple(sorted(values.items()))


__all__ = [
    "NUMERIC_TOLERANCE",
    "clamp01",
    "diagnostics",
    "finite_median",
    "nanmedian_2d",
]
