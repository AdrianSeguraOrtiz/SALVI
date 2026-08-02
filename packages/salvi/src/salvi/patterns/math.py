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


def diagnostics(
    **values: float | int | str | bool | None,
) -> tuple[tuple[str, float | int | str | bool | None], ...]:
    return tuple(sorted(values.items()))


__all__ = ["NUMERIC_TOLERANCE", "clamp01", "diagnostics", "finite_median"]
