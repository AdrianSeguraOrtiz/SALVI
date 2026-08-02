"""Numeric-transformation components."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from salvi.components.protocols import ComponentKind
from salvi.domain.prepared import NumericColumnStatistics, PreparedDataset
from salvi.exceptions import ComponentError

ZERO_SCALE_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class RobustNumericScaling:
    """Add median/P95-P05 statistics and standardized numeric values."""

    component_name: str = "robust_numeric_scaling"
    stage_kind: ComponentKind = ComponentKind.NUMERIC_TRANSFORMATION
    provides: frozenset[str] = frozenset({"robust-numeric-data"})
    requires: frozenset[str] = frozenset({"prepared-dataset", "missing-values-handled"})

    def transform(self, dataset: PreparedDataset) -> PreparedDataset:
        if dataset.has_robust_scaling:
            raise ComponentError("robust numeric scaling cannot be applied more than once")

        standardized = np.full_like(dataset.numeric_values, np.nan)
        statistics: list[NumericColumnStatistics] = []
        for position, column_index in enumerate(dataset.numeric_column_indices):
            values = dataset.numeric_values[:, position]
            support = dataset.support_mask(column_index)
            observed_values = values[support]
            if observed_values.size == 0:
                statistics.append(
                    NumericColumnStatistics(
                        column_index=column_index,
                        observed_count=0,
                        median=None,
                        percentile_05=None,
                        percentile_95=None,
                        robust_range=0.0,
                        zero_scale=True,
                    )
                )
                continue

            percentile_05, median, percentile_95 = np.quantile(
                observed_values,
                (0.05, 0.5, 0.95),
                method="linear",
            )
            robust_range = float(percentile_95 - percentile_05)
            zero_scale = robust_range <= ZERO_SCALE_TOLERANCE
            available = dataset.available_mask(column_index)
            if zero_scale:
                deviations = values[available] - float(median)
                standardized[available, position] = np.where(
                    np.abs(deviations) <= ZERO_SCALE_TOLERANCE,
                    0.0,
                    np.copysign(1.0, deviations),
                )
            else:
                standardized[available, position] = (
                    values[available] - float(median)
                ) / robust_range
            statistics.append(
                NumericColumnStatistics(
                    column_index=column_index,
                    observed_count=int(observed_values.size),
                    median=float(median),
                    percentile_05=float(percentile_05),
                    percentile_95=float(percentile_95),
                    robust_range=robust_range,
                    zero_scale=zero_scale,
                )
            )

        return dataset.with_robust_scaling(tuple(statistics), standardized)


__all__ = ["ZERO_SCALE_TOLERANCE", "RobustNumericScaling"]
