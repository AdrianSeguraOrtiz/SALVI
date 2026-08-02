"""Missing-value policies applied after source filtering."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from salvi.domain.enums import ColumnKind
from salvi.domain.prepared import PreparedColumnMetadata, PreparedDataset
from salvi.exceptions import ComponentError


@dataclass(frozen=True, slots=True)
class PreserveMissingValues:
    """Keep canonical nulls untouched for support-aware scientific evaluation."""

    component_name: str = "preserve"
    provides: frozenset[str] = frozenset({"missing-values-handled"})
    requires: frozenset[str] = frozenset({"prepared-dataset"})

    def apply(self, dataset: PreparedDataset) -> PreparedDataset:
        return dataset


@dataclass(frozen=True, slots=True)
class RejectMissingValues:
    """Reject a dataset when any canonical value is missing."""

    component_name: str = "reject"
    provides: frozenset[str] = frozenset({"missing-values-handled"})
    requires: frozenset[str] = frozenset({"prepared-dataset"})

    def apply(self, dataset: PreparedDataset) -> PreparedDataset:
        if dataset.missing_count:
            raise ComponentError(
                f"missing-values policy 'reject' found {dataset.missing_count} missing values"
            )
        return dataset


def _mode(column: pa.ChunkedArray, metadata: PreparedColumnMetadata) -> bool | str:
    values = tuple(value for value in column.to_pylist() if value is not None)
    if not values:
        raise ComponentError(f"cannot impute all-missing column {metadata.name!r}")
    counts = Counter(values)
    if metadata.kind is ColumnKind.BOOLEAN:
        order: tuple[bool | str, ...] = (False, True)
    else:
        order = metadata.categories
    return max(order, key=lambda value: (counts[value], -order.index(value)))


@dataclass(frozen=True, slots=True)
class MedianModeImputation:
    """Fill nulls while retaining their original-observation mask."""

    component_name: str = "median_mode_imputation"
    provides: frozenset[str] = frozenset({"missing-values-handled", "imputed-data"})
    requires: frozenset[str] = frozenset({"prepared-dataset"})

    def apply(self, dataset: PreparedDataset) -> PreparedDataset:
        if dataset.missing_count == 0:
            return dataset
        table = dataset.raw_table
        for column_index, metadata in enumerate(dataset.columns):
            values = table.column(column_index)
            if values.null_count == 0:
                continue
            if metadata.kind is ColumnKind.NUMERIC:
                numeric = dataset.numeric_column(column_index)
                observed = numeric[dataset.support_mask(column_index)]
                if observed.size == 0:
                    raise ComponentError(f"cannot impute all-missing column {metadata.name!r}")
                replacement: float | bool | str = float(np.median(observed))
                values = pc.cast(values, pa.float64())
            else:
                replacement = _mode(values, metadata)
            filled = pc.fill_null(values, pa.scalar(replacement, type=values.type))
            table = table.set_column(column_index, metadata.name, filled)
        return dataset.with_replaced_values(table)


__all__ = ["MedianModeImputation", "PreserveMissingValues", "RejectMissingValues"]
