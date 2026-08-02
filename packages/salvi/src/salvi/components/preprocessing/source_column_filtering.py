"""Source-column filtering components."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from salvi.components.protocols import ComponentKind
from salvi.domain.prepared import PreparedDataset
from salvi.exceptions import ComponentError


@dataclass(frozen=True, slots=True)
class DropAllMissingColumns:
    """Remove source columns with no originally observed value."""

    component_name: str = "drop_all_missing_columns"
    stage_kind: ComponentKind = ComponentKind.SOURCE_COLUMN_FILTER
    provides: frozenset[str] = frozenset({"columns-filtered"})
    requires: frozenset[str] = frozenset({"prepared-dataset"})

    def transform(self, dataset: PreparedDataset) -> PreparedDataset:
        retained = tuple(
            column.index
            for column in dataset.columns
            if np.any(dataset.source_observed_mask(column.index))
        )
        if not retained:
            raise ComponentError("column filtering would remove every prepared column")
        return (
            dataset if len(retained) == dataset.column_count else dataset.select_columns(retained)
        )


__all__ = ["DropAllMissingColumns"]
