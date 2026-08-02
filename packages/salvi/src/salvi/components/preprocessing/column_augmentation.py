"""Column-augmentation components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from salvi.components.protocols import ComponentKind
from salvi.domain.prepared import PreparedDataset
from salvi.exceptions import ComponentError


class MissingnessIndicatorsConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    min_missing_ratio: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Minimum source-column missing fraction that creates an indicator.",
        ),
    ] = 0.1
    max_missing_ratio: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Maximum source-column missing fraction that creates an indicator.",
        ),
    ] = 1.0

    def model_post_init(self, __context: object) -> None:
        if self.min_missing_ratio > self.max_missing_ratio:
            raise ValueError("min_missing_ratio must not exceed max_missing_ratio")


@dataclass(frozen=True, slots=True)
class MissingnessIndicators:
    """Append explicit Boolean indicators for sufficiently incomplete columns."""

    min_missing_ratio: float = 0.1
    max_missing_ratio: float = 1.0
    component_name: str = "missingness_indicators"
    stage_kind: ComponentKind = ComponentKind.COLUMN_AUGMENTATION
    provides: frozenset[str] = frozenset({"columns-augmented"})
    requires: frozenset[str] = frozenset({"prepared-dataset", "missing-values-handled"})

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_missing_ratio <= 1.0:
            raise ValueError("min_missing_ratio must be in [0, 1]")
        if not 0.0 <= self.max_missing_ratio <= 1.0:
            raise ValueError("max_missing_ratio must be in [0, 1]")
        if self.min_missing_ratio > self.max_missing_ratio:
            raise ValueError("min_missing_ratio must not exceed max_missing_ratio")

    def transform(self, dataset: PreparedDataset) -> PreparedDataset:
        if any(column.derivation == self.component_name for column in dataset.columns):
            raise ComponentError("missingness indicators cannot be applied more than once")
        additions: list[tuple[str, np.ndarray, int, str]] = []
        for column in dataset.columns:
            if column.derivation is not None:
                continue
            observed = dataset.source_observed_mask(column.index)
            missing_count = dataset.row_count - int(np.count_nonzero(observed))
            missing_ratio = missing_count / dataset.row_count
            if missing_count and self.min_missing_ratio <= missing_ratio <= self.max_missing_ratio:
                additions.append(
                    (
                        f"{column.name}__is_missing",
                        np.asarray(~observed, dtype=np.bool_),
                        column.source_column_index,
                        self.component_name,
                    )
                )
        try:
            return dataset.with_appended_boolean_columns(tuple(additions))
        except ValueError as error:
            raise ComponentError(f"cannot add missingness indicators: {error}") from error


__all__ = ["MissingnessIndicators", "MissingnessIndicatorsConfiguration"]
