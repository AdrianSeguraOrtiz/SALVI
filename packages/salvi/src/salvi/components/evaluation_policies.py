"""Structural candidate validity and observed-support policies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from salvi.domain.models import Candidate
from salvi.domain.prepared import PreparedDataset
from salvi.domain.search import CandidateBounds
from salvi.exceptions import ComponentError


class MinimumCardinalityConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    min_rows: Annotated[int, Field(ge=1)] = 2
    min_columns: Annotated[int, Field(ge=1)] = 2


@dataclass(frozen=True, slots=True)
class MinimumCardinality:
    min_rows: int = 2
    min_columns: int = 2
    component_name: str = "minimum_cardinality"
    provides: frozenset[str] = frozenset({"candidate-validity"})
    requires: frozenset[str] = frozenset({"prepared-dataset"})

    def __post_init__(self) -> None:
        if self.min_rows < 1 or self.min_columns < 1:
            raise ValueError("minimum candidate cardinalities must be positive")

    def validate_dataset(self, dataset: PreparedDataset) -> None:
        if self.min_rows > dataset.row_count:
            raise ComponentError(
                f"minimum row cardinality {self.min_rows} exceeds the dataset row count "
                f"{dataset.row_count}"
            )
        if self.min_columns > dataset.column_count:
            raise ComponentError(
                f"minimum column cardinality {self.min_columns} exceeds the prepared column "
                f"count {dataset.column_count}"
            )

    def validate(self, candidate: Candidate, dataset: PreparedDataset) -> None:
        rows = candidate.bicluster.row_indices
        columns = candidate.bicluster.column_indices
        if len(rows) < self.min_rows:
            raise ComponentError(
                f"candidate {candidate.identifier!r} has {len(rows)} rows; "
                f"at least {self.min_rows} are required"
            )
        if len(columns) < self.min_columns:
            raise ComponentError(
                f"candidate {candidate.identifier!r} has {len(columns)} columns; "
                f"at least {self.min_columns} are required"
            )
        if rows[-1] >= dataset.row_count:
            raise ComponentError(
                f"candidate {candidate.identifier!r} references a row outside the dataset"
            )
        if columns[-1] >= dataset.column_count:
            raise ComponentError(
                f"candidate {candidate.identifier!r} references a column outside prepared data"
            )

    def bounds(self, dataset: PreparedDataset) -> CandidateBounds:
        self.validate_dataset(dataset)
        return CandidateBounds(
            min_rows=self.min_rows,
            max_rows=dataset.row_count,
            min_columns=self.min_columns,
            max_columns=dataset.column_count,
        )


class ObservedSupportConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    min_observed_count: Annotated[int, Field(ge=2)] = 2
    min_observed_ratio: Annotated[float, Field(ge=0.0, le=1.0)] = 0.8


@lru_cache(maxsize=4096)
def _required_observations(
    minimum_count: int,
    minimum_ratio: float,
    opportunity_count: int,
) -> int:
    return max(minimum_count, math.ceil(opportunity_count * minimum_ratio))


@dataclass(frozen=True, slots=True)
class MinimumObservedSupport:
    min_observed_count: int = 2
    min_observed_ratio: float = 0.8
    component_name: str = "minimum_observed_support"
    provides: frozenset[str] = frozenset({"evaluation-support"})
    requires: frozenset[str] = frozenset({"prepared-dataset"})

    def __post_init__(self) -> None:
        if self.min_observed_count < 2:
            raise ValueError(
                "min_observed_count must be at least 2 so pattern coherence is not "
                "estimated from a single value"
            )
        if not 0.0 <= self.min_observed_ratio <= 1.0:
            raise ValueError("min_observed_ratio must be in [0, 1]")

    def validate_dataset(self, dataset: PreparedDataset) -> None:
        if self.min_observed_count > dataset.row_count:
            raise ComponentError(
                f"minimum observed count {self.min_observed_count} exceeds the dataset row "
                f"count {dataset.row_count}"
            )

    def required_observations(self, opportunity_count: int) -> int:
        if opportunity_count < 0:
            raise ValueError("opportunity_count must be non-negative")
        return _required_observations(
            self.min_observed_count,
            self.min_observed_ratio,
            opportunity_count,
        )

    def is_sufficient(self, observed_count: int, opportunity_count: int) -> bool:
        if observed_count < 0:
            raise ValueError("observed_count must be non-negative")
        if observed_count > opportunity_count:
            raise ValueError("observed_count cannot exceed opportunity_count")
        return observed_count >= self.required_observations(opportunity_count)


__all__ = [
    "MinimumCardinality",
    "MinimumCardinalityConfiguration",
    "MinimumObservedSupport",
    "ObservedSupportConfiguration",
]
