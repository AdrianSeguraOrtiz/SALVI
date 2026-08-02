"""Canonical ground-truth contract reserved for experiment adapters."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import Field, model_validator

from salvi.domain.enums import PatternKind
from salvi.domain.models import FrozenModel


class GroundTruthColumnPattern(FrozenModel):
    column_index: int = Field(ge=0)
    pattern: PatternKind


class GroundTruthBicluster(FrozenModel):
    identifier: str = Field(min_length=1)
    row_indices: tuple[int, ...]
    column_indices: tuple[int, ...]
    column_patterns: tuple[GroundTruthColumnPattern, ...]
    source_type: str | None = None
    source_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_structure(self) -> Self:
        for values, label in (
            (self.row_indices, "row_indices"),
            (self.column_indices, "column_indices"),
        ):
            if not values:
                raise ValueError(f"{label} must not be empty")
            if tuple(sorted(set(values))) != values or values[0] < 0:
                raise ValueError(f"{label} must be sorted, unique, and non-negative")
        pattern_indices = tuple(item.column_index for item in self.column_patterns)
        if pattern_indices != self.column_indices:
            raise ValueError("column_patterns must cover selected columns in order")
        return self


class GroundTruth(FrozenModel):
    schema_version: Literal[1] = 1
    dataset_identifier: str = Field(min_length=1)
    row_count: int = Field(ge=1)
    column_count: int = Field(ge=1)
    biclusters: tuple[GroundTruthBicluster, ...]
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_dataset_coordinates(self) -> Self:
        identifiers = tuple(bicluster.identifier for bicluster in self.biclusters)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("ground-truth bicluster identifiers must be unique")
        for bicluster in self.biclusters:
            if bicluster.row_indices[-1] >= self.row_count:
                raise ValueError("ground-truth row index is outside dataset dimensions")
            if bicluster.column_indices[-1] >= self.column_count:
                raise ValueError("ground-truth column index is outside dataset dimensions")
        return self


__all__ = ["GroundTruth", "GroundTruthBicluster", "GroundTruthColumnPattern"]
