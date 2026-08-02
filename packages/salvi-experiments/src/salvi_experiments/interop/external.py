"""Shared conversion primitives for external biclustering tools."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from salvi.domain.enums import PatternKind
from salvi.domain.models import (
    Bicluster,
    Candidate,
    CandidateProvenance,
    Evaluation,
    NamedValue,
    Repertoire,
)
from salvi.domain.prepared import PreparedColumnMetadata
from salvi.exceptions import ConversionError
from salvi.infrastructure.bicluster_set import BiclusterSetWriter
from salvi.infrastructure.dataset_bundle import DatasetBundleReader


class ExternalBiclusterRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    identifier: str = Field(min_length=1)
    row_indices: Annotated[tuple[int, ...], Field(min_length=1)]
    column_indices: Annotated[tuple[int, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_indices(self) -> Self:
        for values, label in (
            (self.row_indices, "row_indices"),
            (self.column_indices, "column_indices"),
        ):
            if tuple(sorted(set(values))) != values or values[0] < 0:
                raise ValueError(f"{label} must be sorted, unique and non-negative")
        return self


class ExternalResultDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    dataset_identifier: str = Field(min_length=1)
    algorithm: str = Field(min_length=1)
    pattern: PatternKind
    biclusters: tuple[ExternalBiclusterRecord, ...]

    @model_validator(mode="after")
    def validate_identifiers(self) -> Self:
        identifiers = tuple(item.identifier for item in self.biclusters)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("external bicluster identifiers must be unique")
        return self


@dataclass(frozen=True, slots=True)
class ExternalBiclusterSetConverter:
    """Write externally detected coordinates without fabricating objective scores."""

    dataset_bundle: Path
    overwrite: bool = False

    def write(self, document: ExternalResultDocument, destination: Path) -> Path:
        loaded = DatasetBundleReader().load(self.dataset_bundle)
        if document.dataset_identifier != loaded.dataset.identifier:
            raise ConversionError(
                f"{document.algorithm} result dataset identifier does not match the DatasetBundle"
            )
        evaluations = tuple(
            self._evaluation(
                document.algorithm,
                document.pattern,
                sequence,
                record,
                row_count=loaded.dataset.row_count,
                column_count=loaded.dataset.column_count,
            )
            for sequence, record in enumerate(document.biclusters)
        )
        columns = tuple(
            PreparedColumnMetadata.from_source(column) for column in loaded.dataset.columns
        )
        BiclusterSetWriter().write(
            destination,
            identifier=f"{loaded.dataset.identifier}-{document.algorithm.lower()}",
            dataset_identifier=loaded.dataset.identifier,
            row_count=loaded.dataset.row_count,
            source_column_count=loaded.dataset.column_count,
            columns=columns,
            repertoire=Repertoire(evaluations=evaluations),
            source_run=f"external:{document.algorithm.lower()}:{document.pattern.value}",
            overwrite=self.overwrite,
        )
        return destination.resolve()

    @staticmethod
    def records(
        algorithm: str,
        biclusters: Iterable[tuple[Iterable[int], Iterable[int]]],
    ) -> tuple[ExternalBiclusterRecord, ...]:
        prefix = algorithm.lower().replace("_", "-")
        return tuple(
            ExternalBiclusterRecord(
                identifier=f"{prefix}-{index:06d}",
                row_indices=tuple(sorted(set(rows))),
                column_indices=tuple(sorted(set(columns))),
            )
            for index, (rows, columns) in enumerate(biclusters)
        )

    @staticmethod
    def _evaluation(
        algorithm: str,
        pattern: PatternKind,
        sequence: int,
        record: ExternalBiclusterRecord,
        *,
        row_count: int,
        column_count: int,
    ) -> Evaluation:
        if record.row_indices[-1] >= row_count or record.column_indices[-1] >= column_count:
            raise ConversionError(
                f"{algorithm} bicluster {record.identifier!r} exceeds dataset dimensions"
            )
        candidate = Candidate(
            identifier=record.identifier,
            bicluster=Bicluster(
                row_indices=record.row_indices,
                column_indices=record.column_indices,
            ),
            provenance=CandidateProvenance(
                producer=algorithm.lower(),
                operation="external_result",
                sequence=sequence,
                pattern_hint=pattern,
            ),
        )
        return Evaluation(
            candidate=candidate,
            objectives=(),
            descriptors=(
                NamedValue(name="row_cardinality", value=len(record.row_indices)),
                NamedValue(name="column_cardinality", value=len(record.column_indices)),
            ),
        )


__all__ = [
    "ExternalBiclusterRecord",
    "ExternalBiclusterSetConverter",
    "ExternalResultDocument",
]
