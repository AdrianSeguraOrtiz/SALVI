"""Convert explicit HBIC result documents into canonical BiclusterSets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from salvi.domain.enums import PatternKind
from salvi.exceptions import ConversionError
from salvi.infrastructure.dataset_bundle import DatasetBundleReader
from salvi_experiments.interop.external import (
    ExternalBiclusterRecord,
    ExternalBiclusterSetConverter,
    ExternalResultDocument,
)


class HbicConverterConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_bundle: Path = Path(".")
    overwrite: bool = False


class HbicBiclusterRecord(ExternalBiclusterRecord):
    """HBIC-specific public name for the shared external coordinate record."""


class HbicResultDocument(BaseModel):
    """Versioned disk representation of the list returned by ``Hbic.fit_predict``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    dataset_identifier: str = Field(min_length=1)
    biclusters: tuple[HbicBiclusterRecord, ...]

    @model_validator(mode="after")
    def validate_identifiers(self) -> Self:
        identifiers = tuple(item.identifier for item in self.biclusters)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("HBIC bicluster identifiers must be unique")
        return self


@dataclass(frozen=True, slots=True)
class HbicConverter:
    """Import HBIC indices without fabricating unavailable scientific scores."""

    dataset_bundle: Path = Path(".")
    overwrite: bool = False
    pattern: PatternKind = PatternKind.CONSTANT
    component_name: str = "hbic"
    provides: frozenset[str] = frozenset({"canonical-bicluster-set"})
    requires: frozenset[str] = frozenset({"canonical-dataset-bundle"})

    def convert(self, source: Path, destination: Path) -> Path:
        try:
            document = HbicResultDocument.model_validate_json(source.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ConversionError(f"invalid HBIC result document {source}: {error}") from error
        return self._write(document, destination)

    def convert_result(
        self,
        biclusters: Sequence[tuple[Sequence[Any], Sequence[Any]]],
        destination: Path,
    ) -> Path:
        dataset = DatasetBundleReader().inspect(self.dataset_bundle)
        records = tuple(
            HbicBiclusterRecord(
                identifier=f"hbic-{index:06d}",
                row_indices=self._indices(rows, dataset.row_count, "rows"),
                column_indices=self._indices(columns, dataset.column_count, "columns"),
            )
            for index, (rows, columns) in enumerate(biclusters)
        )
        return self._write(
            HbicResultDocument(
                dataset_identifier=dataset.identifier,
                biclusters=records,
            ),
            destination,
        )

    @staticmethod
    def _indices(values: Sequence[Any], bound: int, label: str) -> tuple[int, ...]:
        raw = tuple(values)
        if len(raw) == bound and all(isinstance(value, bool | np.bool_) for value in raw):
            indices = tuple(index for index, selected in enumerate(raw) if bool(selected))
        else:
            try:
                indices = tuple(sorted({int(value) for value in raw}))
            except (TypeError, ValueError) as error:
                raise ConversionError(f"HBIC {label} are neither a mask nor indices") from error
        if not indices or indices[-1] >= bound or indices[0] < 0:
            raise ConversionError(f"HBIC {label} are empty or outside dataset bounds")
        return indices

    def _write(self, document: HbicResultDocument, destination: Path) -> Path:
        return ExternalBiclusterSetConverter(
            dataset_bundle=self.dataset_bundle,
            overwrite=self.overwrite,
        ).write(
            ExternalResultDocument(
                dataset_identifier=document.dataset_identifier,
                algorithm="HBIC",
                pattern=self.pattern,
                biclusters=tuple(
                    ExternalBiclusterRecord.model_validate(record.model_dump(mode="python"))
                    for record in document.biclusters
                ),
            ),
            destination,
        )


__all__ = [
    "HbicBiclusterRecord",
    "HbicConverter",
    "HbicConverterConfiguration",
    "HbicResultDocument",
]
