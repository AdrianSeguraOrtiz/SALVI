"""Transport-independent models exposed by the local web application."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field

from salvi.domain.enums import ColumnKind, RunStatus
from salvi.domain.models import FrozenModel


class ImportStatus(StrEnum):
    UPLOADED = "uploaded"
    READY = "ready"
    COMPLETED = "completed"
    FAILED = "failed"


class AdapterParameterKind(StrEnum):
    STRING = "STRING"
    INTEGER = "INTEGER"
    NUMBER = "NUMBER"
    BOOLEAN = "BOOLEAN"


class AdapterParameterDescription(FrozenModel):
    name: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    kind: AdapterParameterKind
    required: bool = True
    default: str | int | float | bool | None = None
    minimum: float | None = None
    maximum: float | None = None


class WebColumnProposal(FrozenModel):
    source_index: int = Field(ge=0)
    name: str = Field(min_length=1)
    inferred_kind: ColumnKind
    selected_kind: ColumnKind
    missing_ratio: float = Field(ge=0.0, le=1.0)
    sample_values: tuple[str, ...] = ()
    is_row_identifier: bool = False
    role: str | None = None
    annotation_kind: str | None = None
    units: str | None = None
    description: str | None = None


class DatasetImportPreview(FrozenModel):
    identifier: str = Field(min_length=1)
    adapter: str = Field(min_length=1)
    row_count: int = Field(ge=1)
    column_count: int = Field(ge=1)
    columns: tuple[WebColumnProposal, ...]
    confirmation_required: bool
    ground_truth_attached: bool = False
    clinical_annotations_attached: bool = False
    warnings: tuple[str, ...] = ()
    adapter_configuration: dict[str, Any] = Field(default_factory=dict)


class AdapterFileSlot(FrozenModel):
    name: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    required: bool = True
    accepted_extensions: tuple[str, ...] = ()


class InputAdapterDescription(FrozenModel):
    name: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    files: tuple[AdapterFileSlot, ...]
    parameters: tuple[AdapterParameterDescription, ...] = ()
    supports_ground_truth: bool = False
    requires_confirmation: bool = False


class AnalysisDescription(FrozenModel):
    name: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    requires_ground_truth: bool = True


class WebDatasetRecord(FrozenModel):
    identifier: str = Field(min_length=1)
    adapter: str = Field(min_length=1)
    bundle_path: Path
    storage_path: Path | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ground_truth_attached: bool = False
    clinical_annotations_attached: bool = False


class WebImportRecord(FrozenModel):
    identifier: str = Field(min_length=1)
    adapter: str = Field(min_length=1)
    upload_directory: Path
    files: dict[str, Path]
    parameters: dict[str, str | int | float | bool] = Field(default_factory=dict)
    preview: DatasetImportPreview
    status: ImportStatus = ImportStatus.UPLOADED
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    error: str | None = None


class WebRunRecord(FrozenModel):
    identifier: str = Field(min_length=1)
    dataset_identifier: str = Field(min_length=1)
    pipeline_path: Path
    output_directory: Path
    seed: Annotated[int, Field(ge=0)] = 0
    analyses: tuple[str, ...] = ()
    status: RunStatus = RunStatus.CREATED
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None


class AccuracySummary(FrozenModel):
    relevance: float = Field(ge=0.0, le=1.0)
    recovery: float = Field(ge=0.0, le=1.0)
    biclustering_error: float = Field(ge=0.0, le=1.0)
    detected_count: int = Field(ge=0)
    ground_truth_count: int = Field(ge=0)
    coverage: tuple[tuple[float, float], ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "AccuracySummary",
    "AdapterFileSlot",
    "AdapterParameterDescription",
    "AdapterParameterKind",
    "AnalysisDescription",
    "DatasetImportPreview",
    "ImportStatus",
    "InputAdapterDescription",
    "WebColumnProposal",
    "WebDatasetRecord",
    "WebImportRecord",
    "WebRunRecord",
]
