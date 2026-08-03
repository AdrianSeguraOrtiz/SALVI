"""Versioned BiclusterSet reader and writer with column-level explanations."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self, TypeVar

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, Field, field_validator, model_validator

from salvi.domain.enums import ColumnKind, ObjectiveDirection, ParameterScale, PatternKind
from salvi.domain.models import (
    Bicluster,
    Candidate,
    CandidateProvenance,
    ColumnObjectiveValue,
    ColumnPatternFit,
    ConstraintValue,
    Evaluation,
    EvaluationIssue,
    FinalSelectionProvenance,
    FrozenModel,
    NamedValue,
    ObjectiveValue,
    PatternCandidateFit,
    PatternFit,
    PatternGroupFit,
    Repertoire,
)
from salvi.domain.prepared import PreparedColumnMetadata
from salvi.exceptions import ArtifactError
from salvi.infrastructure.files import atomic_directory, sha256_file

Filename = TypeVar("Filename", bound=str)


class ColumnPatternRecord(FrozenModel):
    bicluster_id: str = Field(min_length=1)
    column_index: int = Field(ge=0)
    pattern: PatternKind | None
    group_identifier: str | None = None
    error: float = Field(ge=0.0, le=1.0)
    parameter: bool | float | str | None = None
    parameter_scale: ParameterScale = ParameterScale.NONE
    source_support: int = Field(ge=0)
    available_support: int = Field(ge=0)
    prototype_support: int = Field(ge=0)
    alternatives: tuple[PatternCandidateFit, ...] = ()
    diagnostics: tuple[tuple[str, float | int | str | bool | None], ...] = ()

    @field_validator("parameter")
    @classmethod
    def validate_parameter(cls, value: bool | float | str | None) -> bool | float | str | None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("numeric pattern parameters must be finite")
        return value


class ColumnObjectiveRecord(FrozenModel):
    bicluster_id: str = Field(min_length=1)
    objective_name: str = Field(min_length=1)
    direction: ObjectiveDirection
    column_index: int = Field(ge=0)
    value: float
    valid: bool = True
    diagnostics: tuple[tuple[str, float | int | str | bool | None], ...] = ()

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("column objective values must be finite")
        return value


class PatternGroupRecord(FrozenModel):
    bicluster_id: str = Field(min_length=1)
    group_identifier: str = Field(min_length=1)
    pattern: PatternKind
    column_indices: tuple[int, ...]
    iterations: int = Field(ge=0)
    converged: bool
    diagnostics: tuple[tuple[str, float | int | str | bool | None], ...] = ()


class PatternRowParameterRecord(FrozenModel):
    bicluster_id: str = Field(min_length=1)
    group_identifier: str = Field(min_length=1)
    row_index: int = Field(ge=0)
    value: float

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("pattern row parameters must be finite")
        return value


class FinalSelectionRecord(FrozenModel):
    bicluster_id: str = Field(min_length=1)
    selector: str = Field(min_length=1)
    group_identifier: str = Field(min_length=1)
    selection_rank: int = Field(ge=0)
    quality_score: float = Field(ge=0.0, le=1.0)
    novelty_score: float = Field(ge=0.0, le=1.0)
    marginal_gain: float = Field(ge=0.0, le=1.0)
    source_candidate_identifiers: tuple[str, ...]
    source_archive_coordinates: tuple[tuple[int, ...], ...] = ()


class BiclusterSetManifest(FrozenModel):
    schema_version: Literal[5, 6, 7] = 7
    identifier: str = Field(min_length=1)
    created_at: datetime
    dataset_identifier: str = Field(min_length=1)
    row_count: int = Field(ge=1)
    source_column_count: int = Field(ge=1)
    column_count: int = Field(ge=1)
    source_run: str | None = None
    source_checkpoint: str | None = None
    source_checkpoint_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    source_checkpoint_evaluations: int | None = Field(default=None, ge=0)
    columns_file: Literal["columns.parquet"] = "columns.parquet"
    biclusters_file: Literal["biclusters.parquet"] = "biclusters.parquet"
    column_patterns_file: Literal["column-patterns.parquet"] | None = None
    column_objectives_file: Literal["column-objectives.parquet"] | None = None
    pattern_groups_file: Literal["pattern-groups.parquet"] | None = None
    pattern_row_parameters_file: Literal["pattern-row-parameters.parquet"] | None = None
    final_selection_file: Literal["final-selection.parquet"] | None = None
    # Accepted but not emitted so schema-7 results produced during development remain readable.
    candidate_assessments_file: Literal["candidate-assessments.parquet"] | None = Field(
        default=None,
        exclude=True,
    )
    column_assessments_file: Literal["column-assessments.parquet"] | None = Field(
        default=None,
        exclude=True,
    )
    checksums: dict[str, str]

    @model_validator(mode="after")
    def validate_checksums(self) -> Self:
        expected: set[str] = {self.columns_file, self.biclusters_file}
        for filename in (
            self.column_patterns_file,
            self.column_objectives_file,
            self.pattern_groups_file,
            self.pattern_row_parameters_file,
            self.final_selection_file,
            self.candidate_assessments_file,
            self.column_assessments_file,
        ):
            if filename is not None:
                expected.add(filename)
        if set(self.checksums) != expected:
            raise ValueError("manifest checksums must cover every declared result file exactly")
        if any(
            len(checksum) != 64
            or any(character not in "0123456789abcdef" for character in checksum)
            for checksum in self.checksums.values()
        ):
            raise ValueError("manifest checksums must be lowercase SHA-256 values")
        checkpoint_fields = (
            self.source_checkpoint,
            self.source_checkpoint_sha256,
            self.source_checkpoint_evaluations,
        )
        if any(value is not None for value in checkpoint_fields) and not all(
            value is not None for value in checkpoint_fields
        ):
            raise ValueError("source checkpoint provenance must be complete or absent")
        return self


class BiclusterSetContents(FrozenModel):
    manifest: BiclusterSetManifest
    columns: tuple[PreparedColumnMetadata, ...]
    repertoire: Repertoire
    column_patterns: tuple[ColumnPatternRecord, ...] = ()
    column_objectives: tuple[ColumnObjectiveRecord, ...] = ()
    pattern_groups: tuple[PatternGroupRecord, ...] = ()
    row_parameters: tuple[PatternRowParameterRecord, ...] = ()
    final_selection: tuple[FinalSelectionRecord, ...] = ()


class BiclusterStructureRecord(FrozenModel):
    """Membership and pattern labels needed by structural analyses."""

    identifier: str = Field(min_length=1)
    bicluster: Bicluster
    column_patterns: tuple[tuple[int, PatternKind | None], ...] = ()

    @model_validator(mode="after")
    def validate_column_patterns(self) -> Self:
        indices = tuple(index for index, _pattern in self.column_patterns)
        if tuple(sorted(set(indices))) != indices:
            raise ValueError("structural column patterns must be sorted and unique")
        if not set(indices).issubset(self.bicluster.column_indices):
            raise ValueError("structural patterns must reference selected columns")
        return self


class BiclusterSetStructures(FrozenModel):
    """Lightweight, validated structural projection of a BiclusterSet."""

    manifest: BiclusterSetManifest
    columns: tuple[PreparedColumnMetadata, ...]
    biclusters: tuple[BiclusterStructureRecord, ...]


def _json(value: object) -> str:
    if isinstance(value, tuple):
        value = list(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _model_json(values: tuple[BaseModel, ...]) -> str:
    return _json([value.model_dump(mode="json") for value in values])


class BiclusterSetWriter:
    def write(
        self,
        destination: Path,
        *,
        identifier: str,
        dataset_identifier: str,
        row_count: int,
        source_column_count: int,
        columns: tuple[PreparedColumnMetadata, ...],
        repertoire: Repertoire,
        source_run: str | None = None,
        source_checkpoint: str | None = None,
        source_checkpoint_sha256: str | None = None,
        source_checkpoint_evaluations: int | None = None,
        overwrite: bool = False,
    ) -> BiclusterSetManifest:
        if row_count < 1 or source_column_count < 1 or not columns:
            raise ArtifactError("result dimensions must be positive")
        self._validate_columns(columns, source_column_count)
        column_count = len(columns)
        self._validate_repertoire(repertoire, row_count, column_count)
        (
            patterns,
            objectives,
            groups,
            row_parameters,
            final_selection,
        ) = self._derive_details(repertoire)

        with atomic_directory(destination, overwrite=overwrite) as temporary:
            self._write_columns(temporary / "columns.parquet", columns)
            self._write_biclusters(temporary / "biclusters.parquet", repertoire)
            column_patterns_file: Literal["column-patterns.parquet"] | None = self._write_optional(
                temporary,
                "column-patterns.parquet",
                [
                    {
                        **record.model_dump(
                            mode="json",
                            exclude={"alternatives", "diagnostics", "parameter"},
                        ),
                        "parameter_json": _json(record.parameter),
                        "alternatives_json": _model_json(record.alternatives),
                        "diagnostics_json": _json(record.diagnostics),
                    }
                    for record in patterns
                ],
            )
            column_objectives_file: Literal["column-objectives.parquet"] | None = (
                self._write_optional(
                    temporary,
                    "column-objectives.parquet",
                    [
                        {
                            **record.model_dump(mode="json", exclude={"diagnostics"}),
                            "diagnostics_json": _json(record.diagnostics),
                        }
                        for record in objectives
                    ],
                )
            )
            pattern_groups_file: Literal["pattern-groups.parquet"] | None = self._write_optional(
                temporary,
                "pattern-groups.parquet",
                [
                    {
                        **record.model_dump(mode="json", exclude={"diagnostics"}),
                        "diagnostics_json": _json(record.diagnostics),
                    }
                    for record in groups
                ],
            )
            pattern_row_parameters_file: Literal["pattern-row-parameters.parquet"] | None = (
                self._write_optional(
                    temporary,
                    "pattern-row-parameters.parquet",
                    [record.model_dump(mode="json") for record in row_parameters],
                )
            )
            final_selection_file: Literal["final-selection.parquet"] | None = self._write_optional(
                temporary,
                "final-selection.parquet",
                [record.model_dump(mode="json") for record in final_selection],
            )
            optional_files = (
                column_patterns_file,
                column_objectives_file,
                pattern_groups_file,
                pattern_row_parameters_file,
                final_selection_file,
            )
            filenames = [
                "columns.parquet",
                "biclusters.parquet",
                *(name for name in optional_files if name is not None),
            ]
            checksums = {name: sha256_file(temporary / name) for name in filenames}
            manifest = BiclusterSetManifest(
                identifier=identifier,
                created_at=datetime.now(UTC),
                dataset_identifier=dataset_identifier,
                row_count=row_count,
                source_column_count=source_column_count,
                column_count=column_count,
                source_run=source_run,
                source_checkpoint=source_checkpoint,
                source_checkpoint_sha256=source_checkpoint_sha256,
                source_checkpoint_evaluations=source_checkpoint_evaluations,
                checksums=checksums,
                column_patterns_file=column_patterns_file,
                column_objectives_file=column_objectives_file,
                pattern_groups_file=pattern_groups_file,
                pattern_row_parameters_file=pattern_row_parameters_file,
                final_selection_file=final_selection_file,
            )
            (temporary / "manifest.json").write_text(
                manifest.model_dump_json(indent=2) + "\n",
                encoding="utf-8",
            )
        return manifest

    @staticmethod
    def _validate_columns(
        columns: tuple[PreparedColumnMetadata, ...],
        source_column_count: int,
    ) -> None:
        if tuple(column.index for column in columns) != tuple(range(len(columns))):
            raise ArtifactError("prepared result column indices must be contiguous and ordered")
        names = tuple(column.name for column in columns)
        if len(set(names)) != len(names):
            raise ArtifactError("prepared result column names must be unique")
        if any(column.source_column_index >= source_column_count for column in columns):
            raise ArtifactError("prepared result column references an unknown source column")

    @staticmethod
    def _validate_repertoire(repertoire: Repertoire, row_count: int, column_count: int) -> None:
        identifiers: set[str] = set()
        for evaluation in repertoire.evaluations:
            candidate = evaluation.candidate
            if candidate.identifier in identifiers:
                raise ArtifactError("bicluster identifiers must be unique")
            identifiers.add(candidate.identifier)
            if candidate.bicluster.row_indices[-1] >= row_count:
                raise ArtifactError("bicluster row index exceeds dataset dimensions")
            if candidate.bicluster.column_indices[-1] >= column_count:
                raise ArtifactError("bicluster column index exceeds dataset dimensions")

    @staticmethod
    def _derive_details(
        repertoire: Repertoire,
    ) -> tuple[
        tuple[ColumnPatternRecord, ...],
        tuple[ColumnObjectiveRecord, ...],
        tuple[PatternGroupRecord, ...],
        tuple[PatternRowParameterRecord, ...],
        tuple[FinalSelectionRecord, ...],
    ]:
        patterns: list[ColumnPatternRecord] = []
        objectives: list[ColumnObjectiveRecord] = []
        groups: list[PatternGroupRecord] = []
        row_parameters: list[PatternRowParameterRecord] = []
        final_selection: list[FinalSelectionRecord] = []
        for evaluation in repertoire.evaluations:
            identifier = evaluation.candidate.identifier
            if evaluation.final_selection is not None:
                final_selection.append(
                    FinalSelectionRecord(
                        bicluster_id=identifier,
                        **evaluation.final_selection.model_dump(mode="python"),
                    )
                )
            for objective in evaluation.objectives:
                objectives.extend(
                    ColumnObjectiveRecord(
                        bicluster_id=identifier,
                        objective_name=objective.name,
                        direction=objective.direction,
                        column_index=column.column_index,
                        value=column.value,
                        valid=column.valid,
                        diagnostics=column.diagnostics,
                    )
                    for column in objective.columns
                )
            if evaluation.pattern_fit is None:
                continue
            fit = evaluation.pattern_fit
            patterns.extend(
                ColumnPatternRecord(
                    bicluster_id=identifier,
                    column_index=column.column_index,
                    pattern=column.pattern,
                    group_identifier=column.group_identifier,
                    error=column.error,
                    parameter=column.parameter,
                    parameter_scale=column.parameter_scale,
                    source_support=column.source_support,
                    available_support=column.available_support,
                    prototype_support=column.prototype_support,
                    alternatives=column.alternatives,
                    diagnostics=column.diagnostics,
                )
                for column in fit.columns
            )
            for group in fit.groups:
                groups.append(
                    PatternGroupRecord(
                        bicluster_id=identifier,
                        group_identifier=group.identifier,
                        pattern=group.pattern,
                        column_indices=group.column_indices,
                        iterations=group.iterations,
                        converged=group.converged,
                        diagnostics=group.diagnostics,
                    )
                )
                row_parameters.extend(
                    PatternRowParameterRecord(
                        bicluster_id=identifier,
                        group_identifier=group.identifier,
                        row_index=row,
                        value=value,
                    )
                    for row, value in group.row_parameters
                )
        return (
            tuple(patterns),
            tuple(objectives),
            tuple(groups),
            tuple(row_parameters),
            tuple(final_selection),
        )

    @staticmethod
    def _write_columns(
        path: Path,
        columns: tuple[PreparedColumnMetadata, ...],
    ) -> None:
        schema = pa.schema(
            [
                ("column_index", pa.int64()),
                ("name", pa.string()),
                ("kind", pa.string()),
                ("categories", pa.list_(pa.string())),
                ("source_column_index", pa.int64()),
                ("derivation", pa.string()),
            ]
        )
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "column_index": column.index,
                        "name": column.name,
                        "kind": column.kind.value,
                        "categories": list(column.categories),
                        "source_column_index": column.source_column_index,
                        "derivation": column.derivation,
                    }
                    for column in columns
                ],
                schema=schema,
            ),
            path,
            compression="zstd",
        )

    @staticmethod
    def _write_biclusters(path: Path, repertoire: Repertoire) -> None:
        rows: list[dict[str, object]] = []
        for evaluation in repertoire.evaluations:
            objectives = tuple(
                objective.model_copy(update={"columns": ()}) for objective in evaluation.objectives
            )
            rows.append(
                {
                    "bicluster_id": evaluation.candidate.identifier,
                    "generation": evaluation.candidate.generation,
                    "provenance_json": (
                        "null"
                        if evaluation.candidate.provenance is None
                        else evaluation.candidate.provenance.model_dump_json()
                    ),
                    "row_indices": list(evaluation.candidate.bicluster.row_indices),
                    "column_indices": list(evaluation.candidate.bicluster.column_indices),
                    "archive_coordinate": (
                        None
                        if evaluation.archive_coordinate is None
                        else list(evaluation.archive_coordinate)
                    ),
                    "objectives_json": _model_json(objectives),
                    "constraints_json": _model_json(evaluation.constraints),
                    "descriptors_json": _model_json(evaluation.descriptors),
                    "issues_json": _model_json(evaluation.issues),
                    "pattern_issues_json": _model_json(
                        evaluation.pattern_fit.issues if evaluation.pattern_fit else ()
                    ),
                    "has_pattern_fit": evaluation.pattern_fit is not None,
                    "valid": evaluation.valid,
                    "feasible": evaluation.feasible,
                }
            )
        schema = pa.schema(
            [
                ("bicluster_id", pa.string()),
                ("generation", pa.int64()),
                ("provenance_json", pa.string()),
                ("row_indices", pa.list_(pa.int64())),
                ("column_indices", pa.list_(pa.int64())),
                ("archive_coordinate", pa.list_(pa.int64())),
                ("objectives_json", pa.string()),
                ("constraints_json", pa.string()),
                ("descriptors_json", pa.string()),
                ("issues_json", pa.string()),
                ("pattern_issues_json", pa.string()),
                ("has_pattern_fit", pa.bool_()),
                ("valid", pa.bool_()),
                ("feasible", pa.bool_()),
            ]
        )
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")

    @staticmethod
    def _write_optional(
        root: Path,
        filename: Filename,
        records: list[dict[str, object]],
    ) -> Filename | None:
        if not records:
            return None
        pq.write_table(pa.Table.from_pylist(records), root / filename, compression="zstd")
        return filename


class BiclusterSetReader:
    def read_manifest(self, directory: Path) -> BiclusterSetManifest:
        root = directory.resolve()
        try:
            manifest = BiclusterSetManifest.model_validate_json(
                (root / "manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as error:
            raise ArtifactError(f"invalid BiclusterSet manifest in {root}: {error}") from error
        for filename, expected in manifest.checksums.items():
            path = root / filename
            if not path.is_file() or sha256_file(path) != expected:
                raise ArtifactError(f"missing or corrupted BiclusterSet artifact: {path}")
        return manifest

    def read(self, directory: Path) -> Repertoire:
        return self.read_contents(directory).repertoire

    def read_structures(self, directory: Path) -> BiclusterSetStructures:
        """Read memberships and pattern labels without rebuilding scientific evaluations."""

        root = directory.resolve()
        manifest = self.read_manifest(root)
        try:
            columns = self._read_columns(root, manifest)
            core_records = pq.read_table(
                root / manifest.biclusters_file,
                columns=["bicluster_id", "row_indices", "column_indices"],
            ).to_pylist()
            pattern_records = (
                []
                if manifest.column_patterns_file is None
                else pq.read_table(
                    root / manifest.column_patterns_file,
                    columns=["bicluster_id", "column_index", "pattern"],
                ).to_pylist()
            )
            structures = self._assemble_structures(core_records, pattern_records, manifest)
        except ArtifactError:
            raise
        except (
            AttributeError,
            IndexError,
            KeyError,
            OSError,
            TypeError,
            ValueError,
            pa.ArrowException,
        ) as error:
            raise ArtifactError(f"invalid BiclusterSet structural contents: {error}") from error
        return BiclusterSetStructures(
            manifest=manifest,
            columns=columns,
            biclusters=structures,
        )

    def read_contents(self, directory: Path) -> BiclusterSetContents:
        root = directory.resolve()
        manifest = self.read_manifest(root)
        try:
            columns = self._read_columns(root, manifest)
            patterns = self._read_records(root, manifest.column_patterns_file, self._pattern_record)
            objectives = self._read_records(
                root, manifest.column_objectives_file, self._objective_record
            )
            groups = self._read_records(root, manifest.pattern_groups_file, self._group_record)
            row_parameters = self._read_records(
                root,
                manifest.pattern_row_parameters_file,
                PatternRowParameterRecord.model_validate,
            )
            final_selection = self._read_records(
                root,
                manifest.final_selection_file,
                FinalSelectionRecord.model_validate,
            )
            repertoire = self._read_repertoire(
                root,
                manifest,
                tuple(patterns),
                tuple(objectives),
                tuple(groups),
                tuple(row_parameters),
                tuple(final_selection),
            )
        except ArtifactError:
            raise
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
            raise ArtifactError(f"invalid BiclusterSet contents: {error}") from error
        return BiclusterSetContents(
            manifest=manifest,
            columns=columns,
            repertoire=repertoire,
            column_patterns=tuple(patterns),
            column_objectives=tuple(objectives),
            pattern_groups=tuple(groups),
            row_parameters=tuple(row_parameters),
            final_selection=tuple(final_selection),
        )

    @staticmethod
    def _read_columns(
        root: Path,
        manifest: BiclusterSetManifest,
    ) -> tuple[PreparedColumnMetadata, ...]:
        try:
            records = pq.read_table(root / manifest.columns_file).to_pylist()
        except (OSError, pa.ArrowException) as error:
            raise ArtifactError(f"cannot read BiclusterSet column metadata: {error}") from error
        columns = tuple(
            PreparedColumnMetadata(
                index=int(record["column_index"]),
                name=str(record["name"]),
                kind=ColumnKind(str(record["kind"])),
                categories=tuple(str(value) for value in record["categories"]),
                source_column_index=int(record["source_column_index"]),
                derivation=(None if record["derivation"] is None else str(record["derivation"])),
            )
            for record in records
        )
        BiclusterSetWriter._validate_columns(columns, manifest.source_column_count)
        if len(columns) != manifest.column_count:
            raise ArtifactError("BiclusterSet column metadata does not match its manifest")
        return columns

    @staticmethod
    def _read_records(
        root: Path,
        filename: str | None,
        factory: Any,
    ) -> tuple[Any, ...]:
        if filename is None:
            return ()
        try:
            return tuple(factory(record) for record in pq.read_table(root / filename).to_pylist())
        except (OSError, pa.ArrowException) as error:
            raise ArtifactError(f"cannot read BiclusterSet detail table: {error}") from error

    @staticmethod
    def _pattern_record(record: dict[str, Any]) -> ColumnPatternRecord:
        return ColumnPatternRecord(
            **{
                key: value
                for key, value in record.items()
                if key not in {"parameter_json", "alternatives_json", "diagnostics_json"}
            },
            parameter=json.loads(record["parameter_json"]),
            alternatives=tuple(
                PatternCandidateFit.model_validate(item)
                for item in json.loads(record["alternatives_json"])
            ),
            diagnostics=tuple(tuple(item) for item in json.loads(record["diagnostics_json"])),
        )

    @staticmethod
    def _objective_record(record: dict[str, Any]) -> ColumnObjectiveRecord:
        return ColumnObjectiveRecord(
            **{key: value for key, value in record.items() if key != "diagnostics_json"},
            diagnostics=tuple(tuple(item) for item in json.loads(record["diagnostics_json"])),
        )

    @staticmethod
    def _group_record(record: dict[str, Any]) -> PatternGroupRecord:
        return PatternGroupRecord(
            **{key: value for key, value in record.items() if key != "diagnostics_json"},
            diagnostics=tuple(tuple(item) for item in json.loads(record["diagnostics_json"])),
        )

    @staticmethod
    def _read_repertoire(
        root: Path,
        manifest: BiclusterSetManifest,
        patterns: tuple[ColumnPatternRecord, ...],
        objective_details: tuple[ColumnObjectiveRecord, ...],
        groups: tuple[PatternGroupRecord, ...],
        row_parameters: tuple[PatternRowParameterRecord, ...],
        final_selection: tuple[FinalSelectionRecord, ...],
    ) -> Repertoire:
        records = pq.read_table(root / manifest.biclusters_file).to_pylist()
        return BiclusterSetReader._assemble_repertoire(
            records,
            manifest,
            patterns,
            objective_details,
            groups,
            row_parameters,
            final_selection,
        )

    @staticmethod
    def _assemble_structures(
        core_records: list[dict[str, Any]],
        pattern_records: list[dict[str, Any]],
        manifest: BiclusterSetManifest,
    ) -> tuple[BiclusterStructureRecord, ...]:
        identifiers = tuple(str(record["bicluster_id"]) for record in core_records)
        if any(not identifier for identifier in identifiers) or len(set(identifiers)) != len(
            identifiers
        ):
            raise ArtifactError("BiclusterSet bicluster identifiers are invalid or duplicated")

        selected_columns: dict[str, frozenset[int]] = {}
        biclusters: dict[str, Bicluster] = {}
        for identifier, record in zip(identifiers, core_records, strict=True):
            bicluster = Bicluster(
                row_indices=tuple(int(value) for value in record["row_indices"]),
                column_indices=tuple(int(value) for value in record["column_indices"]),
            )
            if (
                bicluster.row_indices[-1] >= manifest.row_count
                or bicluster.column_indices[-1] >= manifest.column_count
            ):
                raise ArtifactError("persisted BiclusterSet indices exceed dataset dimensions")
            biclusters[identifier] = bicluster
            selected_columns[identifier] = frozenset(bicluster.column_indices)

        patterns_by_identifier: dict[str, dict[int, PatternKind | None]] = defaultdict(dict)
        for record in pattern_records:
            identifier = str(record["bicluster_id"])
            if identifier not in biclusters:
                raise ArtifactError("column-pattern records reference unknown biclusters")
            column_index = int(record["column_index"])
            if column_index not in selected_columns[identifier]:
                raise ArtifactError("column-pattern records reference unselected columns")
            if column_index in patterns_by_identifier[identifier]:
                raise ArtifactError("column-pattern records contain duplicate columns")
            raw_pattern = record["pattern"]
            patterns_by_identifier[identifier][column_index] = (
                None if raw_pattern is None else PatternKind(str(raw_pattern))
            )

        return tuple(
            BiclusterStructureRecord(
                identifier=identifier,
                bicluster=biclusters[identifier],
                column_patterns=tuple(sorted(patterns_by_identifier[identifier].items())),
            )
            for identifier in identifiers
        )

    @staticmethod
    def _assemble_repertoire(
        records: list[dict[str, Any]],
        manifest: BiclusterSetManifest,
        patterns: tuple[ColumnPatternRecord, ...],
        objective_details: tuple[ColumnObjectiveRecord, ...],
        groups: tuple[PatternGroupRecord, ...],
        row_parameters: tuple[PatternRowParameterRecord, ...],
        final_selection: tuple[FinalSelectionRecord, ...],
    ) -> Repertoire:
        patterns_by_id: dict[str, list[ColumnPatternRecord]] = defaultdict(list)
        objectives_by_key: dict[tuple[str, str], list[ColumnObjectiveRecord]] = defaultdict(list)
        groups_by_id: dict[str, list[PatternGroupRecord]] = defaultdict(list)
        parameters_by_group: dict[tuple[str, str], list[PatternRowParameterRecord]] = defaultdict(
            list
        )
        selection_by_id = {record.bicluster_id: record for record in final_selection}
        if len(selection_by_id) != len(final_selection):
            raise ArtifactError("BiclusterSet contains duplicate final-selection records")
        for pattern_record in patterns:
            patterns_by_id[pattern_record.bicluster_id].append(pattern_record)
        for objective_record in objective_details:
            objectives_by_key[
                (objective_record.bicluster_id, objective_record.objective_name)
            ].append(objective_record)
        for group_record in groups:
            groups_by_id[group_record.bicluster_id].append(group_record)
        for parameter_record in row_parameters:
            parameters_by_group[
                (parameter_record.bicluster_id, parameter_record.group_identifier)
            ].append(parameter_record)

        evaluations: list[Evaluation] = []
        identifiers: set[str] = set()
        for record in records:
            identifier = str(record["bicluster_id"])
            if identifier in identifiers:
                raise ArtifactError("BiclusterSet contains duplicate identifiers")
            identifiers.add(identifier)
            bicluster = Bicluster(
                row_indices=tuple(int(value) for value in record["row_indices"]),
                column_indices=tuple(int(value) for value in record["column_indices"]),
            )
            if (
                bicluster.row_indices[-1] >= manifest.row_count
                or bicluster.column_indices[-1] >= manifest.column_count
            ):
                raise ArtifactError("BiclusterSet contains an out-of-range index")
            candidate = Candidate(
                identifier=identifier,
                generation=int(record["generation"]),
                bicluster=bicluster,
                provenance=(
                    None
                    if json.loads(record["provenance_json"]) is None
                    else CandidateProvenance.model_validate_json(record["provenance_json"])
                ),
            )
            objectives = []
            for raw in json.loads(record["objectives_json"]):
                summary = ObjectiveValue.model_validate(raw)
                details = sorted(
                    objectives_by_key[(identifier, summary.name)],
                    key=lambda item: item.column_index,
                )
                objectives.append(
                    summary.model_copy(
                        update={
                            "columns": tuple(
                                ColumnObjectiveValue(
                                    column_index=item.column_index,
                                    value=item.value,
                                    valid=item.valid,
                                    diagnostics=item.diagnostics,
                                )
                                for item in details
                            )
                        }
                    )
                )
            pattern_fit = None
            if record["has_pattern_fit"]:
                column_records = sorted(
                    patterns_by_id[identifier], key=lambda item: item.column_index
                )
                group_models = []
                for group in groups_by_id[identifier]:
                    parameters = sorted(
                        parameters_by_group[(identifier, group.group_identifier)],
                        key=lambda item: item.row_index,
                    )
                    group_models.append(
                        PatternGroupFit(
                            identifier=group.group_identifier,
                            pattern=group.pattern,
                            column_indices=group.column_indices,
                            row_parameters=tuple(
                                (item.row_index, item.value) for item in parameters
                            ),
                            iterations=group.iterations,
                            converged=group.converged,
                            diagnostics=group.diagnostics,
                        )
                    )
                pattern_fit = PatternFit(
                    candidate_signature=bicluster.signature,
                    row_indices=bicluster.row_indices,
                    column_indices=bicluster.column_indices,
                    columns=tuple(
                        ColumnPatternFit(
                            column_index=item.column_index,
                            pattern=item.pattern,
                            group_identifier=item.group_identifier,
                            error=item.error,
                            parameter=item.parameter,
                            parameter_scale=item.parameter_scale,
                            source_support=item.source_support,
                            available_support=item.available_support,
                            prototype_support=item.prototype_support,
                            alternatives=item.alternatives,
                            diagnostics=item.diagnostics,
                        )
                        for item in column_records
                    ),
                    groups=tuple(group_models),
                    issues=tuple(
                        EvaluationIssue.model_validate(item)
                        for item in json.loads(record["pattern_issues_json"])
                    ),
                )
            evaluation = Evaluation(
                candidate=candidate,
                objectives=tuple(objectives),
                constraints=tuple(
                    ConstraintValue.model_validate(item)
                    for item in json.loads(record.get("constraints_json", "[]"))
                ),
                descriptors=tuple(
                    NamedValue.model_validate(item)
                    for item in json.loads(record["descriptors_json"])
                ),
                pattern_fit=pattern_fit,
                issues=tuple(
                    EvaluationIssue.model_validate(item)
                    for item in json.loads(record["issues_json"])
                ),
                archive_coordinate=(
                    None
                    if record["archive_coordinate"] is None
                    else tuple(int(value) for value in record["archive_coordinate"])
                ),
                final_selection=(
                    None
                    if identifier not in selection_by_id
                    else FinalSelectionProvenance.model_validate(
                        selection_by_id[identifier].model_dump(exclude={"bicluster_id"})
                    )
                ),
            )
            if evaluation.valid is not bool(record["valid"]):
                raise ArtifactError("persisted BiclusterSet validity is inconsistent")
            if "feasible" in record and evaluation.feasible is not bool(record["feasible"]):
                raise ArtifactError("persisted BiclusterSet feasibility is inconsistent")
            evaluations.append(evaluation)
        unknown_selection = set(selection_by_id) - identifiers
        if unknown_selection:
            raise ArtifactError("final-selection records reference unknown biclusters")
        return Repertoire(evaluations=tuple(evaluations))


class PagedBiclusterSetReader:
    """Validate one BiclusterSet and materialize only requested Parquet pages."""

    def __init__(self, directory: Path) -> None:
        self.root = directory.resolve()
        reader = BiclusterSetReader()
        self.manifest = reader.read_manifest(self.root)
        self.columns = reader._read_columns(self.root, self.manifest)
        try:
            self._biclusters = pq.ParquetFile(self.root / self.manifest.biclusters_file)
            identifiers = tuple(
                str(value)
                for value in pq.read_table(
                    self.root / self.manifest.biclusters_file,
                    columns=["bicluster_id"],
                )
                .column("bicluster_id")
                .to_pylist()
            )
        except (OSError, pa.ArrowException) as error:
            raise ArtifactError(f"cannot index BiclusterSet biclusters: {error}") from error
        if (
            len(identifiers) != self._biclusters.metadata.num_rows
            or any(not identifier for identifier in identifiers)
            or len(set(identifiers)) != len(identifiers)
        ):
            raise ArtifactError("BiclusterSet bicluster identifiers are invalid or duplicated")
        self._identifiers = frozenset(identifiers)
        self._validate_detail_references()
        self._row_group_offsets = self._build_row_group_offsets()

    @property
    def row_count(self) -> int:
        return int(self._biclusters.metadata.num_rows)

    def read_page(self, offset: int, *, limit: int = 128) -> tuple[Evaluation, ...]:
        if offset < 0 or limit < 1:
            raise ValueError("offset must be non-negative and limit must be positive")
        records = self._read_core_records(offset, min(limit, self.row_count - offset))
        return self._assemble_records(records)

    def read_identifiers(self, identifiers: tuple[str, ...]) -> tuple[Evaluation, ...]:
        """Read an ordered subset without materializing unrelated biclusters."""

        if not identifiers:
            return ()
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("requested bicluster identifiers must be unique")
        unknown = set(identifiers) - self._identifiers
        if unknown:
            raise ArtifactError("unknown BiclusterSet bicluster(s): " + ", ".join(sorted(unknown)))
        try:
            unordered = pq.read_table(
                self.root / self.manifest.biclusters_file,
                filters=[("bicluster_id", "in", list(identifiers))],
            ).to_pylist()
        except (OSError, pa.ArrowException) as error:
            raise ArtifactError(f"cannot read BiclusterSet biclusters: {error}") from error
        by_identifier = {str(record["bicluster_id"]): record for record in unordered}
        return self._assemble_records([by_identifier[identifier] for identifier in identifiers])

    def core_batches(
        self,
        *,
        columns: tuple[str, ...],
        batch_size: int = 65_536,
    ) -> Iterator[pa.RecordBatch]:
        """Iterate lightweight core metadata batches for web-side filtering."""

        try:
            yield from self._biclusters.iter_batches(
                batch_size=batch_size,
                columns=list(columns),
            )
        except (OSError, pa.ArrowException) as error:
            raise ArtifactError(f"cannot scan BiclusterSet metadata: {error}") from error

    def pattern_identifiers(self, pattern: PatternKind) -> frozenset[str]:
        filename = self.manifest.column_patterns_file
        if filename is None:
            return frozenset()
        try:
            table = pq.read_table(
                self.root / filename,
                columns=["bicluster_id"],
                filters=[("pattern", "=", pattern.value)],
            )
        except (OSError, pa.ArrowException) as error:
            raise ArtifactError(f"cannot scan BiclusterSet patterns: {error}") from error
        return frozenset(str(value) for value in table.column("bicluster_id").to_pylist())

    def _assemble_records(
        self,
        records: list[dict[str, Any]],
    ) -> tuple[Evaluation, ...]:
        if not records:
            return ()
        identifiers = frozenset(str(record["bicluster_id"]) for record in records)
        patterns = self._read_detail_page(
            self.manifest.column_patterns_file,
            identifiers,
            BiclusterSetReader._pattern_record,
        )
        objectives = self._read_detail_page(
            self.manifest.column_objectives_file,
            identifiers,
            BiclusterSetReader._objective_record,
        )
        groups = self._read_detail_page(
            self.manifest.pattern_groups_file,
            identifiers,
            BiclusterSetReader._group_record,
        )
        row_parameters = self._read_detail_page(
            self.manifest.pattern_row_parameters_file,
            identifiers,
            PatternRowParameterRecord.model_validate,
        )
        final_selection = self._read_detail_page(
            self.manifest.final_selection_file,
            identifiers,
            FinalSelectionRecord.model_validate,
        )
        return BiclusterSetReader._assemble_repertoire(
            records,
            self.manifest,
            patterns,
            objectives,
            groups,
            row_parameters,
            final_selection,
        ).evaluations

    def read_by_identifier(self, identifier: str) -> Evaluation:
        """Read one bicluster and its explanation tables without materializing the set."""

        if identifier not in self._identifiers:
            raise ArtifactError(f"unknown BiclusterSet bicluster: {identifier}")
        return self.read_identifiers((identifier,))[0]

    def _build_row_group_offsets(self) -> tuple[int, ...]:
        offsets: list[int] = []
        current = 0
        for index in range(self._biclusters.num_row_groups):
            offsets.append(current)
            current += self._biclusters.metadata.row_group(index).num_rows
        return tuple(offsets)

    def _read_core_records(self, offset: int, limit: int) -> list[dict[str, Any]]:
        if limit <= 0 or offset >= self.row_count:
            return []
        remaining = limit
        tables: list[pa.Table] = []
        try:
            for group_index, group_start in enumerate(self._row_group_offsets):
                group_rows = self._biclusters.metadata.row_group(group_index).num_rows
                group_end = group_start + group_rows
                if group_end <= offset:
                    continue
                local_start = max(0, offset - group_start)
                take = min(group_rows - local_start, remaining)
                tables.append(self._biclusters.read_row_group(group_index).slice(local_start, take))
                remaining -= take
                if remaining == 0:
                    break
        except (OSError, pa.ArrowException) as error:
            raise ArtifactError(f"cannot page BiclusterSet biclusters: {error}") from error
        return [] if not tables else pa.concat_tables(tables).to_pylist()

    def _validate_detail_references(self) -> None:
        for filename in (
            self.manifest.column_patterns_file,
            self.manifest.column_objectives_file,
            self.manifest.pattern_groups_file,
            self.manifest.pattern_row_parameters_file,
            self.manifest.final_selection_file,
        ):
            if filename is None:
                continue
            try:
                detail_file = pq.ParquetFile(self.root / filename)
                unknown = any(
                    str(value) not in self._identifiers
                    for batch in detail_file.iter_batches(
                        batch_size=65_536,
                        columns=["bicluster_id"],
                    )
                    for value in batch.column(0).to_pylist()
                )
            except (OSError, pa.ArrowException) as error:
                raise ArtifactError(f"cannot index BiclusterSet details: {error}") from error
            if unknown:
                raise ArtifactError("BiclusterSet detail records reference unknown biclusters")

    def _read_detail_page(
        self,
        filename: str | None,
        identifiers: frozenset[str],
        factory: Any,
    ) -> tuple[Any, ...]:
        if filename is None:
            return ()
        try:
            records = pq.read_table(
                self.root / filename,
                filters=[("bicluster_id", "in", sorted(identifiers))],
            ).to_pylist()
        except (OSError, pa.ArrowException) as error:
            raise ArtifactError(f"cannot page BiclusterSet details: {error}") from error
        return tuple(factory(record) for record in records)


__all__ = [
    "BiclusterSetContents",
    "BiclusterSetManifest",
    "BiclusterSetReader",
    "BiclusterSetStructures",
    "BiclusterSetWriter",
    "BiclusterStructureRecord",
    "ColumnObjectiveRecord",
    "ColumnPatternRecord",
    "FinalSelectionRecord",
    "PagedBiclusterSetReader",
    "PatternGroupRecord",
    "PatternRowParameterRecord",
]
