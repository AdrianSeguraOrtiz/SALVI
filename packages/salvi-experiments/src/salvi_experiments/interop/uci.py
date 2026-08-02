"""Reproducible UCI imports and clinical DatasetBundle annotations."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Self, TypeAlias

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from salvi.domain.enums import ColumnKind
from salvi.domain.models import ColumnMetadata
from salvi.exceptions import ArtifactError, ConfigurationError, ConversionError
from salvi.infrastructure.dataset_bundle import DatasetBundleReader, DatasetBundleWriter
from salvi.infrastructure.files import atomic_directory, sha256_file
from salvi.infrastructure.yaml import dump_yaml, load_strict_yaml

UCI_API_URL = "https://archive.ics.uci.edu/api/dataset?id={dataset_id}"
UCI_DATA_URL = "https://archive.ics.uci.edu/static/public/{dataset_id}/data.csv"
Primitive: TypeAlias = bool | int | float | str


class ClinicalColumnRole(StrEnum):
    IDENTIFIER = "IDENTIFIER"
    SEARCH = "SEARCH"
    OUTCOME = "OUTCOME"
    COVARIATE = "COVARIATE"
    SUPPLEMENTARY = "SUPPLEMENTARY"
    EXCLUDED = "EXCLUDED"


class ClinicalAnnotationKind(StrEnum):
    NUMERIC = "NUMERIC"
    BOOLEAN = "BOOLEAN"
    CATEGORICAL = "CATEGORICAL"
    ORDINAL = "ORDINAL"
    SURVIVAL_TIME = "SURVIVAL_TIME"


class DerivedOperation(StrEnum):
    EQUAL = "EQUAL"
    NOT_EQUAL = "NOT_EQUAL"
    LESS_EQUAL = "LESS_EQUAL"
    IN = "IN"


class FrozenUciModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)


class UciRoleDefaults(FrozenUciModel):
    identifier: ClinicalColumnRole = ClinicalColumnRole.IDENTIFIER
    feature: ClinicalColumnRole = ClinicalColumnRole.SEARCH
    target: ClinicalColumnRole = ClinicalColumnRole.OUTCOME
    other: ClinicalColumnRole = ClinicalColumnRole.EXCLUDED

    def for_uci_role(self, role: str) -> ClinicalColumnRole:
        normalized = role.strip().lower()
        if normalized == "id":
            return self.identifier
        if normalized == "feature":
            return self.feature
        if normalized == "target":
            return self.target
        return self.other


class UciColumnRule(FrozenUciModel):
    name: str = Field(min_length=1)
    output_name: str | None = Field(default=None, min_length=1)
    role: ClinicalColumnRole | None = None
    annotation_kind: ClinicalAnnotationKind | None = None
    search_kind: ColumnKind | None = None
    units: str | None = None
    mapping: dict[str, Primitive] = Field(default_factory=dict)
    categories: tuple[str, ...] = ()
    missing_tokens: tuple[str, ...] | None = None
    copy_to_annotations: bool = False
    survival_event_column: str | None = None

    @model_validator(mode="after")
    def validate_survival_link(self) -> Self:
        if (
            self.survival_event_column is not None
            and self.annotation_kind is not ClinicalAnnotationKind.SURVIVAL_TIME
        ):
            raise ValueError("survival_event_column requires annotation_kind SURVIVAL_TIME")
        if len(set(self.categories)) != len(self.categories):
            raise ValueError("declared categories must be unique")
        return self

    @field_validator("mapping")
    @classmethod
    def validate_mapping(cls, value: dict[str, Primitive]) -> dict[str, Primitive]:
        if any(not key for key in value):
            raise ValueError("mapping keys must not be blank")
        return value


class DerivedAnnotation(FrozenUciModel):
    name: str = Field(min_length=1)
    source: str = Field(min_length=1)
    operation: DerivedOperation
    value: Primitive | tuple[Primitive, ...]
    role: ClinicalColumnRole
    annotation_kind: ClinicalAnnotationKind
    units: str | None = None
    survival_event_column: str | None = None

    @model_validator(mode="after")
    def validate_derivation(self) -> Self:
        if self.role in {
            ClinicalColumnRole.IDENTIFIER,
            ClinicalColumnRole.SEARCH,
            ClinicalColumnRole.EXCLUDED,
        }:
            raise ValueError("derived annotations must be outcomes, covariates, or supplementary")
        if self.operation is DerivedOperation.IN and not isinstance(self.value, tuple):
            raise ValueError("IN derivations require a list of values")
        if self.operation is not DerivedOperation.IN and isinstance(self.value, tuple):
            raise ValueError(f"{self.operation.value} derivations require one scalar value")
        return self


class UciImportRecipe(FrozenUciModel):
    schema_version: Literal[1] = 1
    identifier: str = Field(min_length=1)
    dataset_id: Annotated[int, Field(ge=1)]
    expected_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    missing_tokens: tuple[str, ...] = ("", "NaN")
    role_defaults: UciRoleDefaults = Field(default_factory=UciRoleDefaults)
    columns: tuple[UciColumnRule, ...] = ()
    derived_annotations: tuple[DerivedAnnotation, ...] = ()

    @model_validator(mode="after")
    def validate_names(self) -> Self:
        column_names = tuple(item.name for item in self.columns)
        if len(set(column_names)) != len(column_names):
            raise ValueError("column rules must have unique source names")
        derived_names = tuple(item.name for item in self.derived_annotations)
        if len(set(derived_names)) != len(derived_names):
            raise ValueError("derived annotation names must be unique")
        return self


class ClinicalAnnotation(FrozenUciModel):
    name: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    role: ClinicalColumnRole
    kind: ClinicalAnnotationKind
    units: str | None = None
    categories: tuple[str, ...] = ()
    search_column_name: str | None = None
    derived: bool = False
    survival_event_column: str | None = None

    @model_validator(mode="after")
    def validate_categories(self) -> Self:
        if self.kind in {ClinicalAnnotationKind.CATEGORICAL, ClinicalAnnotationKind.ORDINAL}:
            if not self.categories:
                raise ValueError("categorical and ordinal annotations require categories")
        elif self.categories:
            raise ValueError("only categorical and ordinal annotations declare categories")
        if self.role is ClinicalColumnRole.SEARCH and self.search_column_name is None:
            raise ValueError("SEARCH annotations must identify their search column")
        return self


class ClinicalDatasetManifest(FrozenUciModel):
    schema_version: Literal[1] = 1
    identifier: str = Field(min_length=1)
    uci_dataset_id: Annotated[int, Field(ge=1)]
    uci_dataset_name: str = Field(min_length=1)
    row_count: Annotated[int, Field(ge=1)]
    dataset_directory: Literal["dataset"] = "dataset"
    annotations_file: Literal["annotations.parquet"] = "annotations.parquet"
    recipe_file: Literal["effective-import-recipe.yaml"] = "effective-import-recipe.yaml"
    source_metadata_file: Literal["source/metadata.json"] = "source/metadata.json"
    source_data_file: Literal["source/data.csv"] = "source/data.csv"
    source_data_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    annotations: tuple[ClinicalAnnotation, ...]
    checksums: dict[str, str]

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        expected = {
            self.annotations_file,
            self.recipe_file,
            self.source_metadata_file,
            self.source_data_file,
        }
        if set(self.checksums) != expected:
            raise ValueError("clinical manifest checksums do not cover declared files")
        names = tuple(annotation.name for annotation in self.annotations)
        if len(set(names)) != len(names):
            raise ValueError("clinical annotation names must be unique")
        return self


class LoadedClinicalDataset(FrozenUciModel):
    root: Path
    manifest: ClinicalDatasetManifest
    dataset_bundle: Path
    annotations: pa.Table


def load_uci_import_recipe(path: str | Path) -> UciImportRecipe:
    source = Path(path).expanduser().resolve()
    try:
        return UciImportRecipe.model_validate(load_strict_yaml(source))
    except ValidationError as error:
        raise ConfigurationError(f"invalid UCI import recipe {source}: {error}") from error


class UciRepositoryClient:
    """Fetch official UCI metadata and CSV resources into a checksum-addressed cache."""

    def __init__(
        self,
        cache_directory: Path | None = None,
        *,
        timeout_seconds: float = 60.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.cache_directory = (
            (cache_directory or Path.home() / ".cache" / "salvi" / "uci").expanduser().resolve()
        )
        self.timeout_seconds = timeout_seconds

    def fetch(self, dataset_id: int, expected_sha256: str) -> tuple[dict[str, Any], Path]:
        directory = self.cache_directory / str(dataset_id) / expected_sha256
        metadata_path = directory / "metadata.json"
        data_path = directory / "data.csv"
        if metadata_path.is_file() and data_path.is_file():
            if sha256_file(data_path) == expected_sha256:
                return self._read_metadata(metadata_path, dataset_id), data_path
            shutil.rmtree(directory)
        metadata, downloaded_path, actual = self.fetch_current(dataset_id)
        if actual != expected_sha256:
            raise ConversionError(
                f"UCI dataset {dataset_id} checksum changed: expected "
                f"{expected_sha256}, received {actual}"
            )
        return metadata, downloaded_path

    def fetch_current(self, dataset_id: int) -> tuple[dict[str, Any], Path, str]:
        """Fetch the current official resource and cache it under its actual checksum."""

        try:
            metadata_bytes = self._download(UCI_API_URL.format(dataset_id=dataset_id))
            data_bytes = self._download(UCI_DATA_URL.format(dataset_id=dataset_id))
        except (OSError, urllib.error.URLError) as error:
            raise ConversionError(f"cannot download UCI dataset {dataset_id}: {error}") from error
        actual = hashlib.sha256(data_bytes).hexdigest()
        try:
            metadata = json.loads(metadata_bytes)
        except json.JSONDecodeError as error:
            raise ConversionError(f"UCI dataset {dataset_id} returned invalid metadata") from error
        payload = self._metadata_payload(metadata, dataset_id)
        directory = self.cache_directory / str(dataset_id) / actual
        metadata_path = directory / "metadata.json"
        data_path = directory / "data.csv"
        directory.mkdir(parents=True, exist_ok=True)
        self._atomic_write(metadata_path, json.dumps(metadata, indent=2, sort_keys=True).encode())
        self._atomic_write(data_path, data_bytes)
        return payload, data_path, actual

    def _download(self, url: str) -> bytes:
        request = urllib.request.Request(url, headers={"User-Agent": "SALVI/0.1 UCI importer"})
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return bytes(response.read())

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
            os.replace(temporary_name, path)
        finally:
            Path(temporary_name).unlink(missing_ok=True)

    @classmethod
    def _read_metadata(cls, path: Path, dataset_id: int) -> dict[str, Any]:
        try:
            return cls._metadata_payload(json.loads(path.read_text(encoding="utf-8")), dataset_id)
        except (OSError, json.JSONDecodeError) as error:
            raise ConversionError(f"invalid cached UCI metadata {path}: {error}") from error

    @staticmethod
    def _metadata_payload(document: object, dataset_id: int) -> dict[str, Any]:
        if not isinstance(document, dict) or document.get("status") != 200:
            raise ConversionError(f"UCI dataset {dataset_id} does not exist")
        payload = document.get("data")
        if not isinstance(payload, dict) or payload.get("uci_id") != dataset_id:
            raise ConversionError(f"UCI metadata identity does not match dataset {dataset_id}")
        return payload


def _annotation_kind(variable_type: str) -> ClinicalAnnotationKind:
    normalized = variable_type.strip().lower()
    if normalized == "binary":
        return ClinicalAnnotationKind.BOOLEAN
    if normalized == "categorical":
        return ClinicalAnnotationKind.CATEGORICAL
    return ClinicalAnnotationKind.NUMERIC


def _search_kind(kind: ClinicalAnnotationKind) -> ColumnKind:
    if kind is ClinicalAnnotationKind.BOOLEAN:
        return ColumnKind.BOOLEAN
    if kind in {ClinicalAnnotationKind.CATEGORICAL, ClinicalAnnotationKind.ORDINAL}:
        return ColumnKind.CATEGORICAL
    return ColumnKind.NUMERIC


def _coerce(
    raw: str,
    *,
    kind: ClinicalAnnotationKind,
    mapping: Mapping[str, Primitive],
    missing_tokens: frozenset[str],
) -> Primitive | None:
    value = raw.strip()
    if value in missing_tokens:
        return None
    mapped: Primitive = mapping.get(value, value)
    if kind is ClinicalAnnotationKind.BOOLEAN:
        if isinstance(mapped, bool):
            return mapped
        normalized = str(mapped).strip().lower()
        if normalized in {"1", "true", "yes", "y", "present"}:
            return True
        if normalized in {"0", "false", "no", "n", "absent"}:
            return False
        raise ConversionError(f"cannot parse {raw!r} as Boolean")
    if kind in {ClinicalAnnotationKind.NUMERIC, ClinicalAnnotationKind.SURVIVAL_TIME}:
        try:
            numeric = float(mapped)
        except (TypeError, ValueError) as error:
            raise ConversionError(f"cannot parse {raw!r} as numeric") from error
        if not (float("-inf") < numeric < float("inf")):
            raise ConversionError(f"non-finite numeric value {raw!r}")
        return numeric
    return str(mapped)


def _categories(values: Sequence[Primitive | None]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values if value is not None))


def _arrow(values: Sequence[Primitive | None], kind: ClinicalAnnotationKind) -> pa.Array:
    if kind is ClinicalAnnotationKind.BOOLEAN:
        return pa.array(values, type=pa.bool_())
    if kind in {ClinicalAnnotationKind.NUMERIC, ClinicalAnnotationKind.SURVIVAL_TIME}:
        return pa.array(values, type=pa.float64())
    return pa.array(values, type=pa.string())


def _derive(values: Sequence[Primitive | None], rule: DerivedAnnotation) -> list[Primitive | None]:
    result: list[Primitive | None] = []
    for value in values:
        if value is None:
            result.append(None)
        elif rule.operation is DerivedOperation.EQUAL:
            result.append(value == rule.value)
        elif rule.operation is DerivedOperation.NOT_EQUAL:
            result.append(value != rule.value)
        elif rule.operation is DerivedOperation.LESS_EQUAL:
            assert not isinstance(rule.value, tuple)
            result.append(float(value) <= float(rule.value))
        else:
            assert isinstance(rule.value, tuple)
            result.append(value in rule.value)
    return result


class ClinicalDatasetBundleWriter:
    def write(
        self,
        destination: Path,
        *,
        recipe: UciImportRecipe,
        metadata: Mapping[str, Any],
        source_data: Path,
        search_table: pa.Table,
        search_columns: tuple[ColumnMetadata, ...],
        row_identifiers: Sequence[str],
        annotation_table: pa.Table,
        annotations: tuple[ClinicalAnnotation, ...],
        overwrite: bool = False,
    ) -> Path:
        target = destination.expanduser().resolve()
        with atomic_directory(target, overwrite=overwrite) as temporary:
            DatasetBundleWriter().write(
                temporary / "dataset",
                identifier=recipe.identifier,
                table=search_table,
                columns=search_columns,
                row_identifiers=row_identifiers,
            )
            pq.write_table(annotation_table, temporary / "annotations.parquet", compression="zstd")
            dump_yaml(recipe.model_dump(mode="json"), temporary / "effective-import-recipe.yaml")
            source = temporary / "source"
            source.mkdir()
            (source / "metadata.json").write_text(
                json.dumps(dict(metadata), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            shutil.copy2(source_data, source / "data.csv")
            files = (
                "annotations.parquet",
                "effective-import-recipe.yaml",
                "source/metadata.json",
                "source/data.csv",
            )
            manifest = ClinicalDatasetManifest(
                identifier=recipe.identifier,
                uci_dataset_id=recipe.dataset_id,
                uci_dataset_name=str(metadata.get("name", f"UCI {recipe.dataset_id}")),
                row_count=search_table.num_rows,
                source_data_sha256=sha256_file(source / "data.csv"),
                dataset_manifest_sha256=sha256_file(temporary / "dataset" / "dataset.yaml"),
                annotations=annotations,
                checksums={name: sha256_file(temporary / name) for name in files},
            )
            dump_yaml(manifest.model_dump(mode="json"), temporary / "clinical-dataset.yaml")
        return target


class ClinicalDatasetBundleReader:
    def read_manifest(self, directory: Path) -> ClinicalDatasetManifest:
        root = directory.expanduser().resolve()
        try:
            manifest = ClinicalDatasetManifest.model_validate(
                load_strict_yaml(root / "clinical-dataset.yaml")
            )
        except (ValidationError, OSError) as error:
            raise ArtifactError(f"invalid ClinicalDatasetBundle {root}: {error}") from error
        for relative, expected in manifest.checksums.items():
            path = root / relative
            if not path.is_file() or sha256_file(path) != expected:
                raise ArtifactError(f"clinical artifact checksum mismatch: {path}")
        dataset = DatasetBundleReader().inspect(root / manifest.dataset_directory)
        if (
            sha256_file(root / manifest.dataset_directory / "dataset.yaml")
            != manifest.dataset_manifest_sha256
        ):
            raise ArtifactError("nested DatasetBundle manifest checksum mismatch")
        if dataset.identifier != manifest.identifier or dataset.row_count != manifest.row_count:
            raise ArtifactError("nested DatasetBundle does not match the clinical manifest")
        table = pq.read_table(root / manifest.annotations_file)
        if table.num_rows != manifest.row_count or "row_identifier" not in table.column_names:
            raise ArtifactError("clinical annotations do not align with the dataset")
        search_names = {column.name for column in dataset.columns}
        leaked = {
            annotation.name
            for annotation in manifest.annotations
            if annotation.role not in {ClinicalColumnRole.SEARCH}
            and annotation.name in search_names
        }
        if leaked:
            raise ArtifactError(
                "non-search annotations leaked into DatasetBundle: " + ", ".join(sorted(leaked))
            )
        return manifest

    def load(self, directory: Path) -> LoadedClinicalDataset:
        root = directory.expanduser().resolve()
        manifest = self.read_manifest(root)
        return LoadedClinicalDataset(
            root=root,
            manifest=manifest,
            dataset_bundle=root / manifest.dataset_directory,
            annotations=pq.read_table(root / manifest.annotations_file),
        )


class UciConverter:
    def __init__(
        self,
        *,
        client: UciRepositoryClient | None = None,
        overwrite: bool = False,
    ) -> None:
        self.client = client or UciRepositoryClient()
        self.overwrite = overwrite

    def inspect(self, recipe: UciImportRecipe) -> tuple[dict[str, Any], Path]:
        return self.client.fetch(recipe.dataset_id, recipe.expected_sha256)

    def convert(self, recipe: UciImportRecipe | Path, destination: Path) -> Path:
        resolved = load_uci_import_recipe(recipe) if isinstance(recipe, Path) else recipe
        metadata, data_path = self.inspect(resolved)
        variables = metadata.get("variables")
        if not isinstance(variables, list) or not variables:
            raise ConversionError(f"UCI dataset {resolved.dataset_id} has no variable metadata")
        by_name: dict[str, Mapping[str, Any]] = {}
        for item in variables:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                raise ConversionError("UCI variable metadata is malformed")
            by_name[str(item["name"])] = item
        rules = {rule.name: rule for rule in resolved.columns}
        unknown_rules = set(rules) - set(by_name)
        if unknown_rules:
            raise ConversionError(
                "UCI recipe references unknown columns: " + ", ".join(sorted(unknown_rules))
            )
        with data_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or tuple(reader.fieldnames) != tuple(by_name):
                raise ConversionError("UCI CSV columns do not match official variable metadata")
            rows = list(reader)
        if not rows:
            raise ConversionError("UCI CSV contains no rows")

        parsed: dict[str, list[Primitive | None]] = {}
        resolved_specs: dict[
            str,
            tuple[
                ClinicalColumnRole,
                ClinicalAnnotationKind,
                ColumnKind,
                str,
                str | None,
                bool,
                str | None,
                tuple[str, ...],
            ],
        ] = {}
        identifier_name: str | None = None
        for source_name, variable in by_name.items():
            rule = rules.get(source_name, UciColumnRule(name=source_name))
            role = rule.role or resolved.role_defaults.for_uci_role(str(variable.get("role", "")))
            annotation_kind = rule.annotation_kind or _annotation_kind(
                str(variable.get("type", ""))
            )
            column_kind = rule.search_kind or _search_kind(annotation_kind)
            output_name = rule.output_name or source_name.strip()
            missing = frozenset(
                resolved.missing_tokens if rule.missing_tokens is None else rule.missing_tokens
            )
            values = [
                _coerce(
                    row[source_name],
                    kind=annotation_kind,
                    mapping=rule.mapping,
                    missing_tokens=missing,
                )
                for row in rows
            ]
            parsed[source_name] = values
            resolved_specs[source_name] = (
                role,
                annotation_kind,
                column_kind,
                output_name,
                rule.units if rule.units is not None else variable.get("units"),
                rule.copy_to_annotations,
                rule.survival_event_column,
                rule.categories,
            )
            if role is ClinicalColumnRole.IDENTIFIER:
                if identifier_name is not None:
                    raise ConversionError("UCI recipe selects more than one identifier column")
                identifier_name = source_name

        if identifier_name is None:
            row_identifiers = tuple(
                f"{resolved.identifier}-row-{index:06d}" for index in range(len(rows))
            )
        else:
            if any(value is None for value in parsed[identifier_name]):
                raise ConversionError("identifier column contains missing values")
            row_identifiers = tuple(str(value) for value in parsed[identifier_name])
            if len(set(row_identifiers)) != len(row_identifiers):
                raise ConversionError("identifier column contains duplicate values")

        search_arrays: list[pa.Array] = []
        search_columns: list[ColumnMetadata] = []
        annotation_arrays: dict[str, pa.Array] = {
            "row_identifier": pa.array(row_identifiers, type=pa.string())
        }
        annotations: list[ClinicalAnnotation] = []
        annotation_values: dict[str, list[Primitive | None]] = {}
        for source_name, values in parsed.items():
            (
                role,
                kind,
                column_kind,
                output_name,
                units,
                copy_to_annotations,
                survival_event_column,
                declared_categories,
            ) = resolved_specs[source_name]
            categories = declared_categories or (
                _categories(values)
                if kind
                in {
                    ClinicalAnnotationKind.CATEGORICAL,
                    ClinicalAnnotationKind.ORDINAL,
                }
                else ()
            )
            if declared_categories:
                unknown = set(_categories(values)) - set(declared_categories)
                if unknown:
                    raise ConversionError(
                        f"column {source_name!r} contains undeclared categories: "
                        + ", ".join(sorted(unknown))
                    )
            if role is ClinicalColumnRole.SEARCH:
                search_arrays.append(_arrow(values, kind))
                search_columns.append(
                    ColumnMetadata(
                        index=len(search_columns),
                        name=output_name,
                        kind=column_kind,
                        categories=categories if column_kind is ColumnKind.CATEGORICAL else (),
                    )
                )
            if role in {
                ClinicalColumnRole.OUTCOME,
                ClinicalColumnRole.COVARIATE,
                ClinicalColumnRole.SUPPLEMENTARY,
            } or (role is ClinicalColumnRole.SEARCH and copy_to_annotations):
                if output_name in annotation_arrays:
                    raise ConversionError(f"duplicate annotation name {output_name!r}")
                annotation_arrays[output_name] = _arrow(values, kind)
                annotation_values[output_name] = values
                annotations.append(
                    ClinicalAnnotation(
                        name=output_name,
                        source_name=source_name,
                        role=role,
                        kind=kind,
                        units=None if units is None else str(units),
                        categories=categories,
                        search_column_name=(
                            output_name if role is ClinicalColumnRole.SEARCH else None
                        ),
                        survival_event_column=survival_event_column,
                    )
                )
        if not search_columns:
            raise ConversionError("UCI recipe selects no SEARCH columns")

        source_or_annotation = {
            **parsed,
            **annotation_values,
        }
        for derivation in resolved.derived_annotations:
            source_values = source_or_annotation.get(derivation.source)
            if source_values is None:
                raise ConversionError(
                    f"derived annotation {derivation.name!r} has unknown source "
                    f"{derivation.source!r}"
                )
            values = _derive(source_values, derivation)
            if derivation.name in annotation_arrays:
                raise ConversionError(f"duplicate derived annotation {derivation.name!r}")
            categories = (
                _categories(values)
                if derivation.annotation_kind
                in {
                    ClinicalAnnotationKind.CATEGORICAL,
                    ClinicalAnnotationKind.ORDINAL,
                }
                else ()
            )
            annotation_arrays[derivation.name] = _arrow(values, derivation.annotation_kind)
            annotation_values[derivation.name] = values
            annotations.append(
                ClinicalAnnotation(
                    name=derivation.name,
                    source_name=derivation.source,
                    role=derivation.role,
                    kind=derivation.annotation_kind,
                    units=derivation.units,
                    categories=categories,
                    derived=True,
                    survival_event_column=derivation.survival_event_column,
                )
            )

        return ClinicalDatasetBundleWriter().write(
            destination,
            recipe=resolved,
            metadata=metadata,
            source_data=data_path,
            search_table=pa.table(
                {
                    column.name: array
                    for column, array in zip(search_columns, search_arrays, strict=True)
                }
            ),
            search_columns=tuple(search_columns),
            row_identifiers=row_identifiers,
            annotation_table=pa.table(annotation_arrays),
            annotations=tuple(annotations),
            overwrite=self.overwrite,
        )


def write_clinical_subsample(
    source: Path,
    destination: Path,
    row_indices: Sequence[int],
    *,
    identifier: str,
    overwrite: bool = False,
) -> Path:
    """Create an aligned clinical bundle for a deterministic row subsample."""

    clinical = ClinicalDatasetBundleReader().load(source)
    indices = tuple(sorted(set(row_indices)))
    if not indices or len(indices) != len(row_indices):
        raise ValueError("clinical subsample row indices must be non-empty and unique")
    if indices[0] < 0 or indices[-1] >= clinical.manifest.row_count:
        raise ValueError("clinical subsample row index is outside the source dataset")
    dataset = DatasetBundleReader().load(clinical.dataset_bundle)
    selection = pa.array(indices, type=pa.int64())
    recipe = load_uci_import_recipe(clinical.root / clinical.manifest.recipe_file).model_copy(
        update={"identifier": identifier}
    )
    metadata = json.loads(
        (clinical.root / clinical.manifest.source_metadata_file).read_text(encoding="utf-8")
    )
    return ClinicalDatasetBundleWriter().write(
        destination,
        recipe=recipe,
        metadata=metadata,
        source_data=clinical.root / clinical.manifest.source_data_file,
        search_table=dataset.table.take(selection),
        search_columns=dataset.dataset.columns,
        row_identifiers=dataset.row_identifiers.take(selection).to_pylist(),
        annotation_table=clinical.annotations.take(selection),
        annotations=clinical.manifest.annotations,
        overwrite=overwrite,
    )


__all__ = [
    "ClinicalAnnotation",
    "ClinicalAnnotationKind",
    "ClinicalColumnRole",
    "ClinicalDatasetBundleReader",
    "ClinicalDatasetBundleWriter",
    "ClinicalDatasetManifest",
    "DerivedAnnotation",
    "DerivedOperation",
    "LoadedClinicalDataset",
    "UciColumnRule",
    "UciConverter",
    "UciImportRecipe",
    "UciRepositoryClient",
    "UciRoleDefaults",
    "load_uci_import_recipe",
    "write_clinical_subsample",
]
