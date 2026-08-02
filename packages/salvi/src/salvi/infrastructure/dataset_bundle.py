"""Versioned DatasetBundle reader and writer."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pydantic import Field, model_validator

from salvi.domain.enums import ColumnKind
from salvi.domain.models import ColumnMetadata, Dataset, FrozenModel
from salvi.exceptions import ArtifactError, ConfigurationError
from salvi.infrastructure.files import atomic_directory, sha256_file
from salvi.infrastructure.ground_truth import GroundTruth
from salvi.infrastructure.yaml import dump_yaml, load_strict_yaml


class DatasetManifest(FrozenModel):
    schema_version: Literal[1] = 1
    identifier: str = Field(min_length=1)
    row_count: int = Field(ge=1)
    column_count: int = Field(ge=1)
    columns: tuple[ColumnMetadata, ...]
    data_file: Literal["data.parquet"] = "data.parquet"
    row_identifiers_file: Literal["row-identifiers.parquet"] | None = None
    ground_truth_file: Literal["ground-truth.json"] | None = None
    checksums: dict[str, str]

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        if self.column_count != len(self.columns):
            raise ValueError("manifest column_count does not match columns")
        if tuple(column.index for column in self.columns) != tuple(range(self.column_count)):
            raise ValueError("manifest column indices must be contiguous and zero-based")
        names = tuple(column.name for column in self.columns)
        if len(set(names)) != len(names):
            raise ValueError("manifest column names must be unique")
        expected: set[str] = {self.data_file}
        if self.row_identifiers_file is not None:
            expected.add(self.row_identifiers_file)
        if self.ground_truth_file is not None:
            expected.add(self.ground_truth_file)
        if set(self.checksums) != expected:
            raise ValueError("manifest checksums must cover every declared data file exactly")
        if any(
            len(checksum) != 64
            or any(character not in "0123456789abcdef" for character in checksum)
            for checksum in self.checksums.values()
        ):
            raise ValueError("manifest checksums must be lowercase SHA-256 values")
        return self


def _is_categorical_type(data_type: pa.DataType) -> bool:
    if pa.types.is_string(data_type) or pa.types.is_large_string(data_type):
        return True
    return bool(
        pa.types.is_dictionary(data_type)
        and (
            pa.types.is_string(data_type.value_type)
            or pa.types.is_large_string(data_type.value_type)
        )
    )


def _validate_arrow_schema(schema: pa.Schema, columns: tuple[ColumnMetadata, ...]) -> None:
    for field, column in zip(schema, columns, strict=True):
        valid = (
            (
                column.kind is ColumnKind.NUMERIC
                and (pa.types.is_integer(field.type) or pa.types.is_floating(field.type))
            )
            or (column.kind is ColumnKind.BOOLEAN and pa.types.is_boolean(field.type))
            or (column.kind is ColumnKind.CATEGORICAL and _is_categorical_type(field.type))
        )
        if not valid:
            raise ArtifactError(
                f"column {column.name!r} has Arrow type {field.type} but is declared "
                f"as {column.kind.value}"
            )


def _validate_categorical_values(
    table: pa.Table,
    columns: tuple[ColumnMetadata, ...],
) -> None:
    for column in columns:
        if column.kind is not ColumnKind.CATEGORICAL:
            continue
        observed = {
            str(value)
            for value in pc.unique(table.column(column.index)).to_pylist()
            if value is not None
        }
        undeclared = observed - set(column.categories)
        if undeclared:
            raise ArtifactError(
                f"column {column.name!r} contains undeclared categories: "
                f"{', '.join(sorted(undeclared))}"
            )


def _validate_numeric_values(table: pa.Table, columns: tuple[ColumnMetadata, ...]) -> None:
    for column in columns:
        field = table.schema.field(column.index)
        if column.kind is not ColumnKind.NUMERIC or not pa.types.is_floating(field.type):
            continue
        values = table.column(column.index)
        non_finite = pc.and_(pc.is_valid(values), pc.invert(pc.is_finite(values)))
        if bool(pc.any(non_finite).as_py()):
            raise ArtifactError(
                f"numeric column {column.name!r} contains non-finite observed values; "
                "missing values must be Arrow nulls"
            )


@dataclass(frozen=True, slots=True)
class LoadedDatasetBundle:
    dataset: Dataset
    table: pa.Table
    row_identifiers: pa.Array


class DatasetBundleWriter:
    def write(
        self,
        destination: Path,
        *,
        identifier: str,
        table: pa.Table,
        columns: tuple[ColumnMetadata, ...],
        row_identifiers: Sequence[str] | None = None,
        ground_truth: GroundTruth | None = None,
        overwrite: bool = False,
    ) -> Dataset:
        if table.num_rows < 1 or table.num_columns < 1:
            raise ArtifactError("dataset tables must contain at least one row and one column")
        expected_names = tuple(column.name for column in columns)
        if tuple(table.column_names) != expected_names:
            raise ArtifactError("table columns must exactly match ordered column metadata names")
        if tuple(column.index for column in columns) != tuple(range(len(columns))):
            raise ArtifactError("column metadata indices must be contiguous and zero-based")
        if len(set(expected_names)) != len(expected_names):
            raise ArtifactError("column names must be unique")
        _validate_arrow_schema(table.schema, columns)
        _validate_categorical_values(table, columns)
        _validate_numeric_values(table, columns)
        identifiers: tuple[str, ...] | None = None
        if row_identifiers is not None:
            identifiers = tuple(row_identifiers)
            if len(identifiers) != table.num_rows:
                raise ArtifactError("row identifier count must match the dataset row count")
            if any(not isinstance(row_identifier, str) for row_identifier in identifiers):
                raise ArtifactError("row identifiers must be strings")
            if any(not identifier.strip() for identifier in identifiers):
                raise ArtifactError("row identifiers must not be blank")
            if len(set(identifiers)) != len(identifiers):
                raise ArtifactError("row identifiers must be unique")
        if ground_truth is not None and (
            ground_truth.dataset_identifier != identifier
            or ground_truth.row_count != table.num_rows
            or ground_truth.column_count != table.num_columns
        ):
            raise ArtifactError("ground truth does not match the DatasetBundle identity or shape")

        with atomic_directory(destination, overwrite=overwrite) as temporary:
            data_path = temporary / "data.parquet"
            pq.write_table(table, data_path, compression="zstd")
            checksums = {"data.parquet": sha256_file(data_path)}
            row_identifiers_file: Literal["row-identifiers.parquet"] | None = None
            if identifiers is not None:
                row_identifiers_file = "row-identifiers.parquet"
                row_path = temporary / row_identifiers_file
                pq.write_table(
                    pa.table({"row_identifier": pa.array(identifiers, type=pa.string())}),
                    row_path,
                    compression="zstd",
                )
                checksums[row_identifiers_file] = sha256_file(row_path)
            ground_truth_file: Literal["ground-truth.json"] | None = None
            if ground_truth is not None:
                ground_truth_file = "ground-truth.json"
                ground_truth_path = temporary / ground_truth_file
                ground_truth_path.write_text(
                    json.dumps(ground_truth.model_dump(mode="json"), indent=2, sort_keys=True)
                    + "\n",
                    encoding="utf-8",
                )
                checksums[ground_truth_file] = sha256_file(ground_truth_path)

            manifest = DatasetManifest(
                identifier=identifier,
                row_count=table.num_rows,
                column_count=table.num_columns,
                columns=columns,
                row_identifiers_file=row_identifiers_file,
                ground_truth_file=ground_truth_file,
                checksums=checksums,
            )
            dump_yaml(manifest.model_dump(mode="json"), temporary / "dataset.yaml")

        return Dataset(
            identifier=identifier,
            bundle_path=destination.resolve(),
            row_count=table.num_rows,
            column_count=table.num_columns,
            columns=columns,
        )


class DatasetBundleReader:
    @staticmethod
    def _load_manifest(bundle: Path) -> DatasetManifest:
        try:
            raw = load_strict_yaml(bundle / "dataset.yaml")
            return DatasetManifest.model_validate(raw)
        except (ConfigurationError, ValueError) as error:
            raise ArtifactError(f"invalid dataset manifest in {bundle}: {error}") from error

    @staticmethod
    def _verify_files(
        bundle: Path,
        manifest: DatasetManifest,
        filenames: Sequence[str],
    ) -> None:
        for filename in filenames:
            expected = manifest.checksums[filename]
            path = bundle / filename
            if not path.is_file():
                raise ArtifactError(f"dataset artifact is missing: {path}")
            actual = sha256_file(path)
            if actual != expected:
                raise ArtifactError(f"checksum mismatch for dataset artifact {path}")

    def read_manifest(self, bundle_path: Path) -> DatasetManifest:
        """Read a manifest and verify every declared artifact."""

        bundle = bundle_path.resolve()
        manifest = self._load_manifest(bundle)
        self._verify_files(bundle, manifest, tuple(manifest.checksums))
        return manifest

    @staticmethod
    def _inspect(bundle: Path, manifest: DatasetManifest) -> Dataset:
        try:
            metadata = pq.read_metadata(bundle / manifest.data_file)
        except (OSError, pa.ArrowException) as error:
            raise ArtifactError(f"cannot read dataset Parquet metadata: {error}") from error
        if metadata.num_rows != manifest.row_count or metadata.num_columns != manifest.column_count:
            raise ArtifactError("dataset Parquet dimensions do not match dataset.yaml")
        parquet_names = tuple(metadata.schema.names)
        expected_names = tuple(column.name for column in manifest.columns)
        if parquet_names != expected_names:
            raise ArtifactError("dataset Parquet columns do not match dataset.yaml")
        _validate_arrow_schema(metadata.schema.to_arrow_schema(), manifest.columns)
        return Dataset(
            identifier=manifest.identifier,
            bundle_path=bundle,
            row_count=manifest.row_count,
            column_count=manifest.column_count,
            columns=manifest.columns,
        )

    def inspect(self, bundle_path: Path) -> Dataset:
        """Inspect identity and Arrow metadata without decoding or hashing data."""

        bundle = bundle_path.resolve()
        return self._inspect(bundle, self._load_manifest(bundle))

    def read(self, bundle_path: Path) -> Dataset:
        """Read runtime metadata and verify runtime-owned artifacts."""

        bundle = bundle_path.resolve()
        manifest = self._load_manifest(bundle)
        runtime_files: list[str] = [manifest.data_file]
        if manifest.row_identifiers_file is not None:
            runtime_files.append(manifest.row_identifiers_file)
        self._verify_files(bundle, manifest, runtime_files)
        return self._inspect(bundle, manifest)

    def load(self, bundle_path: Path) -> LoadedDatasetBundle:
        """Load and validate one complete bundle without exposing ground truth."""

        bundle = bundle_path.resolve()
        manifest = self._load_manifest(bundle)
        runtime_files: list[str] = [manifest.data_file]
        if manifest.row_identifiers_file is not None:
            runtime_files.append(manifest.row_identifiers_file)
        self._verify_files(bundle, manifest, runtime_files)
        try:
            table = pq.read_table(bundle / manifest.data_file)
        except (OSError, pa.ArrowException) as error:
            raise ArtifactError(f"cannot read dataset table: {error}") from error
        if table.num_rows != manifest.row_count or table.num_columns != manifest.column_count:
            raise ArtifactError("dataset Parquet dimensions do not match dataset.yaml")
        expected_names = tuple(column.name for column in manifest.columns)
        if tuple(table.column_names) != expected_names:
            raise ArtifactError("dataset Parquet columns do not match dataset.yaml")
        _validate_arrow_schema(table.schema, manifest.columns)
        _validate_categorical_values(table, manifest.columns)
        _validate_numeric_values(table, manifest.columns)
        dataset = Dataset(
            identifier=manifest.identifier,
            bundle_path=bundle,
            row_count=manifest.row_count,
            column_count=manifest.column_count,
            columns=manifest.columns,
        )
        row_identifiers = self._read_row_identifiers(bundle, manifest)
        return LoadedDatasetBundle(
            dataset=dataset,
            table=table,
            row_identifiers=row_identifiers,
        )

    def read_table(self, bundle_path: Path) -> pa.Table:
        return self.load(bundle_path).table

    def read_ground_truth(self, bundle_path: Path) -> GroundTruth | None:
        bundle = bundle_path.resolve()
        manifest = self._load_manifest(bundle)
        if manifest.ground_truth_file is None:
            return None
        self._verify_files(bundle, manifest, (manifest.ground_truth_file,))
        try:
            raw = json.loads((bundle / manifest.ground_truth_file).read_text(encoding="utf-8"))
            ground_truth = GroundTruth.model_validate(raw)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            raise ArtifactError(f"invalid canonical ground truth in {bundle}: {error}") from error
        if (
            ground_truth.dataset_identifier != manifest.identifier
            or ground_truth.row_count != manifest.row_count
            or ground_truth.column_count != manifest.column_count
        ):
            raise ArtifactError("canonical ground truth does not match its DatasetBundle")
        return ground_truth

    @staticmethod
    def _read_row_identifiers(
        bundle: Path,
        manifest: DatasetManifest,
    ) -> pa.Array:
        if manifest.row_identifiers_file is None:
            generated = pa.array(range(manifest.row_count), type=pa.int64())
            return pc.cast(generated, pa.string())
        try:
            table = pq.read_table(bundle / manifest.row_identifiers_file)
        except (OSError, pa.ArrowException) as error:
            raise ArtifactError(f"cannot read row identifiers: {error}") from error
        if table.column_names != ["row_identifier"] or table.num_columns != 1:
            raise ArtifactError("row identifier table must contain only 'row_identifier'")
        if table.num_rows != manifest.row_count:
            raise ArtifactError("row identifier count does not match dataset dimensions")
        values = table.column(0).combine_chunks()
        if not pa.types.is_string(values.type) or values.null_count:
            raise ArtifactError("row identifiers must be non-null strings")
        if int(pc.count_distinct(values).as_py()) != manifest.row_count:
            raise ArtifactError("row identifiers must be unique")
        if bool(pc.any(pc.equal(pc.utf8_trim_whitespace(values), "")).as_py()):
            raise ArtifactError("row identifiers must not be blank")
        return values


__all__ = [
    "DatasetBundleReader",
    "DatasetBundleWriter",
    "DatasetManifest",
    "LoadedDatasetBundle",
]
