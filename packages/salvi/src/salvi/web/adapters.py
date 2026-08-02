"""Built-in browser upload adapters for canonical and tabular datasets."""

from __future__ import annotations

import csv
import math
import re
import shutil
import stat
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from salvi.domain.enums import ColumnKind
from salvi.domain.models import ColumnMetadata
from salvi.exceptions import ArtifactError, ConversionError
from salvi.infrastructure.dataset_bundle import DatasetBundleReader, DatasetBundleWriter
from salvi.web.models import (
    AdapterFileSlot,
    DatasetImportPreview,
    InputAdapterDescription,
    WebColumnProposal,
)
from salvi.web.providers import InputAdapter

_MISSING_TOKENS = ("", "na", "n/a", "nan", "null", "?")
_TRUE_TOKENS = frozenset({"true", "t", "yes", "y", "1"})
_FALSE_TOKENS = frozenset({"false", "f", "no", "n", "0"})
_IDENTIFIER_NAMES = frozenset({"x", "id", "row", "row_id", "row_identifier"})
_SAFE_IDENTIFIER = re.compile(r"[^A-Za-z0-9._-]+")


def normalized_identifier(value: str) -> str:
    normalized = _SAFE_IDENTIFIER.sub("-", value.strip()).strip("-.")
    if not normalized:
        raise ConversionError("dataset identifier must contain letters or digits")
    return normalized


def canonical_bundle_preview(bundle: Path, *, adapter: str) -> DatasetImportPreview:
    """Project a verified canonical bundle into the browser import contract."""

    manifest = DatasetBundleReader().read_manifest(bundle)
    try:
        parquet = pq.ParquetFile(bundle / manifest.data_file)
        missing_counts: list[int] = []
        for column_index in range(manifest.column_count):
            counts = [
                parquet.metadata.row_group(row_group).column(column_index).statistics.null_count
                if parquet.metadata.row_group(row_group).column(column_index).statistics is not None
                else None
                for row_group in range(parquet.num_row_groups)
            ]
            if any(value is None for value in counts):
                missing_counts.append(
                    pq.read_table(
                        bundle / manifest.data_file,
                        columns=[manifest.columns[column_index].name],
                    )
                    .column(0)
                    .null_count
                )
            else:
                missing_counts.append(sum(int(value) for value in counts if value is not None))
    except (OSError, pa.ArrowException) as error:
        raise ConversionError(f"cannot inspect canonical dataset values: {error}") from error
    return DatasetImportPreview(
        identifier=manifest.identifier,
        adapter=adapter,
        row_count=manifest.row_count,
        column_count=manifest.column_count,
        columns=tuple(
            WebColumnProposal(
                source_index=column.index,
                name=column.name,
                inferred_kind=column.kind,
                selected_kind=column.kind,
                missing_ratio=missing_counts[column.index] / manifest.row_count,
            )
            for column in manifest.columns
        ),
        confirmation_required=False,
        ground_truth_attached=manifest.ground_truth_file is not None,
    )


def _tabular_table(path: Path, delimiter: str) -> pa.Table:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream, delimiter=delimiter))
    except (OSError, StopIteration, csv.Error) as error:
        raise ConversionError(f"cannot read tabular header from {path}: {error}") from error
    if not header or any(not name.strip() for name in header):
        raise ConversionError("tabular columns must have non-empty headers")
    if len(set(header)) != len(header):
        raise ConversionError("tabular column headers must be unique")
    try:
        return pacsv.read_csv(
            path,
            read_options=pacsv.ReadOptions(encoding="utf8"),
            parse_options=pacsv.ParseOptions(delimiter=delimiter),
            convert_options=pacsv.ConvertOptions(
                column_types={name: pa.string() for name in header},
                null_values=list(_MISSING_TOKENS),
                strings_can_be_null=True,
            ),
        )
    except (OSError, pa.ArrowException) as error:
        raise ConversionError(f"cannot parse tabular dataset {path}: {error}") from error


def _observed_strings(column: pa.ChunkedArray) -> list[str]:
    return [str(value) for value in column.to_pylist() if value is not None]


def _parse_number(value: str) -> float:
    parsed = float(value.strip())
    if not math.isfinite(parsed):
        raise ValueError("numeric values must be finite")
    return parsed


def _infer_kind(values: Sequence[str]) -> ColumnKind:
    normalized = {value.strip().lower() for value in values}
    if normalized and normalized <= _TRUE_TOKENS | _FALSE_TOKENS:
        return ColumnKind.BOOLEAN
    try:
        for value in values:
            _parse_number(value)
    except ValueError:
        return ColumnKind.CATEGORICAL
    return ColumnKind.NUMERIC


def _row_identifier_candidate(name: str, values: Sequence[str], row_count: int) -> bool:
    return (
        name.strip().lower() in _IDENTIFIER_NAMES
        and len(values) == row_count
        and len(set(values)) == row_count
    )


def _array(values: Sequence[str | None], kind: ColumnKind) -> pa.Array:
    if kind is ColumnKind.NUMERIC:
        parsed = [None if value is None else _parse_number(value) for value in values]
        return pa.array(parsed, type=pa.float64())
    if kind is ColumnKind.BOOLEAN:
        parsed_boolean: list[bool | None] = []
        for value in values:
            if value is None:
                parsed_boolean.append(None)
                continue
            normalized = value.strip().lower()
            if normalized in _TRUE_TOKENS:
                parsed_boolean.append(True)
            elif normalized in _FALSE_TOKENS:
                parsed_boolean.append(False)
            else:
                raise ConversionError(f"value {value!r} is not boolean")
        return pa.array(parsed_boolean, type=pa.bool_())
    return pa.array(values, type=pa.string())


@dataclass(frozen=True, slots=True)
class TabularInputAdapter:
    delimiter: str
    name: str
    title: str
    extension: str

    @property
    def description(self) -> InputAdapterDescription:
        return InputAdapterDescription(
            name=self.name,
            title=self.title,
            description=(
                "Imports a header-based table, infers semantic column types, and requires "
                "the proposed mapping to be confirmed."
            ),
            files=(
                AdapterFileSlot(
                    name="data",
                    title="Data table",
                    description=f"One {self.title} file containing the source matrix.",
                    accepted_extensions=(self.extension,),
                ),
            ),
            requires_confirmation=True,
        )

    def inspect(
        self,
        files: Mapping[str, Path],
        *,
        parameters: Mapping[str, str | int | float | bool] | None = None,
        identifier: str,
        workspace: Path,
    ) -> DatasetImportPreview:
        del parameters, workspace
        path = files["data"]
        table = _tabular_table(path, self.delimiter)
        if table.num_rows < 1:
            raise ConversionError("tabular dataset must contain at least one data row")
        proposals: list[WebColumnProposal] = []
        for index, name in enumerate(table.column_names):
            column = table.column(index)
            values = _observed_strings(column)
            kind = _infer_kind(values)
            proposals.append(
                WebColumnProposal(
                    source_index=index,
                    name=name,
                    inferred_kind=kind,
                    selected_kind=kind,
                    missing_ratio=column.null_count / table.num_rows,
                    sample_values=tuple(dict.fromkeys(values[:32]))[:5],
                    is_row_identifier=(
                        index == 0 and _row_identifier_candidate(name, values, table.num_rows)
                    ),
                )
            )
        output_columns = sum(not proposal.is_row_identifier for proposal in proposals)
        if output_columns < 1:
            raise ConversionError("at least one non-identifier column is required")
        return DatasetImportPreview(
            identifier=normalized_identifier(identifier),
            adapter=self.name,
            row_count=table.num_rows,
            column_count=output_columns,
            columns=tuple(proposals),
            confirmation_required=True,
        )

    def convert(
        self,
        files: Mapping[str, Path],
        *,
        identifier: str,
        columns: Sequence[WebColumnProposal],
        parameters: Mapping[str, str | int | float | bool] | None = None,
        adapter_configuration: Mapping[str, object] | None = None,
        destination: Path,
        workspace: Path,
    ) -> Path:
        del parameters, adapter_configuration, workspace
        table = _tabular_table(files["data"], self.delimiter)
        if len(columns) != table.num_columns:
            raise ConversionError("confirmed column mapping does not match the uploaded table")
        by_index = {column.source_index: column for column in columns}
        if set(by_index) != set(range(table.num_columns)):
            raise ConversionError("confirmed source column indices must be contiguous")
        identifiers: Sequence[str] | None = None
        selected_arrays: list[pa.Array] = []
        metadata: list[ColumnMetadata] = []
        for source_index in range(table.num_columns):
            proposal = by_index[source_index]
            values = table.column(source_index).to_pylist()
            if proposal.is_row_identifier:
                if identifiers is not None:
                    raise ConversionError("only one row identifier column may be selected")
                if any(value is None for value in values):
                    raise ConversionError("row identifiers cannot contain missing values")
                identifiers = [str(value) for value in values]
                continue
            array = _array(values, proposal.selected_kind)
            output_index = len(metadata)
            categories = (
                tuple(dict.fromkeys(str(value) for value in values if value is not None))
                if proposal.selected_kind is ColumnKind.CATEGORICAL
                else ()
            )
            metadata.append(
                ColumnMetadata(
                    index=output_index,
                    name=proposal.name,
                    kind=proposal.selected_kind,
                    categories=categories,
                )
            )
            selected_arrays.append(array)
        if not metadata:
            raise ConversionError("at least one data column must remain after mapping")
        output_table = pa.table(
            {column.name: array for column, array in zip(metadata, selected_arrays, strict=True)}
        )
        DatasetBundleWriter().write(
            destination,
            identifier=normalized_identifier(identifier),
            table=output_table,
            columns=tuple(metadata),
            row_identifiers=identifiers,
        )
        return destination


def _safe_extract(source: Path, destination: Path, *, maximum_bytes: int) -> Path:
    total = 0
    destination.mkdir(parents=True, exist_ok=False)
    resolved_destination = destination.resolve()
    with zipfile.ZipFile(source) as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            mode = member.external_attr >> 16
            if (
                member_path.is_absolute()
                or ".." in member_path.parts
                or (member.is_dir() and member.file_size)
                or stat.S_ISLNK(mode)
            ):
                raise ConversionError("DatasetBundle ZIP contains an unsafe path")
            total += member.file_size
            if total > maximum_bytes:
                raise ConversionError("DatasetBundle ZIP exceeds the expanded size limit")
            target = (destination / member_path).resolve()
            if not target.is_relative_to(resolved_destination):
                raise ConversionError("DatasetBundle ZIP escapes its extraction directory")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source_stream, target.open("wb") as target_stream:
                shutil.copyfileobj(source_stream, target_stream)
    candidates = (
        [destination]
        if (destination / "dataset.yaml").is_file()
        else [
            item
            for item in destination.iterdir()
            if item.is_dir() and (item / "dataset.yaml").is_file()
        ]
    )
    if len(candidates) != 1:
        raise ConversionError("DatasetBundle ZIP must contain exactly one canonical bundle")
    return candidates[0]


@dataclass(frozen=True, slots=True)
class DatasetBundleZipAdapter:
    maximum_expanded_bytes: int

    @property
    def description(self) -> InputAdapterDescription:
        return InputAdapterDescription(
            name="dataset_bundle",
            title="SALVI DatasetBundle",
            description="Imports one canonical, checksummed DatasetBundle ZIP.",
            files=(
                AdapterFileSlot(
                    name="bundle",
                    title="DatasetBundle ZIP",
                    description="A ZIP containing dataset.yaml and its declared artifacts.",
                    accepted_extensions=(".zip",),
                ),
            ),
        )

    def _extract(self, files: Mapping[str, Path], workspace: Path) -> Path:
        extracted = workspace / "extracted"
        if extracted.exists():
            shutil.rmtree(extracted)
        return _safe_extract(
            files["bundle"],
            extracted,
            maximum_bytes=self.maximum_expanded_bytes,
        )

    def inspect(
        self,
        files: Mapping[str, Path],
        *,
        parameters: Mapping[str, str | int | float | bool] | None = None,
        identifier: str,
        workspace: Path,
    ) -> DatasetImportPreview:
        del identifier, parameters
        bundle = self._extract(files, workspace)
        return canonical_bundle_preview(bundle, adapter="dataset_bundle")

    def convert(
        self,
        files: Mapping[str, Path],
        *,
        identifier: str,
        columns: Sequence[WebColumnProposal],
        parameters: Mapping[str, str | int | float | bool] | None = None,
        adapter_configuration: Mapping[str, object] | None = None,
        destination: Path,
        workspace: Path,
    ) -> Path:
        del identifier, columns, parameters, adapter_configuration
        bundle = self._extract(files, workspace)
        try:
            DatasetBundleReader().read_manifest(bundle)
        except ArtifactError as error:
            raise ConversionError(str(error)) from error
        if destination.exists():
            raise ConversionError(f"dataset destination already exists: {destination}")
        shutil.copytree(bundle, destination)
        return destination


def built_in_adapters(maximum_expanded_bytes: int) -> tuple[InputAdapter, ...]:
    return (
        DatasetBundleZipAdapter(maximum_expanded_bytes=maximum_expanded_bytes),
        TabularInputAdapter(delimiter=",", name="csv", title="CSV", extension=".csv"),
        TabularInputAdapter(delimiter="\t", name="tsv", title="TSV", extension=".tsv"),
    )


__all__ = [
    "DatasetBundleZipAdapter",
    "TabularInputAdapter",
    "built_in_adapters",
    "canonical_bundle_preview",
    "normalized_identifier",
]
