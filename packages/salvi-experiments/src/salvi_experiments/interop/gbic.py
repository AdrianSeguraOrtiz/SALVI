"""G-Bic to canonical DatasetBundle conversion."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import pyarrow as pa
from pydantic import BaseModel, ConfigDict

from salvi.domain.enums import ColumnKind, PatternKind
from salvi.domain.models import ColumnMetadata
from salvi.exceptions import ArtifactError, ConversionError
from salvi.infrastructure.dataset_bundle import DatasetBundleWriter
from salvi.infrastructure.files import atomic_directory, sha256_file
from salvi.infrastructure.ground_truth import (
    GroundTruth,
    GroundTruthBicluster,
    GroundTruthColumnPattern,
)

DATA_SUFFIXES = ("_data.csv", "_data.tsv")
MISSING_TOKENS = frozenset({"", "na", "n/a", "nan", "null", "?"})


class GbicConverterConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    overwrite: bool = False


@dataclass(frozen=True, slots=True)
class _ParsedGroundTruth:
    schema: Literal["legacy", "split"]
    rows: int
    columns: int
    numeric_columns: int
    categorical_alphabet: tuple[str, ...]
    biclusters: tuple[GroundTruthBicluster, ...]


def _error(path: Path, message: str) -> ConversionError:
    return ConversionError(f"{path}: {message}")


def _mapping(value: object, path: Path, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise _error(path, f"{label} must be a JSON object")
    return value


def _integer(mapping: Mapping[str, Any], key: str, path: Path, *, positive: bool) -> int:
    value = mapping.get(key)
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise _error(path, f"{key} must be a {qualifier} integer")
    return value


def _text(mapping: Mapping[str, Any], key: str, path: Path) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise _error(path, f"{key} must be a non-empty string")
    return value.strip()


def _percentage(mapping: Mapping[str, Any], key: str, path: Path) -> float:
    try:
        value = _decimal(mapping.get(key, 0.0))
    except (TypeError, ValueError):
        raise _error(path, f"{key} must be numeric") from None
    if not math.isfinite(value) or not 0.0 <= value <= 100.0:
        raise _error(path, f"{key} must be finite and between 0 and 100")
    return value


def _decimal(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean values are not numeric")
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return float(value.strip().replace(",", "."))
    raise TypeError(f"unsupported numeric value: {type(value).__name__}")


def _indices(value: object, path: Path, label: str, upper_bound: int) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise _error(path, f"{label} must be a non-empty array")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise _error(path, f"{label} must contain integer indices")
        if item < 0 or item >= upper_bound:
            raise _error(path, f"{label} index {item} is outside [0, {upper_bound})")
        result.append(item)
    if len(set(result)) != len(result):
        raise _error(path, f"{label} contains duplicate indices")
    return tuple(sorted(result))


def _pattern(properties: Mapping[str, Any], path: Path, *, numeric: bool) -> PatternKind:
    raw = _text(properties, "RowPattern", path).upper()
    try:
        pattern = PatternKind(raw)
    except ValueError as error:
        raise _error(path, f"unsupported G-Bic row pattern: {raw}") from error
    if not numeric and pattern is not PatternKind.CONSTANT:
        raise _error(path, "categorical G-Bic columns only support the CONSTANT pattern")
    return pattern


def _source_pattern(properties: Mapping[str, Any], path: Path) -> dict[str, str]:
    return {
        "row_pattern": _text(properties, "RowPattern", path),
        "column_pattern": _text(properties, "ColumnPattern", path),
        "plaid_coherency": _text(properties, "PlaidCoherency", path),
    }


def _validate_declared_shape(
    source: Mapping[str, Any],
    rows: tuple[int, ...],
    columns: tuple[int, ...],
    path: Path,
) -> None:
    if "#rows" in source and _integer(source, "#rows", path, positive=False) != len(rows):
        raise _error(path, "#rows does not match the row index count")
    if "#columns" in source and _integer(source, "#columns", path, positive=False) != len(columns):
        raise _error(path, "#columns does not match the column index count")


def _simple_bicluster(
    identifier: str,
    source: Mapping[str, Any],
    *,
    expected_type: Literal["numeric", "symbolic"],
    rows_bound: int,
    columns_bound: int,
    numeric_count: int,
    path: Path,
) -> GroundTruthBicluster:
    location = Path(f"{path}:{expected_type}:{identifier}")
    declared = _text(source, "Type", location).lower()
    if declared != expected_type:
        raise _error(location, f"Type must be {expected_type.title()}")
    rows = _indices(source.get("X"), location, "X", rows_bound)
    columns = _indices(source.get("Y"), location, "Y", columns_bound)
    numeric = expected_type == "numeric"
    if numeric and any(index >= numeric_count for index in columns):
        raise _error(location, "numeric bicluster contains a categorical column")
    if not numeric and any(index < numeric_count for index in columns):
        raise _error(location, "symbolic bicluster contains a numeric column")
    _validate_declared_shape(source, rows, columns, location)
    pattern = _pattern(source, location, numeric=numeric)
    return GroundTruthBicluster(
        identifier=identifier,
        row_indices=rows,
        column_indices=columns,
        column_patterns=tuple(
            GroundTruthColumnPattern(column_index=index, pattern=pattern) for index in columns
        ),
        source_type=expected_type,
        source_metadata={
            "pattern": _source_pattern(source, location),
            "noise_percent": _percentage(source, "%Noise", location),
            "errors_percent": _percentage(source, "%Errors", location),
            "missings_percent": _percentage(source, "%Missings", location),
        },
    )


def _mixed_bicluster(
    identifier: str,
    source: Mapping[str, Any],
    *,
    rows_bound: int,
    columns_bound: int,
    numeric_count: int,
    path: Path,
) -> GroundTruthBicluster:
    location = Path(f"{path}:mixed:{identifier}")
    if _text(source, "Type", location).lower() != "mixed":
        raise _error(location, "Type must be Mixed")
    rows = _indices(source.get("Rows"), location, "Rows", rows_bound)
    numeric_columns = _indices(
        source.get("NumericColumns"), location, "NumericColumns", columns_bound
    )
    categorical_columns = _indices(
        source.get("SymbolicColumns"), location, "SymbolicColumns", columns_bound
    )
    if any(index >= numeric_count for index in numeric_columns):
        raise _error(location, "NumericColumns contains a categorical column")
    if any(index < numeric_count for index in categorical_columns):
        raise _error(location, "SymbolicColumns contains a numeric column")
    if set(numeric_columns).intersection(categorical_columns):
        raise _error(location, "numeric and categorical column sets overlap")
    columns = tuple(sorted((*numeric_columns, *categorical_columns)))
    _validate_declared_shape(source, rows, columns, location)
    numeric_properties = _mapping(source.get("NumericProperties"), location, "NumericProperties")
    categorical_properties = _mapping(
        source.get("SymbolicProperties"), location, "SymbolicProperties"
    )
    assignments = {
        **{
            index: _pattern(numeric_properties, location, numeric=True) for index in numeric_columns
        },
        **{
            index: _pattern(categorical_properties, location, numeric=False)
            for index in categorical_columns
        },
    }
    return GroundTruthBicluster(
        identifier=identifier,
        row_indices=rows,
        column_indices=columns,
        column_patterns=tuple(
            GroundTruthColumnPattern(column_index=index, pattern=assignments[index])
            for index in columns
        ),
        source_type="mixed",
        source_metadata={
            "numeric_pattern": _source_pattern(numeric_properties, location),
            "categorical_pattern": _source_pattern(categorical_properties, location),
            "noise_percent": _percentage(source, "%Noise", location),
            "errors_percent": _percentage(source, "%Errors", location),
            "missings_percent": _percentage(source, "%Missings", location),
        },
    )


def _sort_identifier(identifier: str) -> tuple[int, int | str]:
    return (0, int(identifier)) if identifier.isdigit() else (1, identifier)


def _parse_ground_truth(path: Path) -> _ParsedGroundTruth:
    try:
        root = _mapping(json.loads(path.read_text(encoding="utf-8")), path, "ground truth")
    except (OSError, json.JSONDecodeError) as error:
        raise _error(path, f"cannot read ground truth: {error}") from error
    rows = _integer(root, "#DatasetRows", path, positive=True)
    columns = _integer(root, "#DatasetColumns", path, positive=True)
    schema: Literal["legacy", "split"] = "legacy" if "biclusters" in root else "split"

    if schema == "split":
        numeric_count = _integer(root, "#DatasetNumericColumns", path, positive=False)
        symbolic_count = _integer(root, "#DatasetSymbolicColumns", path, positive=False)
        if numeric_count + symbolic_count != columns:
            raise _error(path, "numeric and symbolic column counts do not match the dataset")
        group_specs: tuple[tuple[str, str], ...] = (
            ("NumericBiclusters", "numeric"),
            ("SymbolicBiclusters", "symbolic"),
            ("MixedBiclusters", "mixed"),
        )
    else:
        legacy = _mapping(root.get("biclusters"), path, "biclusters")
        kinds = {
            _text(_mapping(value, path, f"biclusters.{key}"), "Type", path).lower()
            for key, value in legacy.items()
        }
        if kinds == {"numeric"}:
            numeric_count = columns
        elif kinds == {"symbolic"}:
            numeric_count = 0
        elif "#DatasetMinValue" in root and "#DatasetAlphabet" not in root:
            numeric_count = columns
        elif "#DatasetAlphabet" in root and "#DatasetMinValue" not in root:
            numeric_count = 0
        else:
            raise _error(path, "cannot determine legacy G-Bic column kinds")
        group_specs = (("biclusters", "dynamic"),)

    raw_alphabet = root.get("#DatasetAlphabet", [])
    if not isinstance(raw_alphabet, list) or any(
        not isinstance(item, str) for item in raw_alphabet
    ):
        raise _error(path, "#DatasetAlphabet must be an array of strings")
    alphabet = tuple(dict.fromkeys(item for item in raw_alphabet if item))
    biclusters: list[GroundTruthBicluster] = []
    seen: set[str] = set()
    for group_name, expected in group_specs:
        group = _mapping(root.get(group_name, {}), path, group_name)
        for raw_identifier, raw_bicluster in group.items():
            identifier = str(raw_identifier)
            if identifier in seen:
                raise _error(path, f"duplicate bicluster identifier {identifier}")
            seen.add(identifier)
            source = _mapping(raw_bicluster, path, f"{group_name}.{identifier}")
            source_type = _text(source, "Type", path).lower() if expected == "dynamic" else expected
            if source_type == "mixed":
                bicluster = _mixed_bicluster(
                    identifier,
                    source,
                    rows_bound=rows,
                    columns_bound=columns,
                    numeric_count=numeric_count,
                    path=path,
                )
            elif source_type in {"numeric", "symbolic"}:
                bicluster = _simple_bicluster(
                    identifier,
                    source,
                    expected_type=cast(Literal["numeric", "symbolic"], source_type),
                    rows_bound=rows,
                    columns_bound=columns,
                    numeric_count=numeric_count,
                    path=path,
                )
            else:
                raise _error(path, f"unsupported G-Bic bicluster type: {source_type}")
            biclusters.append(bicluster)
    biclusters.sort(key=lambda item: _sort_identifier(item.identifier))
    return _ParsedGroundTruth(
        schema=schema,
        rows=rows,
        columns=columns,
        numeric_columns=numeric_count,
        categorical_alphabet=alphabet,
        biclusters=tuple(biclusters),
    )


def _dataset_identifier(path: Path) -> str:
    for suffix in DATA_SUFFIXES:
        if path.name.endswith(suffix):
            identifier = path.name[: -len(suffix)]
            if identifier:
                return identifier
    raise _error(path, f"data filename must end with one of {DATA_SUFFIXES}")


def _discover(source: Path) -> tuple[Path, tuple[Path, ...]]:
    resolved = source.resolve()
    if resolved.is_file():
        _dataset_identifier(resolved)
        return resolved.parent, (resolved,)
    if not resolved.is_dir():
        raise _error(resolved, "source path does not exist")
    files = tuple(
        sorted(
            path
            for path in resolved.rglob("*")
            if path.is_file() and any(path.name.endswith(suffix) for suffix in DATA_SUFFIXES)
        )
    )
    if not files:
        raise _error(resolved, "no G-Bic data files were found")
    return resolved, files


def _read_matrix(
    path: Path,
    parsed: _ParsedGroundTruth,
) -> tuple[pa.Table, tuple[ColumnMetadata, ...], tuple[str, ...]]:
    try:
        stream = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as error:
        raise _error(path, f"cannot read matrix: {error}") from error
    with stream:
        reader = csv.reader(stream, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration:
            raise _error(path, "matrix is empty") from None
        if len(header) != parsed.columns + 1 or header[0].strip().lower() != "x":
            raise _error(path, "matrix header must contain X followed by every data column")
        names = tuple(item.strip() for item in header[1:])
        if any(not name for name in names) or len(set(names)) != len(names):
            raise _error(path, "data column names must be non-empty and unique")
        row_identifiers: list[str] = []
        values: list[list[float | str | None]] = [[] for _ in names]
        category_orders: list[dict[str, None]] = [dict() for _ in names]
        for row_number, row in enumerate(reader, start=1):
            if len(row) != parsed.columns + 1:
                raise _error(path, f"row {row_number} has an invalid field count")
            identifier = row[0].strip()
            if not identifier:
                raise _error(path, f"row {row_number} has a blank identifier")
            row_identifiers.append(identifier)
            for column_index, raw in enumerate(row[1:]):
                stripped = raw.strip()
                if stripped.lower() in MISSING_TOKENS:
                    values[column_index].append(None)
                elif column_index < parsed.numeric_columns:
                    try:
                        numeric = _decimal(stripped)
                    except ValueError:
                        raise _error(
                            path,
                            f"row {row_number}, column {column_index} is not numeric: {raw!r}",
                        ) from None
                    if not math.isfinite(numeric):
                        raise _error(path, "observed numeric values must be finite")
                    values[column_index].append(numeric)
                else:
                    values[column_index].append(stripped)
                    category_orders[column_index][stripped] = None

    if len(row_identifiers) != parsed.rows:
        raise _error(path, f"matrix contains {len(row_identifiers)} rows; expected {parsed.rows}")
    if len(set(row_identifiers)) != len(row_identifiers):
        raise _error(path, "row identifiers must be unique")

    arrays: dict[str, pa.Array] = {}
    metadata: list[ColumnMetadata] = []
    for index, name in enumerate(names):
        if index < parsed.numeric_columns:
            arrays[name] = pa.array(values[index], type=pa.float64())
            metadata.append(ColumnMetadata(index=index, name=name, kind=ColumnKind.NUMERIC))
        else:
            categories = tuple(category_orders[index])
            categories = tuple(dict.fromkeys((*categories, *parsed.categorical_alphabet)))
            if not categories:
                raise _error(
                    path,
                    f"all-missing categorical column {name!r} has no declared G-Bic alphabet",
                )
            arrays[name] = pa.array(values[index], type=pa.string())
            metadata.append(
                ColumnMetadata(
                    index=index,
                    name=name,
                    kind=ColumnKind.CATEGORICAL,
                    categories=categories,
                )
            )
    return pa.table(arrays), tuple(metadata), tuple(row_identifiers)


@dataclass(frozen=True, slots=True)
class GbicConverter:
    overwrite: bool = False
    component_name: str = "gbic"
    provides: frozenset[str] = frozenset({"canonical-dataset-bundle"})
    requires: frozenset[str] = frozenset()

    def convert(self, source: Path, destination: Path) -> Path:
        resolved_source = source.resolve()
        target = destination.resolve()
        if resolved_source.is_dir() and target.is_relative_to(resolved_source):
            raise _error(target, "destination cannot be inside the source directory tree")
        if resolved_source.is_file() and target == resolved_source:
            raise _error(target, "destination cannot replace the source data file")
        source_root, data_files = _discover(resolved_source)

        records: list[dict[str, str]] = []
        try:
            with atomic_directory(target, overwrite=self.overwrite) as temporary:
                for data_path in data_files:
                    identifier = _dataset_identifier(data_path)
                    ground_truth_path = data_path.with_name(f"{identifier}_bics.json")
                    if not ground_truth_path.is_file():
                        raise _error(
                            data_path, f"paired ground truth is missing: {ground_truth_path}"
                        )
                    parsed = _parse_ground_truth(ground_truth_path)
                    table, columns, row_identifiers = _read_matrix(data_path, parsed)
                    relative_parent = data_path.parent.relative_to(source_root)
                    bundle_relative = relative_parent / identifier
                    source_data = data_path.relative_to(source_root).as_posix()
                    source_ground_truth = ground_truth_path.relative_to(source_root).as_posix()
                    provenance = {
                        "format": "G-Bic",
                        "source_schema": parsed.schema,
                        "source_data": source_data,
                        "source_ground_truth": source_ground_truth,
                        "source_data_sha256": sha256_file(data_path),
                        "source_ground_truth_sha256": sha256_file(ground_truth_path),
                    }
                    ground_truth = GroundTruth(
                        dataset_identifier=identifier,
                        row_count=parsed.rows,
                        column_count=parsed.columns,
                        biclusters=parsed.biclusters,
                        provenance=provenance,
                    )
                    DatasetBundleWriter().write(
                        temporary / bundle_relative,
                        identifier=identifier,
                        table=table,
                        columns=columns,
                        row_identifiers=row_identifiers,
                        ground_truth=ground_truth,
                    )
                    records.append(
                        {
                            "identifier": identifier,
                            "bundle": bundle_relative.as_posix(),
                            "source_data": source_data,
                            "source_ground_truth": source_ground_truth,
                        }
                    )
                (temporary / "conversion-manifest.json").write_text(
                    json.dumps(
                        {"schema_version": 1, "converter": "gbic", "datasets": records},
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
        except ConversionError:
            raise
        except ArtifactError as error:
            raise _error(target, str(error)) from error
        return target


__all__ = ["GbicConverter", "GbicConverterConfiguration"]
