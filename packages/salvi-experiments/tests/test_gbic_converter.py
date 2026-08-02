from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from salvi.domain import ColumnKind, PatternKind
from salvi.exceptions import ConversionError
from salvi.infrastructure.dataset_bundle import DatasetBundleReader
from salvi_experiments.cli import main
from salvi_experiments.interop import GbicConverter


def _pattern(name: str) -> dict[str, str]:
    return {
        "RowPattern": name,
        "ColumnPattern": "None",
        "PlaidCoherency": "No Overlapping",
    }


def _simple(kind: str, pattern: str, rows: list[int], columns: list[int]) -> dict[str, Any]:
    return {
        "Type": kind,
        "X": rows,
        "Y": columns,
        "#rows": len(rows),
        "#columns": len(columns),
        "%Noise": "0",
        "%Errors": "0",
        "%Missings": "0",
        "Data": [["must not be copied"]],
        **_pattern(pattern),
    }


def _write_pair(
    directory: Path,
    identifier: str,
    header: list[str],
    rows: list[list[str]],
    ground_truth: dict[str, Any],
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    data = directory / f"{identifier}_data.csv"
    with data.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["X", *header])
        for index, values in enumerate(rows):
            writer.writerow([f"subject-{index}", *values])
    (directory / f"{identifier}_bics.json").write_text(json.dumps(ground_truth), encoding="utf-8")
    return data


def test_gbic_converter_round_trips_all_supported_pattern_data(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_pair(
        source / "constant",
        "constant",
        ["n0", "n1"],
        [["1", "2,5"], ["1", "NA"], ["1", "4"]],
        {
            "#DatasetRows": 3,
            "#DatasetColumns": 2,
            "#DatasetMinValue": 0,
            "#DatasetMaxValue": 4,
            "biclusters": {"0": _simple("Numeric", "Constant", [0, 1], [0, 1])},
        },
    )
    _write_pair(
        source / "additive",
        "additive",
        ["n0", "n1"],
        [["1", "10"], ["2", "11"], ["3", "12"]],
        {
            "#DatasetRows": 3,
            "#DatasetColumns": 2,
            "#DatasetMinValue": 0,
            "#DatasetMaxValue": 12,
            "biclusters": {"0": _simple("Numeric", "Additive", [0, 1, 2], [0, 1])},
        },
    )
    _write_pair(
        source / "multiplicative",
        "multiplicative",
        ["n0", "n1"],
        [["1", "3"], ["2", "6"], ["4", "12"]],
        {
            "#DatasetRows": 3,
            "#DatasetColumns": 2,
            "#DatasetMinValue": 0,
            "#DatasetMaxValue": 12,
            "biclusters": {
                "0": _simple(
                    "Numeric",
                    "Multiplicative",
                    [0, 1, 2],
                    [0, 1],
                )
            },
        },
    )
    _write_pair(
        source / "split_multiplicative",
        "split_multiplicative",
        ["n0", "n1"],
        [["1", "3"], ["2", "6"], ["4", "12"]],
        {
            "#DatasetRows": 3,
            "#DatasetColumns": 2,
            "#DatasetNumericColumns": 2,
            "#DatasetSymbolicColumns": 0,
            "#DatasetMinValue": 0,
            "#DatasetMaxValue": 12,
            "NumericBiclusters": {
                "0": _simple(
                    "Numeric",
                    "Multiplicative",
                    [0, 1, 2],
                    [0, 1],
                )
            },
            "SymbolicBiclusters": {},
            "MixedBiclusters": {},
        },
    )
    mixed = {
        "Type": "Mixed",
        "Rows": [0, 2],
        "NumericColumns": [0, 1],
        "SymbolicColumns": [2],
        "#rows": 2,
        "#columns": 3,
        "%Noise": "1",
        "%Errors": "2",
        "%Missings": "3,5",
        "NumericProperties": _pattern("Additive"),
        "SymbolicProperties": _pattern("Constant"),
        "Data": [["must not be copied"]],
    }
    _write_pair(
        source / "mixed",
        "mixed",
        ["n0", "n1", "group"],
        [["1", "10", "case"], ["2", "11", "?"], ["3", "12", "control"]],
        {
            "#DatasetRows": 3,
            "#DatasetColumns": 3,
            "#DatasetNumericColumns": 2,
            "#DatasetSymbolicColumns": 1,
            "#DatasetAlphabet": ["case", "control", "unused"],
            "NumericBiclusters": {},
            "SymbolicBiclusters": {},
            "MixedBiclusters": {"7": mixed},
        },
    )

    destination = GbicConverter().convert(source, tmp_path / "converted")
    manifest = json.loads((destination / "conversion-manifest.json").read_text())
    assert [item["identifier"] for item in manifest["datasets"]] == [
        "additive",
        "constant",
        "mixed",
        "multiplicative",
        "split_multiplicative",
    ]

    reader = DatasetBundleReader()
    constant = reader.load(destination / "constant" / "constant")
    assert constant.row_identifiers.to_pylist() == ["subject-0", "subject-1", "subject-2"]
    assert constant.table.column("n1").to_pylist() == [2.5, None, 4.0]
    constant_truth = reader.read_ground_truth(destination / "constant" / "constant")
    assert constant_truth is not None
    assert {item.pattern for item in constant_truth.biclusters[0].column_patterns} == {
        PatternKind.CONSTANT
    }

    additive_truth = reader.read_ground_truth(destination / "additive" / "additive")
    assert additive_truth is not None
    assert {item.pattern for item in additive_truth.biclusters[0].column_patterns} == {
        PatternKind.ADDITIVE
    }
    multiplicative_truth = reader.read_ground_truth(
        destination / "multiplicative" / "multiplicative"
    )
    assert multiplicative_truth is not None
    assert {item.pattern for item in multiplicative_truth.biclusters[0].column_patterns} == {
        PatternKind.MULTIPLICATIVE
    }
    split_multiplicative_truth = reader.read_ground_truth(
        destination / "split_multiplicative" / "split_multiplicative"
    )
    assert split_multiplicative_truth is not None
    assert {item.pattern for item in split_multiplicative_truth.biclusters[0].column_patterns} == {
        PatternKind.MULTIPLICATIVE
    }

    mixed_bundle = destination / "mixed" / "mixed"
    mixed_dataset = reader.read(mixed_bundle)
    assert tuple(column.kind for column in mixed_dataset.columns) == (
        ColumnKind.NUMERIC,
        ColumnKind.NUMERIC,
        ColumnKind.CATEGORICAL,
    )
    assert mixed_dataset.columns[2].categories == ("case", "control", "unused")
    mixed_truth = reader.read_ground_truth(mixed_bundle)
    assert mixed_truth is not None
    assert tuple(item.pattern for item in mixed_truth.biclusters[0].column_patterns) == (
        PatternKind.ADDITIVE,
        PatternKind.ADDITIVE,
        PatternKind.CONSTANT,
    )
    assert mixed_truth.biclusters[0].source_metadata["missings_percent"] == 3.5
    assert "must not be copied" not in (mixed_bundle / "ground-truth.json").read_text()


def test_gbic_conversion_is_atomic_and_cli_exposes_the_adapter(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "source"
    data = _write_pair(
        source,
        "numeric",
        ["n0"],
        [["1"], ["2"]],
        {
            "#DatasetRows": 2,
            "#DatasetColumns": 1,
            "#DatasetMinValue": 0,
            "biclusters": {"0": _simple("Numeric", "Constant", [0, 1], [0])},
        },
    )
    destination = tmp_path / "converted"
    assert main(["convert", "gbic", str(data), str(destination)]) == 0
    assert json.loads(capsys.readouterr().out)["converter"] == "gbic"
    assert (destination / "numeric" / "dataset.yaml").is_file()
    with pytest.raises(ConversionError, match="destination already exists"):
        GbicConverter().convert(source, destination)
    assert GbicConverter(overwrite=True).convert(data, destination) == destination

    invalid_destination = tmp_path / "invalid-output"
    data.with_name("numeric_bics.json").unlink()
    with pytest.raises(ConversionError, match="paired ground truth"):
        GbicConverter().convert(source, invalid_destination)
    assert not invalid_destination.exists()


def test_gbic_converter_rejects_destination_inside_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ConversionError, match="inside"):
        GbicConverter().convert(source, source / "converted")


@pytest.mark.parametrize(
    "pattern, kind, split, values, message",
    (
        ("Additive", "Symbolic", True, [["A"], ["B"]], "only support"),
        ("Constant", "Numeric", False, [["bad"], ["2"]], "not numeric"),
        ("Constant", "Numeric", False, [["1"]], "contains 1 rows"),
    ),
)
def test_gbic_converter_rejects_unsupported_patterns_and_malformed_matrices(
    tmp_path: Path,
    pattern: str,
    kind: str,
    split: bool,
    values: list[list[str]],
    message: str,
) -> None:
    source = tmp_path / message.replace(" ", "-")
    if split:
        ground_truth = {
            "#DatasetRows": 2,
            "#DatasetColumns": 1,
            "#DatasetNumericColumns": 0,
            "#DatasetSymbolicColumns": 1,
            "#DatasetAlphabet": ["A", "B"],
            "NumericBiclusters": {},
            "SymbolicBiclusters": {
                "0": _simple(kind, pattern, [0, 1], [0]),
            },
            "MixedBiclusters": {},
        }
    else:
        ground_truth = {
            "#DatasetRows": 2,
            "#DatasetColumns": 1,
            "#DatasetMinValue": 0,
            "biclusters": {"0": _simple(kind, pattern, [0, 1], [0])},
        }
    _write_pair(source, "invalid", ["value"], values, ground_truth)
    with pytest.raises(ConversionError, match=message):
        GbicConverter().convert(source, tmp_path / f"output-{source.name}")


def test_gbic_converter_rejects_missing_sources_and_source_replacement(tmp_path: Path) -> None:
    with pytest.raises(ConversionError, match="does not exist"):
        GbicConverter().convert(tmp_path / "missing", tmp_path / "output")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ConversionError, match="no G-Bic"):
        GbicConverter().convert(empty, tmp_path / "empty-output")

    source = _write_pair(
        tmp_path / "source",
        "dataset",
        ["value"],
        [["1"], ["2"]],
        {
            "#DatasetRows": 2,
            "#DatasetColumns": 1,
            "#DatasetMinValue": 0,
            "biclusters": {"0": _simple("Numeric", "Constant", [0, 1], [0])},
        },
    )
    with pytest.raises(ConversionError, match="replace"):
        GbicConverter(overwrite=True).convert(source, source)
