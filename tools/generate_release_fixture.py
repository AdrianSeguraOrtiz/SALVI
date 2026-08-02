"""Generate the deterministic heterogeneous dataset used by release profiling."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa

from salvi.domain import ColumnKind, ColumnMetadata
from salvi.infrastructure.dataset_bundle import DatasetBundleWriter


def _masked(values: np.ndarray, missing: np.ndarray, arrow_type: pa.DataType) -> pa.Array:
    return pa.array(values.tolist(), mask=missing.tolist(), type=arrow_type)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "destination",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "generated"
        / "release-dataset",
    )
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()

    rows = 240
    rng = np.random.default_rng(20260724)
    numeric = rng.normal(0.0, 1.0, size=(rows, 8))
    row_effect = np.linspace(-1.5, 1.5, 48)
    numeric[:48, :4] = row_effect[:, None] + np.array([0.0, 0.4, -0.7, 1.1])
    numeric[48:96, 4:] = np.array([2.0, -1.0, 0.5, 3.0])
    boolean = rng.integers(0, 2, size=(rows, 2)).astype(bool)
    categorical = rng.choice(np.array(["control", "case", "other"]), size=(rows, 2))
    boolean[96:144, :] = np.array([True, False])
    categorical[96:144, :] = np.array(["case", "control"])
    missing = rng.random(size=(rows, 12)) < 0.03

    arrays: dict[str, pa.Array] = {}
    columns: list[ColumnMetadata] = []
    for index in range(8):
        name = f"numeric_{index + 1:02d}"
        arrays[name] = _masked(numeric[:, index], missing[:, index], pa.float64())
        columns.append(ColumnMetadata(index=index, name=name, kind=ColumnKind.NUMERIC))
    for offset in range(2):
        index = 8 + offset
        name = f"boolean_{offset + 1:02d}"
        arrays[name] = _masked(boolean[:, offset], missing[:, index], pa.bool_())
        columns.append(ColumnMetadata(index=index, name=name, kind=ColumnKind.BOOLEAN))
    for offset in range(2):
        index = 10 + offset
        name = f"categorical_{offset + 1:02d}"
        arrays[name] = _masked(categorical[:, offset], missing[:, index], pa.string())
        columns.append(
            ColumnMetadata(
                index=index,
                name=name,
                kind=ColumnKind.CATEGORICAL,
                categories=("case", "control", "other"),
            )
        )

    dataset = DatasetBundleWriter().write(
        arguments.destination,
        identifier="salvi-release-profile",
        table=pa.table(arrays),
        columns=tuple(columns),
        row_identifiers=tuple(f"row-{index:04d}" for index in range(rows)),
        overwrite=arguments.overwrite,
    )
    print(dataset.bundle_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
