"""Create the small canonical DatasetBundle used by SALVI examples."""

from pathlib import Path

import pyarrow as pa

from salvi.domain import ColumnKind, ColumnMetadata, PatternKind
from salvi.infrastructure.dataset_bundle import DatasetBundleWriter
from salvi.infrastructure.ground_truth import (
    GroundTruth,
    GroundTruthBicluster,
    GroundTruthColumnPattern,
)

root = Path(__file__).resolve().parent
table = pa.table(
    {
        "age": pa.array(
            [38.0, 57.0, 46.0, None, 62.0, 41.0, 53.0, 35.0, 49.0, 68.0],
            type=pa.float64(),
        ),
        "smoker": pa.array(
            [False, True, False, None, True, False, True, False, False, True],
            type=pa.bool_(),
        ),
        "group": pa.array(
            [
                "control",
                "case",
                "control",
                "case",
                "case",
                "control",
                "case",
                "control",
                "control",
                "case",
            ],
            type=pa.string(),
        ),
    }
)
columns = (
    ColumnMetadata(index=0, name="age", kind=ColumnKind.NUMERIC),
    ColumnMetadata(index=1, name="smoker", kind=ColumnKind.BOOLEAN),
    ColumnMetadata(
        index=2,
        name="group",
        kind=ColumnKind.CATEGORICAL,
        categories=("control", "case"),
    ),
)
dataset = DatasetBundleWriter().write(
    root / "example-dataset",
    identifier="example-dataset",
    table=table,
    columns=columns,
    row_identifiers=tuple(f"patient-{index:03d}" for index in range(1, 11)),
    ground_truth=GroundTruth(
        dataset_identifier="example-dataset",
        row_count=10,
        column_count=3,
        biclusters=(
            GroundTruthBicluster(
                identifier="control-profile",
                row_indices=(0, 2, 5, 7, 8),
                column_indices=(1, 2),
                column_patterns=(
                    GroundTruthColumnPattern(
                        column_index=1,
                        pattern=PatternKind.CONSTANT,
                    ),
                    GroundTruthColumnPattern(
                        column_index=2,
                        pattern=PatternKind.CONSTANT,
                    ),
                ),
                source_type="CONSTANT",
            ),
        ),
    ),
    overwrite=True,
)
print(dataset.bundle_path)
