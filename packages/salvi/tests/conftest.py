from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
import yaml

from salvi.application.context import NamedRandomStreams, QdRunContext, RunContext
from salvi.components.evaluation_policies import MinimumCardinality, MinimumObservedSupport
from salvi.components.parent_selection import RepertoireUniformParentSelection
from salvi.components.preprocessing import RobustNumericScaling
from salvi.domain import ColumnKind, ColumnMetadata
from salvi.domain.prepared import PreparedDataset
from salvi.infrastructure.dataset_bundle import DatasetBundleReader, DatasetBundleWriter
from salvi.infrastructure.ground_truth import GroundTruth
from salvi.patterns import PatternConfiguration


def create_dataset_bundle(
    destination: Path,
    *,
    rows: int = 4,
    columns: int = 3,
) -> Path:
    available = {
        "numeric": pa.array([float(index) for index in range(rows)], type=pa.float64()),
        "boolean": pa.array([index % 2 == 0 for index in range(rows)], type=pa.bool_()),
        "categorical": pa.array(
            ["A" if index % 2 == 0 else "B" for index in range(rows)],
            type=pa.string(),
        ),
    }
    names = tuple(available)[:columns]
    table = pa.table({name: available[name] for name in names})
    metadata = tuple(
        ColumnMetadata(index=index, name=name, kind=kind, categories=categories)
        for index, (name, kind, categories) in enumerate(
            (
                ("numeric", ColumnKind.NUMERIC, ()),
                ("boolean", ColumnKind.BOOLEAN, ()),
                ("categorical", ColumnKind.CATEGORICAL, ("A", "B")),
            )[:columns]
        )
    )
    DatasetBundleWriter().write(
        destination,
        identifier="test-dataset",
        table=table,
        columns=metadata,
        row_identifiers=tuple(f"row-{index}" for index in range(rows)),
        ground_truth=GroundTruth(
            dataset_identifier="test-dataset",
            row_count=rows,
            column_count=columns,
            biclusters=(),
        ),
    )
    return destination


def configuration_mapping(
    dataset: Path,
    output: Path,
    *,
    patterns: list[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run": {"identifier": "test-run", "seed": 7},
        "dataset": {"bundle": str(dataset)},
        "patterns": {
            "allowed": patterns or ["CONSTANT"],
            "min_improvement": 0.1,
            "max_iterations": 25,
            "convergence_tolerance": 1e-6,
        },
        "preprocessing": {
            "source_column_filters": [],
            "missing_values": {"name": "preserve", "parameters": {}},
            "column_augmentations": [],
            "numeric_transformations": [{"name": "robust_numeric_scaling", "parameters": {}}],
        },
        "evaluation": {
            "candidate_validity": {
                "name": "minimum_cardinality",
                "parameters": {"min_rows": 2, "min_columns": 2},
            },
            "observed_support": {
                "name": "minimum_observed_support",
                "parameters": {"min_observed_count": 2, "min_observed_ratio": 0.8},
            },
        },
        "search": {
            "engine": {
                "name": "serial_mome",
                "parameters": {
                    "initial_population_size": 2,
                    "batch_size": 1,
                },
            },
            "objectives": [
                {"name": "internal_coherence", "parameters": {}},
                {
                    "name": "contrast",
                    "parameters": {"min_background_ratio": 0.1},
                },
            ],
            "descriptors": [
                {"name": "row_cardinality", "parameters": {}},
                {"name": "column_cardinality", "parameters": {}},
            ],
            "archive": {
                "name": "deep_grid_mome",
                "parameters": {
                    "axes": [
                        {"descriptor": "row_cardinality", "binning": "EXACT"},
                        {"descriptor": "column_cardinality", "binning": "EXACT"},
                    ],
                    "cell_capacity": 2,
                },
            },
            "parent_selection": {"name": "repertoire_uniform", "parameters": {}},
            "initialization": {"name": "uniform_random", "parameters": {}},
            "emitters": [{"name": "random_move", "parameters": {}}],
            "scheduler": {"name": "first", "parameters": {}},
            "termination": {
                "name": "evaluation_budget",
                "parameters": {"max_evaluations": 2},
            },
        },
        "execution": {
            "executor": {"name": "serial", "parameters": {}},
            "workers": 1,
            "cancellation_grace_seconds": 1,
        },
        "monitoring": {
            "queue_capacity": 64,
            "observers": [{"name": "search_progress", "parameters": {}}],
        },
        "final_selection": None,
        "output": {"directory": str(output), "overwrite": overwrite},
    }


def write_configuration(path: Path, mapping: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def dataset_bundle(tmp_path: Path) -> Path:
    return create_dataset_bundle(tmp_path / "dataset")


@pytest.fixture
def configuration_path(tmp_path: Path, dataset_bundle: Path) -> Path:
    return write_configuration(
        tmp_path / "configuration.yaml",
        configuration_mapping(dataset_bundle, tmp_path / "output"),
    )


@pytest.fixture
def run_context(dataset_bundle: Path) -> RunContext:
    loaded = DatasetBundleReader().load(dataset_bundle)
    prepared = RobustNumericScaling().transform(
        PreparedDataset.from_arrow(loaded.dataset, loaded.table, loaded.row_identifiers)
    )
    return QdRunContext(
        dataset=prepared,
        patterns=PatternConfiguration(),
        random_streams=NamedRandomStreams(7),
        parent_selection_policy=RepertoireUniformParentSelection(),
        candidate_validity_policy=MinimumCardinality(),
        evaluation_support_policy=MinimumObservedSupport(),
    )
