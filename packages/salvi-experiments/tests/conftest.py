from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
import yaml

from salvi import (
    Bicluster,
    BiclusterSetWriter,
    Candidate,
    DatasetBundleWriter,
    Evaluation,
    GroundTruth,
    GroundTruthBicluster,
    GroundTruthColumnPattern,
    PatternKind,
)
from salvi.domain.enums import ColumnKind
from salvi.domain.models import (
    CandidateProvenance,
    ColumnMetadata,
    NamedValue,
    Repertoire,
)
from salvi.domain.prepared import PreparedColumnMetadata


@pytest.fixture
def experiment_dataset(tmp_path: Path) -> Path:
    bundle = tmp_path / "dataset"
    table = pa.table(
        {
            "numeric": pa.array([1.0, 1.1, 0.9, 1.0, 9.0, 10.0, 11.0, 10.0]),
            "boolean": pa.array([True, True, True, True, False, False, False, False]),
            "category": pa.array(["A", "A", "A", "A", "B", "B", "B", "B"]),
        }
    )
    columns = (
        ColumnMetadata(index=0, name="numeric", kind=ColumnKind.NUMERIC),
        ColumnMetadata(index=1, name="boolean", kind=ColumnKind.BOOLEAN),
        ColumnMetadata(
            index=2,
            name="category",
            kind=ColumnKind.CATEGORICAL,
            categories=("A", "B"),
        ),
    )
    truth = GroundTruth(
        dataset_identifier="experiment-dataset",
        row_count=8,
        column_count=3,
        biclusters=(
            GroundTruthBicluster(
                identifier="truth-0",
                row_indices=(0, 1, 2, 3),
                column_indices=(0, 1, 2),
                column_patterns=tuple(
                    GroundTruthColumnPattern(
                        column_index=index,
                        pattern=PatternKind.CONSTANT,
                    )
                    for index in range(3)
                ),
                source_type="CONSTANT",
            ),
        ),
    )
    DatasetBundleWriter().write(
        bundle,
        identifier="experiment-dataset",
        table=table,
        columns=columns,
        ground_truth=truth,
    )
    return bundle


@pytest.fixture
def dataset_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "dataset"
    table = pa.table(
        {
            "numeric": pa.array([0.0, 1.0, 2.0, 3.0], type=pa.float64()),
            "boolean": pa.array([True, False, True, False], type=pa.bool_()),
            "categorical": pa.array(["A", "B", "A", "B"], type=pa.string()),
        }
    )
    columns = (
        ColumnMetadata(index=0, name="numeric", kind=ColumnKind.NUMERIC),
        ColumnMetadata(index=1, name="boolean", kind=ColumnKind.BOOLEAN),
        ColumnMetadata(
            index=2,
            name="categorical",
            kind=ColumnKind.CATEGORICAL,
            categories=("A", "B"),
        ),
    )
    DatasetBundleWriter().write(
        bundle,
        identifier="test-dataset",
        table=table,
        columns=columns,
        row_identifiers=tuple(f"row-{index}" for index in range(4)),
        ground_truth=GroundTruth(
            dataset_identifier="test-dataset",
            row_count=4,
            column_count=3,
            biclusters=(),
        ),
    )
    return bundle


@pytest.fixture
def perfect_bicluster_set(tmp_path: Path, experiment_dataset: Path) -> Path:
    destination = tmp_path / "perfect-result"
    candidate = Candidate(
        identifier="detected-0",
        bicluster=Bicluster(
            row_indices=(0, 1, 2, 3),
            column_indices=(0, 1, 2),
        ),
        provenance=CandidateProvenance(
            producer="test",
            operation="fixture",
            sequence=0,
        ),
    )
    BiclusterSetWriter().write(
        destination,
        identifier="perfect-result",
        dataset_identifier="experiment-dataset",
        row_count=8,
        source_column_count=3,
        columns=(
            PreparedColumnMetadata(
                index=0,
                name="numeric",
                kind=ColumnKind.NUMERIC,
                categories=(),
                source_column_index=0,
            ),
            PreparedColumnMetadata(
                index=1,
                name="boolean",
                kind=ColumnKind.BOOLEAN,
                categories=(),
                source_column_index=1,
            ),
            PreparedColumnMetadata(
                index=2,
                name="category",
                kind=ColumnKind.CATEGORICAL,
                categories=("A", "B"),
                source_column_index=2,
            ),
        ),
        repertoire=Repertoire(
            evaluations=(
                Evaluation(
                    candidate=candidate,
                    objectives=(),
                    descriptors=(
                        NamedValue(name="row_cardinality", value=4),
                        NamedValue(name="column_cardinality", value=3),
                    ),
                ),
            )
        ),
        source_run="test",
    )
    return destination


def scientific_configuration_mapping(
    dataset: Path,
    output: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run": {"identifier": "alignment-run", "seed": 17},
        "dataset": {"bundle": str(dataset)},
        "patterns": {
            "allowed": ["CONSTANT"],
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
                "parameters": {
                    "min_observed_count": 2,
                    "min_observed_ratio": 0.8,
                },
            },
        },
        "search": {
            "engine": {
                "name": "serial_mome",
                "parameters": {"initial_population_size": 4, "batch_size": 2},
            },
            "objectives": [
                {"name": "internal_coherence", "parameters": {}},
                {"name": "contrast", "parameters": {"min_background_ratio": 0.1}},
            ],
            "descriptors": [
                {"name": "row_cardinality", "parameters": {}},
                {"name": "column_cardinality", "parameters": {}},
            ],
            "archive": {
                "name": "deep_grid_mome",
                "parameters": {
                    "axes": [
                        {
                            "descriptor": "row_cardinality",
                            "binning": "EXACT",
                        },
                        {
                            "descriptor": "column_cardinality",
                            "binning": "EXACT",
                        },
                    ],
                    "cell_capacity": 4,
                },
            },
            "parent_selection": {
                "name": "repertoire_uniform",
                "parameters": {},
            },
            "initialization": {
                "name": "uniform_random",
                "parameters": {},
            },
            "emitters": [{"name": "random_move", "parameters": {}}],
            "scheduler": {"name": "first", "parameters": {}},
            "termination": {
                "name": "evaluation_budget",
                "parameters": {"max_evaluations": 4},
            },
        },
        "execution": {
            "executor": {"name": "serial", "parameters": {}},
            "workers": 1,
            "cancellation_grace_seconds": 1.0,
        },
        "monitoring": {"queue_capacity": 64, "observers": []},
        "final_selection": None,
        "output": {"directory": str(output), "overwrite": True},
    }


@pytest.fixture
def scientific_configuration(
    tmp_path: Path,
    experiment_dataset: Path,
) -> Path:
    path = tmp_path / "run.yaml"
    path.write_text(
        yaml.safe_dump(
            scientific_configuration_mapping(
                experiment_dataset,
                tmp_path / "unused-run-output",
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def scientific_pipeline(scientific_configuration: Path, tmp_path: Path) -> Path:
    """Reusable pipeline counterpart to the effective-run fixture."""

    mapping = yaml.safe_load(scientific_configuration.read_text(encoding="utf-8"))
    assert isinstance(mapping, dict)
    for key in ("run", "dataset", "output"):
        mapping.pop(key)
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")
    return path
