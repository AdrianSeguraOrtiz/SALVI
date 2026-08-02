from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import yaml

from salvi import Bicluster, ScientificEvaluationService, load_configuration
from salvi_experiments.cli import main
from salvi_experiments.configuration import (
    AccuracyConfiguration,
    AlgorithmRunInformation,
    ObjectiveAlignmentConfiguration,
    ObjectiveAlignmentSampling,
    TaskScope,
    UncertaintyConfiguration,
    load_experiment_configuration,
)
from salvi_experiments.dataset import run_accuracy, run_objective_alignment
from salvi_experiments.exceptions import ExperimentConfigurationError


def test_public_scientific_evaluation_uses_configured_objectives(
    scientific_configuration: Path,
) -> None:
    configuration = load_configuration(scientific_configuration).configuration
    with ScientificEvaluationService(configuration) as service:
        batch = service.evaluate(
            (
                Bicluster(
                    row_indices=(0, 1, 2, 3),
                    column_indices=(0, 1, 2),
                ),
            ),
            identifiers=("truth",),
        )
    assert service.objective_names == ("internal_coherence", "contrast")
    assert tuple(item.name for item in batch.evaluations[0].objectives) == (
        "internal_coherence",
        "contrast",
    )
    assert batch.evaluations[0].pattern_fit is not None


def test_objective_alignment_writes_exact_tables_and_plots(
    tmp_path: Path,
    scientific_pipeline: Path,
    experiment_dataset: Path,
) -> None:
    output = tmp_path / "alignment"
    run_objective_alignment(
        ObjectiveAlignmentConfiguration(
            identifier="alignment",
            pipeline_configuration=scientific_pipeline,
            dataset_bundle=experiment_dataset,
            output_directory=output,
            sampling=ObjectiveAlignmentSampling(
                random_controls=3,
                perturbations=2,
                perturbation_ratio=0.25,
            ),
        )
    )
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["experiment_type"] == "dataset.objective_alignment"
    assert report["ground_truth_count"] == 1
    assert report["candidate_count"] == 6
    assert {
        "candidates.parquet",
        "objective-alignment.parquet",
        "objective-alignment.png",
        "objective-distributions.svg",
        "manifest.json",
    }.issubset({path.name for path in output.iterdir()})
    records = pq.read_table(output / "candidates.parquet").to_pylist()
    exact = next(record for record in records if record["candidate_type"] == "GROUND_TRUTH")
    assert exact["internal_coherence"] < 0.1
    assert exact["contrast"] > 0.5


def test_accuracy_accepts_any_canonical_bicluster_set(
    tmp_path: Path,
    experiment_dataset: Path,
    perfect_bicluster_set: Path,
) -> None:
    output = tmp_path / "accuracy"
    run_accuracy(
        AccuracyConfiguration(
            identifier="accuracy",
            dataset_bundle=experiment_dataset,
            bicluster_set=perfect_bicluster_set,
            output_directory=output,
            task=TaskScope(),
            algorithm=AlgorithmRunInformation(
                algorithm="fixture",
                target_count_known=False,
                postprocessing_policy="none",
                final_selection_policy="none",
            ),
            uncertainty=UncertaintyConfiguration(bootstrap_samples=0),
        )
    )
    metrics = pq.read_table(output / "metrics.parquet").to_pylist()
    assert len(metrics) == 1
    assert metrics[0]["relevance"] == pytest.approx(1.0)
    assert metrics[0]["recovery"] == pytest.approx(1.0)
    assert metrics[0]["biclustering_error"] == pytest.approx(1.0)
    assert (output / "accuracy-summary.png").is_file()


def test_experiment_cli_runs_accuracy(
    tmp_path: Path,
    experiment_dataset: Path,
    perfect_bicluster_set: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configuration = tmp_path / "accuracy.yaml"
    output = tmp_path / "cli-accuracy"
    configuration.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "identifier": "cli-accuracy",
                "dataset_bundle": str(experiment_dataset),
                "bicluster_set": str(perfect_bicluster_set),
                "output_directory": str(output),
                "algorithm": {
                    "algorithm": "fixture",
                    "target_count_known": False,
                    "postprocessing_policy": "none",
                    "final_selection_policy": "none",
                },
                "uncertainty": {
                    "bootstrap_samples": 0,
                    "confidence_level": 0.95,
                    "seed": 0,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    assert main(["dataset", "accuracy", str(configuration)]) == 0
    assert str(output) in capsys.readouterr().out


def test_duplicate_experiment_yaml_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text(
        "schema_version: 1\nidentifier: first\nidentifier: second\n",
        encoding="utf-8",
    )
    with pytest.raises(ExperimentConfigurationError, match="duplicate YAML key"):
        load_experiment_configuration(path, ObjectiveAlignmentConfiguration)


def test_experiment_configuration_rejects_ambiguous_scientific_scopes(
    tmp_path: Path,
) -> None:
    assert TaskScope().resolved(tmp_path) is not None
    with pytest.raises(ValueError, match="at least one pattern"):
        TaskScope(included_patterns=())
    with pytest.raises(ValueError, match="duplicates"):
        TaskScope(included_patterns=("CONSTANT", "CONSTANT"))

    algorithm = {
        "algorithm": "fixture",
        "postprocessing_policy": "none",
        "final_selection_policy": "none",
    }
    with pytest.raises(ValueError, match="target_count_value"):
        AlgorithmRunInformation(**algorithm, target_count_known=True)
    with pytest.raises(ValueError, match="target_count_value"):
        AlgorithmRunInformation(
            **algorithm,
            target_count_known=False,
            target_count_value=2,
        )

    accuracy = {
        "identifier": "accuracy",
        "dataset_bundle": tmp_path / "dataset",
        "bicluster_set": tmp_path / "result",
        "output_directory": tmp_path / "output",
        "algorithm": AlgorithmRunInformation(**algorithm),
    }
    with pytest.raises(ValueError, match="at least one coverage threshold"):
        AccuracyConfiguration(**accuracy, coverage_thresholds=())
    with pytest.raises(ValueError, match="sorted and unique"):
        AccuracyConfiguration(**accuracy, coverage_thresholds=(0.5, 0.25))


def test_experiment_configuration_reports_file_and_yaml_errors(tmp_path: Path) -> None:
    with pytest.raises(ExperimentConfigurationError, match="cannot read"):
        load_experiment_configuration(tmp_path / "missing.yaml", ObjectiveAlignmentConfiguration)
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("value: [", encoding="utf-8")
    with pytest.raises(ExperimentConfigurationError, match="cannot read"):
        load_experiment_configuration(invalid, ObjectiveAlignmentConfiguration)
