"""Dataset-level accuracy evaluation for any canonical BiclusterSet."""

from __future__ import annotations

from pathlib import Path

from salvi import BiclusterSetReader, DatasetBundleReader
from salvi_experiments.artifacts import (
    atomic_experiment_directory,
    sha256_file,
    write_json,
    write_manifest,
    write_table,
)
from salvi_experiments.configuration import AccuracyConfiguration
from salvi_experiments.dataset.common import (
    detected_memberships,
    ground_truth_memberships,
    read_scoped_ground_truth,
)
from salvi_experiments.metrics import AccuracyResult, calculate_accuracy
from salvi_experiments.plots import plot_accuracy
from salvi_experiments.progress import ProgressReporter, progress_or_null


def _threshold_name(threshold: float) -> str:
    return f"coverage_at_{threshold:.4f}".rstrip("0").rstrip(".").replace(".", "_")


def accuracy_record(
    result: AccuracyResult,
    configuration: AccuracyConfiguration,
    *,
    dataset_identifier: str,
) -> dict[str, object]:
    record: dict[str, object] = {
        "experiment_id": configuration.identifier,
        "dataset_identifier": dataset_identifier,
        "algorithm": configuration.algorithm.algorithm,
        "algorithm_version": configuration.algorithm.version,
        "relevance": result.relevance,
        "relevance_ci_lower": result.relevance_interval.lower,
        "relevance_ci_upper": result.relevance_interval.upper,
        "recovery": result.recovery,
        "recovery_ci_lower": result.recovery_interval.lower,
        "recovery_ci_upper": result.recovery_interval.upper,
        "biclustering_error": result.biclustering_error,
        "biclustering_error_ci_lower": result.biclustering_error_interval.lower,
        "biclustering_error_ci_upper": result.biclustering_error_interval.upper,
        "detected_count": result.detected_count,
        "ground_truth_count": result.ground_truth_count,
        "target_count_known": configuration.algorithm.target_count_known,
        "target_count_value": configuration.algorithm.target_count_value,
        "evaluation_budget": configuration.algorithm.evaluation_budget,
        "wall_clock_budget_seconds": configuration.algorithm.wall_clock_budget_seconds,
        "wall_time_seconds": configuration.algorithm.wall_time_seconds,
        "cpu_time_seconds": configuration.algorithm.cpu_time_seconds,
        "peak_memory_bytes": configuration.algorithm.peak_memory_bytes,
        "postprocessing_policy": configuration.algorithm.postprocessing_policy,
        "final_selection_policy": configuration.algorithm.final_selection_policy,
    }
    record.update({_threshold_name(threshold): value for threshold, value in result.coverage})
    return record


def run_accuracy(
    configuration: AccuracyConfiguration,
    *,
    progress: ProgressReporter | None = None,
) -> Path:
    reporter = progress_or_null(progress)
    reporter.stage("loading dataset and detected bicluster artifacts")
    dataset = DatasetBundleReader().inspect(configuration.dataset_bundle)
    result_manifest = BiclusterSetReader().read_manifest(configuration.bicluster_set)
    reporter.stage("loading scoped ground truth")
    _, scoped_ground_truth = read_scoped_ground_truth(
        configuration.dataset_bundle,
        configuration.task,
    )
    reporter.stage("calculating accuracy metrics")
    result = calculate_accuracy(
        detected_memberships(
            configuration.dataset_bundle,
            configuration.bicluster_set,
        ),
        ground_truth_memberships(scoped_ground_truth),
        uncertainty=configuration.uncertainty,
        coverage_thresholds=configuration.coverage_thresholds,
    )
    record = accuracy_record(
        result,
        configuration,
        dataset_identifier=dataset.identifier,
    )
    run_configuration_sha256 = (
        None
        if configuration.run_configuration is None
        else sha256_file(configuration.run_configuration)
    )

    reporter.stage(f"writing accuracy artifacts to {configuration.output_directory}")
    with atomic_experiment_directory(
        configuration.output_directory,
        overwrite=configuration.overwrite,
    ) as temporary:
        write_table(temporary, "metrics", [record])
        write_table(
            temporary,
            "matches",
            [match.model_dump(mode="json") for match in result.matches],
        )
        plot_accuracy(
            temporary,
            relevance=result.relevance,
            recovery=result.recovery,
            biclustering_error=result.biclustering_error,
            coverage=result.coverage,
        )
        report = {
            "schema_version": 1,
            "experiment_type": "dataset.accuracy",
            "identifier": configuration.identifier,
            "dataset_identifier": dataset.identifier,
            "dataset_bundle": str(configuration.dataset_bundle),
            "dataset_manifest_sha256": sha256_file(configuration.dataset_bundle / "dataset.yaml"),
            "bicluster_set": str(configuration.bicluster_set),
            "bicluster_set_manifest_sha256": sha256_file(
                configuration.bicluster_set / "manifest.json"
            ),
            "bicluster_set_identifier": result_manifest.identifier,
            "run_configuration": (
                None
                if configuration.run_configuration is None
                else str(configuration.run_configuration)
            ),
            "run_configuration_sha256": run_configuration_sha256,
            "task": configuration.task.model_dump(mode="json"),
            "algorithm": configuration.algorithm.model_dump(mode="json"),
            "uncertainty": configuration.uncertainty.model_dump(mode="json"),
            "coverage_thresholds": list(configuration.coverage_thresholds),
            "metrics": {
                "relevance": result.relevance,
                "recovery": result.recovery,
                "biclustering_error": result.biclustering_error,
                "relevance_interval": result.relevance_interval.model_dump(mode="json"),
                "recovery_interval": result.recovery_interval.model_dump(mode="json"),
                "biclustering_error_interval": result.biclustering_error_interval.model_dump(
                    mode="json"
                ),
                "coverage": [
                    {"threshold": threshold, "value": value} for threshold, value in result.coverage
                ],
                "detected_count": result.detected_count,
                "ground_truth_count": result.ground_truth_count,
            },
        }
        write_json(temporary / "report.json", report)
        write_manifest(
            temporary,
            experiment_type="dataset.accuracy",
            identifier=configuration.identifier,
            metadata={
                "dataset_identifier": dataset.identifier,
                "algorithm": configuration.algorithm.algorithm,
                "task": configuration.task.model_dump(mode="json"),
            },
        )
    return configuration.output_directory.resolve()


__all__ = ["accuracy_record", "run_accuracy"]
