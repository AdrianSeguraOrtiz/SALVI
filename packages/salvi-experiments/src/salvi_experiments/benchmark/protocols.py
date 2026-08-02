"""Benchmark iteration, aggregation, and algorithm comparison."""

from __future__ import annotations

import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from salvi import load_configuration, load_pipeline_configuration
from salvi_experiments.artifacts import (
    atomic_experiment_directory,
    read_report,
    write_json,
    write_manifest,
    write_table,
)
from salvi_experiments.benchmark.execution import validate_benchmark_parallelism
from salvi_experiments.configuration import (
    AccuracyBenchmarkConfiguration,
    AccuracyConfiguration,
    ComparisonConfiguration,
    ObjectiveAlignmentBenchmarkConfiguration,
    ObjectiveAlignmentConfiguration,
)
from salvi_experiments.dataset import run_accuracy, run_objective_alignment
from salvi_experiments.exceptions import ExperimentArtifactError
from salvi_experiments.plots import (
    plot_algorithm_comparison,
    plot_alignment_benchmark,
    plot_benchmark_metrics,
)
from salvi_experiments.progress import ProgressReporter, progress_or_null
from salvi_experiments.reporting import summarize_metric

ACCURACY_METRICS = ("relevance", "recovery", "biclustering_error")
RESOURCE_METRICS = (
    "detected_count",
    "wall_time_seconds",
    "cpu_time_seconds",
    "peak_memory_bytes",
)
CONTROLS = ("RANDOM_MATCHED", "REMOVED", "ADDED")


def _number(value: object) -> float:
    if not isinstance(value, int | float):
        raise ExperimentArtifactError(
            f"expected numeric experiment value, received {type(value).__name__}"
        )
    return float(value)


def _records(path: Path) -> list[dict[str, Any]]:
    try:
        return [dict(record) for record in pq.read_table(path).to_pylist()]
    except Exception as error:
        raise ExperimentArtifactError(f"cannot read experiment table {path}: {error}") from error


def _available_numeric_values(
    records: list[dict[str, object]] | list[dict[str, Any]],
    field: str,
) -> list[float]:
    return [_number(value) for record in records if (value := record.get(field)) is not None]


def _summary_metrics(records: list[dict[str, object]] | list[dict[str, Any]]) -> tuple[str, ...]:
    coverage = sorted(
        {field for record in records for field in record if field.startswith("coverage_at_")}
    )
    optional_resources = tuple(
        metric for metric in RESOURCE_METRICS if _available_numeric_values(records, metric)
    )
    return (*ACCURACY_METRICS, *coverage, *optional_resources)


def _run_objective_alignment_case(configuration: ObjectiveAlignmentConfiguration) -> Path:
    return run_objective_alignment(configuration)


def _run_accuracy_case(configuration: AccuracyConfiguration) -> Path:
    return run_accuracy(configuration)


def _pipeline_workers(pipeline_configuration: Path | None) -> int:
    if pipeline_configuration is None:
        return 1
    return load_pipeline_configuration(pipeline_configuration).pipeline.execution.workers


def _effective_configuration_workers(run_configuration: Path | None) -> int:
    if run_configuration is None:
        return 1
    return load_configuration(run_configuration).configuration.execution.workers


def _objective_alignment_case_configurations(
    configuration: ObjectiveAlignmentBenchmarkConfiguration,
    root: Path,
) -> tuple[tuple[str, ObjectiveAlignmentConfiguration], ...]:
    return tuple(
        (
            case.identifier,
            ObjectiveAlignmentConfiguration(
                identifier=f"{configuration.identifier}:{case.identifier}",
                pipeline_configuration=case.pipeline_configuration,
                dataset_bundle=case.dataset_bundle,
                output_directory=root / "cases" / case.identifier,
                analysis_seed=configuration.analysis_seed,
                task=configuration.task,
                sampling=configuration.sampling,
            ),
        )
        for case in configuration.cases
    )


def _accuracy_case_configurations(
    configuration: AccuracyBenchmarkConfiguration,
    root: Path,
) -> tuple[tuple[str, AccuracyConfiguration], ...]:
    return tuple(
        (
            case.identifier,
            AccuracyConfiguration(
                identifier=f"{configuration.identifier}:{case.identifier}",
                dataset_bundle=case.dataset_bundle,
                bicluster_set=case.bicluster_set,
                output_directory=root / "cases" / case.identifier,
                run_configuration=case.run_configuration,
                task=configuration.task,
                algorithm=case.algorithm,
                uncertainty=configuration.uncertainty,
                coverage_thresholds=configuration.coverage_thresholds,
            ),
        )
        for case in configuration.cases
    )


def _run_objective_alignment_cases(
    case_configurations: tuple[tuple[str, ObjectiveAlignmentConfiguration], ...],
    *,
    workers: int,
    reporter: ProgressReporter,
) -> None:
    total = len(case_configurations)
    if workers <= 1:
        for index, (case_id, case_configuration) in enumerate(case_configurations, start=1):
            reporter.step(f"objective-alignment case {case_id}", index, total)
            run_objective_alignment(case_configuration, progress=reporter)
        return
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
    ) as executor:
        futures = {
            executor.submit(_run_objective_alignment_case, case_configuration): case_id
            for case_id, case_configuration in case_configurations
        }
        for index, future in enumerate(as_completed(futures), start=1):
            case_id = futures[future]
            future.result()
            reporter.step(f"completed objective-alignment case {case_id}", index, total)


def _run_accuracy_cases(
    case_configurations: tuple[tuple[str, AccuracyConfiguration], ...],
    *,
    workers: int,
    reporter: ProgressReporter,
) -> None:
    total = len(case_configurations)
    if workers <= 1:
        for index, (case_id, case_configuration) in enumerate(case_configurations, start=1):
            reporter.step(f"accuracy case {case_id}", index, total)
            run_accuracy(case_configuration, progress=reporter)
        return
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
    ) as executor:
        futures = {
            executor.submit(_run_accuracy_case, case_configuration): case_id
            for case_id, case_configuration in case_configurations
        }
        for index, future in enumerate(as_completed(futures), start=1):
            case_id = futures[future]
            future.result()
            reporter.step(f"completed accuracy case {case_id}", index, total)


def run_objective_alignment_benchmark(
    configuration: ObjectiveAlignmentBenchmarkConfiguration,
    *,
    progress: ProgressReporter | None = None,
) -> Path:
    reporter = progress_or_null(progress)
    workers = validate_benchmark_parallelism(
        configuration.execution,
        tuple(_pipeline_workers(case.pipeline_configuration) for case in configuration.cases),
    )
    reporter.stage(
        f"running {len(configuration.cases)} objective-alignment benchmark cases "
        f"with {workers} benchmark worker(s)"
    )
    with atomic_experiment_directory(
        configuration.output_directory,
        overwrite=configuration.overwrite,
    ) as temporary:
        case_configurations = _objective_alignment_case_configurations(
            configuration,
            temporary,
        )
        _run_objective_alignment_cases(
            case_configurations,
            workers=workers,
            reporter=reporter,
        )
        case_records: list[dict[str, object]] = []
        for case_id, case_configuration in case_configurations:
            report = read_report(case_configuration.output_directory)
            objectives = report.get("objectives")
            aggregate = report.get("aggregate_alignment")
            if not isinstance(objectives, list) or not isinstance(aggregate, dict):
                raise ExperimentArtifactError(
                    f"invalid objective-alignment report in {case_configuration.output_directory}"
                )
            for objective in objectives:
                if not isinstance(objective, dict) or not isinstance(objective.get("name"), str):
                    raise ExperimentArtifactError(
                        f"invalid objective declaration in {case_configuration.output_directory}"
                    )
                objective_name = objective["name"]
                for control in CONTROLS:
                    key = f"{objective_name}_{control.lower()}_favorable_fraction"
                    value = aggregate.get(key)
                    if value is None:
                        continue
                    case_records.append(
                        {
                            "case_id": case_id,
                            "dataset_identifier": report["dataset_identifier"],
                            "objective": objective_name,
                            "direction": objective.get("direction"),
                            "control_type": control,
                            "favorable_fraction": _number(value),
                        }
                    )
        if not case_records:
            raise ExperimentArtifactError(
                "objective-alignment benchmark produced no comparable controls"
            )
        reporter.stage("aggregating objective-alignment benchmark results")
        summary_records: list[dict[str, object]] = []
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        for record in case_records:
            grouped[(str(record["objective"]), str(record["control_type"]))].append(
                _number(record["favorable_fraction"])
            )
        for index, ((objective, control), values) in enumerate(sorted(grouped.items())):
            summary_records.append(
                {
                    "objective": objective,
                    "control_type": control,
                    **summarize_metric(
                        values,
                        configuration.uncertainty,
                        seed_offset=index,
                    ),
                }
            )
        write_table(temporary, "case-alignment", case_records)
        write_table(temporary, "summary", summary_records)
        plot_alignment_benchmark(temporary, records=case_records)
        report = {
            "schema_version": 1,
            "experiment_type": "benchmark.objective_alignment",
            "identifier": configuration.identifier,
            "case_count": len(configuration.cases),
            "analysis_seed": configuration.analysis_seed,
            "task": configuration.task.model_dump(mode="json"),
            "sampling": configuration.sampling.model_dump(mode="json"),
            "uncertainty": configuration.uncertainty.model_dump(mode="json"),
            "execution": {
                **configuration.execution.model_dump(mode="json"),
                "effective_workers": workers,
            },
            "summary": summary_records,
        }
        write_json(temporary / "report.json", report)
        write_manifest(
            temporary,
            experiment_type="benchmark.objective_alignment",
            identifier=configuration.identifier,
            metadata={"case_count": len(configuration.cases)},
        )
    return configuration.output_directory.resolve()


def run_accuracy_benchmark(
    configuration: AccuracyBenchmarkConfiguration,
    *,
    progress: ProgressReporter | None = None,
) -> Path:
    reporter = progress_or_null(progress)
    workers = validate_benchmark_parallelism(
        configuration.execution,
        tuple(
            _effective_configuration_workers(case.run_configuration) for case in configuration.cases
        ),
    )
    reporter.stage(
        f"running {len(configuration.cases)} accuracy benchmark cases "
        f"with {workers} benchmark worker(s)"
    )
    with atomic_experiment_directory(
        configuration.output_directory,
        overwrite=configuration.overwrite,
    ) as temporary:
        case_configurations = _accuracy_case_configurations(configuration, temporary)
        _run_accuracy_cases(
            case_configurations,
            workers=workers,
            reporter=reporter,
        )
        dataset_records: list[dict[str, object]] = []
        for case_id, case_configuration in case_configurations:
            records = _records(case_configuration.output_directory / "metrics.parquet")
            if len(records) != 1:
                raise ExperimentArtifactError(
                    f"accuracy case {case_id!r} produced an invalid metric table"
                )
            dataset_records.append({"case_id": case_id, **records[0]})

        reporter.stage("aggregating accuracy benchmark results")
        summary_records: list[dict[str, object]] = []
        for metric_index, metric in enumerate(_summary_metrics(dataset_records)):
            summary_records.append(
                {
                    "metric": metric,
                    **summarize_metric(
                        _available_numeric_values(dataset_records, metric),
                        configuration.uncertainty,
                        seed_offset=metric_index,
                    ),
                }
            )
        write_table(temporary, "dataset-metrics", dataset_records)
        write_table(temporary, "summary", summary_records)
        plot_benchmark_metrics(
            temporary,
            records=dataset_records,
            stem="accuracy-benchmark",
        )
        report = {
            "schema_version": 1,
            "experiment_type": "benchmark.accuracy",
            "identifier": configuration.identifier,
            "algorithm": configuration.cases[0].algorithm.algorithm,
            "case_count": len(dataset_records),
            "task": configuration.task.model_dump(mode="json"),
            "uncertainty": configuration.uncertainty.model_dump(mode="json"),
            "coverage_thresholds": list(configuration.coverage_thresholds),
            "execution": {
                **configuration.execution.model_dump(mode="json"),
                "effective_workers": workers,
            },
            "summary": summary_records,
        }
        write_json(temporary / "report.json", report)
        write_manifest(
            temporary,
            experiment_type="benchmark.accuracy",
            identifier=configuration.identifier,
            metadata={
                "algorithm": configuration.cases[0].algorithm.algorithm,
                "case_count": len(dataset_records),
                "task": configuration.task.model_dump(mode="json"),
            },
        )
    return configuration.output_directory.resolve()


def _accuracy_records(directory: Path) -> tuple[dict[str, object], list[dict[str, Any]]]:
    report = read_report(directory)
    experiment_type = report.get("experiment_type")
    if experiment_type == "dataset.accuracy":
        records = _records(directory / "metrics.parquet")
    elif experiment_type == "benchmark.accuracy":
        records = _records(directory / "dataset-metrics.parquet")
    else:
        raise ExperimentArtifactError(f"{directory} is not a dataset or benchmark accuracy result")
    return report, records


def run_comparison(
    configuration: ComparisonConfiguration,
    *,
    progress: ProgressReporter | None = None,
) -> Path:
    reporter = progress_or_null(progress)
    reporter.stage(f"loading accuracy results for {len(configuration.algorithms)} algorithms")
    per_dataset: list[dict[str, object]] = []
    task: object | None = None
    expected_datasets: set[str] | None = None
    for algorithm_index, algorithm in enumerate(configuration.algorithms, start=1):
        reporter.step(
            f"comparison algorithm {algorithm.identifier}",
            algorithm_index,
            len(configuration.algorithms),
        )
        algorithm_records: list[dict[str, Any]] = []
        for result_index, result_directory in enumerate(algorithm.accuracy_results, start=1):
            reporter.step(
                f"reading {algorithm.identifier} result {result_index}",
                result_index,
                len(algorithm.accuracy_results),
            )
            report, records = _accuracy_records(result_directory)
            report_task = report.get("task")
            if task is None:
                task = report_task
            elif report_task != task:
                raise ExperimentArtifactError(
                    "algorithm comparison inputs use different scientific task scopes"
                )
            algorithm_records.extend(records)
        datasets = [str(record["dataset_identifier"]) for record in algorithm_records]
        if len(set(datasets)) != len(datasets):
            raise ExperimentArtifactError(
                f"algorithm {algorithm.identifier!r} contains duplicate dataset results"
            )
        dataset_set = set(datasets)
        if expected_datasets is None:
            expected_datasets = dataset_set
        elif dataset_set != expected_datasets:
            raise ExperimentArtifactError(
                "every compared algorithm must cover exactly the same datasets"
            )
        per_dataset.extend(
            {
                **record,
                "algorithm": algorithm.identifier,
                "reported_algorithm": record.get("algorithm"),
            }
            for record in algorithm_records
        )
    if task is None or expected_datasets is None:
        raise ExperimentArtifactError("algorithm comparison contains no accuracy results")

    reporter.stage("aggregating algorithm comparison metrics")
    summaries: list[dict[str, object]] = []
    for algorithm_index, algorithm in enumerate(configuration.algorithms):
        records = [record for record in per_dataset if record["algorithm"] == algorithm.identifier]
        summary: dict[str, object] = {
            "algorithm": algorithm.identifier,
            "dataset_count": len(records),
        }
        summary_metrics = _summary_metrics(records)
        for metric_index, metric in enumerate(summary_metrics):
            statistics = summarize_metric(
                _available_numeric_values(records, metric),
                configuration.uncertainty,
                seed_offset=algorithm_index * len(summary_metrics) + metric_index,
            )
            summary.update({f"{metric}_{name}": value for name, value in statistics.items()})
        summaries.append(summary)

    baseline = configuration.algorithms[0].identifier
    by_key = {
        (str(record["algorithm"]), str(record["dataset_identifier"])): record
        for record in per_dataset
    }
    deltas: list[dict[str, object]] = []
    for algorithm in configuration.algorithms[1:]:
        for dataset_identifier in sorted(expected_datasets):
            baseline_record = by_key[(baseline, dataset_identifier)]
            compared_record = by_key[(algorithm.identifier, dataset_identifier)]
            deltas.append(
                {
                    "baseline": baseline,
                    "algorithm": algorithm.identifier,
                    "dataset_identifier": dataset_identifier,
                    **{
                        f"{metric}_delta": _number(compared_record[metric])
                        - _number(baseline_record[metric])
                        for metric in ACCURACY_METRICS
                    },
                }
            )

    reporter.stage(f"writing comparison artifacts to {configuration.output_directory}")
    with atomic_experiment_directory(
        configuration.output_directory,
        overwrite=configuration.overwrite,
    ) as temporary:
        write_table(temporary, "per-dataset", per_dataset)
        write_table(temporary, "summary", summaries)
        write_table(temporary, "paired-deltas", deltas)
        plot_algorithm_comparison(temporary, summaries=summaries)
        report = {
            "schema_version": 1,
            "experiment_type": "benchmark.compare",
            "identifier": configuration.identifier,
            "algorithms": [algorithm.identifier for algorithm in configuration.algorithms],
            "baseline_algorithm": baseline,
            "dataset_count": len(expected_datasets),
            "task": task,
            "uncertainty": configuration.uncertainty.model_dump(mode="json"),
            "summary": summaries,
        }
        write_json(temporary / "report.json", report)
        write_manifest(
            temporary,
            experiment_type="benchmark.compare",
            identifier=configuration.identifier,
            metadata={
                "algorithms": report["algorithms"],
                "dataset_count": len(expected_datasets),
                "task": task,
            },
        )
    return configuration.output_directory.resolve()


__all__ = [
    "run_accuracy_benchmark",
    "run_comparison",
    "run_objective_alignment_benchmark",
]
