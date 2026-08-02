from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import yaml

import salvi_experiments.benchmark.execution as benchmark_execution
from salvi_experiments.benchmark import (
    run_accuracy_benchmark,
    run_comparison,
    run_objective_alignment_benchmark,
)
from salvi_experiments.configuration import (
    AccuracyBenchmarkCase,
    AccuracyBenchmarkConfiguration,
    AccuracyConfiguration,
    AlgorithmRunInformation,
    BenchmarkExecutionConfiguration,
    ComparisonAlgorithm,
    ComparisonConfiguration,
    ObjectiveAlignmentBenchmarkCase,
    ObjectiveAlignmentBenchmarkConfiguration,
    ObjectiveAlignmentSampling,
    UncertaintyConfiguration,
)
from salvi_experiments.dataset import run_accuracy
from salvi_experiments.exceptions import ExperimentArtifactError


def _algorithm(name: str) -> AlgorithmRunInformation:
    return AlgorithmRunInformation(
        algorithm=name,
        target_count_known=False,
        evaluation_budget=100,
        wall_time_seconds=2.0,
        cpu_time_seconds=3.0,
        peak_memory_bytes=1024,
        postprocessing_policy="none",
        final_selection_policy="none",
    )


def test_benchmark_configuration_rejects_ambiguous_case_sets(tmp_path: Path) -> None:
    alignment_case = ObjectiveAlignmentBenchmarkCase(
        identifier="same",
        pipeline_configuration=tmp_path / "pipeline.yaml",
        dataset_bundle=tmp_path / "dataset",
    )
    with pytest.raises(ValueError, match="case identifiers"):
        ObjectiveAlignmentBenchmarkConfiguration(
            identifier="invalid",
            cases=(alignment_case, alignment_case),
            output_directory=tmp_path / "alignment",
        )

    first = AccuracyBenchmarkCase(
        identifier="same",
        dataset_bundle=tmp_path / "dataset",
        bicluster_set=tmp_path / "first",
        algorithm=_algorithm("SALVI"),
    )
    duplicate = first.model_copy(update={"bicluster_set": tmp_path / "second"})
    with pytest.raises(ValueError, match="case identifiers"):
        AccuracyBenchmarkConfiguration(
            identifier="invalid",
            cases=(first, duplicate),
            output_directory=tmp_path / "accuracy",
        )

    second_algorithm = first.model_copy(
        update={"identifier": "other", "algorithm": _algorithm("HBIC")}
    )
    with pytest.raises(ValueError, match="exactly one algorithm"):
        AccuracyBenchmarkConfiguration(
            identifier="invalid",
            cases=(first, second_algorithm),
            output_directory=tmp_path / "accuracy",
        )
    with pytest.raises(ValueError, match="at least one coverage threshold"):
        AccuracyBenchmarkConfiguration(
            identifier="invalid",
            cases=(first,),
            output_directory=tmp_path / "accuracy",
            coverage_thresholds=(),
        )
    with pytest.raises(ValueError, match="sorted and unique"):
        AccuracyBenchmarkConfiguration(
            identifier="invalid",
            cases=(first,),
            output_directory=tmp_path / "accuracy",
            coverage_thresholds=(0.5, 0.25),
        )

    algorithm = ComparisonAlgorithm(
        identifier="same",
        accuracy_results=(tmp_path / "accuracy",),
    )
    with pytest.raises(ValueError, match="algorithm identifiers"):
        ComparisonConfiguration(
            identifier="invalid",
            algorithms=(algorithm, algorithm),
            output_directory=tmp_path / "comparison",
        )


def test_accuracy_benchmark_aggregates_dataset_results(
    tmp_path: Path,
    experiment_dataset: Path,
    perfect_bicluster_set: Path,
) -> None:
    output = tmp_path / "accuracy-benchmark"
    run_accuracy_benchmark(
        AccuracyBenchmarkConfiguration(
            identifier="benchmark",
            cases=(
                AccuracyBenchmarkCase(
                    identifier="case-a",
                    dataset_bundle=experiment_dataset,
                    bicluster_set=perfect_bicluster_set,
                    algorithm=_algorithm("SALVI"),
                ),
                AccuracyBenchmarkCase(
                    identifier="case-b",
                    dataset_bundle=experiment_dataset,
                    bicluster_set=perfect_bicluster_set,
                    algorithm=_algorithm("SALVI"),
                ),
            ),
            output_directory=output,
            uncertainty=UncertaintyConfiguration(bootstrap_samples=10),
        )
    )
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["experiment_type"] == "benchmark.accuracy"
    assert report["case_count"] == 2
    summary = pq.read_table(output / "summary.parquet").to_pylist()
    assert {record["metric"] for record in summary} == {
        "relevance",
        "recovery",
        "biclustering_error",
        "detected_count",
        "coverage_at_0_25",
        "coverage_at_0_5",
        "coverage_at_0_75",
        "wall_time_seconds",
        "cpu_time_seconds",
        "peak_memory_bytes",
    }


def test_accuracy_benchmark_can_run_cases_in_parallel(
    tmp_path: Path,
    experiment_dataset: Path,
    perfect_bicluster_set: Path,
) -> None:
    output = tmp_path / "parallel-accuracy-benchmark"
    run_accuracy_benchmark(
        AccuracyBenchmarkConfiguration(
            identifier="parallel-benchmark",
            cases=(
                AccuracyBenchmarkCase(
                    identifier="case-a",
                    dataset_bundle=experiment_dataset,
                    bicluster_set=perfect_bicluster_set,
                    algorithm=_algorithm("SALVI"),
                ),
                AccuracyBenchmarkCase(
                    identifier="case-b",
                    dataset_bundle=experiment_dataset,
                    bicluster_set=perfect_bicluster_set,
                    algorithm=_algorithm("SALVI"),
                ),
            ),
            output_directory=output,
            uncertainty=UncertaintyConfiguration(bootstrap_samples=0),
            execution=BenchmarkExecutionConfiguration(
                workers=2,
                allow_nested_parallelism=True,
            ),
        )
    )
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["execution"]["effective_workers"] == 2
    metrics = pq.read_table(output / "dataset-metrics.parquet").to_pylist()
    assert {record["case_id"] for record in metrics} == {"case-a", "case-b"}


def test_benchmark_parallelism_rejects_nested_salvi_workers(
    tmp_path: Path,
    scientific_configuration: Path,
    experiment_dataset: Path,
) -> None:
    nested = tmp_path / "nested-run.yaml"
    mapping = yaml.safe_load(scientific_configuration.read_text(encoding="utf-8"))
    for key in ("run", "dataset", "output"):
        mapping.pop(key)
    mapping["execution"]["executor"]["name"] = "thread_pool"
    mapping["execution"]["workers"] = 2
    nested.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")

    with pytest.raises(ExperimentArtifactError, match="nested parallelism"):
        run_objective_alignment_benchmark(
            ObjectiveAlignmentBenchmarkConfiguration(
                identifier="nested-alignment-benchmark",
                cases=(
                    ObjectiveAlignmentBenchmarkCase(
                        identifier="case-a",
                        pipeline_configuration=nested,
                        dataset_bundle=experiment_dataset,
                    ),
                    ObjectiveAlignmentBenchmarkCase(
                        identifier="case-b",
                        pipeline_configuration=nested,
                        dataset_bundle=experiment_dataset,
                    ),
                ),
                output_directory=tmp_path / "nested-alignment-output",
                execution=BenchmarkExecutionConfiguration(workers=2),
            )
        )


def test_benchmark_parallelism_rejects_cpu_oversubscription_independently(
    tmp_path: Path,
    scientific_configuration: Path,
    experiment_dataset: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nested = tmp_path / "nested-run.yaml"
    mapping = yaml.safe_load(scientific_configuration.read_text(encoding="utf-8"))
    for key in ("run", "dataset", "output"):
        mapping.pop(key)
    mapping["execution"]["executor"]["name"] = "thread_pool"
    mapping["execution"]["workers"] = 2
    nested.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")
    configuration = ObjectiveAlignmentBenchmarkConfiguration(
        identifier="oversubscribed-alignment-benchmark",
        cases=(
            ObjectiveAlignmentBenchmarkCase(
                identifier="case-a",
                pipeline_configuration=nested,
                dataset_bundle=experiment_dataset,
            ),
            ObjectiveAlignmentBenchmarkCase(
                identifier="case-b",
                pipeline_configuration=nested,
                dataset_bundle=experiment_dataset,
            ),
        ),
        output_directory=tmp_path / "oversubscribed-alignment-output",
        execution=BenchmarkExecutionConfiguration(
            workers=2,
            allow_nested_parallelism=True,
        ),
    )
    monkeypatch.setattr(benchmark_execution.os, "cpu_count", lambda: 2)

    with pytest.raises(ExperimentArtifactError, match="oversubscribe"):
        run_objective_alignment_benchmark(configuration)


def test_benchmark_parallelism_can_explicitly_allow_cpu_oversubscription(
    tmp_path: Path,
    scientific_configuration: Path,
    experiment_dataset: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nested = tmp_path / "nested-run.yaml"
    mapping = yaml.safe_load(scientific_configuration.read_text(encoding="utf-8"))
    for key in ("run", "dataset", "output"):
        mapping.pop(key)
    mapping["execution"]["executor"]["name"] = "thread_pool"
    mapping["execution"]["workers"] = 2
    nested.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")
    output = tmp_path / "oversubscribed-allowed-output"
    monkeypatch.setattr(benchmark_execution.os, "cpu_count", lambda: 2)

    run_objective_alignment_benchmark(
        ObjectiveAlignmentBenchmarkConfiguration(
            identifier="oversubscribed-allowed",
            cases=(
                ObjectiveAlignmentBenchmarkCase(
                    identifier="case-a",
                    pipeline_configuration=nested,
                    dataset_bundle=experiment_dataset,
                ),
                ObjectiveAlignmentBenchmarkCase(
                    identifier="case-b",
                    pipeline_configuration=nested,
                    dataset_bundle=experiment_dataset,
                ),
            ),
            output_directory=output,
            sampling=ObjectiveAlignmentSampling(
                random_controls=1,
                perturbations=0,
            ),
            uncertainty=UncertaintyConfiguration(bootstrap_samples=0),
            execution=BenchmarkExecutionConfiguration(
                workers=2,
                allow_nested_parallelism=True,
                allow_cpu_oversubscription=True,
            ),
        )
    )
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["execution"]["allow_cpu_oversubscription"] is True
    assert report["execution"]["effective_workers"] == 2


def test_comparison_consumes_prior_accuracy_outputs(
    tmp_path: Path,
    experiment_dataset: Path,
    perfect_bicluster_set: Path,
) -> None:
    inputs: list[Path] = []
    for algorithm in ("SALVI", "HBIC"):
        output = tmp_path / f"{algorithm.lower()}-accuracy"
        run_accuracy(
            AccuracyConfiguration(
                identifier=f"{algorithm}-accuracy",
                dataset_bundle=experiment_dataset,
                bicluster_set=perfect_bicluster_set,
                output_directory=output,
                algorithm=_algorithm(algorithm),
                uncertainty=UncertaintyConfiguration(bootstrap_samples=0),
            )
        )
        inputs.append(output)
    comparison = tmp_path / "comparison"
    run_comparison(
        ComparisonConfiguration(
            identifier="comparison",
            algorithms=(
                ComparisonAlgorithm(
                    identifier="SALVI",
                    accuracy_results=(inputs[0],),
                ),
                ComparisonAlgorithm(
                    identifier="HBIC",
                    accuracy_results=(inputs[1],),
                ),
            ),
            output_directory=comparison,
            uncertainty=UncertaintyConfiguration(bootstrap_samples=10),
        )
    )
    report = json.loads((comparison / "report.json").read_text(encoding="utf-8"))
    assert report["algorithms"] == ["SALVI", "HBIC"]
    deltas = pq.read_table(comparison / "paired-deltas.parquet").to_pylist()
    assert deltas[0]["relevance_delta"] == 0.0


def test_objective_alignment_benchmark_iterates_reusable_pipelines(
    tmp_path: Path,
    scientific_pipeline: Path,
    experiment_dataset: Path,
) -> None:
    output = tmp_path / "alignment-benchmark"
    run_objective_alignment_benchmark(
        ObjectiveAlignmentBenchmarkConfiguration(
            identifier="alignment-benchmark",
            cases=(
                ObjectiveAlignmentBenchmarkCase(
                    identifier="case",
                    pipeline_configuration=scientific_pipeline,
                    dataset_bundle=experiment_dataset,
                ),
            ),
            output_directory=output,
            sampling=ObjectiveAlignmentSampling(
                random_controls=2,
                perturbations=1,
                perturbation_ratio=0.25,
            ),
            uncertainty=UncertaintyConfiguration(bootstrap_samples=10),
        )
    )
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["experiment_type"] == "benchmark.objective_alignment"
    assert report["case_count"] == 1
    assert (output / "objective-alignment-benchmark.svg").is_file()
