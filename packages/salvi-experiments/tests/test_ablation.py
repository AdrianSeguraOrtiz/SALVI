from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import yaml

from salvi.domain import (
    EventType,
    RunEvent,
)
from salvi.infrastructure.events import SQLiteEventStore
from salvi_experiments.benchmark import run_salvi_ablation
from salvi_experiments.benchmark.ablation import _aggregate_paired_deltas_by_dataset
from salvi_experiments.cli import main
from salvi_experiments.configuration import (
    AblationDatasetSelection,
    AblationMetricsConfiguration,
    AblationPairwiseComparison,
    AblationPipeline,
    AblationSelector,
    SalviAblationConfiguration,
    UncertaintyConfiguration,
)
from salvi_experiments.exceptions import ExperimentArtifactError
from salvi_experiments.metrics import (
    analyze_run_event_store,
    flatten_configuration,
)


def _configuration(
    tmp_path: Path,
    dataset: Path,
    pipeline: Path,
) -> SalviAblationConfiguration:
    no_bootstrap = UncertaintyConfiguration(bootstrap_samples=0)
    return SalviAblationConfiguration(
        identifier="test-ablation",
        benchmark_root=dataset,
        datasets=AblationDatasetSelection(
            identifiers=("experiment-dataset",),
        ),
        pattern_binding="GROUND_TRUTH",
        pipelines=(
            AblationPipeline(
                identifier="baseline",
                pipeline_configuration=pipeline,
            ),
            AblationPipeline(
                identifier="variant",
                pipeline_configuration=pipeline,
            ),
        ),
        run_seeds=(7,),
        metrics=AblationMetricsConfiguration(
            case_uncertainty=no_bootstrap,
            aggregate_uncertainty=no_bootstrap,
            diversity_sample_size=16,
        ),
        output_directory=tmp_path / "ablation",
    )


def test_ablation_configuration_accepts_single_pipeline_validation(
    tmp_path: Path,
    scientific_pipeline: Path,
) -> None:
    configuration = SalviAblationConfiguration(
        identifier="candidate-validation",
        benchmark_root=tmp_path,
        pattern_binding="GROUND_TRUTH",
        pipelines=(
            AblationPipeline(
                identifier="candidate",
                pipeline_configuration=scientific_pipeline,
            ),
        ),
        run_seeds=(42, 43, 44),
        output_directory=tmp_path / "validation",
    )

    assert tuple(entry.identifier for entry in configuration.pipelines) == ("candidate",)


def test_ablation_configuration_rejects_ambiguous_selections(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="non-empty list"):
        AblationDatasetSelection(replicates=())
    with pytest.raises(ValueError, match="non-empty list"):
        AblationDatasetSelection(replicates=(101, 101))
    with pytest.raises(ValueError, match="must not be blank"):
        AblationDatasetSelection(identifiers=(" ",))
    with pytest.raises(ValueError, match="must not contain duplicates"):
        AblationDatasetSelection(identifiers=("same", "same"))
    with pytest.raises(ValueError, match="raw selector"):
        AblationSelector(identifier="raw", parameters={"unexpected": True})
    with pytest.raises(ValueError, match="non-empty list"):
        AblationMetricsConfiguration(artifacts=())
    with pytest.raises(ValueError, match="at least one coverage threshold"):
        AblationMetricsConfiguration(coverage_thresholds=())
    with pytest.raises(ValueError, match="sorted and unique"):
        AblationMetricsConfiguration(coverage_thresholds=(0.5, 0.25))

    pipeline = AblationPipeline(
        identifier="same",
        pipeline_configuration=tmp_path / "pipeline.yaml",
    )
    base = {
        "identifier": "invalid",
        "benchmark_root": tmp_path,
        "pattern_binding": "PIPELINE",
        "output_directory": tmp_path / "output",
    }
    with pytest.raises(ValueError, match="pipeline identifiers"):
        SalviAblationConfiguration(**base, pipelines=(pipeline, pipeline))
    selector = AblationSelector(identifier="same")
    with pytest.raises(ValueError, match="selector identifiers"):
        SalviAblationConfiguration(
            **base,
            pipelines=(pipeline,),
            selectors=(selector, selector),
        )
    with pytest.raises(ValueError, match="run seeds"):
        SalviAblationConfiguration(
            **base,
            pipelines=(pipeline,),
            run_seeds=(1, 1),
        )
    with pytest.raises(ValueError, match="must be distinct"):
        AblationPairwiseComparison(
            baseline_pipeline="same",
            compared_pipeline="same",
        )
    with pytest.raises(ValueError, match="unknown pipelines"):
        SalviAblationConfiguration(
            **base,
            pipelines=(pipeline,),
            paired_comparisons=(
                AblationPairwiseComparison(
                    baseline_pipeline="same",
                    compared_pipeline="missing",
                ),
            ),
        )
    other_pipeline = AblationPipeline(
        identifier="other",
        pipeline_configuration=tmp_path / "other.yaml",
    )
    repeated_comparison = AblationPairwiseComparison(
        baseline_pipeline="same",
        compared_pipeline="other",
    )
    with pytest.raises(ValueError, match="paired comparisons must be unique"):
        SalviAblationConfiguration(
            **base,
            pipelines=(pipeline, other_pipeline),
            paired_comparisons=(repeated_comparison, repeated_comparison),
        )


def test_salvi_ablation_runs_complete_pipelines_and_measures_both_outputs(
    tmp_path: Path,
    experiment_dataset: Path,
    scientific_pipeline: Path,
) -> None:
    pipeline_mapping = yaml.safe_load(scientific_pipeline.read_text(encoding="utf-8"))
    pipeline_mapping["final_selection"] = {
        "name": "containment_marginal_quality",
        "parameters": {},
    }
    pipeline_with_selector = tmp_path / "pipeline-with-selector.yaml"
    pipeline_with_selector.write_text(
        yaml.safe_dump(pipeline_mapping, sort_keys=False),
        encoding="utf-8",
    )
    configuration = _configuration(
        tmp_path,
        experiment_dataset,
        pipeline_with_selector,
    )
    output = run_salvi_ablation(configuration)

    case_status = pq.read_table(output / "case-status.parquet").to_pylist()
    run_metrics = pq.read_table(output / "run-metrics.parquet").to_pylist()
    repertoire_metrics = pq.read_table(output / "repertoire-metrics.parquet").to_pylist()
    assert len(case_status) == 2
    assert {record["status"] for record in case_status} == {"completed"}
    assert len(run_metrics) == 2
    assert {record["configured_evaluation_budget"] for record in run_metrics} == {4}
    assert {record["initial_population_size"] for record in run_metrics} == {4}
    assert all(record["total_seconds"] >= record["search_seconds"] for record in run_metrics)
    assert len(repertoire_metrics) == 4
    assert {record["artifact"] for record in repertoire_metrics} == {"SEARCH", "FINAL"}
    assert all(0.0 <= record["relevance"] <= 1.0 for record in repertoire_metrics)
    assert (output / "configuration-differences.parquet").exists() is False
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["completed_count"] == 2
    assert report["baseline_pipeline"] == "baseline"
    assert (output / "ablation-accuracy.svg").is_file()
    effective = yaml.safe_load(
        (
            output
            / "cases"
            / "baseline"
            / "experiment-dataset"
            / "seed-7"
            / "effective-pipeline.yaml"
        ).read_text(encoding="utf-8")
    )
    assert effective["final_selection"] is not None


def test_salvi_ablation_reuses_matching_completed_cases(
    tmp_path: Path,
    experiment_dataset: Path,
    scientific_pipeline: Path,
) -> None:
    configuration = _configuration(
        tmp_path,
        experiment_dataset,
        scientific_pipeline,
    )
    output = run_salvi_ablation(configuration)
    metadata = (
        output
        / "cases"
        / "baseline"
        / "experiment-dataset"
        / "seed-7"
        / "run"
        / "run-metadata.json"
    )
    first_timestamp = metadata.stat().st_mtime_ns

    run_salvi_ablation(configuration)

    assert metadata.stat().st_mtime_ns == first_timestamp


def test_ablation_reports_population_and_evaluation_budget_differences(
    tmp_path: Path,
    experiment_dataset: Path,
    scientific_pipeline: Path,
) -> None:
    variant = tmp_path / "larger-search.yaml"
    mapping = yaml.safe_load(scientific_pipeline.read_text(encoding="utf-8"))
    mapping["search"]["engine"]["parameters"]["initial_population_size"] = 6
    mapping["search"]["termination"]["parameters"]["max_evaluations"] = 6
    variant.write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")
    no_bootstrap = UncertaintyConfiguration(bootstrap_samples=0)
    configuration = SalviAblationConfiguration(
        identifier="budget-ablation",
        benchmark_root=experiment_dataset,
        datasets=AblationDatasetSelection(
            identifiers=("experiment-dataset",),
        ),
        pattern_binding="PIPELINE",
        pipelines=(
            AblationPipeline(
                identifier="baseline",
                pipeline_configuration=scientific_pipeline,
            ),
            AblationPipeline(
                identifier="larger-search",
                pipeline_configuration=variant,
            ),
        ),
        run_seeds=(11,),
        metrics=AblationMetricsConfiguration(
            artifacts=("FINAL",),
            case_uncertainty=no_bootstrap,
            aggregate_uncertainty=no_bootstrap,
        ),
        output_directory=tmp_path / "budget-ablation",
    )

    output = run_salvi_ablation(configuration)

    differences = pq.read_table(output / "configuration-differences.parquet").to_pylist()
    parameters = {record["parameter"] for record in differences}
    assert "search.engine.parameters.initial_population_size" in parameters
    assert "search.termination.parameters.max_evaluations" in parameters
    run_metrics = pq.read_table(output / "run-metrics.parquet").to_pylist()
    assert {record["initial_population_size"] for record in run_metrics} == {4, 6}
    assert {record["configured_evaluation_budget"] for record in run_metrics} == {4, 6}


def test_ablation_applies_multiple_selectors_without_repeating_search(
    tmp_path: Path,
    experiment_dataset: Path,
    scientific_pipeline: Path,
) -> None:
    base = _configuration(
        tmp_path,
        experiment_dataset,
        scientific_pipeline,
    )
    configuration = base.model_copy(
        update={
            "selectors": (
                AblationSelector(
                    identifier="raw",
                    name=None,
                ),
                AblationSelector(
                    identifier="containment",
                    name="containment_marginal_quality",
                    parameters={
                        "max_objective_degradation": 0.15,
                        "max_degradation_per_log_area_gain": 0.20,
                    },
                ),
            ),
            "metrics": base.metrics.model_copy(update={"artifacts": ("FINAL",)}),
        }
    )

    output = run_salvi_ablation(configuration)
    metadata = (
        output
        / "cases"
        / "baseline"
        / "experiment-dataset"
        / "seed-7"
        / "run"
        / "run-metadata.json"
    )
    first_timestamp = metadata.stat().st_mtime_ns

    selections = pq.read_table(output / "selection-status.parquet").to_pylist()
    metrics = pq.read_table(output / "repertoire-metrics.parquet").to_pylist()
    effective = yaml.safe_load(
        (
            output
            / "cases"
            / "baseline"
            / "experiment-dataset"
            / "seed-7"
            / "effective-pipeline.yaml"
        ).read_text(encoding="utf-8")
    )
    assert effective["final_selection"] is None
    assert len(selections) == 4
    assert {record["status"] for record in selections} == {"completed"}
    assert len(metrics) == 4
    assert {record["selector_id"] for record in metrics} == {
        "raw",
        "containment",
    }
    assert {record["pipeline_id"] for record in metrics} == {
        "baseline::raw",
        "baseline::containment",
        "variant::raw",
        "variant::containment",
    }

    run_salvi_ablation(configuration)
    assert metadata.stat().st_mtime_ns == first_timestamp


def test_ablation_can_use_dataset_as_the_paired_analysis_unit(
    tmp_path: Path,
    experiment_dataset: Path,
    scientific_pipeline: Path,
) -> None:
    base = _configuration(tmp_path, experiment_dataset, scientific_pipeline)
    configuration = base.model_copy(
        update={
            "run_seeds": (7, 8),
            "metrics": base.metrics.model_copy(
                update={
                    "artifacts": ("FINAL",),
                    "paired_analysis_unit": "DATASET",
                    "paired_seed_aggregation": "MEAN",
                }
            ),
        }
    )

    output = run_salvi_ablation(configuration)

    raw = pq.read_table(output / "paired-deltas.parquet").to_pylist()
    analyzed = pq.read_table(output / "paired-analysis-deltas.parquet").to_pylist()
    summary = pq.read_table(output / "paired-summary.parquet").to_pylist()
    assert len(raw) == 2 * len(analyzed)
    assert len(summary) == len(analyzed)
    assert {record["run_seed_count"] for record in analyzed} == {2}
    assert {record["seed_aggregation"] for record in analyzed} == {"MEAN"}
    assert {record["sample_count"] for record in summary} == {1}
    assert all(record["holm_adjusted_p_value"] == 1.0 for record in summary)


def test_dataset_level_ablation_excludes_incomplete_seed_pairs() -> None:
    record = {
        "comparison_scope": "SEARCH_REPERTOIRE",
        "baseline_pipeline": "baseline",
        "compared_pipeline": "variant",
        "artifact": "SEARCH",
        "dataset_identifier": "dataset",
        "run_seed": 7,
        "metric": "recovery",
        "preferred_direction": "HIGHER",
        "baseline_value": 0.4,
        "compared_value": 0.5,
        "delta": 0.1,
        "favorable_delta": 0.1,
    }
    aggregated, incomplete = _aggregate_paired_deltas_by_dataset(
        [record],
        aggregation="MEAN",
        required_run_seeds=(7, 8),
    )

    assert aggregated == []
    assert len(incomplete) == 1
    assert incomplete[0]["expected_run_seeds"] == [7, 8]
    assert incomplete[0]["observed_run_seeds"] == [7]

    with pytest.raises(ExperimentArtifactError, match="duplicate run seeds"):
        _aggregate_paired_deltas_by_dataset(
            [record, record],
            aggregation="MEDIAN",
            required_run_seeds=(7,),
        )


def test_ablation_cli_loads_strict_yaml(
    tmp_path: Path,
    experiment_dataset: Path,
    scientific_pipeline: Path,
) -> None:
    configuration = _configuration(
        tmp_path,
        experiment_dataset,
        scientific_pipeline,
    )
    path = tmp_path / "ablation.yaml"
    path.write_text(
        yaml.safe_dump(configuration.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )

    assert main(["--quiet", "benchmark", "ablation", str(path)]) == 0


def test_event_store_analysis_extracts_runtime_archive_and_resource_metrics(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run.sqlite"
    store = SQLiteEventStore(path)
    store.append(
        RunEvent(
            event_type=EventType.CANDIDATES_ASKED,
            payload={
                "runtime": {"duration_seconds": 0.25},
                "candidates": [],
            },
        )
    )
    store.append(
        RunEvent(
            event_type=EventType.CANDIDATES_EVALUATED,
            payload={
                "runtime": {"duration_seconds": 0.50},
                "items": [
                    "malformed",
                    {
                        "signature": "signal",
                        "valid": True,
                        "objectives": [],
                        "constraints": [],
                    },
                    {
                        "signature": "signal",
                        "valid": False,
                        "objectives": [],
                        "constraints": [],
                    },
                ],
            },
        )
    )
    store.append(
        RunEvent(
            event_type=EventType.ARCHIVE_UPDATED,
            payload={
                "runtime": {"duration_seconds": 0.10},
                "outcomes": [
                    "malformed",
                    {
                        "status": "INSERTED",
                        "created_cell": True,
                        "coordinate": {"indices": [0, 1]},
                    },
                    {
                        "status": "INSERTED_WITH_EVICTIONS",
                        "created_cell": False,
                        "coordinate": [0, 1],
                    },
                    {
                        "status": "REJECTED_DUPLICATE",
                        "created_cell": False,
                        "coordinate": [1, 1],
                    },
                ],
            },
        )
    )
    store.append(
        RunEvent(
            event_type=EventType.EMITTER_CREDIT_UPDATED,
            payload={
                "reports": [
                    "malformed",
                    {
                        "emitter_name": "random_move",
                        "credit": 0.75,
                        "evaluations": 4,
                    },
                ]
            },
        )
    )
    store.record_metric("resource.process_cpu_seconds", 1.5)
    store.record_metric("resource.resident_memory_bytes", 2048)
    store.record_metric("resource.active_threads", 3)
    store.record_metric("diversity.window_size", 2)
    store.record_metric("diversity.window_duplicate_ratio", 0.5)
    store.record_metric("diversity.nearest_distance.minimum", 0.2)
    store.record_metric("diversity.nearest_distance.median", 0.3)
    store.record_metric("diversity.nearest_distance.mean", 0.4)
    store.record_metric("diversity.nearest_distance.maximum", 0.5)
    store.close()

    diagnostics, emitters = analyze_run_event_store(path)

    assert diagnostics["evaluated_candidates"] == 2
    assert diagnostics["unique_evaluated_candidates"] == 1
    assert diagnostics["invalid_candidates"] == 1
    assert diagnostics["accepted_candidates"] == 2
    assert diagnostics["created_cells"] == 1
    assert diagnostics["process_cpu_seconds"] == 1.5
    assert diagnostics["peak_resident_memory_bytes"] == 2048
    assert diagnostics["peak_active_threads"] == 3
    assert emitters == (
        {
            "emitter_name": "random_move",
            "credit": 0.75,
            "evaluations": 4,
        },
    )
    assert flatten_configuration(4, "budget") == {"budget": 4}
