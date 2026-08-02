"""Reproducible SALVI-only ablations over canonical DatasetBundle benchmarks."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml

import salvi
from salvi import (
    BiclusterSetReader,
    DatasetBundleReader,
    FinalSelectionService,
    PatternKind,
    PipelineConfiguration,
    RunBinding,
    RunService,
    load_pipeline_configuration,
)
from salvi.application.configuration import ComponentSpec
from salvi.components.defaults import default_component_registry
from salvi.components.protocols import ComponentKind
from salvi_experiments.artifacts import sha256_file, write_json, write_table
from salvi_experiments.benchmark.execution import validate_benchmark_parallelism
from salvi_experiments.configuration import (
    SalviAblationConfiguration,
    UncertaintyConfiguration,
)
from salvi_experiments.dataset.common import (
    detected_memberships,
    ground_truth_memberships,
    read_scoped_ground_truth,
)
from salvi_experiments.exceptions import ExperimentArtifactError
from salvi_experiments.metrics import (
    analyze_run_event_store,
    calculate_accuracy,
    flatten_configuration,
    repertoire_diversity,
)
from salvi_experiments.plots import plot_salvi_ablation
from salvi_experiments.progress import ProgressReporter, progress_or_null
from salvi_experiments.reporting import summarize_metric

_REPLICATE = re.compile(r"_(\d+)$")
_SAFE_SEGMENT = re.compile(r"[^A-Za-z0-9._-]+")
_ACCURACY_METRICS = ("relevance", "recovery", "biclustering_error")
_RUN_METRICS = (
    "wall_time_seconds",
    "evaluated_candidates",
    "unique_evaluated_candidates",
    "candidate_duplicate_ratio",
    "invalid_candidates",
    "accepted_candidates",
    "acceptance_ratio",
    "created_cells",
    "visited_cells",
    "candidate_generation_seconds",
    "evaluation_seconds",
    "archive_update_seconds",
    "evaluations_per_second",
    "process_cpu_seconds",
    "peak_resident_memory_bytes",
    "peak_active_threads",
    "candidate_window_size",
    "candidate_window_duplicate_ratio",
    "candidate_window_nearest_distance_minimum",
    "candidate_window_nearest_distance_median",
    "candidate_window_nearest_distance_mean",
    "candidate_window_nearest_distance_maximum",
    "search_seconds",
    "selection_seconds",
    "total_seconds",
)
_REPERTOIRE_METRICS = (
    *_ACCURACY_METRICS,
    "detected_count",
    "repertoire_count",
    "repertoire_unique_structures",
    "repertoire_duplicate_ratio",
    "repertoire_occupied_coordinates",
    "repertoire_nearest_distance_minimum",
    "repertoire_nearest_distance_median",
    "repertoire_nearest_distance_mean",
    "repertoire_nearest_distance_maximum",
)


@dataclass(frozen=True, slots=True)
class _DatasetCase:
    identifier: str
    bundle: Path
    patterns: tuple[PatternKind, ...]
    search_fingerprint: str
    ground_truth_fingerprint: str


@dataclass(frozen=True, slots=True)
class _RunCase:
    pipeline_id: str
    pipeline_path: Path
    dataset: _DatasetCase
    allowed_patterns: tuple[PatternKind, ...]
    run_seed: int
    case_root: Path
    pipeline_sha256: str
    implementation_sha256: str
    run_fingerprint: str
    resume: bool
    retry_failed: bool
    strip_final_selection: bool


def _segment(value: str) -> str:
    normalized = _SAFE_SEGMENT.sub("-", value).strip(".-")
    if normalized:
        return (
            normalized
            if normalized == value
            else f"{normalized}-{hashlib.sha256(value.encode('utf-8')).hexdigest()[:8]}"
        )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _hash_files(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _hash_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _salvi_implementation_fingerprint() -> str:
    package_root = Path(salvi.__file__).resolve().parent
    scientific_roots = (
        "application",
        "components",
        "domain",
        "engine",
        "evaluation",
        "infrastructure",
        "patterns",
    )
    sources = tuple(
        sorted(
            path
            for root in scientific_roots
            for path in (package_root / root).rglob("*.py")
            if "__pycache__" not in path.parts
        )
    )
    digest = hashlib.sha256()
    runtime = {
        "python": sys.version,
        "salvi": salvi.__version__,
        **{
            distribution: importlib.metadata.version(distribution)
            for distribution in ("numpy", "pyarrow", "pydantic", "scipy")
        },
    }
    digest.update(json.dumps(runtime, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for path in sources:
        digest.update(path.relative_to(package_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _dataset_case(bundle: Path) -> _DatasetCase:
    source = bundle.expanduser().resolve()
    reader = DatasetBundleReader()
    dataset = reader.inspect(source)
    ground_truth = reader.read_ground_truth(source)
    if ground_truth is None or not ground_truth.biclusters:
        raise ExperimentArtifactError(f"ablation dataset has no non-empty ground truth: {source}")
    patterns = tuple(
        sorted(
            {
                column.pattern
                for bicluster in ground_truth.biclusters
                for column in bicluster.column_patterns
            },
            key=lambda pattern: pattern.value,
        )
    )
    if not patterns:
        raise ExperimentArtifactError(
            f"ablation dataset ground truth declares no patterns: {source}"
        )
    search_files = tuple(
        path
        for name in ("dataset.yaml", "data.parquet", "row-identifiers.parquet")
        if (path := source / name).is_file()
    )
    ground_truth_path = source / "ground-truth.json"
    if len(search_files) < 2 or not ground_truth_path.is_file():
        raise ExperimentArtifactError(f"incomplete DatasetBundle: {source}")
    return _DatasetCase(
        identifier=dataset.identifier,
        bundle=source,
        patterns=patterns,
        search_fingerprint=_hash_files(search_files),
        ground_truth_fingerprint=sha256_file(ground_truth_path),
    )


def _discover_datasets(configuration: SalviAblationConfiguration) -> tuple[_DatasetCase, ...]:
    root = configuration.benchmark_root
    bundles: tuple[Path, ...]
    if (root / "dataset.yaml").is_file() and (root / "data.parquet").is_file():
        bundles = (root,)
    else:
        bundles = tuple(
            sorted(
                manifest.parent
                for manifest in root.rglob("dataset.yaml")
                if (manifest.parent / "data.parquet").is_file()
            )
        )
    cases = tuple(_dataset_case(bundle) for bundle in bundles)
    by_identifier = {case.identifier: case for case in cases}
    if len(by_identifier) != len(cases):
        raise ExperimentArtifactError("benchmark contains duplicate dataset identifiers")

    replicates = configuration.datasets.replicates
    if replicates != "ALL":
        allowed = set(replicates)
        cases = tuple(
            case
            for case in cases
            if (match := _REPLICATE.search(case.identifier)) is not None
            and int(match.group(1)) in allowed
        )
    requested = set(configuration.datasets.identifiers)
    if requested:
        missing = requested - by_identifier.keys()
        if missing:
            raise ExperimentArtifactError(
                "requested ablation datasets were not found: " + ", ".join(sorted(missing))
            )
        cases = tuple(case for case in cases if case.identifier in requested)
    if not cases:
        raise ExperimentArtifactError(f"no DatasetBundles match the ablation filters under {root}")
    return tuple(sorted(cases, key=lambda case: case.identifier))


def _effective_pipeline(
    pipeline: PipelineConfiguration,
    *,
    pattern_binding: str,
    dataset_patterns: tuple[PatternKind, ...],
) -> PipelineConfiguration:
    if pattern_binding == "PIPELINE":
        return pipeline
    return pipeline.model_copy(
        update={"patterns": pipeline.patterns.model_copy(update={"allowed": dataset_patterns})}
    )


def _without_final_selection_dependents(
    pipeline: PipelineConfiguration,
) -> PipelineConfiguration:
    """Remove a selector and observers that consume capabilities provided only by it."""

    selector = pipeline.final_selection
    if selector is None:
        return pipeline
    registry = default_component_registry()
    removed_capabilities = registry.get(
        ComponentKind.FINAL_SELECTOR,
        selector.name,
    ).provides
    observers = tuple(
        observer
        for observer in pipeline.monitoring.observers
        if not (registry.get(ComponentKind.OBSERVER, observer.name).requires & removed_capabilities)
    )
    return pipeline.model_copy(
        update={
            "final_selection": None,
            "monitoring": pipeline.monitoring.model_copy(update={"observers": observers}),
        }
    )


def _write_pipeline(path: Path, pipeline: PipelineConfiguration) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            pipeline.model_dump(mode="json"),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )


def _read_case_record(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _valid_cached_case(case: _RunCase) -> dict[str, Any] | None:
    if not case.resume:
        return None
    record = _read_case_record(case.case_root / "case.json")
    if record is None or record.get("run_fingerprint") != case.run_fingerprint:
        return None
    status = record.get("status")
    if status == "failed" and not case.retry_failed:
        return record
    if status != "completed":
        return None
    run_root = case.case_root / "run"
    required = (
        run_root / "run-metadata.json",
        run_root / "run.sqlite",
        run_root / "artifacts" / "repertoire" / "manifest.json",
    )
    if not all(path.is_file() for path in required):
        return None
    record["ground_truth_sha256"] = case.dataset.ground_truth_fingerprint
    write_json(case.case_root / "case.json", record)
    return record


def _execute_case(case: _RunCase) -> dict[str, Any]:
    cached = _valid_cached_case(case)
    if cached is not None:
        return cached
    case.case_root.mkdir(parents=True, exist_ok=True)
    pipeline = load_pipeline_configuration(case.pipeline_path).pipeline
    effective = pipeline.model_copy(
        update={
            "patterns": pipeline.patterns.model_copy(update={"allowed": case.allowed_patterns}),
        }
    )
    if case.strip_final_selection:
        effective = _without_final_selection_dependents(effective)
    effective_path = case.case_root / "effective-pipeline.yaml"
    _write_pipeline(effective_path, effective)
    run_root = case.case_root / "run"
    started = perf_counter()
    record: dict[str, Any] = {
        "schema_version": 1,
        "pipeline_id": case.pipeline_id,
        "pipeline_source": str(case.pipeline_path),
        "pipeline_sha256": case.pipeline_sha256,
        "salvi_implementation_sha256": case.implementation_sha256,
        "dataset_identifier": case.dataset.identifier,
        "dataset_bundle": str(case.dataset.bundle),
        "dataset_search_sha256": case.dataset.search_fingerprint,
        "ground_truth_sha256": case.dataset.ground_truth_fingerprint,
        "patterns": [pattern.value for pattern in case.allowed_patterns],
        "run_seed": case.run_seed,
        "run_fingerprint": case.run_fingerprint,
        "status": "running",
        "run_output": str(run_root),
        "error": None,
    }
    write_json(case.case_root / "case.json", record)
    try:
        RunService().run_pipeline(
            effective_path,
            RunBinding(
                identifier=(f"{case.pipeline_id}-{case.dataset.identifier}-seed-{case.run_seed}"),
                dataset_bundle=case.dataset.bundle,
                output_directory=run_root,
                seed=case.run_seed,
                overwrite=True,
            ),
        )
        record["status"] = "completed"
    except Exception as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
    record["orchestration_wall_time_seconds"] = perf_counter() - started
    write_json(case.case_root / "case.json", record)
    return record


def _apply_selector(
    configuration: SalviAblationConfiguration,
    case_record: dict[str, Any],
    selector_index: int,
) -> dict[str, Any]:
    selector = configuration.selectors[selector_index]
    case_root = Path(str(case_record["run_output"])).parent
    selection_root = case_root / "selections" / _segment(selector.identifier)
    output = selection_root / "repertoire"
    fingerprint = _hash_json(
        {
            "run": case_record["run_fingerprint"],
            "implementation": case_record["salvi_implementation_sha256"],
            "selector": selector.model_dump(mode="json"),
        }
    )
    record_path = selection_root / "selection.json"
    if configuration.execution.resume:
        cached = _read_case_record(record_path)
        if cached is not None and cached.get("selection_fingerprint") == fingerprint:
            if cached.get("status") == "failed" and not configuration.execution.retry_failed:
                return cached
            cached_output = Path(str(cached.get("bicluster_set", output)))
            if cached.get("status") == "completed" and (cached_output / "manifest.json").is_file():
                return cached

    selection_root.mkdir(parents=True, exist_ok=True)
    source_repertoire = Path(str(case_record["run_output"])) / "artifacts" / "repertoire"
    if selector.name is None:
        repertoire = BiclusterSetReader().read(source_repertoire)
        raw_record = {
            "schema_version": 1,
            "pipeline_id": case_record["pipeline_id"],
            "selector_id": selector.identifier,
            "selector_name": None,
            "dataset_identifier": case_record["dataset_identifier"],
            "run_seed": case_record["run_seed"],
            "selection_fingerprint": fingerprint,
            "status": "completed",
            "bicluster_set": str(source_repertoire),
            "input_count": len(repertoire.evaluations),
            "output_count": len(repertoire.evaluations),
            "wall_time_seconds": 0.0,
            "error": None,
        }
        write_json(record_path, raw_record)
        return raw_record

    source_pipeline = load_pipeline_configuration(case_root / "effective-pipeline.yaml").pipeline
    effective = source_pipeline.model_copy(
        update={
            "final_selection": ComponentSpec(
                name=selector.name,
                parameters=selector.parameters,
            )
        }
    )
    effective_path = selection_root / "effective-pipeline.yaml"
    _write_pipeline(effective_path, effective)
    record: dict[str, Any] = {
        "schema_version": 1,
        "pipeline_id": case_record["pipeline_id"],
        "selector_id": selector.identifier,
        "selector_name": selector.name,
        "dataset_identifier": case_record["dataset_identifier"],
        "run_seed": case_record["run_seed"],
        "selection_fingerprint": fingerprint,
        "status": "running",
        "bicluster_set": str(output),
        "error": None,
    }
    write_json(record_path, record)
    started = perf_counter()
    try:
        result = FinalSelectionService().select(
            effective_path,
            dataset_bundle=Path(str(case_record["dataset_bundle"])),
            repertoire=source_repertoire,
            output=output,
            identifier=(
                f"{case_record['pipeline_id']}-{selector.identifier}-"
                f"{case_record['dataset_identifier']}-seed-{case_record['run_seed']}"
            ),
            overwrite=True,
        )
        record.update(
            {
                "status": "completed",
                "input_count": result.input_count,
                "output_count": result.output_count,
            }
        )
    except Exception as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
    record["wall_time_seconds"] = perf_counter() - started
    write_json(record_path, record)
    return record


def _configuration_value(
    configuration: dict[str, Any],
    *path: str,
) -> object | None:
    value: object = configuration
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _wall_time(metadata: dict[str, Any]) -> float | None:
    started = metadata.get("started_at")
    finished = metadata.get("finished_at")
    if not isinstance(started, str) or not isinstance(finished, str):
        return None
    return (datetime.fromisoformat(finished) - datetime.fromisoformat(started)).total_seconds()


def _run_record(
    case_record: dict[str, Any],
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    run_root = Path(str(case_record["run_output"]))
    try:
        metadata = json.loads((run_root / "run-metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExperimentArtifactError(
            f"invalid SALVI run metadata in {run_root}: {error}"
        ) from error
    if not isinstance(metadata, dict) or metadata.get("status") != "completed":
        raise ExperimentArtifactError(f"SALVI run is not complete: {run_root}")
    configuration = metadata.get("configuration")
    if not isinstance(configuration, dict):
        raise ExperimentArtifactError(f"SALVI run has no effective configuration: {run_root}")
    diagnostics, emitters = analyze_run_event_store(run_root / "run.sqlite")
    search = metadata.get("search")
    search_values = search if isinstance(search, dict) else {}
    timing = metadata.get("timing_seconds")
    timing_values = timing if isinstance(timing, dict) else {}
    engine_parameters = _configuration_value(
        configuration,
        "search",
        "engine",
        "parameters",
    )
    archive_parameters = _configuration_value(
        configuration,
        "search",
        "archive",
        "parameters",
    )
    termination_parameters = _configuration_value(
        configuration,
        "search",
        "termination",
        "parameters",
    )
    record: dict[str, object] = {
        "pipeline_id": case_record["pipeline_id"],
        "dataset_identifier": case_record["dataset_identifier"],
        "patterns": case_record["patterns"],
        "run_seed": case_record["run_seed"],
        "run_fingerprint": case_record["run_fingerprint"],
        "pipeline_sha256": case_record["pipeline_sha256"],
        "salvi_implementation_sha256": case_record["salvi_implementation_sha256"],
        "run_output": str(run_root),
        "wall_time_seconds": _wall_time(metadata),
        "configured_evaluation_budget": (
            termination_parameters.get("max_evaluations")
            if isinstance(termination_parameters, dict)
            else None
        ),
        "initial_population_size": (
            engine_parameters.get("initial_population_size")
            if isinstance(engine_parameters, dict)
            else None
        ),
        "batch_size": (
            engine_parameters.get("batch_size") if isinstance(engine_parameters, dict) else None
        ),
        "execution_workers": _configuration_value(
            configuration,
            "execution",
            "workers",
        ),
        "archive_cell_capacity": (
            archive_parameters.get("cell_capacity")
            if isinstance(archive_parameters, dict)
            else None
        ),
        "final_selector": _configuration_value(
            configuration,
            "final_selection",
            "name",
        ),
        "search_accepted": search_values.get("accepted"),
        "search_rejected": search_values.get("rejected"),
        "search_occupied_cells": search_values.get("occupied_cells"),
        "search_repertoire_size": search_values.get("repertoire_size"),
        "search_seconds": timing_values.get("search"),
        "selection_seconds": timing_values.get("selection"),
        "total_seconds": timing_values.get("total"),
        **diagnostics,
    }
    emitter_records = tuple(
        {
            "pipeline_id": case_record["pipeline_id"],
            "dataset_identifier": case_record["dataset_identifier"],
            "run_seed": case_record["run_seed"],
            **emitter,
        }
        for emitter in emitters
    )
    return record, emitter_records


def _coverage_name(threshold: float) -> str:
    return f"coverage_at_{threshold:.4f}".rstrip("0").rstrip(".").replace(".", "_")


def _repertoire_record(
    configuration: SalviAblationConfiguration,
    case_record: dict[str, Any],
    *,
    artifact: str,
    artifact_directory: Path | None = None,
    analysis_pipeline_id: str | None = None,
    selector_id: str | None = None,
) -> dict[str, object]:
    run_root = Path(str(case_record["run_output"]))
    if artifact_directory is None:
        search_directory = run_root / "artifacts" / "search-repertoire"
        artifact_directory = (
            search_directory
            if artifact == "SEARCH" and (search_directory / "manifest.json").is_file()
            else run_root / "artifacts" / "repertoire"
        )
    reader = BiclusterSetReader()
    repertoire = reader.read(artifact_directory)
    _, scoped_ground_truth = read_scoped_ground_truth(
        Path(str(case_record["dataset_bundle"])),
        configuration.task,
    )
    accuracy = calculate_accuracy(
        detected_memberships(
            Path(str(case_record["dataset_bundle"])),
            artifact_directory,
        ),
        ground_truth_memberships(scoped_ground_truth),
        uncertainty=configuration.metrics.case_uncertainty,
        coverage_thresholds=configuration.metrics.coverage_thresholds,
    )
    return {
        "pipeline_id": analysis_pipeline_id or case_record["pipeline_id"],
        "search_pipeline_id": case_record["pipeline_id"],
        "selector_id": selector_id,
        "dataset_identifier": case_record["dataset_identifier"],
        "patterns": case_record["patterns"],
        "run_seed": case_record["run_seed"],
        "artifact": artifact,
        "bicluster_set": str(artifact_directory),
        "relevance": accuracy.relevance,
        "recovery": accuracy.recovery,
        "biclustering_error": accuracy.biclustering_error,
        "detected_count": accuracy.detected_count,
        "ground_truth_count": accuracy.ground_truth_count,
        **{_coverage_name(threshold): value for threshold, value in accuracy.coverage},
        **repertoire_diversity(
            repertoire,
            row_weight=configuration.metrics.structural_row_weight,
            sample_size=configuration.metrics.diversity_sample_size,
        ),
    }


def _numeric(record: dict[str, object], field: str) -> float | None:
    value = record.get(field)
    return float(value) if isinstance(value, int | float) else None


def _pattern_label(record: dict[str, object]) -> str:
    patterns = record.get("patterns")
    if not isinstance(patterns, list | tuple):
        raise ExperimentArtifactError("ablation record has no pattern list")
    return ",".join(str(value) for value in patterns)


def _aggregate(
    records: list[dict[str, object]],
    *,
    group_fields: tuple[str, ...],
    metrics: tuple[str, ...],
    uncertainty: UncertaintyConfiguration,
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[tuple(record[field] for field in group_fields)].append(record)
    results: list[dict[str, object]] = []
    offset = 0
    for group, samples in sorted(grouped.items(), key=lambda item: str(item[0])):
        for pattern in (
            "ALL",
            *sorted({_pattern_label(sample) for sample in samples}),
        ):
            selected = (
                samples
                if pattern == "ALL"
                else [sample for sample in samples if _pattern_label(sample) == pattern]
            )
            for metric in metrics:
                values = tuple(
                    value for sample in selected if (value := _numeric(sample, metric)) is not None
                )
                if not values:
                    continue
                results.append(
                    {
                        **dict(zip(group_fields, group, strict=True)),
                        "pattern_scope": pattern,
                        "metric": metric,
                        **summarize_metric(
                            values,
                            uncertainty,
                            seed_offset=offset,
                        ),
                    }
                )
                offset += 1
    return results


def _paired_deltas(
    records: list[dict[str, object]],
    *,
    baseline: str,
    artifact_sensitive: bool,
    metrics: tuple[str, ...],
    comparison_scope: str,
) -> list[dict[str, object]]:
    key_fields = ("dataset_identifier", "run_seed", *(("artifact",) if artifact_sensitive else ()))
    baseline_records = {
        tuple(record[field] for field in key_fields): record
        for record in records
        if record["pipeline_id"] == baseline
    }
    deltas: list[dict[str, object]] = []
    for record in records:
        pipeline = str(record["pipeline_id"])
        if pipeline == baseline:
            continue
        key = tuple(record[field] for field in key_fields)
        reference = baseline_records.get(key)
        if reference is None:
            continue
        for metric in metrics:
            baseline_value = _numeric(reference, metric)
            compared_value = _numeric(record, metric)
            if baseline_value is None or compared_value is None:
                continue
            direction = "LOWER" if metric == "wall_time_seconds" else "HIGHER"
            delta = compared_value - baseline_value
            deltas.append(
                {
                    "comparison_scope": comparison_scope,
                    "baseline_pipeline": baseline,
                    "compared_pipeline": pipeline,
                    **{field: record[field] for field in key_fields},
                    "metric": metric,
                    "preferred_direction": direction,
                    "baseline_value": baseline_value,
                    "compared_value": compared_value,
                    "delta": delta,
                    "favorable_delta": -delta if direction == "LOWER" else delta,
                }
            )
    return deltas


def _configuration_differences(
    configurations: dict[str, PipelineConfiguration],
) -> list[dict[str, object]]:
    flattened = {
        identifier: flatten_configuration(configuration.model_dump(mode="json"))
        for identifier, configuration in configurations.items()
    }
    paths = sorted({path for values in flattened.values() for path in values})
    varying = tuple(
        path
        for path in paths
        if len(
            {
                json.dumps(values.get(path), sort_keys=True, separators=(",", ":"))
                for values in flattened.values()
            }
        )
        > 1
    )
    return [
        {
            "parameter": path,
            "pipeline_id": identifier,
            "value_json": json.dumps(
                flattened[identifier].get(path),
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
        for path in varying
        for identifier in configurations
    ]


def _selector_differences(
    configuration: SalviAblationConfiguration,
) -> list[dict[str, object]]:
    flattened = {
        selector.identifier: flatten_configuration(selector.model_dump(mode="json"))
        for selector in configuration.selectors
    }
    paths = sorted({path for values in flattened.values() for path in values})
    varying = tuple(
        path
        for path in paths
        if len(
            {
                json.dumps(values.get(path), sort_keys=True, separators=(",", ":"))
                for values in flattened.values()
            }
        )
        > 1
    )
    return [
        {
            "parameter": path,
            "selector_id": identifier,
            "value_json": json.dumps(
                flattened[identifier].get(path),
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
        for path in varying
        for identifier in flattened
    ]


def _paired_summary(
    deltas: list[dict[str, object]],
    uncertainty: UncertaintyConfiguration,
) -> list[dict[str, object]]:
    grouped: dict[
        tuple[str, str, str, str, str, str],
        list[tuple[float, float]],
    ] = defaultdict(list)
    for record in deltas:
        delta = record.get("delta")
        favorable = record.get("favorable_delta")
        if not isinstance(delta, int | float) or not isinstance(
            favorable,
            int | float,
        ):
            raise ExperimentArtifactError("paired ablation delta is not numeric")
        grouped[
            (
                str(record["comparison_scope"]),
                str(record["baseline_pipeline"]),
                str(record["compared_pipeline"]),
                str(record.get("artifact", "RUN")),
                str(record["metric"]),
                str(record["preferred_direction"]),
            )
        ].append((float(delta), float(favorable)))
    results: list[dict[str, object]] = []
    for index, (
        (scope, baseline, pipeline, artifact, metric, direction),
        values,
    ) in enumerate(sorted(grouped.items())):
        delta_summary = summarize_metric(
            [value[0] for value in values],
            uncertainty,
            seed_offset=index * 2,
        )
        favorable_summary = summarize_metric(
            [value[1] for value in values],
            uncertainty,
            seed_offset=index * 2 + 1,
        )
        results.append(
            {
                "comparison_scope": scope,
                "baseline_pipeline": baseline,
                "compared_pipeline": pipeline,
                "artifact": artifact,
                "metric": metric,
                "preferred_direction": direction,
                **{f"delta_{name}": value for name, value in delta_summary.items()},
                **{f"favorable_delta_{name}": value for name, value in favorable_summary.items()},
            }
        )
    return results


def _study_manifest(root: Path, configuration: SalviAblationConfiguration) -> None:
    outputs = tuple(
        sorted(path for path in root.iterdir() if path.is_file() and path.name != "manifest.json")
    )
    write_json(
        root / "manifest.json",
        {
            "schema_version": 1,
            "experiment_type": "benchmark.salvi_ablation",
            "identifier": configuration.identifier,
            "checksums": {path.relative_to(root).as_posix(): sha256_file(path) for path in outputs},
        },
    )


def run_salvi_ablation(
    configuration: SalviAblationConfiguration,
    *,
    progress: ProgressReporter | None = None,
) -> Path:
    """Execute and compare complete SALVI pipelines over one benchmark."""

    reporter = progress_or_null(progress)
    datasets = _discover_datasets(configuration)
    loaded_pipelines = {
        entry.identifier: load_pipeline_configuration(entry.pipeline_configuration).pipeline
        for entry in configuration.pipelines
    }
    implementation_sha256 = _salvi_implementation_fingerprint()
    root = configuration.output_directory
    root.mkdir(parents=True, exist_ok=True)
    (root / "effective-experiment-configuration.yaml").write_text(
        yaml.safe_dump(
            configuration.model_dump(mode="json"),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    for entry in configuration.pipelines:
        pipeline_snapshot = root / "pipeline-sources" / f"{_segment(entry.identifier)}.yaml"
        _write_pipeline(
            pipeline_snapshot,
            loaded_pipelines[entry.identifier],
        )

    cases: list[_RunCase] = []
    for entry in configuration.pipelines:
        source_pipeline = loaded_pipelines[entry.identifier]
        pipeline_snapshot = root / "pipeline-sources" / f"{_segment(entry.identifier)}.yaml"
        for dataset in datasets:
            effective = _effective_pipeline(
                source_pipeline,
                pattern_binding=configuration.pattern_binding,
                dataset_patterns=dataset.patterns,
            )
            search_pipeline = (
                _without_final_selection_dependents(effective)
                if configuration.selectors
                else effective
            )
            pipeline_sha256 = _hash_json(search_pipeline.model_dump(mode="json"))
            allowed_patterns = effective.patterns.allowed
            for run_seed in configuration.run_seeds:
                fingerprint = _hash_json(
                    {
                        "pipeline": pipeline_sha256,
                        "implementation": implementation_sha256,
                        "dataset": dataset.search_fingerprint,
                        "patterns": [pattern.value for pattern in allowed_patterns],
                        "run_seed": run_seed,
                    }
                )
                cases.append(
                    _RunCase(
                        pipeline_id=entry.identifier,
                        pipeline_path=pipeline_snapshot,
                        dataset=dataset,
                        allowed_patterns=allowed_patterns,
                        run_seed=run_seed,
                        case_root=(
                            root
                            / "cases"
                            / _segment(entry.identifier)
                            / _segment(dataset.identifier)
                            / f"seed-{run_seed}"
                        ),
                        pipeline_sha256=pipeline_sha256,
                        implementation_sha256=implementation_sha256,
                        run_fingerprint=fingerprint,
                        resume=configuration.execution.resume,
                        retry_failed=configuration.execution.retry_failed,
                        strip_final_selection=bool(configuration.selectors),
                    )
                )

    workers = validate_benchmark_parallelism(
        configuration.execution,
        tuple(loaded_pipelines[case.pipeline_id].execution.workers for case in cases),
    )
    reporter.stage(
        f"running {len(cases)} SALVI ablation cases "
        f"({len(configuration.pipelines)} pipelines, {len(datasets)} datasets, "
        f"{len(configuration.run_seeds)} run seeds) with {workers} benchmark worker(s)"
    )
    case_records: list[dict[str, Any]] = []
    if workers <= 1:
        for index, case in enumerate(cases, start=1):
            record = _execute_case(case)
            case_records.append(record)
            reporter.step(
                f"{record['status']}: {case.pipeline_id} / "
                f"{case.dataset.identifier} / seed {case.run_seed}",
                index,
                len(cases),
            )
            if record["status"] == "failed" and configuration.execution.fail_fast:
                raise ExperimentArtifactError(str(record["error"]))
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
        ) as executor:
            futures = {executor.submit(_execute_case, case): case for case in cases}
            for index, future in enumerate(as_completed(futures), start=1):
                case = futures[future]
                record = future.result()
                case_records.append(record)
                reporter.step(
                    f"{record['status']}: {case.pipeline_id} / "
                    f"{case.dataset.identifier} / seed {case.run_seed}",
                    index,
                    len(cases),
                )
                if record["status"] == "failed" and configuration.execution.fail_fast:
                    raise ExperimentArtifactError(str(record["error"]))

    case_records.sort(
        key=lambda record: (
            str(record["pipeline_id"]),
            str(record["dataset_identifier"]),
            int(record["run_seed"]),
        )
    )
    completed = [record for record in case_records if record["status"] == "completed"]
    selection_records: list[dict[str, Any]] = []
    if configuration.selectors and "FINAL" in configuration.metrics.artifacts:
        total_selections = len(completed) * len(configuration.selectors)
        reporter.stage(
            f"applying {len(configuration.selectors)} selector variants offline "
            f"to {len(completed)} completed archives"
        )
        selection_index = 0
        selection_report_interval = max(1, total_selections // 20)
        for case_record in completed:
            for selector_index, selector in enumerate(configuration.selectors):
                selection_index += 1
                selection_record = _apply_selector(
                    configuration,
                    case_record,
                    selector_index,
                )
                selection_records.append(selection_record)
                if (
                    selection_index == 1
                    or selection_index == total_selections
                    or selection_index % selection_report_interval == 0
                    or selection_record["status"] == "failed"
                ):
                    reporter.step(
                        f"{selection_record['status']}: {case_record['pipeline_id']} / "
                        f"{selector.identifier} / {case_record['dataset_identifier']} / "
                        f"seed {case_record['run_seed']}",
                        selection_index,
                        total_selections,
                    )
                if selection_record["status"] == "failed" and configuration.execution.fail_fast:
                    raise ExperimentArtifactError(str(selection_record["error"]))

    selections_by_case: dict[
        tuple[str, str, int],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for selection_record in selection_records:
        selections_by_case[
            (
                str(selection_record["pipeline_id"]),
                str(selection_record["dataset_identifier"]),
                int(selection_record["run_seed"]),
            )
        ].append(selection_record)

    reporter.stage(f"analyzing {len(completed)} completed SALVI runs")
    run_records: list[dict[str, object]] = []
    emitter_records: list[dict[str, object]] = []
    repertoire_records: list[dict[str, object]] = []
    for index, case_record in enumerate(completed, start=1):
        run_record, run_emitters = _run_record(case_record)
        run_records.append(run_record)
        emitter_records.extend(run_emitters)
        if "SEARCH" in configuration.metrics.artifacts:
            repertoire_records.append(
                _repertoire_record(
                    configuration,
                    case_record,
                    artifact="SEARCH",
                )
            )
        if "FINAL" in configuration.metrics.artifacts:
            if not configuration.selectors:
                repertoire_records.append(
                    _repertoire_record(
                        configuration,
                        case_record,
                        artifact="FINAL",
                    )
                )
            else:
                case_key = (
                    str(case_record["pipeline_id"]),
                    str(case_record["dataset_identifier"]),
                    int(case_record["run_seed"]),
                )
                for selection_record in selections_by_case[case_key]:
                    if selection_record["status"] != "completed":
                        continue
                    selector_id = str(selection_record["selector_id"])
                    repertoire_records.append(
                        _repertoire_record(
                            configuration,
                            case_record,
                            artifact="FINAL",
                            artifact_directory=Path(str(selection_record["bicluster_set"])),
                            analysis_pipeline_id=(f"{case_record['pipeline_id']}::{selector_id}"),
                            selector_id=selector_id,
                        )
                    )
        reporter.step(
            f"measured {case_record['pipeline_id']} / "
            f"{case_record['dataset_identifier']} / seed {case_record['run_seed']}",
            index,
            len(completed),
        )

    coverage_metrics = tuple(
        _coverage_name(threshold) for threshold in configuration.metrics.coverage_thresholds
    )
    run_summary = _aggregate(
        run_records,
        group_fields=("pipeline_id",),
        metrics=_RUN_METRICS,
        uncertainty=configuration.metrics.aggregate_uncertainty,
    )
    repertoire_summary = _aggregate(
        repertoire_records,
        group_fields=("pipeline_id", "artifact"),
        metrics=(*_REPERTOIRE_METRICS, *coverage_metrics),
        uncertainty=configuration.metrics.aggregate_uncertainty,
    )
    baseline = configuration.pipelines[0].identifier
    repertoire_paired: list[dict[str, object]] = []
    if configuration.selectors:
        archive_records = [
            record for record in repertoire_records if record["artifact"] == "SEARCH"
        ]
        final_records = [record for record in repertoire_records if record["artifact"] == "FINAL"]
        repertoire_paired.extend(
            _paired_deltas(
                archive_records,
                baseline=baseline,
                artifact_sensitive=True,
                metrics=(*_ACCURACY_METRICS, *coverage_metrics),
                comparison_scope="SEARCH_REPERTOIRE",
            )
        )
        for selector in configuration.selectors:
            repertoire_paired.extend(
                _paired_deltas(
                    [
                        record
                        for record in final_records
                        if record["selector_id"] == selector.identifier
                    ],
                    baseline=f"{baseline}::{selector.identifier}",
                    artifact_sensitive=True,
                    metrics=(*_ACCURACY_METRICS, *coverage_metrics),
                    comparison_scope="SEARCH_WITH_SELECTOR",
                )
            )
        for pipeline in configuration.pipelines:
            repertoire_paired.extend(
                _paired_deltas(
                    [
                        record
                        for record in final_records
                        if record["search_pipeline_id"] == pipeline.identifier
                    ],
                    baseline=(f"{pipeline.identifier}::{configuration.selectors[0].identifier}"),
                    artifact_sensitive=True,
                    metrics=(*_ACCURACY_METRICS, *coverage_metrics),
                    comparison_scope="SELECTOR_WITHIN_SEARCH",
                )
            )
    else:
        repertoire_paired.extend(
            _paired_deltas(
                repertoire_records,
                baseline=baseline,
                artifact_sensitive=True,
                metrics=(*_ACCURACY_METRICS, *coverage_metrics),
                comparison_scope="SEARCH_REPERTOIRE",
            )
        )
    paired = [
        *repertoire_paired,
        *_paired_deltas(
            run_records,
            baseline=baseline,
            artifact_sensitive=False,
            metrics=("wall_time_seconds", "evaluations_per_second"),
            comparison_scope="SEARCH_RUNTIME",
        ),
    ]
    paired_summary = _paired_summary(
        paired,
        configuration.metrics.aggregate_uncertainty,
    )
    differences = _configuration_differences(loaded_pipelines)
    selector_differences = _selector_differences(configuration)

    write_table(root, "case-status", case_records)
    if selection_records:
        write_table(root, "selection-status", selection_records)
    if run_records:
        write_table(root, "run-metrics", run_records)
    if emitter_records:
        write_table(root, "emitter-metrics", emitter_records)
    if repertoire_records:
        write_table(root, "repertoire-metrics", repertoire_records)
    if run_summary:
        write_table(root, "run-summary", run_summary)
    if repertoire_summary:
        write_table(root, "repertoire-summary", repertoire_summary)
    if paired:
        write_table(root, "paired-deltas", paired)
        write_table(root, "paired-summary", paired_summary)
    if differences:
        write_table(root, "configuration-differences", differences)
    if selector_differences:
        write_table(root, "selector-differences", selector_differences)
    if run_records and repertoire_records:
        plot_salvi_ablation(
            root,
            repertoire_records=repertoire_records,
            run_records=run_records,
        )
    report = {
        "schema_version": 1,
        "experiment_type": "benchmark.salvi_ablation",
        "identifier": configuration.identifier,
        "benchmark_root": str(configuration.benchmark_root),
        "dataset_count": len(datasets),
        "pipeline_count": len(configuration.pipelines),
        "selector_count": len(configuration.selectors),
        "run_seed_count": len(configuration.run_seeds),
        "case_count": len(cases),
        "completed_count": len(completed),
        "failed_count": len(cases) - len(completed),
        "selection_count": len(selection_records),
        "completed_selection_count": sum(
            record["status"] == "completed" for record in selection_records
        ),
        "failed_selection_count": sum(record["status"] == "failed" for record in selection_records),
        "baseline_pipeline": baseline,
        "baseline_selector": (
            None if not configuration.selectors else configuration.selectors[0].identifier
        ),
        "pattern_binding": configuration.pattern_binding,
        "artifacts": list(configuration.metrics.artifacts),
        "salvi_implementation_sha256": implementation_sha256,
        "effective_workers": workers,
        "summary": {
            "run": run_summary,
            "repertoire": repertoire_summary,
            "paired": paired_summary,
        },
    }
    write_json(root / "report.json", report)
    _study_manifest(root, configuration)
    return root.resolve()


__all__ = ["run_salvi_ablation"]
