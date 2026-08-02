"""Dataset-level objective alignment against ground truth and matched controls."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from salvi import (
    Bicluster,
    Evaluation,
    ObjectiveDirection,
    ObjectiveValue,
    PatternKind,
    RunBinding,
    ScientificEvaluationBatch,
    ScientificEvaluationService,
    bind_pipeline,
    load_pipeline_configuration,
)
from salvi_experiments.artifacts import (
    atomic_experiment_directory,
    sha256_file,
    write_json,
    write_manifest,
    write_table,
)
from salvi_experiments.configuration import ObjectiveAlignmentConfiguration
from salvi_experiments.dataset.common import read_scoped_ground_truth
from salvi_experiments.exceptions import ExperimentArtifactError
from salvi_experiments.plots import plot_objective_alignment
from salvi_experiments.progress import ProgressReporter, progress_or_null

CONTROL_TYPES = ("RANDOM_MATCHED", "REMOVED", "ADDED")
PROGRESS_BATCH_SIZE = 512


def _number(value: object) -> float:
    if not isinstance(value, int | float):
        raise TypeError(f"expected numeric objective value, received {type(value).__name__}")
    return float(value)


@dataclass(frozen=True, slots=True)
class _AlignmentCandidate:
    identifier: str
    ground_truth_id: str
    source_type: str
    candidate_type: str
    replicate: int
    source_rows: tuple[int, ...]
    source_columns: tuple[int, ...]
    prepared_columns: tuple[int, ...]


def _source_type(patterns: tuple[PatternKind, ...], declared: str | None) -> str:
    if declared:
        return declared
    unique = tuple(sorted({pattern.value for pattern in patterns}))
    return unique[0] if len(unique) == 1 else "MIXED"


def _remove(
    values: tuple[int, ...],
    ratio: float,
    random_source: random.Random,
) -> tuple[int, ...] | None:
    if len(values) <= 1:
        return None
    count = min(len(values) - 1, max(1, math.ceil(len(values) * ratio)))
    removed = set(random_source.sample(values, count))
    return tuple(value for value in values if value not in removed)


def _add(
    values: tuple[int, ...],
    outside: tuple[int, ...],
    ratio: float,
    random_source: random.Random,
) -> tuple[int, ...] | None:
    if not outside:
        return None
    count = min(len(outside), max(1, math.ceil(len(values) * ratio)))
    return tuple(sorted((*values, *random_source.sample(outside, count))))


def _objective_map(values: tuple[ObjectiveValue, ...]) -> dict[str, float]:
    return {objective.name: objective.value for objective in values}


def _favorable(
    ground_truth: float,
    control: float,
    direction: ObjectiveDirection,
) -> bool:
    return (
        ground_truth <= control
        if direction is ObjectiveDirection.MINIMIZE
        else ground_truth >= control
    )


def _improvement(
    ground_truth: float,
    control: float,
    direction: ObjectiveDirection,
) -> float:
    return (
        control - ground_truth
        if direction is ObjectiveDirection.MINIMIZE
        else ground_truth - control
    )


def _generate_candidates(
    configuration: ObjectiveAlignmentConfiguration,
    service: ScientificEvaluationService,
) -> tuple[_AlignmentCandidate, ...]:
    ground_truth, selected = read_scoped_ground_truth(
        service.dataset.metadata.bundle_path,
        configuration.task,
    )
    allowed_patterns = set(service.allowed_patterns)
    selected_patterns = {
        column.pattern for bicluster in selected for column in bicluster.column_patterns
    }
    if not selected_patterns.issubset(allowed_patterns):
        unavailable = ", ".join(
            sorted(pattern.value for pattern in selected_patterns - allowed_patterns)
        )
        raise ExperimentArtifactError(
            f"ground-truth task contains patterns disabled by the run configuration: {unavailable}"
        )

    direct_by_source = {
        column.source_column_index: column.index
        for column in service.dataset.columns
        if column.derivation is None
    }
    missing_sources = {
        column_index
        for bicluster in selected
        for column_index in bicluster.column_indices
        if column_index not in direct_by_source
    }
    if missing_sources:
        raise ExperimentArtifactError(
            "preprocessing removed ground-truth source columns: "
            + ", ".join(map(str, sorted(missing_sources)))
        )
    direct_columns = tuple(
        column for column in service.dataset.columns if column.derivation is None
    )
    pools_by_kind: dict[object, tuple[int, ...]] = {}
    for kind in {column.kind for column in direct_columns}:
        pools_by_kind[kind] = tuple(
            column.index for column in direct_columns if column.kind is kind
        )
    kind_by_prepared = {column.index: column.kind for column in direct_columns}
    source_by_prepared = {column.index: column.source_column_index for column in direct_columns}
    random_source = random.Random(configuration.analysis_seed)
    row_pool = tuple(range(ground_truth.row_count))
    prepared_pool = tuple(column.index for column in direct_columns)
    candidates: list[_AlignmentCandidate] = []

    for target in selected:
        prepared_columns = tuple(direct_by_source[index] for index in target.column_indices)
        patterns = tuple(column.pattern for column in target.column_patterns)
        target_source_type = _source_type(patterns, target.source_type)
        candidates.append(
            _AlignmentCandidate(
                identifier=f"{target.identifier}:GROUND_TRUTH:0000",
                ground_truth_id=target.identifier,
                source_type=target_source_type,
                candidate_type="GROUND_TRUTH",
                replicate=0,
                source_rows=target.row_indices,
                source_columns=target.column_indices,
                prepared_columns=prepared_columns,
            )
        )
        for replicate in range(1, configuration.sampling.random_controls + 1):
            sampled_columns: list[int] = []
            for kind in sorted(
                {kind_by_prepared[index] for index in prepared_columns},
                key=lambda item: item.value,
            ):
                count = sum(kind_by_prepared[index] is kind for index in prepared_columns)
                pool = pools_by_kind[kind]
                if count > len(pool):
                    raise ExperimentArtifactError(
                        f"cannot sample {count} columns of kind {kind.value}"
                    )
                sampled_columns.extend(random_source.sample(pool, count))
            sampled_prepared = tuple(sorted(sampled_columns))
            candidates.append(
                _AlignmentCandidate(
                    identifier=f"{target.identifier}:RANDOM_MATCHED:{replicate:04d}",
                    ground_truth_id=target.identifier,
                    source_type=target_source_type,
                    candidate_type="RANDOM_MATCHED",
                    replicate=replicate,
                    source_rows=tuple(
                        sorted(random_source.sample(row_pool, len(target.row_indices)))
                    ),
                    source_columns=tuple(
                        sorted(source_by_prepared[index] for index in sampled_prepared)
                    ),
                    prepared_columns=sampled_prepared,
                )
            )
        target_rows = set(target.row_indices)
        target_prepared = set(prepared_columns)
        outside_rows = tuple(index for index in row_pool if index not in target_rows)
        outside_columns = tuple(index for index in prepared_pool if index not in target_prepared)
        for replicate in range(1, configuration.sampling.perturbations + 1):
            removed_rows = _remove(
                target.row_indices,
                configuration.sampling.perturbation_ratio,
                random_source,
            )
            removed_columns = _remove(
                prepared_columns,
                configuration.sampling.perturbation_ratio,
                random_source,
            )
            if removed_rows is not None and removed_columns is not None:
                candidates.append(
                    _AlignmentCandidate(
                        identifier=f"{target.identifier}:REMOVED:{replicate:04d}",
                        ground_truth_id=target.identifier,
                        source_type=target_source_type,
                        candidate_type="REMOVED",
                        replicate=replicate,
                        source_rows=removed_rows,
                        source_columns=tuple(
                            source_by_prepared[index] for index in removed_columns
                        ),
                        prepared_columns=removed_columns,
                    )
                )
            added_rows = _add(
                target.row_indices,
                outside_rows,
                configuration.sampling.perturbation_ratio,
                random_source,
            )
            added_columns = _add(
                prepared_columns,
                outside_columns,
                configuration.sampling.perturbation_ratio,
                random_source,
            )
            if added_rows is not None and added_columns is not None:
                candidates.append(
                    _AlignmentCandidate(
                        identifier=f"{target.identifier}:ADDED:{replicate:04d}",
                        ground_truth_id=target.identifier,
                        source_type=target_source_type,
                        candidate_type="ADDED",
                        replicate=replicate,
                        source_rows=added_rows,
                        source_columns=tuple(
                            sorted(source_by_prepared[index] for index in added_columns)
                        ),
                        prepared_columns=added_columns,
                    )
                )
    return tuple(candidates)


def _evaluate_candidates(
    service: ScientificEvaluationService,
    candidates: tuple[_AlignmentCandidate, ...],
    progress: ProgressReporter,
) -> ScientificEvaluationBatch:
    total = len(candidates)
    if total <= PROGRESS_BATCH_SIZE:
        progress.stage(f"evaluating {total} candidates")
        return service.evaluate(
            tuple(
                Bicluster(
                    row_indices=candidate.source_rows,
                    column_indices=candidate.prepared_columns,
                )
                for candidate in candidates
            ),
            identifiers=tuple(candidate.identifier for candidate in candidates),
        )

    evaluations: list[Evaluation] = []
    evaluation_seconds = 0.0
    loading_seconds = 0.0
    preprocessing_seconds = 0.0
    for start in range(0, total, PROGRESS_BATCH_SIZE):
        chunk = candidates[start : start + PROGRESS_BATCH_SIZE]
        batch = service.evaluate(
            tuple(
                Bicluster(
                    row_indices=candidate.source_rows,
                    column_indices=candidate.prepared_columns,
                )
                for candidate in chunk
            ),
            identifiers=tuple(candidate.identifier for candidate in chunk),
        )
        evaluations.extend(batch.evaluations)
        evaluation_seconds += batch.evaluation_seconds
        loading_seconds = batch.loading_seconds
        preprocessing_seconds = batch.preprocessing_seconds
        progress.step(
            "evaluated objective-alignment candidates",
            min(start + len(chunk), total),
            total,
        )
    return ScientificEvaluationBatch(
        evaluations=tuple(evaluations),
        evaluation_seconds=evaluation_seconds,
        loading_seconds=loading_seconds,
        preprocessing_seconds=preprocessing_seconds,
    )


def run_objective_alignment(
    configuration: ObjectiveAlignmentConfiguration,
    *,
    progress: ProgressReporter | None = None,
) -> Path:
    reporter = progress_or_null(progress)
    reporter.stage("loading reusable pipeline")
    pipeline = load_pipeline_configuration(configuration.pipeline_configuration).pipeline
    bound = bind_pipeline(
        pipeline,
        RunBinding(
            identifier=f"{configuration.identifier}-evaluation",
            dataset_bundle=configuration.dataset_bundle,
            output_directory=configuration.output_directory / ".evaluation-runtime",
            seed=configuration.analysis_seed,
        ),
    )
    dataset_bundle = configuration.dataset_bundle
    with ScientificEvaluationService(bound) as service:
        reporter.stage("preparing ground-truth candidates and controls")
        candidates = _generate_candidates(configuration, service)
        reporter.stage(f"generated {len(candidates)} candidates for objective alignment")
        batch = _evaluate_candidates(service, candidates, reporter)
        objective_names = service.objective_names

    if len(batch.evaluations) != len(candidates):
        raise ExperimentArtifactError("scientific evaluation returned an incomplete batch")
    directions: dict[str, ObjectiveDirection] = {}
    candidate_records: list[dict[str, object]] = []
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for definition, evaluation in zip(candidates, batch.evaluations, strict=True):
        objective_values = _objective_map(evaluation.objectives)
        directions.update(
            {objective.name: objective.direction for objective in evaluation.objectives}
        )
        record: dict[str, object] = {
            "candidate_id": definition.identifier,
            "ground_truth_id": definition.ground_truth_id,
            "source_type": definition.source_type,
            "candidate_type": definition.candidate_type,
            "replicate": definition.replicate,
            "row_count": len(definition.source_rows),
            "column_count": len(definition.source_columns),
            "row_indices": list(definition.source_rows),
            "source_column_indices": list(definition.source_columns),
            "prepared_column_indices": list(definition.prepared_columns),
            "valid": evaluation.valid,
            "feasible": evaluation.feasible,
            "constraint_violation": evaluation.constraint_violation,
            "issues": [issue.code.value for issue in evaluation.issues],
        }
        record.update({name: objective_values.get(name) for name in objective_names})
        candidate_records.append(record)
        grouped[definition.ground_truth_id].append(record)

    if set(directions) != set(objective_names):
        raise ExperimentArtifactError(
            "no valid evaluation exposed every configured objective direction"
        )
    summary_records: list[dict[str, object]] = []
    for ground_truth_id, records in grouped.items():
        exact = next(record for record in records if record["candidate_type"] == "GROUND_TRUTH")
        summary: dict[str, object] = {
            "ground_truth_id": ground_truth_id,
            "source_type": exact["source_type"],
            "row_count": exact["row_count"],
            "column_count": exact["column_count"],
            "valid": exact["valid"],
        }
        for objective_name in objective_names:
            exact_value = exact[objective_name]
            summary[objective_name] = exact_value
            for candidate_type in CONTROL_TYPES:
                slug = candidate_type.lower()
                control_values = [
                    _number(record[objective_name])
                    for record in records
                    if record["candidate_type"] == candidate_type
                    and bool(record["valid"])
                    and record[objective_name] is not None
                ]
                summary[f"{objective_name}_{slug}_count"] = len(control_values)
                if exact_value is None or not control_values:
                    summary[f"{objective_name}_{slug}_median"] = None
                    summary[f"{objective_name}_{slug}_favorable_fraction"] = None
                    summary[f"{objective_name}_{slug}_mean_improvement"] = None
                else:
                    sorted_controls = sorted(control_values)
                    middle = len(sorted_controls) // 2
                    median = (
                        sorted_controls[middle]
                        if len(sorted_controls) % 2
                        else (sorted_controls[middle - 1] + sorted_controls[middle]) / 2.0
                    )
                    direction = directions[objective_name]
                    summary[f"{objective_name}_{slug}_median"] = median
                    summary[f"{objective_name}_{slug}_favorable_fraction"] = sum(
                        1
                        for control in control_values
                        if _favorable(_number(exact_value), control, direction)
                    ) / len(control_values)
                    summary[f"{objective_name}_{slug}_mean_improvement"] = sum(
                        _improvement(_number(exact_value), control, direction)
                        for control in control_values
                    ) / len(control_values)
        summary_records.append(summary)

    reporter.stage("summarizing objective-alignment controls")
    aggregates: dict[str, object] = {}
    for objective_name in objective_names:
        for candidate_type in CONTROL_TYPES:
            slug = candidate_type.lower()
            key = f"{objective_name}_{slug}_favorable_fraction"
            aggregate_values = [
                _number(record[key]) for record in summary_records if record[key] is not None
            ]
            aggregates[key] = (
                sum(aggregate_values) / len(aggregate_values) if aggregate_values else None
            )

    reporter.stage(f"writing objective-alignment artifacts to {configuration.output_directory}")
    with atomic_experiment_directory(
        configuration.output_directory,
        overwrite=configuration.overwrite,
    ) as temporary:
        write_table(temporary, "candidates", candidate_records)
        write_table(temporary, "objective-alignment", summary_records)
        plot_objective_alignment(
            temporary,
            objective_names=objective_names,
            candidate_records=candidate_records,
            summary_records=summary_records,
        )
        report = {
            "schema_version": 1,
            "experiment_type": "dataset.objective_alignment",
            "identifier": configuration.identifier,
            "dataset_identifier": service.dataset.metadata.identifier,
            "dataset_bundle": str(dataset_bundle),
            "pipeline_configuration": str(configuration.pipeline_configuration),
            "pipeline_configuration_sha256": sha256_file(configuration.pipeline_configuration),
            "task": configuration.task.model_dump(mode="json"),
            "sampling": configuration.sampling.model_dump(mode="json"),
            "analysis_seed": configuration.analysis_seed,
            "objectives": [
                {"name": name, "direction": directions[name].value} for name in objective_names
            ],
            "ground_truth_count": len(summary_records),
            "candidate_count": len(candidate_records),
            "valid_candidate_count": sum(bool(record["valid"]) for record in candidate_records),
            "timing_seconds": {
                "loading": batch.loading_seconds,
                "preprocessing": batch.preprocessing_seconds,
                "evaluation": batch.evaluation_seconds,
            },
            "aggregate_alignment": aggregates,
        }
        write_json(temporary / "report.json", report)
        write_manifest(
            temporary,
            experiment_type="dataset.objective_alignment",
            identifier=configuration.identifier,
            metadata={
                "dataset_identifier": service.dataset.metadata.identifier,
                "pipeline_configuration_sha256": report["pipeline_configuration_sha256"],
            },
        )
    return configuration.output_directory.resolve()


__all__ = ["run_objective_alignment"]
