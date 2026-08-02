"""Bounded durable-event serialization driven by observer requirements."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from salvi.components.protocols import (
    EventPayloadRequirement,
    Observer,
    ObserverPayloadRequirements,
)
from salvi.domain.models import Evaluation
from salvi.domain.search import EvaluationBatch

_EVALUATION_ITEM_REQUIREMENTS = frozenset(
    {
        EventPayloadRequirement.CANDIDATE_STRUCTURE,
        EventPayloadRequirement.EVALUATION_CONSTRAINTS,
        EventPayloadRequirement.EVALUATION_DESCRIPTORS,
        EventPayloadRequirement.EVALUATION_ISSUES,
        EventPayloadRequirement.EVALUATION_OBJECTIVES,
    }
)


def collect_payload_requirements(
    observers: Sequence[Observer],
) -> frozenset[EventPayloadRequirement]:
    return frozenset(
        requirement
        for observer in observers
        if isinstance(observer, ObserverPayloadRequirements)
        for requirement in observer.event_payload_requirements
    )


def evaluation_batch_payload(
    batch: EvaluationBatch,
    *,
    evaluations: int,
    requirements: frozenset[EventPayloadRequirement],
) -> dict[str, Any]:
    runtime: dict[str, Any] = {
        "duration_seconds": batch.duration_seconds,
        "worker_count": batch.worker_count,
        "peak_in_flight": batch.peak_in_flight,
        "integration_mode": batch.integration_mode.value,
    }
    if EventPayloadRequirement.COMPONENT_TIMINGS in requirements:
        runtime.update(
            candidate_duration_seconds=list(batch.candidate_duration_seconds),
            component_duration_seconds=dict(batch.component_duration_seconds),
        )
    payload: dict[str, Any] = {
        "count": len(batch.evaluations),
        "evaluations": evaluations,
        "runtime": runtime,
    }
    if requirements & _EVALUATION_ITEM_REQUIREMENTS:
        payload["items"] = [
            _evaluation_payload(evaluation, requirements) for evaluation in batch.evaluations
        ]
    return payload


def runtime_payload(
    duration_seconds: float,
    component_duration_seconds: dict[str, float],
    requirements: frozenset[EventPayloadRequirement],
) -> dict[str, Any]:
    payload: dict[str, Any] = {"duration_seconds": duration_seconds}
    if EventPayloadRequirement.COMPONENT_TIMINGS in requirements:
        payload["component_duration_seconds"] = component_duration_seconds
    return payload


def _evaluation_payload(
    evaluation: Evaluation,
    requirements: frozenset[EventPayloadRequirement],
) -> dict[str, Any]:
    bicluster = evaluation.candidate.bicluster
    payload: dict[str, Any] = {
        "identifier": evaluation.candidate.identifier,
        "signature": bicluster.signature,
        "valid": evaluation.valid,
        "feasible": evaluation.feasible,
        "constraint_violation": evaluation.constraint_violation,
        "row_count": len(bicluster.row_indices),
        "column_count": len(bicluster.column_indices),
    }
    if EventPayloadRequirement.CANDIDATE_STRUCTURE in requirements:
        payload["rows"] = list(bicluster.row_indices)
        payload["columns"] = list(bicluster.column_indices)
    if EventPayloadRequirement.EVALUATION_ISSUES in requirements:
        payload["issues"] = [issue.code.value for issue in evaluation.issues]
    if EventPayloadRequirement.EVALUATION_OBJECTIVES in requirements:
        payload["objectives"] = [
            {
                "name": objective.name,
                "value": objective.value,
                "direction": objective.direction.value,
            }
            for objective in evaluation.objectives
        ]
    if EventPayloadRequirement.EVALUATION_DESCRIPTORS in requirements:
        payload["descriptors"] = [
            {"name": descriptor.name, "value": descriptor.value}
            for descriptor in evaluation.descriptors
        ]
    if EventPayloadRequirement.EVALUATION_CONSTRAINTS in requirements:
        payload["constraints"] = [
            {
                "name": constraint.name,
                "value": constraint.value,
                "violation": constraint.violation,
            }
            for constraint in evaluation.constraints
        ]
    return payload


__all__ = [
    "collect_payload_requirements",
    "evaluation_batch_payload",
    "runtime_payload",
]
