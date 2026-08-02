"""Public service for evaluating explicit biclusters without running a search."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Self

from salvi.application.configuration import SalviConfiguration
from salvi.application.factory import build_specification, prepare_run
from salvi.components.registry import ComponentRegistry
from salvi.domain.enums import PatternKind
from salvi.domain.models import (
    Bicluster,
    Candidate,
    CandidateProvenance,
    Evaluation,
)
from salvi.domain.prepared import PreparedDataset
from salvi.evaluation.workspace import EvaluationWorkspace


@dataclass(frozen=True, slots=True)
class ScientificEvaluationBatch:
    """Exact evaluations and measured wall time for one explicit batch."""

    evaluations: tuple[Evaluation, ...]
    evaluation_seconds: float
    loading_seconds: float
    preprocessing_seconds: float


class ScientificEvaluationService:
    """Prepare one configured scientific runtime and evaluate supplied biclusters.

    The service uses the same preprocessing pipeline, pattern configuration,
    objectives, policies, executor, and worker count as a normal SALVI run. Search,
    archive, emitter, and final-selection components are composed and validated
    from the fully bound effective configuration but are never executed.
    """

    def __init__(
        self,
        configuration: SalviConfiguration,
        *,
        registry: ComponentRegistry | None = None,
    ) -> None:
        specification = build_specification(configuration, registry)
        prepared = prepare_run(specification)
        self._specification = specification
        self._prepared = prepared
        self._workspace = EvaluationWorkspace(prepared.context)
        self._closed = False

    @property
    def dataset(self) -> PreparedDataset:
        return self._prepared.context.dataset

    @property
    def objective_names(self) -> tuple[str, ...]:
        return tuple(objective.component_name for objective in self._specification.objectives)

    @property
    def constraint_names(self) -> tuple[str, ...]:
        return tuple(constraint.component_name for constraint in self._specification.constraints)

    @property
    def allowed_patterns(self) -> tuple[PatternKind, ...]:
        return self._prepared.context.patterns.allowed

    def evaluate(
        self,
        biclusters: Sequence[Bicluster],
        *,
        identifiers: Sequence[str] | None = None,
    ) -> ScientificEvaluationBatch:
        if self._closed:
            raise RuntimeError("scientific evaluation service is closed")
        if identifiers is None:
            identifiers = tuple(f"explicit-{index:06d}" for index in range(len(biclusters)))
        if len(identifiers) != len(biclusters):
            raise ValueError("identifiers must align with supplied biclusters")
        normalized_identifiers = tuple(str(identifier) for identifier in identifiers)
        if any(not identifier.strip() for identifier in normalized_identifiers):
            raise ValueError("evaluation identifiers must not be blank")
        if len(set(normalized_identifiers)) != len(normalized_identifiers):
            raise ValueError("evaluation identifiers must be unique")

        candidates = tuple(
            Candidate(
                identifier=identifier,
                bicluster=bicluster,
                provenance=CandidateProvenance(
                    producer="scientific_evaluation",
                    operation="explicit_candidate",
                    sequence=index,
                ),
            )
            for index, (identifier, bicluster) in enumerate(
                zip(normalized_identifiers, biclusters, strict=True)
            )
        )
        started = perf_counter()
        batch = self._specification.executor.evaluate(
            candidates,
            self._specification.objectives,
            (),
            self._workspace,
            constraints=self._specification.constraints,
            worker_count=self._specification.worker_count,
        )
        evaluation_seconds = perf_counter() - started
        preprocessing_seconds = sum(
            step.duration_seconds for step in self._prepared.preprocessing.steps
        )
        return ScientificEvaluationBatch(
            evaluations=batch.evaluations,
            evaluation_seconds=evaluation_seconds,
            loading_seconds=self._prepared.preprocessing.loading_seconds,
            preprocessing_seconds=preprocessing_seconds,
        )

    def close(self) -> None:
        if not self._closed:
            self._specification.executor.close()
            self._closed = True

    def __enter__(self) -> Self:
        if self._closed:
            raise RuntimeError("scientific evaluation service is closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = ["ScientificEvaluationBatch", "ScientificEvaluationService"]
