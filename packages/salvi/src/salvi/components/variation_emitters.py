"""Generic QD emitters backed by reusable variation operators."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Annotated

import numpy.random as npr
from pydantic import BaseModel, ConfigDict, Field

from salvi.application.context import QdRunContext, RunContext, require_qd_run_context
from salvi.components.candidate_initialization import _candidate, _random_candidate
from salvi.domain.models import Candidate, Evaluation, Repertoire
from salvi.exceptions import ComponentError


class CrossoverEmitterConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_attempts: Annotated[int, Field(ge=1)] = 8


class MutationEmitterConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    guided_parent_selection: bool = False
    parent_pool_size: Annotated[int, Field(ge=1)] = 16
    max_attempts: Annotated[int, Field(ge=1)] = 8


@dataclass(slots=True)
class CrossoverEmitter:
    """Apply the configured mate-selection policy and crossover operator."""

    max_attempts: int = 8
    component_name: str = "crossover"
    provides: frozenset[str] = frozenset({"emitter"})
    requires: frozenset[str] = frozenset(
        {
            "archive",
            "candidate-validity",
            "prepared-dataset",
            "mate-selection",
            "crossover-operator",
        }
    )
    _component_timings: dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        CrossoverEmitterConfiguration(max_attempts=self.max_attempts)

    def emit(
        self,
        context: RunContext,
        repertoire: Repertoire,
        count: int,
        *,
        start_sequence: int = 0,
    ) -> Sequence[Candidate]:
        if count < 0:
            raise ValueError("emitter count must be non-negative")
        qd_context = require_qd_run_context(context)
        if qd_context.mate_selection_policy is None or qd_context.crossover_operator is None:
            raise ComponentError(
                "crossover emitter requires configured mate selection and crossover operator"
            )
        generator = context.random_generator("emitter.crossover")
        candidates: list[Candidate] = []
        for index in range(count):
            sequence = start_sequence + index
            child: Candidate | None = None
            for _ in range(self.max_attempts):
                selection_started = perf_counter()
                pair = qd_context.mate_selection_policy.select(repertoire, generator)
                self._record_timing(
                    f"mate_selection.{qd_context.mate_selection_policy.component_name}",
                    perf_counter() - selection_started,
                )
                if pair is None:
                    break
                first, second = pair
                crossover_started = perf_counter()
                bicluster = qd_context.crossover_operator.cross(
                    qd_context,
                    first,
                    second,
                    generator,
                )
                self._record_timing(
                    f"crossover.{qd_context.crossover_operator.component_name}",
                    perf_counter() - crossover_started,
                )
                proposal = _candidate(
                    producer=self.component_name,
                    operation=qd_context.crossover_operator.component_name,
                    sequence=sequence,
                    rows=bicluster.row_indices,
                    columns=bicluster.column_indices,
                    generation=max(
                        first.candidate.generation,
                        second.candidate.generation,
                    )
                    + 1,
                    parents=(first.candidate.identifier, second.candidate.identifier),
                )
                if proposal.bicluster.signature not in {
                    first.candidate.bicluster.signature,
                    second.candidate.bicluster.signature,
                }:
                    child = proposal
                    break
            if child is None:
                child = _random_candidate(
                    qd_context,
                    generator,
                    producer=self.component_name,
                    operation="restart_fallback",
                    sequence=sequence,
                )
            qd_context.candidate_validity_policy.validate(child, qd_context.dataset)
            candidates.append(child)
        return tuple(candidates)

    def drain_component_timings(self) -> tuple[tuple[str, float], ...]:
        timings = tuple(sorted(self._component_timings.items()))
        self._component_timings.clear()
        return timings

    def _record_timing(self, name: str, duration: float) -> None:
        self._component_timings[name] = self._component_timings.get(name, 0.0) + duration


@dataclass(slots=True)
class MutationEmitter:
    """Apply the configured mutation operator to selected archive members."""

    guided_parent_selection: bool = False
    parent_pool_size: int = 16
    max_attempts: int = 8
    component_name: str = "mutation"
    provides: frozenset[str] = frozenset({"emitter"})
    requires: frozenset[str] = frozenset(
        {
            "archive",
            "candidate-validity",
            "prepared-dataset",
            "parent-selection",
            "mutation-operator",
        }
    )
    _component_timings: dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        MutationEmitterConfiguration(
            guided_parent_selection=self.guided_parent_selection,
            parent_pool_size=self.parent_pool_size,
            max_attempts=self.max_attempts,
        )

    def _select_parent(
        self,
        context: QdRunContext,
        repertoire: Repertoire,
        generator: npr.Generator,
    ) -> Evaluation | None:
        if context.parent_selection_policy is None:
            raise ComponentError("mutation emitter requires a parent-selection policy")
        started = perf_counter()
        selected = context.parent_selection_policy.select(
            repertoire,
            generator,
            pool_size=self.parent_pool_size,
            eligible=lambda _: True,
            guided=self.guided_parent_selection,
        )
        self._record_timing(
            f"parent_selection.{context.parent_selection_policy.component_name}",
            perf_counter() - started,
        )
        return selected

    def emit(
        self,
        context: RunContext,
        repertoire: Repertoire,
        count: int,
        *,
        start_sequence: int = 0,
    ) -> Sequence[Candidate]:
        if count < 0:
            raise ValueError("emitter count must be non-negative")
        qd_context = require_qd_run_context(context)
        if qd_context.mutation_operator is None:
            raise ComponentError("mutation emitter requires a configured mutation operator")
        generator = context.random_generator("emitter.mutation")
        candidates: list[Candidate] = []
        for index in range(count):
            sequence = start_sequence + index
            child: Candidate | None = None
            for _ in range(self.max_attempts):
                parent = self._select_parent(qd_context, repertoire, generator)
                if parent is None:
                    break
                mutation_started = perf_counter()
                bicluster = qd_context.mutation_operator.mutate(qd_context, parent, generator)
                self._record_timing(
                    f"mutation.{qd_context.mutation_operator.component_name}",
                    perf_counter() - mutation_started,
                )
                proposal = _candidate(
                    producer=self.component_name,
                    operation=qd_context.mutation_operator.component_name,
                    sequence=sequence,
                    rows=bicluster.row_indices,
                    columns=bicluster.column_indices,
                    generation=parent.candidate.generation + 1,
                    parents=(parent.candidate.identifier,),
                )
                if proposal.bicluster.signature != parent.candidate.bicluster.signature:
                    child = proposal
                    break
            if child is None:
                child = _random_candidate(
                    qd_context,
                    generator,
                    producer=self.component_name,
                    operation="restart_fallback",
                    sequence=sequence,
                )
            qd_context.candidate_validity_policy.validate(child, qd_context.dataset)
            candidates.append(child)
        return tuple(candidates)

    def drain_component_timings(self) -> tuple[tuple[str, float], ...]:
        timings = tuple(sorted(self._component_timings.items()))
        self._component_timings.clear()
        return timings

    def _record_timing(self, name: str, duration: float) -> None:
        self._component_timings[name] = self._component_timings.get(name, 0.0) + duration


__all__ = [
    "CrossoverEmitter",
    "CrossoverEmitterConfiguration",
    "MutationEmitter",
    "MutationEmitterConfiguration",
]
