"""Bounded pattern-aware local-refinement emitters."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, cast

import numpy.random as npr
from pydantic import BaseModel, ConfigDict, Field

from salvi.application.context import RunContext
from salvi.components.candidate_initialization import (
    Dimension,
    MembershipOperation,
    _candidate,
    _random_candidate,
)
from salvi.components.membership_emitters import (
    _guided_membership_move,
    _membership_options,
    _select_parent,
)
from salvi.domain.models import Candidate, Evaluation, Repertoire
from salvi.exceptions import ComponentError


class AlternatingPatternLocalSearchEmitterConfiguration(BaseModel):
    """Configuration for one-step alternating pattern-guided refinement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    parent_pool_size: Annotated[int, Field(ge=1)] = 16
    candidate_pool_size: Annotated[int, Field(ge=1)] = 64
    cardinality_change_probability: Annotated[float, Field(ge=0.0, le=1.0)] = 0.25
    quality_parent_probability: Annotated[float, Field(ge=0.0, le=1.0)] = 0.25


def _continuation_dimension(
    parent: Evaluation,
    sequence: int,
    component_name: str,
) -> Dimension:
    provenance = parent.candidate.provenance
    if provenance is not None and provenance.producer == component_name:
        if provenance.operation.endswith("_row"):
            return "columns"
        if provenance.operation.endswith("_column"):
            return "rows"
    return "rows" if sequence % 2 == 0 else "columns"


def _valid_operations(
    context: RunContext,
    parent: Evaluation,
    dimension: Dimension,
) -> tuple[MembershipOperation, ...]:
    bounds = context.candidate_validity_policy.bounds(context.dataset)
    selected = (
        parent.candidate.bicluster.row_indices
        if dimension == "rows"
        else parent.candidate.bicluster.column_indices
    )
    population = context.dataset.row_count if dimension == "rows" else context.dataset.column_count
    minimum = bounds.min_rows if dimension == "rows" else bounds.min_columns
    return tuple(
        operation
        for operation in cast(
            tuple[MembershipOperation, ...],
            ("add", "remove", "swap"),
        )
        if _membership_options(selected, population, minimum, operation)
    )


def _choose_local_operation(
    operations: tuple[MembershipOperation, ...],
    generator: npr.Generator,
    cardinality_change_probability: float,
) -> MembershipOperation:
    if "swap" in operations and generator.random() >= cardinality_change_probability:
        return "swap"
    cardinality_changes = tuple(
        operation for operation in operations if operation in {"add", "remove"}
    )
    choices = cardinality_changes or operations
    return choices[int(generator.integers(0, len(choices)))]


@dataclass(frozen=True, slots=True)
class AlternatingPatternLocalSearchEmitter:
    """Propose one guided move and alternate rows and columns across continuations."""

    parent_pool_size: int = 16
    candidate_pool_size: int = 64
    cardinality_change_probability: float = 0.25
    quality_parent_probability: float = 0.25
    component_name: str = "alternating_pattern_local_search"
    provides: frozenset[str] = frozenset({"emitter"})
    requires: frozenset[str] = frozenset(
        {"archive", "candidate-validity", "parent-selection", "prepared-dataset"}
    )

    def __post_init__(self) -> None:
        AlternatingPatternLocalSearchEmitterConfiguration(
            parent_pool_size=self.parent_pool_size,
            candidate_pool_size=self.candidate_pool_size,
            cardinality_change_probability=self.cardinality_change_probability,
            quality_parent_probability=self.quality_parent_probability,
        )

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
        generator = context.random_generator(f"emitter.{self.component_name}")
        candidates: list[Candidate] = []
        for index in range(count):
            sequence = start_sequence + index
            parent = _select_parent(
                context,
                repertoire,
                generator,
                pool_size=self.parent_pool_size,
                guided=generator.random() < self.quality_parent_probability,
                eligible=lambda evaluation: evaluation.pattern_fit is not None,
            )
            if parent is None:
                candidates.append(
                    _random_candidate(
                        context,
                        generator,
                        producer=self.component_name,
                        operation="restart_fallback",
                        sequence=sequence,
                    )
                )
                continue

            preferred = _continuation_dimension(parent, sequence, self.component_name)
            alternate: Dimension = "columns" if preferred == "rows" else "rows"
            dimension = preferred
            operations = _valid_operations(context, parent, dimension)
            if not operations:
                dimension = alternate
                operations = _valid_operations(context, parent, dimension)
            if not operations:
                raise ComponentError("local-search parent has no valid membership move")
            operation = _choose_local_operation(
                operations,
                generator,
                self.cardinality_change_probability,
            )
            moved = _guided_membership_move(
                context,
                parent,
                dimension,
                operation,
                generator,
                self.candidate_pool_size,
            )
            if moved is None:
                raise ComponentError("local-search move could not be generated")
            bicluster = parent.candidate.bicluster
            rows = moved if dimension == "rows" else bicluster.row_indices
            columns = moved if dimension == "columns" else bicluster.column_indices
            candidate = _candidate(
                producer=self.component_name,
                operation=f"{operation}_{dimension[:-1]}",
                sequence=sequence,
                rows=rows,
                columns=columns,
                generation=parent.candidate.generation + 1,
                parents=(parent.candidate.identifier,),
            )
            context.candidate_validity_policy.validate(candidate, context.dataset)
            candidates.append(candidate)
        return tuple(candidates)


__all__ = [
    "AlternatingPatternLocalSearchEmitter",
    "AlternatingPatternLocalSearchEmitterConfiguration",
]
