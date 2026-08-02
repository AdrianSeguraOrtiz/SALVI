"""Reusable crossover and mutation operators over bicluster membership."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Annotated

import numpy as np
import numpy.random as npr
from pydantic import BaseModel, ConfigDict, Field

from salvi.application.context import RunContext
from salvi.components.backend_operators import BackendOperatorSpec
from salvi.components.parent_selection import evaluation_loss
from salvi.domain.enums import PatternKind
from salvi.domain.models import Bicluster, Evaluation


class MembershipRecombinationConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    application_probability: Annotated[float, Field(ge=0.0, le=1.0)] = 0.9
    row_exchange_probability: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    column_exchange_probability: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5


class BitFlipMutationConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    application_probability: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    bit_probability: Annotated[float, Field(gt=0.0, le=1.0)] | None = None


def _recombine_membership(
    first: tuple[int, ...],
    second: tuple[int, ...],
    probability: float,
    minimum: int,
    maximum: int,
    generator: npr.Generator,
) -> tuple[int, ...]:
    if generator.random() >= probability:
        return first
    first_set = set(first)
    second_set = set(second)
    union = tuple(sorted(first_set | second_set))
    lower = max(minimum, min(len(first), len(second)))
    upper = min(maximum, max(len(first), len(second)), len(union))
    target = int(generator.integers(lower, upper + 1))
    shared = tuple(sorted(first_set & second_set))
    if len(shared) >= target:
        positions = generator.choice(len(shared), target, replace=False)
        return tuple(sorted(shared[int(position)] for position in positions))
    shared_set = set(shared)
    additions = tuple(value for value in union if value not in shared_set)
    selected_additions = tuple(
        additions[int(position)]
        for position in generator.choice(
            len(additions),
            target - len(shared),
            replace=False,
        )
    )
    return tuple(sorted((*shared, *selected_additions)))


def _column_evidence(first: Evaluation, second: Evaluation) -> dict[int, float]:
    by_column: dict[int, list[float]] = {}
    for evaluation in (first, second):
        fallback = evaluation_loss(evaluation)
        losses = evaluation.mean_column_losses
        for column in evaluation.candidate.bicluster.column_indices:
            by_column.setdefault(column, []).append(losses.get(column, fallback))
    return {
        column: (
            sum(finite_values) / len(finite_values)
            if (finite_values := tuple(value for value in values if math.isfinite(value)))
            else 1.0
        )
        for column, values in by_column.items()
    }


def _weighted_without_replacement(
    values: tuple[int, ...],
    count: int,
    losses: dict[int, float],
    generator: npr.Generator,
) -> tuple[int, ...]:
    if count <= 0:
        return ()
    if count >= len(values):
        return values
    weights = np.asarray(
        [0.05 + max(0.0, 1.0 - min(1.0, losses.get(column, 1.0))) for column in values],
        dtype=np.float64,
    )
    positions = generator.choice(
        len(values),
        count,
        replace=False,
        p=weights / float(np.sum(weights)),
    )
    return tuple(values[int(position)] for position in positions)


def _joint_groups(evaluation: Evaluation) -> tuple[tuple[int, ...], ...]:
    if evaluation.pattern_fit is None:
        return ()
    return tuple(
        group.column_indices
        for group in evaluation.pattern_fit.groups
        if group.pattern in {PatternKind.ADDITIVE, PatternKind.MULTIPLICATIVE}
    )


def _repair_joint_group_support(
    columns: tuple[int, ...],
    shared: frozenset[int],
    first: Evaluation,
    second: Evaluation,
    losses: dict[int, float],
) -> tuple[int, ...]:
    selected = set(columns)
    available = set(first.candidate.bicluster.column_indices) | set(
        second.candidate.bicluster.column_indices
    )
    for group in (*_joint_groups(first), *_joint_groups(second)):
        if len(selected & set(group)) != 1:
            continue
        siblings = tuple(
            sorted(
                (column for column in group if column in available and column not in selected),
                key=lambda column: (losses.get(column, 1.0), column),
            )
        )
        removable = tuple(
            sorted(
                (column for column in selected if column not in shared),
                key=lambda column: (-losses.get(column, 1.0), column),
            )
        )
        if siblings and removable:
            selected.remove(removable[0])
            selected.add(siblings[0])
    return tuple(sorted(selected))


def _evidence_weighted_columns(
    first: Evaluation,
    second: Evaluation,
    probability: float,
    minimum: int,
    maximum: int,
    generator: npr.Generator,
) -> tuple[int, ...]:
    first_columns = first.candidate.bicluster.column_indices
    second_columns = second.candidate.bicluster.column_indices
    if generator.random() >= probability:
        return first_columns
    first_set = set(first_columns)
    second_set = set(second_columns)
    shared = frozenset(first_set & second_set)
    union = tuple(sorted(first_set | second_set))
    lower = max(minimum, min(len(first_columns), len(second_columns)))
    upper = min(maximum, max(len(first_columns), len(second_columns)), len(union))
    target = int(generator.integers(lower, upper + 1))
    additions = tuple(column for column in union if column not in shared)
    losses = _column_evidence(first, second)
    selected = tuple(
        sorted(
            (
                *shared,
                *_weighted_without_replacement(
                    additions,
                    target - len(shared),
                    losses,
                    generator,
                ),
            )
        )
    )
    return _repair_joint_group_support(selected, shared, first, second, losses)


@dataclass(frozen=True, slots=True)
class MembershipRecombinationCrossover:
    """Recombine row and column memberships without scientific reevaluation."""

    application_probability: float = 0.9
    row_exchange_probability: float = 0.5
    column_exchange_probability: float = 0.5
    component_name: str = "membership_recombination"
    provides: frozenset[str] = frozenset({"crossover-operator"})
    requires: frozenset[str] = frozenset({"candidate-validity", "prepared-dataset"})

    def __post_init__(self) -> None:
        MembershipRecombinationConfiguration(
            application_probability=self.application_probability,
            row_exchange_probability=self.row_exchange_probability,
            column_exchange_probability=self.column_exchange_probability,
        )

    def cross(
        self,
        context: RunContext,
        first: Evaluation,
        second: Evaluation,
        generator: npr.Generator,
    ) -> Bicluster:
        base = first.candidate.bicluster
        if generator.random() >= self.application_probability:
            return base
        bounds = context.candidate_validity_policy.bounds(context.dataset)
        return Bicluster(
            row_indices=_recombine_membership(
                base.row_indices,
                second.candidate.bicluster.row_indices,
                self.row_exchange_probability,
                bounds.min_rows,
                bounds.max_rows,
                generator,
            ),
            column_indices=_recombine_membership(
                base.column_indices,
                second.candidate.bicluster.column_indices,
                self.column_exchange_probability,
                bounds.min_columns,
                bounds.max_columns,
                generator,
            ),
        )


@dataclass(frozen=True, slots=True)
class EvidenceWeightedRecombinationCrossover(MembershipRecombinationCrossover):
    """Favor columns with strong persisted objective evidence in both parents."""

    component_name: str = "evidence_weighted_recombination"
    requires: frozenset[str] = frozenset({"candidate-validity", "prepared-dataset", "objective"})

    def cross(
        self,
        context: RunContext,
        first: Evaluation,
        second: Evaluation,
        generator: npr.Generator,
    ) -> Bicluster:
        base = first.candidate.bicluster
        if generator.random() >= self.application_probability:
            return base
        bounds = context.candidate_validity_policy.bounds(context.dataset)
        return Bicluster(
            row_indices=_recombine_membership(
                base.row_indices,
                second.candidate.bicluster.row_indices,
                self.row_exchange_probability,
                bounds.min_rows,
                bounds.max_rows,
                generator,
            ),
            column_indices=_evidence_weighted_columns(
                first,
                second,
                self.column_exchange_probability,
                bounds.min_columns,
                bounds.max_columns,
                generator,
            ),
        )


def _repair_minimum(
    selected: set[int],
    *,
    population: int,
    minimum: int,
    generator: npr.Generator,
) -> tuple[int, ...]:
    missing = minimum - len(selected)
    if missing > 0:
        available = np.asarray(
            tuple(index for index in range(population) if index not in selected),
            dtype=np.int64,
        )
        additions = generator.choice(available, missing, replace=False)
        selected.update(int(value) for value in additions)
    return tuple(sorted(selected))


@dataclass(frozen=True, slots=True)
class BitFlipMembershipMutation:
    """Flip row and column memberships and repair configured minima."""

    application_probability: float = 1.0
    bit_probability: float | None = None
    component_name: str = "bit_flip_membership"
    provides: frozenset[str] = frozenset({"mutation-operator", "pymoo-mutation"})
    requires: frozenset[str] = frozenset({"candidate-validity", "prepared-dataset"})

    def __post_init__(self) -> None:
        BitFlipMutationConfiguration(
            application_probability=self.application_probability,
            bit_probability=self.bit_probability,
        )

    def backend_operator_spec(self, backend: str) -> BackendOperatorSpec:
        if backend != "pymoo":
            raise ValueError(f"bit_flip_membership does not support backend {backend!r}")
        arguments: list[tuple[str, object]] = [
            ("prob", self.application_probability),
        ]
        if self.bit_probability is not None:
            arguments.append(("prob_var", self.bit_probability))
        return BackendOperatorSpec(
            factory_path="pymoo.operators.mutation.bitflip:BitflipMutation",
            keyword_arguments=tuple(arguments),
        )

    def mutate(
        self,
        context: RunContext,
        parent: Evaluation,
        generator: npr.Generator,
    ) -> Bicluster:
        bicluster = parent.candidate.bicluster
        if generator.random() >= self.application_probability:
            return bicluster
        dimension = context.dataset.row_count + context.dataset.column_count
        probability = self.bit_probability if self.bit_probability is not None else 1.0 / dimension
        rows = set(bicluster.row_indices)
        columns = set(bicluster.column_indices)
        for row in range(context.dataset.row_count):
            if generator.random() < probability:
                rows.symmetric_difference_update((row,))
        for column in range(context.dataset.column_count):
            if generator.random() < probability:
                columns.symmetric_difference_update((column,))
        bounds = context.candidate_validity_policy.bounds(context.dataset)
        return Bicluster(
            row_indices=_repair_minimum(
                rows,
                population=context.dataset.row_count,
                minimum=bounds.min_rows,
                generator=generator,
            ),
            column_indices=_repair_minimum(
                columns,
                population=context.dataset.column_count,
                minimum=bounds.min_columns,
                generator=generator,
            ),
        )


@dataclass(frozen=True, slots=True)
class HalfUniformMembershipCrossover:
    """Exchange differing memberships independently between two parents."""

    application_probability: float = 0.9
    row_exchange_probability: float = 0.5
    column_exchange_probability: float = 0.5
    component_name: str = "half_uniform_membership"
    provides: frozenset[str] = frozenset({"crossover-operator"})
    requires: frozenset[str] = frozenset({"candidate-validity", "prepared-dataset"})

    def __post_init__(self) -> None:
        MembershipRecombinationConfiguration(
            application_probability=self.application_probability,
            row_exchange_probability=self.row_exchange_probability,
            column_exchange_probability=self.column_exchange_probability,
        )

    @staticmethod
    def _exchange(
        first: tuple[int, ...],
        second: tuple[int, ...],
        probability: float,
        generator: npr.Generator,
    ) -> set[int]:
        selected = set(first)
        for value in sorted(set(first) ^ set(second)):
            if generator.random() < probability:
                selected.symmetric_difference_update((value,))
        return selected

    def cross(
        self,
        context: RunContext,
        first: Evaluation,
        second: Evaluation,
        generator: npr.Generator,
    ) -> Bicluster:
        base = first.candidate.bicluster
        if generator.random() >= self.application_probability:
            return base
        other = second.candidate.bicluster
        bounds = context.candidate_validity_policy.bounds(context.dataset)
        return Bicluster(
            row_indices=_repair_minimum(
                self._exchange(
                    base.row_indices,
                    other.row_indices,
                    self.row_exchange_probability,
                    generator,
                ),
                population=context.dataset.row_count,
                minimum=bounds.min_rows,
                generator=generator,
            ),
            column_indices=_repair_minimum(
                self._exchange(
                    base.column_indices,
                    other.column_indices,
                    self.column_exchange_probability,
                    generator,
                ),
                population=context.dataset.column_count,
                minimum=bounds.min_columns,
                generator=generator,
            ),
        )


__all__ = [
    "BitFlipMembershipMutation",
    "BitFlipMutationConfiguration",
    "EvidenceWeightedRecombinationCrossover",
    "HalfUniformMembershipCrossover",
    "MembershipRecombinationConfiguration",
    "MembershipRecombinationCrossover",
]
