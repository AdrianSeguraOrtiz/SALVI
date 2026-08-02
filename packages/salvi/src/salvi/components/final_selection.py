"""Containment-based final repertoire extraction."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from salvi.application.context import RunContext
from salvi.domain.enums import ObjectiveDirection
from salvi.domain.models import (
    Evaluation,
    FinalSelectionProvenance,
    ObjectiveValue,
    Repertoire,
)
from salvi.evaluation.structure import structural_distance
from salvi.exceptions import ComponentError

_EPSILON = 1e-12


class ContainmentMarginalQualityConfiguration(BaseModel):
    """Configuration for containment-chain elbow selection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_objective_degradation: Annotated[float, Field(ge=0.0, le=1.0)] = 0.15
    max_degradation_per_log_area_gain: Annotated[float, Field(ge=0.0)] = 0.20
    objective_names: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def validate_objective_names(self) -> Self:
        if self.objective_names is None:
            return self
        if not self.objective_names:
            raise ValueError("objective_names must be null or a non-empty list")
        if any(not name.strip() for name in self.objective_names):
            raise ValueError("objective_names must not contain blank names")
        if len(set(self.objective_names)) != len(self.objective_names):
            raise ValueError("objective_names must be unique")
        return self


@dataclass(frozen=True, slots=True)
class _ContainmentRepresentative:
    evaluation: Evaluation
    sources: tuple[Evaluation, ...]
    maximum_degradation: float


def _selected_objectives(
    evaluation: Evaluation,
    objective_names: tuple[str, ...] | None,
) -> tuple[ObjectiveValue, ...]:
    if objective_names is None:
        return evaluation.objectives
    by_name = {objective.name: objective for objective in evaluation.objectives}
    missing = tuple(name for name in objective_names if name not in by_name)
    if missing:
        raise ComponentError(
            f"final selection cannot find configured objectives: {', '.join(missing)}"
        )
    return tuple(by_name[name] for name in objective_names)


def _objective_schema(
    evaluation: Evaluation,
    objective_names: tuple[str, ...] | None,
) -> tuple[tuple[str, ObjectiveDirection], ...]:
    return tuple(
        (objective.name, objective.direction)
        for objective in _selected_objectives(evaluation, objective_names)
    )


def _validate_objectives(
    evaluations: Sequence[Evaluation],
    objective_names: tuple[str, ...] | None,
) -> None:
    if not evaluations:
        return
    expected = _objective_schema(evaluations[0], objective_names)
    if not expected:
        raise ComponentError("final selection requires evaluated objectives")
    for evaluation in evaluations[1:]:
        if _objective_schema(evaluation, objective_names) != expected:
            raise ComponentError("final selection requires one consistent objective schema")


def _continuous_utilities(
    evaluations: Sequence[Evaluation],
    objective_names: tuple[str, ...] | None,
) -> dict[str, tuple[float, ...]]:
    selected = tuple(
        _selected_objectives(evaluation, objective_names) for evaluation in evaluations
    )
    utilities = [[1.0] * len(selected[0]) for _ in evaluations]
    for objective_index, objective in enumerate(selected[0]):
        values = tuple(items[objective_index].value for items in selected)
        low = min(values)
        high = max(values)
        span = high - low
        if span <= _EPSILON:
            continue
        for evaluation_index, value in enumerate(values):
            utilities[evaluation_index][objective_index] = (
                (high - value) / span
                if objective.direction is ObjectiveDirection.MINIMIZE
                else (value - low) / span
            )
    return {
        evaluation.candidate.identifier: tuple(values)
        for evaluation, values in zip(evaluations, utilities, strict=True)
    }


def _duplicate_preference(
    evaluation: Evaluation,
    objective_names: tuple[str, ...] | None,
) -> tuple[object, ...]:
    objectives = _selected_objectives(evaluation, objective_names)
    directional_values = tuple(
        objective.value if objective.direction is ObjectiveDirection.MINIMIZE else -objective.value
        for objective in objectives
    )
    return (
        *directional_values,
        evaluation.candidate.generation,
        evaluation.candidate.identifier,
    )


def _deduplicate(
    evaluations: Sequence[Evaluation],
    objective_names: tuple[str, ...] | None,
) -> tuple[tuple[Evaluation, ...], dict[str, tuple[Evaluation, ...]]]:
    grouped: dict[str, list[Evaluation]] = defaultdict(list)
    for evaluation in evaluations:
        grouped[evaluation.candidate.bicluster.signature].append(evaluation)
    sources = {
        signature: tuple(sorted(items, key=lambda item: item.candidate.identifier))
        for signature, items in grouped.items()
    }
    representatives = tuple(
        min(items, key=lambda item: _duplicate_preference(item, objective_names))
        for _signature, items in sorted(grouped.items())
    )
    return representatives, sources


def _group_identifier(source_identifiers: Sequence[str]) -> str:
    material = "\0".join(source_identifiers).encode()
    return f"selection-{hashlib.sha256(material).hexdigest()[:16]}"


def _distance_matrix(
    evaluations: Sequence[Evaluation],
    row_weight: float,
) -> dict[tuple[str, str], float]:
    structures = {
        evaluation.candidate.identifier: (
            frozenset(evaluation.candidate.bicluster.row_indices),
            frozenset(evaluation.candidate.bicluster.column_indices),
        )
        for evaluation in evaluations
    }
    distances: dict[tuple[str, str], float] = {}
    for left_index, left in enumerate(evaluations):
        left_identifier = left.candidate.identifier
        left_rows, left_columns = structures[left_identifier]
        distances[(left_identifier, left_identifier)] = 0.0
        for right in evaluations[left_index + 1 :]:
            right_identifier = right.candidate.identifier
            right_rows, right_columns = structures[right_identifier]
            distance = structural_distance(
                left_rows,
                left_columns,
                right_rows,
                right_columns,
                row_weight=row_weight,
            )
            distances[(left_identifier, right_identifier)] = distance
            distances[(right_identifier, left_identifier)] = distance
    return distances


def _strictly_contains(container: Evaluation, contained: Evaluation) -> bool:
    container_bicluster = container.candidate.bicluster
    contained_bicluster = contained.candidate.bicluster
    container_rows = set(container_bicluster.row_indices)
    container_columns = set(container_bicluster.column_indices)
    contained_rows = set(contained_bicluster.row_indices)
    contained_columns = set(contained_bicluster.column_indices)
    return (
        contained_rows <= container_rows
        and contained_columns <= container_columns
        and (contained_rows != container_rows or contained_columns != container_columns)
    )


def _area(evaluation: Evaluation) -> int:
    bicluster = evaluation.candidate.bicluster
    return len(bicluster.row_indices) * len(bicluster.column_indices)


@dataclass(frozen=True, slots=True)
class ContainmentMarginalQualitySelector:
    """Select the largest nested bicluster before quality degrades materially."""

    max_objective_degradation: float = 0.15
    max_degradation_per_log_area_gain: float = 0.20
    objective_names: tuple[str, ...] | None = None
    component_name: str = "containment_marginal_quality"
    provides: frozenset[str] = frozenset({"final-selection"})
    requires: frozenset[str] = frozenset({"search-result", "objective"})

    def __post_init__(self) -> None:
        ContainmentMarginalQualityConfiguration(
            max_objective_degradation=self.max_objective_degradation,
            max_degradation_per_log_area_gain=self.max_degradation_per_log_area_gain,
            objective_names=self.objective_names,
        )

    def select(
        self,
        context: RunContext,
        repertoire: Repertoire,
    ) -> Repertoire:
        del context
        feasible = tuple(evaluation for evaluation in repertoire.evaluations if evaluation.feasible)
        if not feasible:
            return Repertoire()
        identifiers = tuple(evaluation.candidate.identifier for evaluation in feasible)
        if len(set(identifiers)) != len(identifiers):
            raise ComponentError(
                "containment final selection requires unique candidate identifiers"
            )
        _validate_objectives(feasible, self.objective_names)
        representatives, duplicates = _deduplicate(feasible, self.objective_names)
        utilities = _continuous_utilities(representatives, self.objective_names)
        by_identifier = {
            evaluation.candidate.identifier: evaluation for evaluation in representatives
        }
        parent_by_identifier = self._containment_parents(representatives, utilities)
        children: dict[str, list[str]] = defaultdict(list)
        for child, parent in parent_by_identifier.items():
            if parent is not None:
                children[parent].append(child)
        roots = tuple(
            evaluation.candidate.identifier
            for evaluation in representatives
            if parent_by_identifier[evaluation.candidate.identifier] is None
        )

        def resolve(identifier: str) -> tuple[_ContainmentRepresentative, ...]:
            evaluation = by_identifier[identifier]
            descendants = tuple(
                item for child in sorted(children[identifier]) for item in resolve(child)
            )
            own_sources = duplicates[evaluation.candidate.bicluster.signature]
            if not descendants:
                return (
                    _ContainmentRepresentative(
                        evaluation=evaluation,
                        sources=own_sources,
                        maximum_degradation=0.0,
                    ),
                )
            losses = tuple(
                self._replacement_loss(evaluation, descendant.evaluation, utilities)
                for descendant in descendants
            )
            if all(
                degradation <= self.max_objective_degradation + _EPSILON
                and rate <= self.max_degradation_per_log_area_gain + _EPSILON
                for degradation, rate in losses
            ):
                sources = tuple(
                    sorted(
                        (
                            *own_sources,
                            *(source for item in descendants for source in item.sources),
                        ),
                        key=lambda item: item.candidate.identifier,
                    )
                )
                return (
                    _ContainmentRepresentative(
                        evaluation=evaluation,
                        sources=sources,
                        maximum_degradation=max(
                            (
                                *(degradation for degradation, _rate in losses),
                                *(item.maximum_degradation for item in descendants),
                            ),
                            default=0.0,
                        ),
                    ),
                )
            return descendants

        selected = tuple(item for root in sorted(roots) for item in resolve(root))
        ordered = tuple(
            sorted(
                selected,
                key=lambda item: (
                    -_area(item.evaluation),
                    -sum(utilities[item.evaluation.candidate.identifier]),
                    item.evaluation.candidate.bicluster.signature,
                ),
            )
        )
        distances = _distance_matrix(
            tuple(item.evaluation for item in ordered),
            row_weight=0.5,
        )
        output: list[Evaluation] = []
        for rank, item in enumerate(ordered):
            evaluation = item.evaluation
            identifier = evaluation.candidate.identifier
            source_identifiers = tuple(
                sorted({source.candidate.identifier for source in item.sources})
            )
            source_coordinates = tuple(
                sorted(
                    {
                        source.archive_coordinate
                        for source in item.sources
                        if source.archive_coordinate is not None
                    }
                )
            )
            novelty = min(
                (
                    distances[(identifier, other.evaluation.candidate.identifier)]
                    for other in ordered
                    if other.evaluation.candidate.identifier != identifier
                ),
                default=1.0,
            )
            objective_utilities = utilities[identifier]
            provenance = FinalSelectionProvenance(
                selector=self.component_name,
                group_identifier=_group_identifier(source_identifiers),
                selection_rank=rank,
                quality_score=sum(objective_utilities) / len(objective_utilities),
                novelty_score=novelty,
                marginal_gain=max(0.0, 1.0 - item.maximum_degradation),
                source_candidate_identifiers=source_identifiers,
                source_archive_coordinates=source_coordinates,
            )
            output.append(evaluation.model_copy(update={"final_selection": provenance}))
        return Repertoire(evaluations=tuple(output))

    @staticmethod
    def _containment_parents(
        evaluations: Sequence[Evaluation],
        utilities: dict[str, tuple[float, ...]],
    ) -> dict[str, str | None]:
        parents: dict[str, str | None] = {}
        for evaluation in evaluations:
            containers = tuple(
                candidate for candidate in evaluations if _strictly_contains(candidate, evaluation)
            )
            parent = min(
                containers,
                key=lambda candidate: (
                    _area(candidate),
                    -sum(utilities[candidate.candidate.identifier]),
                    candidate.candidate.bicluster.signature,
                ),
                default=None,
            )
            parents[evaluation.candidate.identifier] = (
                None if parent is None else parent.candidate.identifier
            )
        return parents

    @staticmethod
    def _replacement_loss(
        container: Evaluation,
        contained: Evaluation,
        utilities: dict[str, tuple[float, ...]],
    ) -> tuple[float, float]:
        container_utilities = utilities[container.candidate.identifier]
        contained_utilities = utilities[contained.candidate.identifier]
        degradation = max(
            (
                max(0.0, contained_value - container_value)
                for contained_value, container_value in zip(
                    contained_utilities,
                    container_utilities,
                    strict=True,
                )
            ),
            default=0.0,
        )
        log_area_gain = math.log(_area(container) / _area(contained))
        return degradation, degradation / max(log_area_gain, _EPSILON)


__all__ = [
    "ContainmentMarginalQualityConfiguration",
    "ContainmentMarginalQualitySelector",
]
