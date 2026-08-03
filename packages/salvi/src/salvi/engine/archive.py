"""Sparse bounded deep-grid MOME archive."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import product
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from salvi.application.context import RunContext
from salvi.components.protocols import Component, ComponentKind, Constraint, Descriptor, Objective
from salvi.domain.enums import ArchiveInsertionStatus, BinningStrategy, ObjectiveDirection
from salvi.domain.models import Evaluation, Repertoire
from salvi.domain.search import (
    ArchiveCellCoordinate,
    ArchiveCellTarget,
    ArchiveInsertionOutcome,
)
from salvi.engine.dominance import (
    constrained_dominates,
    validate_constraint_schema,
    validate_objective_schema,
)
from salvi.engine.grid import ArchiveAxisConfiguration, AxisBinner
from salvi.exceptions import ComponentError


class DeepGridMomeConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    axes: Annotated[tuple[ArchiveAxisConfiguration, ...], Field(min_length=1)] = (
        ArchiveAxisConfiguration(
            descriptor="row_cardinality",
            binning=BinningStrategy.GEOMETRIC,
            bins=8,
        ),
        ArchiveAxisConfiguration(
            descriptor="column_cardinality",
            binning=BinningStrategy.GEOMETRIC,
            bins=8,
        ),
    )
    cell_capacity: Annotated[int, Field(ge=1)] = 8


def _crowding_distances(evaluations: Sequence[Evaluation]) -> dict[str, float]:
    distances = {evaluation.candidate.bicluster.signature: 0.0 for evaluation in evaluations}
    if len(evaluations) <= 2:
        return dict.fromkeys(distances, math.inf)
    for objective_index in range(len(evaluations[0].objectives)):
        ordered = sorted(
            evaluations,
            key=lambda evaluation: (
                evaluation.objectives[objective_index].value,
                evaluation.candidate.bicluster.signature,
            ),
        )
        low = ordered[0].objectives[objective_index].value
        high = ordered[-1].objectives[objective_index].value
        if math.isclose(low, high, abs_tol=1e-15):
            continue
        distances[ordered[0].candidate.bicluster.signature] = math.inf
        distances[ordered[-1].candidate.bicluster.signature] = math.inf
        span = high - low
        for index in range(1, len(ordered) - 1):
            signature = ordered[index].candidate.bicluster.signature
            if math.isinf(distances[signature]):
                continue
            previous = ordered[index - 1].objectives[objective_index].value
            following = ordered[index + 1].objectives[objective_index].value
            distances[signature] += (following - previous) / span
    return distances


@dataclass(slots=True)
class DeepGridMomeArchive:
    axes: tuple[ArchiveAxisConfiguration, ...]
    cell_capacity: int = 8
    component_name: str = "deep_grid_mome"
    provides: frozenset[str] = frozenset({"archive", "archive-cell-targets"})
    requires: frozenset[str] = frozenset({"objective", "descriptor"})
    _binners: tuple[AxisBinner, ...] = ()
    _objective_schema: tuple[tuple[str, ObjectiveDirection], ...] = ()
    _constraint_schema: tuple[str, ...] = ()
    _cells: dict[tuple[int, ...], list[Evaluation]] = field(default_factory=dict)
    _repertoire_cache: Repertoire | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        configuration = DeepGridMomeConfiguration(
            axes=self.axes,
            cell_capacity=self.cell_capacity,
        )
        self.axes = configuration.axes
        self.cell_capacity = configuration.cell_capacity

    def composition_issues(
        self,
        components: Sequence[tuple[ComponentKind, Component]],
    ) -> Sequence[str]:
        descriptor_names = {
            component.component_name
            for kind, component in components
            if kind is ComponentKind.DESCRIPTOR
        }
        axis_names = {axis.descriptor for axis in self.axes}
        issues: list[str] = []
        if descriptor_names and axis_names != descriptor_names:
            issues.append("archive axes must reference every configured descriptor exactly once")
        target_consumed = any(
            kind is not ComponentKind.ARCHIVE and "archive-cell-targets" in component.requires
            for kind, component in components
        )
        if target_consumed and axis_names != {"row_cardinality", "column_cardinality"}:
            issues.append(
                "cell-target consumers require exactly row_cardinality and "
                "column_cardinality archive axes"
            )
        return tuple(issues)

    def initialize(
        self,
        context: RunContext,
        objectives: Sequence[Objective],
        descriptors: Sequence[Descriptor],
        constraints: Sequence[Constraint] = (),
    ) -> None:
        descriptor_by_name = {descriptor.component_name: descriptor for descriptor in descriptors}
        configured = tuple(axis.descriptor for axis in self.axes)
        if len(set(configured)) != len(configured):
            raise ComponentError("archive axes must reference unique descriptors")
        if set(configured) != set(descriptor_by_name):
            raise ComponentError(
                "archive axes must reference every configured descriptor exactly once"
            )
        self._binners = tuple(
            AxisBinner.create(axis, descriptor_by_name[axis.descriptor].domain(context))
            for axis in self.axes
        )
        self._objective_schema = tuple(
            (objective.component_name, objective.direction) for objective in objectives
        )
        self._constraint_schema = tuple(constraint.component_name for constraint in constraints)
        self._cells.clear()
        self._repertoire_cache = None

    def add(
        self,
        evaluations: Sequence[Evaluation],
    ) -> Sequence[ArchiveInsertionOutcome]:
        self._require_initialized()
        return tuple(self._add_one(evaluation) for evaluation in evaluations)

    def repertoire(self) -> Repertoire:
        if self._repertoire_cache is None:
            evaluations = tuple(
                evaluation.model_copy(
                    update={
                        "archive_coordinate": coordinate,
                        "final_selection": None,
                    }
                )
                for coordinate in sorted(self._cells)
                for evaluation in sorted(
                    self._cells[coordinate],
                    key=lambda item: item.candidate.bicluster.signature,
                )
            )
            self._repertoire_cache = Repertoire(evaluations=evaluations)
        return self._repertoire_cache

    def cell_targets(self) -> tuple[ArchiveCellTarget, ...]:
        """Enumerate reachable row/column-cardinality cells for guided coverage."""

        self._require_initialized()
        descriptors = tuple(binner.descriptor for binner in self._binners)
        if set(descriptors) != {"row_cardinality", "column_cardinality"}:
            raise ComponentError(
                "cardinality cell targets require row_cardinality and column_cardinality axes"
            )
        representatives = tuple(
            tuple(
                (index, value, bounds)
                for index in range(binner.bin_count)
                if (value := binner.representative_integer(index)) is not None
                and (bounds := binner.integer_bounds(index)) is not None
            )
            for binner in self._binners
        )
        targets: list[ArchiveCellTarget] = []
        for values in product(*representatives):
            coordinate = tuple(index for index, _value, _bounds in values)
            by_descriptor = {
                descriptor: (value, bounds)
                for descriptor, (_index, value, bounds) in zip(
                    descriptors, values, strict=True
                )
            }
            row_count, row_bounds = by_descriptor["row_cardinality"]
            column_count, column_bounds = by_descriptor["column_cardinality"]
            targets.append(
                ArchiveCellTarget(
                    coordinate=ArchiveCellCoordinate(indices=coordinate),
                    row_count=row_count,
                    column_count=column_count,
                    minimum_row_count=row_bounds[0],
                    maximum_row_count=row_bounds[1],
                    minimum_column_count=column_bounds[0],
                    maximum_column_count=column_bounds[1],
                )
            )
        return tuple(sorted(targets, key=lambda target: target.coordinate.indices))

    def restore(self, repertoire: Repertoire) -> None:
        self._require_initialized()
        self._cells.clear()
        self._repertoire_cache = None
        evaluations = tuple(
            evaluation.model_copy(update={"final_selection": None})
            for evaluation in repertoire.evaluations
        )
        outcomes = self.add(evaluations)
        if not all(outcome.accepted for outcome in outcomes):
            rejected = tuple(
                outcome.candidate_identifier for outcome in outcomes if not outcome.accepted
            )
            raise ComponentError(f"checkpoint repertoire could not be restored: {rejected!r}")

    @property
    def occupied_cell_count(self) -> int:
        return len(self._cells)

    @property
    def repertoire_size(self) -> int:
        return sum(len(cell) for cell in self._cells.values())

    def _require_initialized(self) -> None:
        if not self._binners or not self._objective_schema:
            raise ComponentError("archive is not initialized")

    def _coordinate(self, evaluation: Evaluation) -> ArchiveCellCoordinate | None:
        values = {descriptor.name: descriptor.value for descriptor in evaluation.descriptors}
        if set(values) != {binner.descriptor for binner in self._binners}:
            raise ComponentError("evaluation descriptors do not match configured archive axes")
        indices: list[int] = []
        for binner in self._binners:
            index = binner.index(values[binner.descriptor])
            if index is None:
                return None
            indices.append(index)
        return ArchiveCellCoordinate(indices=tuple(indices))

    def _add_one(self, evaluation: Evaluation) -> ArchiveInsertionOutcome:
        identifier = evaluation.candidate.identifier
        if not evaluation.valid:
            return ArchiveInsertionOutcome(
                candidate_identifier=identifier,
                status=ArchiveInsertionStatus.REJECTED_INVALID,
            )
        validate_objective_schema(evaluation, self._objective_schema)
        validate_constraint_schema(evaluation, self._constraint_schema)
        coordinate = self._coordinate(evaluation)
        if coordinate is None:
            return ArchiveInsertionOutcome(
                candidate_identifier=identifier,
                status=ArchiveInsertionStatus.REJECTED_OUT_OF_BOUNDS,
            )
        key = coordinate.indices
        cell = self._cells.get(key)
        if cell is None:
            self._cells[key] = [evaluation]
            self._repertoire_cache = None
            return ArchiveInsertionOutcome(
                candidate_identifier=identifier,
                status=ArchiveInsertionStatus.INSERTED,
                coordinate=coordinate,
                created_cell=True,
            )
        signature = evaluation.candidate.bicluster.signature
        if any(item.candidate.bicluster.signature == signature for item in cell):
            return ArchiveInsertionOutcome(
                candidate_identifier=identifier,
                status=ArchiveInsertionStatus.REJECTED_DUPLICATE,
                coordinate=coordinate,
            )
        if any(constrained_dominates(item, evaluation) for item in cell):
            return ArchiveInsertionOutcome(
                candidate_identifier=identifier,
                status=ArchiveInsertionStatus.REJECTED_DOMINATED,
                coordinate=coordinate,
            )

        dominated = [item for item in cell if constrained_dominates(evaluation, item)]
        retained = [item for item in cell if item not in dominated]
        retained.append(evaluation)
        evicted = [item.candidate.identifier for item in dominated]
        if len(retained) > self.cell_capacity:
            victim = self._select_capacity_victim(retained)
            retained.remove(victim)
            if victim is evaluation:
                return ArchiveInsertionOutcome(
                    candidate_identifier=identifier,
                    status=ArchiveInsertionStatus.REJECTED_CAPACITY,
                    coordinate=coordinate,
                )
            evicted.append(victim.candidate.identifier)
        self._cells[key] = retained
        self._repertoire_cache = None
        return ArchiveInsertionOutcome(
            candidate_identifier=identifier,
            status=(
                ArchiveInsertionStatus.INSERTED_WITH_EVICTIONS
                if evicted
                else ArchiveInsertionStatus.INSERTED
            ),
            coordinate=coordinate,
            evicted_candidate_identifiers=tuple(evicted),
        )

    def _select_capacity_victim(self, retained: Sequence[Evaluation]) -> Evaluation:
        distances = _crowding_distances(retained)
        minimum_distance = min(distances[item.candidate.bicluster.signature] for item in retained)
        return max(
            (
                item
                for item in retained
                if distances[item.candidate.bicluster.signature] == minimum_distance
            ),
            key=lambda item: item.candidate.bicluster.signature,
        )


__all__ = [
    "DeepGridMomeArchive",
    "DeepGridMomeConfiguration",
]
