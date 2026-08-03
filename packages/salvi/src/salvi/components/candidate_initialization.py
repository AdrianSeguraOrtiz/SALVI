"""Composable candidate initialization and variation for serial QD search."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Literal, Self

import numpy as np
import numpy.random as npr
from pydantic import BaseModel, ConfigDict, Field, model_validator

from salvi.application.context import RunContext
from salvi.domain.enums import PatternKind
from salvi.domain.models import (
    Bicluster,
    Candidate,
    CandidateProvenance,
)
from salvi.domain.search import ArchiveCellTarget, BootstrapCellState, CandidateBounds
from salvi.exceptions import ComponentError
from salvi.patterns.catalog import PatternCatalog, default_pattern_catalog
from salvi.patterns.contracts import PatternSeedStrategy

Dimension = Literal["rows", "columns"]
MembershipOperation = Literal["add", "remove", "swap"]
RestartStrategy = Literal["stratified", "pattern_aware"]


class StratifiedInitializerConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cardinality_levels: Annotated[int, Field(ge=2)] = 8


class PatternAwareInitializerConfiguration(StratifiedInitializerConfiguration):
    joint_column_candidate_pool_size: Annotated[int, Field(ge=2)] = 32


class CellCoveragePatternAwareInitializerConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    seeds_per_cell: Annotated[int, Field(ge=1)] = 4
    max_attempts_per_cell: Annotated[int, Field(ge=1)] = 12
    joint_column_candidate_pool_size: Annotated[int, Field(ge=2)] = 32

    @model_validator(mode="after")
    def validate_attempt_budget(self) -> Self:
        if self.max_attempts_per_cell < self.seeds_per_cell:
            raise ValueError("max_attempts_per_cell cannot be smaller than seeds_per_cell")
        return self


def _sample_indices(
    generator: npr.Generator,
    population: int,
    count: int,
) -> tuple[int, ...]:
    return tuple(sorted(int(value) for value in generator.choice(population, count, replace=False)))


def _candidate_identifier(producer: str, sequence: int) -> str:
    return f"{producer}-{sequence:012d}"


def _candidate(
    *,
    producer: str,
    operation: str,
    sequence: int,
    rows: tuple[int, ...],
    columns: tuple[int, ...],
    generation: int = 0,
    parents: tuple[str, ...] = (),
    pattern_hint: PatternKind | None = None,
    target_archive_coordinate: tuple[int, ...] | None = None,
    diagnostics: tuple[tuple[str, float | int | str | bool | None], ...] = (),
) -> Candidate:
    return Candidate(
        identifier=_candidate_identifier(producer, sequence),
        generation=generation,
        bicluster=Bicluster(row_indices=rows, column_indices=columns),
        provenance=CandidateProvenance(
            producer=producer,
            operation=operation,
            sequence=sequence,
            parent_identifiers=parents,
            pattern_hint=pattern_hint,
            target_archive_coordinate=target_archive_coordinate,
            diagnostics=diagnostics,
        ),
    )


def _cardinality_levels(minimum: int, maximum: int, count: int) -> tuple[int, ...]:
    if minimum == maximum:
        return (minimum,)
    raw = np.geomspace(minimum, maximum, num=min(count, maximum - minimum + 1))
    levels = tuple(sorted({minimum, maximum, *(round(value) for value in raw)}))
    return levels


def _stratified_shapes(
    bounds: CandidateBounds,
    count: int,
    levels: int,
    generator: npr.Generator,
) -> tuple[tuple[int, int], ...]:
    row_levels = _cardinality_levels(bounds.min_rows, bounds.max_rows, levels)
    column_levels = _cardinality_levels(bounds.min_columns, bounds.max_columns, levels)
    grid = tuple((rows, columns) for rows in row_levels for columns in column_levels)
    shapes: list[tuple[int, int]] = []
    while len(shapes) < count:
        permutation = generator.permutation(len(grid))
        remaining = count - len(shapes)
        shapes.extend(grid[int(index)] for index in permutation[:remaining])
    return tuple(shapes)


def _random_candidate(
    context: RunContext,
    generator: npr.Generator,
    *,
    producer: str,
    operation: str,
    sequence: int,
    row_count: int | None = None,
    column_count: int | None = None,
    pattern_hint: PatternKind | None = None,
) -> Candidate:
    bounds = context.candidate_validity_policy.bounds(context.dataset)
    actual_rows = (
        int(generator.integers(bounds.min_rows, bounds.max_rows + 1))
        if row_count is None
        else row_count
    )
    actual_columns = (
        int(generator.integers(bounds.min_columns, bounds.max_columns + 1))
        if column_count is None
        else column_count
    )
    candidate = _candidate(
        producer=producer,
        operation=operation,
        sequence=sequence,
        rows=_sample_indices(generator, context.dataset.row_count, actual_rows),
        columns=_sample_indices(generator, context.dataset.column_count, actual_columns),
        pattern_hint=pattern_hint,
    )
    context.candidate_validity_policy.validate(candidate, context.dataset)
    return candidate


@dataclass(frozen=True, slots=True)
class UniformRandomInitializer:
    component_name: str = "uniform_random"
    provides: frozenset[str] = frozenset({"initialization"})
    requires: frozenset[str] = frozenset(
        {"prepared-dataset", "missing-values-handled", "candidate-validity"}
    )

    def initialize(
        self,
        context: RunContext,
        count: int,
        *,
        start_sequence: int = 0,
    ) -> Sequence[Candidate]:
        if count < 0:
            raise ValueError("initializer count must be non-negative")
        generator = context.random_generator("initializer.uniform_random")
        return tuple(
            _random_candidate(
                context,
                generator,
                producer=self.component_name,
                operation="uniform_random",
                sequence=start_sequence + index,
            )
            for index in range(count)
        )


@dataclass(frozen=True, slots=True)
class StratifiedInitializer:
    cardinality_levels: int = 8
    component_name: str = "stratified"
    provides: frozenset[str] = frozenset({"initialization"})
    requires: frozenset[str] = frozenset(
        {"prepared-dataset", "missing-values-handled", "candidate-validity"}
    )

    def __post_init__(self) -> None:
        StratifiedInitializerConfiguration(cardinality_levels=self.cardinality_levels)

    def initialize(
        self,
        context: RunContext,
        count: int,
        *,
        start_sequence: int = 0,
    ) -> Sequence[Candidate]:
        if count < 0:
            raise ValueError("initializer count must be non-negative")
        generator = context.random_generator("initializer.stratified")
        bounds = context.candidate_validity_policy.bounds(context.dataset)
        shapes = _stratified_shapes(bounds, count, self.cardinality_levels, generator)
        return tuple(
            _random_candidate(
                context,
                generator,
                producer=self.component_name,
                operation="stratified",
                sequence=start_sequence + index,
                row_count=row_count,
                column_count=column_count,
            )
            for index, (row_count, column_count) in enumerate(shapes)
        )


def _seed_strategy(catalog: PatternCatalog, pattern: PatternKind) -> PatternSeedStrategy:
    strategy = catalog.implementation(pattern).seed_strategy
    if strategy is None:
        raise ComponentError(f"{pattern.value} has no registered pattern-aware seed strategy")
    return strategy


@dataclass(frozen=True, slots=True)
class PatternAwareInitializer:
    cardinality_levels: int = 8
    joint_column_candidate_pool_size: int = 32
    component_name: str = "pattern_aware"
    provides: frozenset[str] = frozenset({"initialization"})
    requires: frozenset[str] = frozenset(
        {
            "prepared-dataset",
            "missing-values-handled",
            "candidate-validity",
            "robust-numeric-data",
        }
    )

    def __post_init__(self) -> None:
        PatternAwareInitializerConfiguration(
            cardinality_levels=self.cardinality_levels,
            joint_column_candidate_pool_size=self.joint_column_candidate_pool_size,
        )

    def initialize(
        self,
        context: RunContext,
        count: int,
        *,
        start_sequence: int = 0,
    ) -> Sequence[Candidate]:
        if count < 0:
            raise ValueError("initializer count must be non-negative")
        generator = context.random_generator(f"initializer.{self.component_name}")
        bounds = context.candidate_validity_policy.bounds(context.dataset)
        shapes = _stratified_shapes(bounds, count, self.cardinality_levels, generator)
        allowed = context.patterns.allowed
        catalog = default_pattern_catalog(allowed)
        candidates: list[Candidate] = []
        for index, (row_count, requested_columns) in enumerate(shapes):
            pattern = allowed[index % len(allowed)]
            strategy = _seed_strategy(catalog, pattern)
            shape = strategy.project_shape(
                context,
                preferred_row_count=row_count,
                row_range=(row_count, row_count),
                preferred_column_count=requested_columns,
                column_range=(bounds.min_columns, requested_columns),
            )
            if shape is None:
                supported = ", ".join(
                    sorted(
                        kind.value.lower()
                        for kind in catalog.implementation(
                            pattern
                        ).definition.supported_column_kinds
                    )
                )
                raise ComponentError(
                    f"{pattern.value} initialization requires enough support-eligible "
                    f"{supported} columns for the requested cardinalities"
                )
            seed = strategy.generate(
                context,
                generator,
                shape,
                joint_column_candidate_pool_size=self.joint_column_candidate_pool_size,
            )
            candidate = _candidate(
                producer=self.component_name,
                operation=f"{pattern.value.lower()}_anchor",
                sequence=start_sequence + index,
                rows=seed.rows,
                columns=seed.columns,
                pattern_hint=pattern,
            )
            context.candidate_validity_policy.validate(candidate, context.dataset)
            candidates.append(candidate)
        return tuple(candidates)


@dataclass(frozen=True, slots=True)
class CellCoveragePatternAwareInitializer:
    """Generate pattern-aware seeds against explicit reachable QD cells."""

    seeds_per_cell: int = 4
    max_attempts_per_cell: int = 12
    joint_column_candidate_pool_size: int = 32
    component_name: str = "cell_coverage_pattern_aware"
    provides: frozenset[str] = frozenset({"initialization", "cell-coverage-initialization"})
    requires: frozenset[str] = frozenset(
        {
            "prepared-dataset",
            "missing-values-handled",
            "candidate-validity",
            "robust-numeric-data",
            "archive-cell-targets",
            "descriptor:row-cardinality",
            "descriptor:column-cardinality",
        }
    )

    def __post_init__(self) -> None:
        CellCoveragePatternAwareInitializerConfiguration(
            seeds_per_cell=self.seeds_per_cell,
            max_attempts_per_cell=self.max_attempts_per_cell,
            joint_column_candidate_pool_size=self.joint_column_candidate_pool_size,
        )

    def initialize(
        self,
        context: RunContext,
        count: int,
        *,
        start_sequence: int = 0,
    ) -> Sequence[Candidate]:
        """Provide a deterministic fallback for engines without cell-target support."""

        return PatternAwareInitializer(
            cardinality_levels=max(2, round(math.sqrt(max(1, count)))),
            joint_column_candidate_pool_size=self.joint_column_candidate_pool_size,
            component_name=self.component_name,
        ).initialize(context, count, start_sequence=start_sequence)

    def bootstrap_plan(
        self,
        context: RunContext,
        targets: Sequence[ArchiveCellTarget],
    ) -> Sequence[BootstrapCellState]:
        allowed = context.patterns.allowed
        catalog = default_pattern_catalog(allowed)
        if self.seeds_per_cell < len(allowed):
            raise ComponentError(
                "cell-coverage initialization requires seeds_per_cell to be at least "
                "the number of allowed patterns"
            )
        states: list[BootstrapCellState] = []
        for target in targets:
            reachable = tuple(
                pattern
                for pattern in allowed
                if _seed_strategy(catalog, pattern).project_target(context, target) is not None
            )
            if not reachable:
                continue
            required = tuple(
                reachable[index % len(reachable)] for index in range(self.seeds_per_cell)
            )
            states.append(
                BootstrapCellState(
                    target=target,
                    required_patterns=required,
                    maximum_attempts=self.max_attempts_per_cell,
                )
            )
        if not states:
            raise ComponentError("cell-coverage initialization found no reachable archive cells")
        return tuple(states)

    def initialize_bootstrap(
        self,
        context: RunContext,
        states: Sequence[BootstrapCellState],
        *,
        start_sequence: int = 0,
    ) -> Sequence[Candidate]:
        generator = context.random_generator(f"initializer.{self.component_name}")
        catalog = default_pattern_catalog(context.patterns.allowed)
        candidates: list[Candidate] = []
        for index, state in enumerate(states):
            pattern = state.pending_pattern
            if pattern is None:
                raise ComponentError("completed bootstrap state was requested for initialization")
            target = state.target
            strategy = _seed_strategy(catalog, pattern)
            shape = strategy.project_target(context, target)
            if shape is None:
                raise ComponentError(
                    f"{pattern.value} seed cannot reach archive cell {target.coordinate.indices}"
                )
            seed = strategy.generate(
                context,
                generator,
                shape,
                joint_column_candidate_pool_size=self.joint_column_candidate_pool_size,
            )
            candidate = _candidate(
                producer=self.component_name,
                operation=f"{pattern.value.lower()}_cell_seed",
                sequence=start_sequence + index,
                rows=seed.rows,
                columns=seed.columns,
                pattern_hint=pattern,
                target_archive_coordinate=target.coordinate.indices,
            )
            context.candidate_validity_policy.validate(candidate, context.dataset)
            candidates.append(candidate)
        return tuple(candidates)


__all__ = [
    "CellCoveragePatternAwareInitializer",
    "CellCoveragePatternAwareInitializerConfiguration",
    "Dimension",
    "MembershipOperation",
    "PatternAwareInitializer",
    "PatternAwareInitializerConfiguration",
    "RestartStrategy",
    "StratifiedInitializer",
    "StratifiedInitializerConfiguration",
    "UniformRandomInitializer",
]
