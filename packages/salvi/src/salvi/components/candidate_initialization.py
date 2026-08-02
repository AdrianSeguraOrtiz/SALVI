"""Composable candidate initialization and variation for serial QD search."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Literal, Self

import numpy as np
import numpy.random as npr
import numpy.typing as npt
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


def _finite_median(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return None if finite.size == 0 else float(np.median(finite))


def _constant_anchor(
    context: RunContext,
    generator: npr.Generator,
    row_count: int,
    column_count: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    dataset = context.dataset
    anchor = int(generator.integers(0, dataset.row_count))
    observed_columns = tuple(
        column
        for column in range(dataset.column_count)
        if bool(dataset.support_mask(column)[anchor])
    )
    if not observed_columns:
        return (
            _sample_indices(generator, dataset.row_count, row_count),
            _sample_indices(generator, dataset.column_count, column_count),
        )
    anchor_column = observed_columns[int(generator.integers(0, len(observed_columns)))]
    seed_scores = _constant_row_scores(context, anchor, (anchor_column,))
    seed_rows = tuple(sorted(int(index) for index in np.argsort(seed_scores)[:row_count]))
    candidate_columns = np.arange(dataset.column_count, dtype=np.int64)
    column_errors = _constant_column_errors(context, seed_rows, candidate_columns)
    ranked_columns = candidate_columns[np.lexsort((candidate_columns, column_errors))]
    columns = tuple(sorted(int(column) for column in ranked_columns[:column_count]))
    refined_scores = _constant_row_scores(context, anchor, columns)
    rows = tuple(sorted(int(index) for index in np.argsort(refined_scores)[:row_count]))
    return rows, columns


def _constant_row_scores(
    context: RunContext,
    anchor: int,
    columns: tuple[int, ...],
) -> npt.NDArray[np.float64]:
    dataset = context.dataset
    scores = np.zeros(dataset.row_count, dtype=np.float64)
    support = np.zeros(dataset.row_count, dtype=np.int32)
    numeric_positions = tuple(
        dataset.numeric_positions[column]
        for column in columns
        if dataset.numeric_positions[column] >= 0
    )
    if numeric_positions:
        numeric_matrix = dataset.numeric_matrix(standardized=True)[:, numeric_positions]
        anchor_values = numeric_matrix[anchor]
        usable = np.isfinite(numeric_matrix) & np.isfinite(anchor_values)[np.newaxis, :]
        contributions = np.minimum(
            1.0,
            np.abs(numeric_matrix - anchor_values[np.newaxis, :]),
        )
        scores += np.sum(np.where(usable, contributions, 0.0), axis=1)
        support += np.count_nonzero(usable, axis=1).astype(np.int32)
    discrete_positions = tuple(
        dataset.discrete_positions[column]
        for column in columns
        if dataset.discrete_positions[column] >= 0
    )
    if discrete_positions:
        discrete_matrix = dataset.discrete_matrix()[:, discrete_positions]
        anchor_values = discrete_matrix[anchor]
        usable = (discrete_matrix >= 0) & (anchor_values >= 0)[np.newaxis, :]
        scores += np.sum(
            np.where(
                usable,
                discrete_matrix != anchor_values[np.newaxis, :],
                False,
            ),
            axis=1,
        )
        support += np.count_nonzero(usable, axis=1).astype(np.int32)
    normalized = np.full(dataset.row_count, math.inf, dtype=np.float64)
    np.divide(
        scores,
        support,
        out=normalized,
        where=support > 0,
    )
    normalized[anchor] = -1.0
    return normalized


def _constant_column_error(
    context: RunContext,
    rows: tuple[int, ...],
    column: int,
) -> float:
    return float(
        _constant_column_errors(
            context,
            rows,
            np.asarray((column,), dtype=np.int64),
        )[0]
    )


def _constant_column_errors(
    context: RunContext,
    rows: tuple[int, ...],
    columns: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    """Score constant compatibility for many columns with shared matrix scans."""

    if columns.size == 0:
        return np.empty(0, dtype=np.float64)
    dataset = context.dataset
    row_indices = np.asarray(rows, dtype=np.int64)
    source_supports = np.count_nonzero(
        dataset.support_matrix()[np.ix_(row_indices, columns)],
        axis=0,
    )
    sufficient = np.fromiter(
        (
            context.evaluation_support_policy.is_sufficient(
                int(count),
                len(rows),
            )
            for count in source_supports
        ),
        dtype=np.bool_,
        count=len(columns),
    )
    errors = np.full(len(columns), math.inf, dtype=np.float64)
    numeric_offsets = tuple(
        offset
        for offset, column in enumerate(columns)
        if dataset.numeric_positions[int(column)] >= 0 and sufficient[offset]
    )
    if numeric_offsets:
        positions = np.asarray(
            [dataset.numeric_positions[int(columns[offset])] for offset in numeric_offsets],
            dtype=np.int64,
        )
        matrix = dataset.numeric_matrix(standardized=True)[np.ix_(row_indices, positions)]
        with warnings.catch_warnings(), np.errstate(invalid="ignore"):
            warnings.simplefilter("ignore", category=RuntimeWarning)
            centers = np.nanmedian(matrix, axis=0)
            deviations = np.abs(matrix - centers[np.newaxis, :])
            counts = np.count_nonzero(np.isfinite(matrix), axis=0)
            means = np.nanmean(deviations, axis=0)
        for position, offset in enumerate(numeric_offsets):
            if counts[position] >= 2:
                errors[offset] = min(1.0, float(means[position]))
    for offset, raw_column in enumerate(columns):
        column = int(raw_column)
        if not sufficient[offset] or dataset.discrete_positions[column] < 0:
            continue
        values = dataset.discrete_matrix()[
            row_indices,
            dataset.discrete_positions[column],
        ]
        available = values[values >= 0]
        if available.size >= 2:
            counts = np.bincount(available)
            errors[offset] = 1.0 - float(np.max(counts)) / float(available.size)
    return errors


def _joint_anchor(
    context: RunContext,
    generator: npr.Generator,
    row_count: int,
    column_count: int,
    pattern: PatternKind,
    joint_column_candidate_pool_size: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    dataset = context.dataset
    numeric = dataset.numeric_column_indices
    if len(numeric) < column_count:
        raise ComponentError(
            f"{pattern.value} initialization requires at least {column_count} numeric columns"
        )
    first, second = _sample_indices(generator, dataset.row_count, 2)
    candidate_columns = tuple(
        int(value)
        for value in generator.choice(
            numeric,
            min(len(numeric), max(column_count, joint_column_candidate_pool_size)),
            replace=False,
        )
    )
    raw = np.column_stack([dataset.numeric_column(column) for column in candidate_columns])
    standardized = np.column_stack(
        [dataset.numeric_column(column, standardized=True) for column in candidate_columns]
    )
    if pattern is PatternKind.ADDITIVE:
        relation = standardized[second] - standardized[first]
        center = _finite_median(relation)
        ranking = (
            np.full(len(candidate_columns), math.inf)
            if center is None
            else np.abs(relation - center)
        )
    elif pattern is PatternKind.MULTIPLICATIVE:
        denominator = raw[first]
        usable = np.isfinite(raw[second]) & np.isfinite(denominator) & (np.abs(denominator) > 1e-12)
        relation = np.full(len(candidate_columns), np.nan)
        relation[usable] = raw[second, usable] / denominator[usable]
        center = _finite_median(relation)
        ranking = (
            np.full(len(candidate_columns), math.inf)
            if center is None
            else np.abs(relation - center)
        )
    else:
        raise ComponentError(f"{pattern.value} has no joint-anchor strategy")
    order = sorted(
        range(len(candidate_columns)),
        key=lambda position: (float(ranking[position]), candidate_columns[position]),
    )
    columns = tuple(sorted(candidate_columns[position] for position in order[:column_count]))
    matrix = np.column_stack(
        [
            (
                dataset.numeric_column(column, standardized=True)
                if pattern is PatternKind.ADDITIVE
                else dataset.numeric_column(column)
            )
            for column in columns
        ]
    )
    row_scores = np.full(dataset.row_count, math.inf)
    if pattern is PatternKind.ADDITIVE:
        anchor_profile = matrix[first]
        anchor_center = _finite_median(anchor_profile)
        if anchor_center is not None:
            anchor_profile = anchor_profile - anchor_center
            usable = np.isfinite(matrix) & np.isfinite(anchor_profile)[np.newaxis, :]
            with warnings.catch_warnings(), np.errstate(invalid="ignore"):
                warnings.simplefilter("ignore", category=RuntimeWarning)
                centers = np.nanmedian(matrix, axis=1)
                residuals = np.where(
                    usable,
                    np.abs(matrix - centers[:, np.newaxis] - anchor_profile[np.newaxis, :]),
                    np.nan,
                )
                scores = np.nanmedian(residuals, axis=1)
            valid = (
                np.isfinite(centers) & (np.count_nonzero(usable, axis=1) >= 2) & np.isfinite(scores)
            )
            row_scores[valid] = scores[valid]
    elif pattern is PatternKind.MULTIPLICATIVE:
        anchor_profile = matrix[first]
        usable = (
            np.isfinite(matrix)
            & np.isfinite(anchor_profile)[np.newaxis, :]
            & (np.abs(anchor_profile) > 1e-12)[np.newaxis, :]
        )
        ratios = np.full_like(matrix, np.nan)
        np.divide(
            matrix,
            anchor_profile[np.newaxis, :],
            out=ratios,
            where=usable,
        )
        with warnings.catch_warnings(), np.errstate(invalid="ignore"):
            warnings.simplefilter("ignore", category=RuntimeWarning)
            centers = np.nanmedian(ratios, axis=1)
            scores = np.nanmedian(np.abs(ratios - centers[:, np.newaxis]), axis=1)
        valid = (np.count_nonzero(usable, axis=1) >= 2) & np.isfinite(centers) & np.isfinite(scores)
        row_scores[valid] = scores[valid]
    else:
        raise ComponentError(f"{pattern.value} has no joint-anchor strategy")
    row_scores[first] = -2.0
    row_scores[second] = -1.0
    rows = tuple(sorted(int(index) for index in np.argsort(row_scores)[:row_count]))
    return rows, columns


def _pattern_anchor(
    context: RunContext,
    generator: npr.Generator,
    row_count: int,
    column_count: int,
    pattern: PatternKind,
    joint_column_candidate_pool_size: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if pattern is PatternKind.CONSTANT:
        return _constant_anchor(context, generator, row_count, column_count)
    if pattern in {PatternKind.ADDITIVE, PatternKind.MULTIPLICATIVE}:
        return _joint_anchor(
            context,
            generator,
            row_count,
            column_count,
            pattern,
            joint_column_candidate_pool_size,
        )
    raise ComponentError(f"{pattern.value} has no pattern-aware anchor strategy")


def _pattern_column_capacity(context: RunContext, pattern: PatternKind) -> int:
    if pattern is PatternKind.CONSTANT:
        return context.dataset.column_count
    if pattern in {PatternKind.ADDITIVE, PatternKind.MULTIPLICATIVE}:
        return len(context.dataset.numeric_column_indices)
    raise ComponentError(f"{pattern.value} has no pattern-aware anchor strategy")


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
        candidates: list[Candidate] = []
        for index, (row_count, requested_columns) in enumerate(shapes):
            pattern = allowed[index % len(allowed)]
            column_count = requested_columns
            if pattern in {PatternKind.ADDITIVE, PatternKind.MULTIPLICATIVE}:
                column_count = min(column_count, _pattern_column_capacity(context, pattern))
                if column_count < max(bounds.min_columns, 2):
                    raise ComponentError(
                        f"{pattern.value} initialization requires at least two eligible "
                        "numeric columns"
                    )
            rows, columns = _pattern_anchor(
                context,
                generator,
                row_count,
                column_count,
                pattern,
                self.joint_column_candidate_pool_size,
            )
            candidate = _candidate(
                producer=self.component_name,
                operation=f"{pattern.value.lower()}_anchor",
                sequence=start_sequence + index,
                rows=rows,
                columns=columns,
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
                if target.column_count <= _pattern_column_capacity(context, pattern)
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
        candidates: list[Candidate] = []
        for index, state in enumerate(states):
            pattern = state.pending_pattern
            if pattern is None:
                raise ComponentError("completed bootstrap state was requested for initialization")
            target = state.target
            rows, columns = _pattern_anchor(
                context,
                generator,
                target.row_count,
                target.column_count,
                pattern,
                self.joint_column_candidate_pool_size,
            )
            candidate = _candidate(
                producer=self.component_name,
                operation=f"{pattern.value.lower()}_cell_seed",
                sequence=start_sequence + index,
                rows=rows,
                columns=columns,
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
