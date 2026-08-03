"""Membership-move and restart emitters for QD search."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, Literal

import numpy as np
import numpy.random as npr
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from salvi.application.context import RunContext, require_qd_run_context
from salvi.components.candidate_initialization import (
    CellCoveragePatternAwareInitializer,
    Dimension,
    MembershipOperation,
    PatternAwareInitializer,
    RestartStrategy,
    StratifiedInitializer,
    StratifiedInitializerConfiguration,
    _candidate,
    _random_candidate,
)
from salvi.domain.enums import ColumnKind, PatternKind
from salvi.domain.models import Candidate, Evaluation, PatternFit, Repertoire
from salvi.domain.search import BootstrapCellState
from salvi.exceptions import ComponentError
from salvi.patterns.catalog import default_pattern_catalog
from salvi.patterns.joint_models import robust_column_scales
from salvi.patterns.math import nanmedian_2d
from salvi.patterns.seeding import constant_column_errors


class MembershipEmitterConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    guided: bool = False
    parent_pool_size: Annotated[int, Field(ge=1)] = 16
    candidate_pool_size: Annotated[int, Field(ge=1)] = 64


class ShapeMoveEmitterConfiguration(MembershipEmitterConfiguration):
    pass


class RestartEmitterConfiguration(StratifiedInitializerConfiguration):
    strategy: RestartStrategy = "stratified"
    joint_column_candidate_pool_size: Annotated[int, Field(ge=2)] = 32


class CellCoverageRestartEmitterConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    joint_column_candidate_pool_size: Annotated[int, Field(ge=2)] = 32


def _membership_options(
    selected: tuple[int, ...],
    population: int,
    minimum: int,
    operation: MembershipOperation,
) -> bool:
    if operation == "add":
        return len(selected) < population
    if operation == "remove":
        return len(selected) > minimum
    return len(selected) < population


def _random_membership_move(
    selected: tuple[int, ...],
    population: int,
    minimum: int,
    operation: MembershipOperation,
    generator: npr.Generator,
) -> tuple[int, ...] | None:
    if not _membership_options(selected, population, minimum, operation):
        return None
    selected_set = set(selected)
    if operation == "remove":
        removed = selected[int(generator.integers(0, len(selected)))]
        return tuple(sorted(selected_set - {removed}))
    available = tuple(index for index in range(population) if index not in selected_set)
    added = available[int(generator.integers(0, len(available)))]
    if operation == "add":
        return tuple(sorted(selected_set | {added}))
    removed = selected[int(generator.integers(0, len(selected)))]
    return tuple(sorted((selected_set - {removed}) | {added}))


def _column_losses(evaluation: Evaluation) -> dict[int, float]:
    return evaluation.mean_column_losses


def _row_compatibilities(
    context: RunContext,
    fit: PatternFit,
    row_indices: tuple[int, ...],
) -> dict[int, float]:
    if not row_indices:
        return {}
    dataset = context.dataset
    rows = np.asarray(row_indices, dtype=np.int64)
    error_sum = np.zeros(len(rows), dtype=np.float64)
    error_count = np.zeros(len(rows), dtype=np.int32)
    groups = {group.identifier: group for group in fit.groups}
    row_effects: dict[str, npt.NDArray[np.float64]] = {}
    group_scales: dict[tuple[str, int], float] = {}
    for group in fit.groups:
        fitted_columns = tuple(
            column
            for column in fit.columns
            if column.group_identifier == group.identifier and isinstance(column.parameter, float)
        )
        if len(fitted_columns) < 2:
            continue
        columns = tuple(column.column_index for column in fitted_columns)
        parameters = np.asarray(
            [column.parameter for column in fitted_columns],
            dtype=np.float64,
        )
        positions = np.asarray(
            [dataset.numeric_positions[column] for column in columns],
            dtype=np.int64,
        )
        scales = robust_column_scales(context, columns)
        group_scales.update(
            {
                (group.identifier, column): float(scales[position])
                for position, column in enumerate(columns)
            }
        )
        if group.pattern is PatternKind.ADDITIVE:
            matrix = dataset.numeric_matrix()[np.ix_(rows, positions)]
            estimates = matrix - parameters[np.newaxis, :]
        elif group.pattern is PatternKind.MULTIPLICATIVE:
            matrix = dataset.numeric_matrix()[np.ix_(rows, positions)]
            usable_parameters = np.abs(parameters) > 1e-12
            estimates = np.full_like(matrix, np.nan)
            np.divide(
                matrix,
                scales[np.newaxis, :] * parameters[np.newaxis, :],
                out=estimates,
                where=np.isfinite(matrix) & usable_parameters[np.newaxis, :],
            )
        else:
            raise ComponentError(f"{group.pattern.value} has no guided row-compatibility strategy")
        usable_count = np.count_nonzero(np.isfinite(estimates), axis=1)
        effects = nanmedian_2d(estimates, axis=1)
        effects[usable_count < 2] = np.nan
        row_effects[group.identifier] = effects
    for column in fit.columns:
        if column.pattern is None or column.parameter is None:
            continue
        column_index = column.column_index
        metadata = dataset.columns[column_index]
        if column.pattern is PatternKind.CONSTANT:
            if metadata.kind is ColumnKind.NUMERIC:
                position = dataset.numeric_positions[column_index]
                values = dataset.numeric_matrix()[rows, position]
                scale = dataset.numeric_statistics[position].robust_range
                if isinstance(column.parameter, float):
                    usable = np.isfinite(values)
                    errors = (
                        (np.abs(values - column.parameter) > 1e-12).astype(np.float64)
                        if scale <= 1e-12
                        else np.minimum(1.0, np.abs(values - column.parameter) / scale)
                    )
                    error_sum[usable] += errors[usable]
                    error_count[usable] += 1
            else:
                position = dataset.discrete_positions[column_index]
                codes = dataset.discrete_matrix()[rows, position]
                usable = codes >= 0
                if not isinstance(column.parameter, (bool, str)):
                    continue
                prototype = dataset.discrete_code(column_index, column.parameter)
                error_sum[usable] += (codes[usable] != prototype).astype(np.float64)
                error_count[usable] += 1
            continue
        if column.group_identifier is None or not isinstance(column.parameter, float):
            continue
        group = groups[column.group_identifier]
        group_effects = row_effects.get(column.group_identifier)
        if group_effects is None:
            continue
        if column.pattern is PatternKind.ADDITIVE:
            position = dataset.numeric_positions[column_index]
            values = dataset.numeric_matrix()[rows, position]
            scale = group_scales[(column.group_identifier, column_index)]
            usable = np.isfinite(values) & np.isfinite(group_effects)
            errors = np.minimum(
                1.0,
                np.abs(values - group_effects - column.parameter) / scale,
            )
            error_sum[usable] += errors[usable]
            error_count[usable] += 1
        elif column.pattern is PatternKind.MULTIPLICATIVE:
            position = dataset.numeric_positions[column_index]
            values = dataset.numeric_matrix()[rows, position]
            scale = dataset.numeric_statistics[position].robust_range
            usable = np.isfinite(values) & np.isfinite(group_effects) & (scale > 1e-12)
            if scale > 1e-12:
                errors = np.minimum(
                    1.0,
                    np.abs(values / scale - group_effects * column.parameter),
                )
                error_sum[usable] += errors[usable]
                error_count[usable] += 1
        else:
            raise ComponentError(f"{column.pattern.value} has no guided row-compatibility strategy")
    scores = np.full(len(rows), math.inf, dtype=np.float64)
    np.divide(error_sum, error_count, out=scores, where=error_count > 0)
    return {int(row): float(score) for row, score in zip(rows, scores, strict=True)}


def _row_compatibility(context: RunContext, fit: PatternFit, row_index: int) -> float:
    return _row_compatibilities(context, fit, (row_index,))[row_index]


def _guided_membership_move(
    context: RunContext,
    evaluation: Evaluation,
    dimension: Dimension,
    operation: MembershipOperation,
    generator: npr.Generator,
    candidate_pool_size: int,
) -> tuple[int, ...] | None:
    selected = (
        evaluation.candidate.bicluster.row_indices
        if dimension == "rows"
        else evaluation.candidate.bicluster.column_indices
    )
    population = context.dataset.row_count if dimension == "rows" else context.dataset.column_count
    bounds = context.candidate_validity_policy.bounds(context.dataset)
    minimum = bounds.min_rows if dimension == "rows" else bounds.min_columns
    if not _membership_options(selected, population, minimum, operation):
        return None
    selected_set = set(selected)
    available = tuple(index for index in range(population) if index not in selected_set)
    removal_pool = (
        _sample_membership_pool(selected, candidate_pool_size, generator)
        if operation != "add"
        else ()
    )
    addition_pool = (
        _sample_membership_pool(available, candidate_pool_size, generator)
        if operation != "remove"
        else ()
    )
    if dimension == "columns":
        column_losses = _column_losses(evaluation)
        addition_compatibility = _external_column_compatibilities(
            context,
            evaluation,
            addition_pool,
        )
        removed = (
            max(
                removal_pool,
                key=lambda index: (column_losses.get(index, 0.0), index),
            )
            if removal_pool
            else None
        )
        if addition_pool:
            added = min(
                addition_pool,
                key=lambda index: (
                    addition_compatibility[index],
                    index,
                ),
            )
        else:
            added = None
    elif evaluation.pattern_fit is not None:
        fit = evaluation.pattern_fit
        row_compatibility = _row_compatibilities(
            context,
            fit,
            tuple(dict.fromkeys((*removal_pool, *addition_pool))),
        )
        removed = (
            max(
                removal_pool,
                key=lambda index: (row_compatibility[index], index),
            )
            if removal_pool
            else None
        )
        added = (
            min(
                addition_pool,
                key=lambda index: (
                    row_compatibility[index],
                    index,
                ),
            )
            if addition_pool
            else None
        )
    else:
        return _random_membership_move(
            selected,
            population,
            minimum,
            operation,
            generator,
        )
    if operation == "remove":
        assert removed is not None
        return tuple(sorted(selected_set - {removed}))
    if added is None:
        return None
    if operation == "add":
        return tuple(sorted(selected_set | {added}))
    assert removed is not None
    return tuple(sorted((selected_set - {removed}) | {added}))


def _sample_membership_pool(
    values: tuple[int, ...],
    pool_size: int,
    generator: npr.Generator,
) -> tuple[int, ...]:
    if len(values) <= pool_size:
        return values
    positions = generator.choice(len(values), pool_size, replace=False)
    return tuple(values[int(position)] for position in positions)


def _external_column_compatibility(
    context: RunContext,
    evaluation: Evaluation,
    column_index: int,
) -> float:
    return _external_column_compatibilities(
        context,
        evaluation,
        (column_index,),
    )[column_index]


def _external_column_compatibilities(
    context: RunContext,
    evaluation: Evaluation,
    column_indices: tuple[int, ...],
) -> dict[int, float]:
    if not column_indices:
        return {}
    rows = evaluation.candidate.bicluster.row_indices
    columns = np.asarray(column_indices, dtype=np.int64)
    losses: dict[int, list[float]] = {column: [] for column in column_indices}
    if PatternKind.CONSTANT in context.patterns.allowed:
        constant_errors = constant_column_errors(context, rows, columns)
        for column, error in zip(column_indices, constant_errors, strict=True):
            losses[column].append(float(error))
    fit = evaluation.pattern_fit
    if fit is None:
        return {column: min(values, default=math.inf) for column, values in losses.items()}
    dataset = context.dataset
    row_indices = np.asarray(rows, dtype=np.int64)
    source_supports = np.count_nonzero(
        dataset.support_matrix()[np.ix_(row_indices, columns)],
        axis=0,
    )
    eligible = {
        column
        for column, support in zip(column_indices, source_supports, strict=True)
        if dataset.numeric_positions[column] >= 0
        and context.evaluation_support_policy.is_sufficient(int(support), len(rows))
    }
    ordered_eligible = tuple(sorted(eligible))
    candidate_scales = dict(
        zip(
            ordered_eligible,
            robust_column_scales(context, ordered_eligible),
            strict=True,
        )
    )
    for group in fit.groups:
        if group.pattern not in context.patterns.allowed:
            continue
        alpha_by_row = dict(group.row_parameters)
        alpha = np.asarray(
            [alpha_by_row.get(row, math.nan) for row in rows],
            dtype=np.float64,
        )
        for column_index in ordered_eligible:
            position = dataset.numeric_positions[column_index]
            if group.pattern is PatternKind.ADDITIVE:
                values = dataset.numeric_matrix()[row_indices, position]
                usable = np.isfinite(values) & np.isfinite(alpha)
                if np.count_nonzero(usable) < 2:
                    continue
                beta = float(np.median(values[usable] - alpha[usable]))
                scale = candidate_scales[column_index]
                losses[column_index].append(
                    min(
                        1.0,
                        float(
                            np.mean(
                                np.abs(values[usable] - alpha[usable] - beta) / scale
                            )
                        ),
                    )
                )
            elif group.pattern is PatternKind.MULTIPLICATIVE:
                scale = candidate_scales[column_index]
                values = dataset.numeric_matrix()[row_indices, position] / scale
                usable = np.isfinite(values) & np.isfinite(alpha) & (np.abs(alpha) > 1e-12)
                if np.count_nonzero(usable) < 2:
                    continue
                beta = float(np.median(values[usable] / alpha[usable]))
                losses[column_index].append(
                    min(1.0, float(np.mean(np.abs(values[usable] - alpha[usable] * beta))))
                )
            else:
                raise ComponentError(
                    f"{group.pattern.value} has no guided column-compatibility strategy"
                )
    return {column: min(values, default=math.inf) for column, values in losses.items()}


def _select_parent(
    context: RunContext,
    repertoire: Repertoire,
    generator: npr.Generator,
    *,
    pool_size: int,
    eligible: Callable[[Evaluation], bool],
    guided: bool,
) -> Evaluation | None:
    qd_context = require_qd_run_context(context)
    if qd_context.parent_selection_policy is None:
        raise ComponentError("the active emitter requires a parent-selection policy")
    return qd_context.parent_selection_policy.select(
        repertoire,
        generator,
        pool_size=pool_size,
        eligible=eligible,
        guided=guided,
    )


@dataclass(frozen=True, slots=True)
class MembershipMoveEmitter:
    dimension: Dimension
    operation: MembershipOperation
    guided: bool = False
    parent_pool_size: int = 16
    candidate_pool_size: int = 64
    component_name: str = "membership_move"
    provides: frozenset[str] = frozenset({"emitter"})
    requires: frozenset[str] = frozenset(
        {"archive", "candidate-validity", "parent-selection", "prepared-dataset"}
    )

    def __post_init__(self) -> None:
        MembershipEmitterConfiguration(
            guided=self.guided,
            parent_pool_size=self.parent_pool_size,
            candidate_pool_size=self.candidate_pool_size,
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
        bounds = context.candidate_validity_policy.bounds(context.dataset)
        population = (
            context.dataset.row_count if self.dimension == "rows" else context.dataset.column_count
        )
        minimum = bounds.min_rows if self.dimension == "rows" else bounds.min_columns
        candidates: list[Candidate] = []
        for index in range(count):
            parent = _select_parent(
                context,
                repertoire,
                generator,
                pool_size=self.parent_pool_size,
                guided=self.guided,
                eligible=lambda evaluation: _membership_options(
                    (
                        evaluation.candidate.bicluster.row_indices
                        if self.dimension == "rows"
                        else evaluation.candidate.bicluster.column_indices
                    ),
                    population,
                    minimum,
                    self.operation,
                ),
            )
            sequence = start_sequence + index
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
            bicluster = parent.candidate.bicluster
            selected = (
                bicluster.row_indices if self.dimension == "rows" else bicluster.column_indices
            )
            moved = (
                _guided_membership_move(
                    context,
                    parent,
                    self.dimension,
                    self.operation,
                    generator,
                    self.candidate_pool_size,
                )
                if self.guided
                else _random_membership_move(
                    selected,
                    population,
                    minimum,
                    self.operation,
                    generator,
                )
            )
            if moved is None:
                raise ComponentError("eligible parent did not support its configured move")
            rows = moved if self.dimension == "rows" else bicluster.row_indices
            columns = moved if self.dimension == "columns" else bicluster.column_indices
            candidate = _candidate(
                producer=self.component_name,
                operation=f"{self.operation}_{self.dimension[:-1]}",
                sequence=sequence,
                rows=rows,
                columns=columns,
                generation=parent.candidate.generation + 1,
                parents=(parent.candidate.identifier,),
            )
            context.candidate_validity_policy.validate(candidate, context.dataset)
            candidates.append(candidate)
        return tuple(candidates)


@dataclass(frozen=True, slots=True)
class RandomMoveEmitter:
    """Neutral baseline choosing any valid one-dimensional move."""

    component_name: str = "random_move"
    provides: frozenset[str] = frozenset({"emitter"})
    requires: frozenset[str] = frozenset(
        {"archive", "candidate-validity", "parent-selection", "prepared-dataset"}
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
        generator = context.random_generator("emitter.random_move")
        bounds = context.candidate_validity_policy.bounds(context.dataset)
        candidates: list[Candidate] = []
        for index in range(count):
            sequence = start_sequence + index
            parent = _select_parent(
                context,
                repertoire,
                generator,
                pool_size=1,
                guided=False,
                eligible=lambda _: True,
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
            choices: list[tuple[Dimension, MembershipOperation]] = []
            dimension_options: tuple[tuple[Dimension, tuple[int, ...], int, int], ...] = (
                (
                    "rows",
                    parent.candidate.bicluster.row_indices,
                    context.dataset.row_count,
                    bounds.min_rows,
                ),
                (
                    "columns",
                    parent.candidate.bicluster.column_indices,
                    context.dataset.column_count,
                    bounds.min_columns,
                ),
            )
            operations: tuple[MembershipOperation, ...] = ("add", "remove", "swap")
            for dimension, selected, population, minimum in dimension_options:
                for operation in operations:
                    if _membership_options(selected, population, minimum, operation):
                        choices.append((dimension, operation))
            if not choices:
                rows = parent.candidate.bicluster.row_indices
                columns = parent.candidate.bicluster.column_indices
                operation_name = "copy_fallback"
            else:
                dimension, operation = choices[int(generator.integers(0, len(choices)))]
                selected = (
                    parent.candidate.bicluster.row_indices
                    if dimension == "rows"
                    else parent.candidate.bicluster.column_indices
                )
                population = (
                    context.dataset.row_count
                    if dimension == "rows"
                    else context.dataset.column_count
                )
                minimum = bounds.min_rows if dimension == "rows" else bounds.min_columns
                moved = _random_membership_move(
                    selected,
                    population,
                    minimum,
                    operation,
                    generator,
                )
                assert moved is not None
                rows = moved if dimension == "rows" else parent.candidate.bicluster.row_indices
                columns = (
                    moved if dimension == "columns" else parent.candidate.bicluster.column_indices
                )
                operation_name = f"{operation}_{dimension[:-1]}"
            candidate = _candidate(
                producer=self.component_name,
                operation=operation_name,
                sequence=sequence,
                rows=rows,
                columns=columns,
                generation=parent.candidate.generation + 1,
                parents=(parent.candidate.identifier,),
            )
            context.candidate_validity_policy.validate(candidate, context.dataset)
            candidates.append(candidate)
        return tuple(candidates)


@dataclass(frozen=True, slots=True)
class ShapeMoveEmitter:
    guided: bool = False
    parent_pool_size: int = 16
    candidate_pool_size: int = 64
    component_name: str = "shape_move"
    provides: frozenset[str] = frozenset({"emitter"})
    requires: frozenset[str] = frozenset(
        {"archive", "candidate-validity", "parent-selection", "prepared-dataset"}
    )

    def __post_init__(self) -> None:
        ShapeMoveEmitterConfiguration(
            guided=self.guided,
            parent_pool_size=self.parent_pool_size,
            candidate_pool_size=self.candidate_pool_size,
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
        generator = context.random_generator("emitter.shape_move")
        bounds = context.candidate_validity_policy.bounds(context.dataset)
        candidates: list[Candidate] = []
        for index in range(count):

            def eligible(evaluation: Evaluation) -> bool:
                bicluster = evaluation.candidate.bicluster
                return (
                    len(bicluster.row_indices) < bounds.max_rows
                    and len(bicluster.column_indices) > bounds.min_columns
                ) or (
                    len(bicluster.column_indices) < bounds.max_columns
                    and len(bicluster.row_indices) > bounds.min_rows
                )

            parent = _select_parent(
                context,
                repertoire,
                generator,
                pool_size=self.parent_pool_size,
                guided=self.guided,
                eligible=eligible,
            )
            sequence = start_sequence + index
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
            bicluster = parent.candidate.bicluster
            directions: list[Literal["row_expand", "column_expand"]] = []
            if (
                len(bicluster.row_indices) < bounds.max_rows
                and len(bicluster.column_indices) > bounds.min_columns
            ):
                directions.append("row_expand")
            if (
                len(bicluster.column_indices) < bounds.max_columns
                and len(bicluster.row_indices) > bounds.min_rows
            ):
                directions.append("column_expand")
            direction = (
                min(directions)
                if self.guided
                else directions[int(generator.integers(0, len(directions)))]
            )
            if direction == "row_expand":
                rows = _random_membership_move(
                    bicluster.row_indices,
                    context.dataset.row_count,
                    bounds.min_rows,
                    "add",
                    generator,
                )
                columns = (
                    _guided_membership_move(
                        context,
                        parent,
                        "columns",
                        "remove",
                        generator,
                        self.candidate_pool_size,
                    )
                    if self.guided
                    else _random_membership_move(
                        bicluster.column_indices,
                        context.dataset.column_count,
                        bounds.min_columns,
                        "remove",
                        generator,
                    )
                )
            else:
                columns = _random_membership_move(
                    bicluster.column_indices,
                    context.dataset.column_count,
                    bounds.min_columns,
                    "add",
                    generator,
                )
                rows = (
                    _guided_membership_move(
                        context,
                        parent,
                        "rows",
                        "remove",
                        generator,
                        self.candidate_pool_size,
                    )
                    if self.guided
                    else _random_membership_move(
                        bicluster.row_indices,
                        context.dataset.row_count,
                        bounds.min_rows,
                        "remove",
                        generator,
                    )
                )
            assert rows is not None and columns is not None
            candidate = _candidate(
                producer=self.component_name,
                operation=direction,
                sequence=sequence,
                rows=rows,
                columns=columns,
                generation=parent.candidate.generation + 1,
                parents=(parent.candidate.identifier,),
            )
            context.candidate_validity_policy.validate(candidate, context.dataset)
            candidates.append(candidate)
        return tuple(candidates)


@dataclass(frozen=True, slots=True)
class RestartEmitter:
    strategy: RestartStrategy = "stratified"
    cardinality_levels: int = 8
    joint_column_candidate_pool_size: int = 32
    component_name: str = "restart"
    provides: frozenset[str] = frozenset({"emitter"})
    requires: frozenset[str] = frozenset(
        {"archive", "candidate-validity", "prepared-dataset", "robust-numeric-data"}
    )

    def __post_init__(self) -> None:
        RestartEmitterConfiguration(
            strategy=self.strategy,
            cardinality_levels=self.cardinality_levels,
            joint_column_candidate_pool_size=self.joint_column_candidate_pool_size,
        )

    def emit(
        self,
        context: RunContext,
        repertoire: Repertoire,
        count: int,
        *,
        start_sequence: int = 0,
    ) -> Sequence[Candidate]:
        del repertoire
        initializer: StratifiedInitializer | PatternAwareInitializer
        if self.strategy == "pattern_aware":
            initializer = PatternAwareInitializer(
                cardinality_levels=self.cardinality_levels,
                joint_column_candidate_pool_size=self.joint_column_candidate_pool_size,
                component_name=self.component_name,
            )
        else:
            initializer = StratifiedInitializer(
                cardinality_levels=self.cardinality_levels,
                component_name=self.component_name,
            )
        generated = initializer.initialize(
            context,
            count,
            start_sequence=start_sequence,
        )
        return tuple(
            candidate.model_copy(
                update={
                    "provenance": candidate.provenance.model_copy(
                        update={"operation": f"{self.strategy}_restart"}
                    )
                    if candidate.provenance is not None
                    else None
                }
            )
            for candidate in generated
        )


@dataclass(frozen=True, slots=True)
class CellCoverageRestartEmitter:
    """Generate pattern-aware restarts in the least represented reachable cells."""

    joint_column_candidate_pool_size: int = 32
    component_name: str = "cell_coverage_restart"
    provides: frozenset[str] = frozenset({"emitter"})
    requires: frozenset[str] = frozenset(
        {
            "archive",
            "candidate-validity",
            "prepared-dataset",
            "robust-numeric-data",
            "archive-cell-targets",
            "descriptor:row-cardinality",
            "descriptor:column-cardinality",
        }
    )

    def __post_init__(self) -> None:
        CellCoverageRestartEmitterConfiguration(
            joint_column_candidate_pool_size=self.joint_column_candidate_pool_size,
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
        if count == 0:
            return ()
        qd_context = require_qd_run_context(context)
        if not qd_context.archive_cell_targets:
            raise ComponentError("cell-coverage restart requires archive cardinality targets")
        occupancy = {target.coordinate.indices: 0 for target in qd_context.archive_cell_targets}
        for evaluation in repertoire.evaluations:
            coordinate = evaluation.archive_coordinate
            if coordinate in occupancy:
                occupancy[coordinate] += 1
        generated: list[Candidate] = []
        initializer = CellCoveragePatternAwareInitializer(
            seeds_per_cell=1,
            max_attempts_per_cell=1,
            joint_column_candidate_pool_size=self.joint_column_candidate_pool_size,
            component_name=self.component_name,
        )
        catalog = default_pattern_catalog(context.patterns.allowed)
        reachable_by_pattern = []
        for pattern in context.patterns.allowed:
            strategy = catalog.implementation(pattern).seed_strategy
            if strategy is None:
                raise ComponentError(
                    f"{pattern.value} has no registered pattern-aware seed strategy"
                )
            reachable = tuple(
                target
                for target in qd_context.archive_cell_targets
                if strategy.project_target(context, target) is not None
            )
            if reachable:
                reachable_by_pattern.append((pattern, reachable))
        if not reachable_by_pattern:
            raise ComponentError("cell-coverage restart found no reachable archive cell")
        for index in range(count):
            pattern, reachable = reachable_by_pattern[
                (start_sequence + index) % len(reachable_by_pattern)
            ]
            target = min(
                reachable,
                key=lambda item: (
                    occupancy[item.coordinate.indices],
                    item.coordinate.indices,
                ),
            )
            state = BootstrapCellState(
                target=target,
                required_patterns=(pattern,),
                maximum_attempts=1,
            )
            candidate = initializer.initialize_bootstrap(
                qd_context,
                (state,),
                start_sequence=start_sequence + index,
            )[0]
            provenance = candidate.provenance
            assert provenance is not None
            generated.append(
                candidate.model_copy(
                    update={
                        "provenance": provenance.model_copy(
                            update={
                                "producer": self.component_name,
                                "operation": "cell_coverage_restart",
                            }
                        )
                    }
                )
            )
            occupancy[target.coordinate.indices] += 1
        return tuple(generated)


__all__ = [
    "CellCoverageRestartEmitter",
    "CellCoverageRestartEmitterConfiguration",
    "MembershipEmitterConfiguration",
    "MembershipMoveEmitter",
    "RandomMoveEmitter",
    "RestartEmitter",
    "RestartEmitterConfiguration",
    "ShapeMoveEmitter",
    "ShapeMoveEmitterConfiguration",
]
