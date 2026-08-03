"""Pattern-owned strategies for feasible, support-aware candidate seeding."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import numpy.random as npr
import numpy.typing as npt

from salvi.application.context import RunContext
from salvi.domain.enums import PatternKind
from salvi.domain.search import ArchiveCellTarget
from salvi.exceptions import ComponentError
from salvi.patterns.contracts import (
    PatternDefinition,
    PatternSeed,
    PatternSeedShape,
)
from salvi.patterns.joint_models import robust_column_scales
from salvi.patterns.math import NUMERIC_TOLERANCE, nanmedian_2d


def _sample_indices(
    generator: npr.Generator,
    population: int,
    count: int,
) -> tuple[int, ...]:
    return tuple(sorted(int(value) for value in generator.choice(population, count, replace=False)))


def _ordered_cardinalities(preferred: int, minimum: int, maximum: int) -> tuple[int, ...]:
    bounded = min(max(preferred, minimum), maximum)
    return (
        bounded,
        *range(bounded - 1, minimum - 1, -1),
        *range(bounded + 1, maximum + 1),
    )


def _finite_median(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return None if finite.size == 0 else float(np.median(finite))


def constant_column_errors(
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
    required_support = context.evaluation_support_policy.required_observations(len(rows))
    sufficient = source_supports >= required_support
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
        with np.errstate(invalid="ignore"):
            centers = nanmedian_2d(matrix, axis=0)
            deviations = np.abs(matrix - centers[np.newaxis, :])
            counts = np.count_nonzero(np.isfinite(matrix), axis=0)
            means = np.divide(
                np.nansum(deviations, axis=0),
                counts,
                out=np.full(len(counts), np.nan, dtype=np.float64),
                where=counts > 0,
            )
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
        contributions = np.minimum(1.0, np.abs(numeric_matrix - anchor_values[np.newaxis, :]))
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
            np.where(usable, discrete_matrix != anchor_values[np.newaxis, :], False),
            axis=1,
        )
        support += np.count_nonzero(usable, axis=1).astype(np.int32)
    normalized = np.full(dataset.row_count, math.inf, dtype=np.float64)
    np.divide(scores, support, out=normalized, where=support > 0)
    normalized[anchor] = -1.0
    return normalized


@dataclass(frozen=True, slots=True)
class _BasePatternSeedStrategy:
    definition: PatternDefinition

    @property
    def pattern(self) -> PatternKind:
        return self.definition.kind

    def _eligible_columns(self, context: RunContext, row_count: int) -> tuple[int, ...]:
        required = context.evaluation_support_policy.required_observations(row_count)
        support_counts = np.count_nonzero(context.dataset.support_matrix(), axis=0)
        return tuple(
            column.index
            for column in context.dataset.columns
            if self.definition.supports(column.kind) and support_counts[column.index] >= required
        )

    def project_shape(
        self,
        context: RunContext,
        *,
        preferred_row_count: int,
        row_range: tuple[int, int],
        preferred_column_count: int,
        column_range: tuple[int, int],
    ) -> PatternSeedShape | None:
        limits = self._shape_limits(context, row_range=row_range, column_range=column_range)
        if limits is None:
            return None
        minimum_rows, maximum_rows, minimum_columns, maximum_columns = limits
        row_counts = _ordered_cardinalities(preferred_row_count, minimum_rows, maximum_rows)
        for row_count in row_counts:
            capacity = len(self._eligible_columns(context, row_count))
            available_maximum = min(maximum_columns, capacity)
            if available_maximum < minimum_columns:
                continue
            column_count = min(
                max(preferred_column_count, minimum_columns),
                available_maximum,
            )
            return PatternSeedShape(row_count=row_count, column_count=column_count)
        return None

    def _shape_limits(
        self,
        context: RunContext,
        *,
        row_range: tuple[int, int],
        column_range: tuple[int, int],
    ) -> tuple[int, int, int, int] | None:
        bounds = context.candidate_validity_policy.bounds(context.dataset)
        minimum_rows = max(bounds.min_rows, row_range[0])
        maximum_rows = min(bounds.max_rows, row_range[1])
        minimum_columns = max(
            bounds.min_columns,
            self.definition.minimum_columns,
            column_range[0],
        )
        maximum_columns = min(bounds.max_columns, column_range[1])
        if minimum_rows > maximum_rows or minimum_columns > maximum_columns:
            return None
        return minimum_rows, maximum_rows, minimum_columns, maximum_columns

    def project_target(
        self,
        context: RunContext,
        target: ArchiveCellTarget,
    ) -> PatternSeedShape | None:
        return self.project_shape(
            context,
            preferred_row_count=target.row_count,
            row_range=target.row_range,
            preferred_column_count=target.column_count,
            column_range=target.column_range,
        )


@dataclass(frozen=True, slots=True)
class ConstantPatternSeedStrategy(_BasePatternSeedStrategy):
    def generate(
        self,
        context: RunContext,
        generator: npr.Generator,
        shape: PatternSeedShape,
        *,
        joint_column_candidate_pool_size: int,
    ) -> PatternSeed:
        del joint_column_candidate_pool_size
        dataset = context.dataset
        eligible = self._eligible_columns(context, shape.row_count)
        if len(eligible) < shape.column_count:
            raise ComponentError(
                f"{self.pattern.value} seeding has fewer support-eligible columns than requested"
            )
        support = dataset.support_matrix()[:, np.asarray(eligible, dtype=np.int64)]
        anchor_candidates = np.flatnonzero(np.any(support, axis=1))
        if anchor_candidates.size == 0:
            raise ComponentError(f"{self.pattern.value} seeding found no observed anchor row")
        attempts = min(16, len(anchor_candidates))
        anchors = generator.choice(anchor_candidates, attempts, replace=False)
        candidate_columns = np.asarray(eligible, dtype=np.int64)
        for raw_anchor in anchors:
            anchor = int(raw_anchor)
            observed = candidate_columns[support[anchor]]
            if observed.size == 0:
                continue
            anchor_column = int(observed[int(generator.integers(0, len(observed)))])
            seed_scores = _constant_row_scores(context, anchor, (anchor_column,))
            seed_rows = tuple(
                sorted(int(index) for index in np.argsort(seed_scores)[: shape.row_count])
            )
            column_errors = constant_column_errors(context, seed_rows, candidate_columns)
            finite = np.flatnonzero(np.isfinite(column_errors))
            if finite.size < shape.column_count:
                continue
            order = finite[np.lexsort((candidate_columns[finite], column_errors[finite]))]
            columns = tuple(
                sorted(int(column) for column in candidate_columns[order[: shape.column_count]])
            )
            refined_scores = _constant_row_scores(context, anchor, columns)
            if np.count_nonzero(np.isfinite(refined_scores)) < shape.row_count:
                continue
            rows = tuple(
                sorted(int(index) for index in np.argsort(refined_scores)[: shape.row_count])
            )
            final_errors = constant_column_errors(
                context,
                rows,
                np.asarray(columns, dtype=np.int64),
            )
            if np.all(np.isfinite(final_errors)):
                return PatternSeed(rows=rows, columns=columns)
        raise ComponentError(
            f"{self.pattern.value} seeding could not construct a support-valid anchor"
        )


def _joint_relation_ranking(
    pattern: PatternKind,
    raw: npt.NDArray[np.float64],
    first: int,
    second: int,
    scales: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    if pattern is PatternKind.ADDITIVE:
        relation = raw[second] - raw[first]
        center = _finite_median(relation)
        return (
            np.full(raw.shape[1], math.inf)
            if center is None
            else np.abs(relation - center) / scales
        )
    if pattern is PatternKind.MULTIPLICATIVE:
        denominator = raw[first]
        usable = (
            np.isfinite(raw[second])
            & np.isfinite(denominator)
            & (np.abs(denominator) > NUMERIC_TOLERANCE)
        )
        relation = np.full(raw.shape[1], np.nan)
        relation[usable] = raw[second, usable] / denominator[usable]
        center = _finite_median(relation)
        return np.full(raw.shape[1], math.inf) if center is None else np.abs(relation - center)
    raise ComponentError(f"{pattern.value} has no joint seed relation")


def _joint_row_scores(
    context: RunContext,
    pattern: PatternKind,
    matrix: npt.NDArray[np.float64],
    first: int,
    scales: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    row_scores = np.full(context.dataset.row_count, math.inf)
    if pattern is PatternKind.ADDITIVE:
        anchor_profile = matrix[first]
        anchor_center = _finite_median(anchor_profile)
        if anchor_center is None:
            return row_scores
        anchor_profile = anchor_profile - anchor_center
        usable = np.isfinite(matrix) & np.isfinite(anchor_profile)[np.newaxis, :]
        with np.errstate(invalid="ignore"):
            centers = nanmedian_2d(matrix, axis=1)
            residuals = np.where(
                usable,
                np.abs(matrix - centers[:, np.newaxis] - anchor_profile[np.newaxis, :])
                / scales[np.newaxis, :],
                np.nan,
            )
            scores = nanmedian_2d(residuals, axis=1)
    elif pattern is PatternKind.MULTIPLICATIVE:
        anchor_profile = matrix[first]
        usable = (
            np.isfinite(matrix)
            & np.isfinite(anchor_profile)[np.newaxis, :]
            & (np.abs(anchor_profile) > NUMERIC_TOLERANCE)[np.newaxis, :]
        )
        ratios = np.full_like(matrix, np.nan)
        np.divide(matrix, anchor_profile[np.newaxis, :], out=ratios, where=usable)
        with np.errstate(invalid="ignore"):
            centers = nanmedian_2d(ratios, axis=1)
            scores = nanmedian_2d(np.abs(ratios - centers[:, np.newaxis]), axis=1)
    else:
        raise ComponentError(f"{pattern.value} has no joint seed relation")
    required = context.evaluation_support_policy.required_observations(matrix.shape[1])
    valid = (
        np.isfinite(centers) & np.isfinite(scores) & (np.count_nonzero(usable, axis=1) >= required)
    )
    row_scores[valid] = scores[valid]
    return row_scores


def _support_valid_row_selection(
    context: RunContext,
    scores: npt.NDArray[np.float64],
    support: npt.NDArray[np.bool_],
    row_count: int,
    mandatory_rows: tuple[int, ...],
) -> tuple[int, ...] | None:
    """Select high-affinity rows, repairing marginal column support by swaps."""

    required_per_row = context.evaluation_support_policy.required_observations(support.shape[1])
    eligible = np.isfinite(scores) & (np.count_nonzero(support, axis=1) >= required_per_row)
    if np.count_nonzero(eligible) < row_count or not all(eligible[row] for row in mandatory_rows):
        return None

    ranked = np.flatnonzero(eligible)[np.lexsort((np.flatnonzero(eligible), scores[eligible]))]
    selected_mask = np.zeros(len(scores), dtype=np.bool_)
    selected_mask[ranked[:row_count]] = True
    selected_mask[np.asarray(mandatory_rows, dtype=np.int64)] = True
    if np.count_nonzero(selected_mask) > row_count:
        removable = np.flatnonzero(selected_mask)
        mandatory_mask = np.isin(removable, np.asarray(mandatory_rows, dtype=np.int64))
        drop_order = removable[~mandatory_mask][
            np.lexsort(
                (
                    -removable[~mandatory_mask],
                    -scores[removable[~mandatory_mask]],
                )
            )
        ]
        selected_mask[drop_order[: np.count_nonzero(selected_mask) - row_count]] = False

    required_per_column = context.evaluation_support_policy.required_observations(row_count)
    selected_support = support[selected_mask]
    column_counts = np.count_nonzero(selected_support, axis=0).astype(np.int64)
    mandatory = frozenset(mandatory_rows)
    while np.any(column_counts < required_per_column):
        current_deficit = int(np.sum(np.maximum(required_per_column - column_counts, 0)))
        deficit_columns = column_counts < required_per_column
        outside = np.flatnonzero(eligible & ~selected_mask)
        if outside.size == 0:
            return None
        gains = np.count_nonzero(support[outside][:, deficit_columns], axis=1)
        useful = gains > 0
        additions = outside[useful]
        if additions.size == 0:
            return None
        additions = additions[np.lexsort((additions, scores[additions], -gains[useful]))]

        selected = np.flatnonzero(selected_mask)
        removable = np.asarray(
            [row for row in selected if int(row) not in mandatory],
            dtype=np.int64,
        )
        if removable.size == 0:
            return None
        repaired = False
        for added in additions:
            counts_after_addition = column_counts + support[added].astype(np.int64)
            projected = counts_after_addition[np.newaxis, :] - support[removable].astype(np.int64)
            deficits = np.sum(
                np.maximum(required_per_column - projected, 0),
                axis=1,
            )
            best_deficit = int(np.min(deficits))
            if best_deficit >= current_deficit:
                continue
            options = removable[deficits == best_deficit]
            removed = int(options[np.lexsort((-options, -scores[options]))][0])
            selected_mask[removed] = False
            selected_mask[int(added)] = True
            column_counts = counts_after_addition - support[removed].astype(np.int64)
            repaired = True
            break
        if not repaired:
            return None

    return tuple(int(row) for row in np.flatnonzero(selected_mask))


@dataclass(frozen=True, slots=True)
class _JointPatternSeedStrategy(_BasePatternSeedStrategy):
    def project_shape(
        self,
        context: RunContext,
        *,
        preferred_row_count: int,
        row_range: tuple[int, int],
        preferred_column_count: int,
        column_range: tuple[int, int],
    ) -> PatternSeedShape | None:
        limits = self._shape_limits(context, row_range=row_range, column_range=column_range)
        if limits is None:
            return None
        minimum_rows, maximum_rows, minimum_columns, maximum_columns = limits
        support = context.dataset.support_matrix()
        global_support = np.count_nonzero(support, axis=0)
        for row_count in _ordered_cardinalities(
            preferred_row_count,
            minimum_rows,
            maximum_rows,
        ):
            eligible = self._eligible_columns(context, row_count)
            available_maximum = min(maximum_columns, len(eligible))
            if available_maximum < minimum_columns:
                continue
            for column_count in _ordered_cardinalities(
                preferred_column_count,
                minimum_columns,
                available_maximum,
            ):
                eligible_array = np.asarray(eligible, dtype=np.int64)
                order = np.lexsort((eligible_array, -global_support[eligible_array]))
                columns = eligible_array[order[:column_count]]
                selected_support = support[:, columns]
                required_per_row = context.evaluation_support_policy.required_observations(
                    column_count
                )
                support_valid_rows = np.count_nonzero(selected_support, axis=1) >= required_per_row
                if np.count_nonzero(support_valid_rows) < row_count:
                    continue
                required_per_column = context.evaluation_support_policy.required_observations(
                    row_count
                )
                if np.all(
                    np.count_nonzero(selected_support[support_valid_rows], axis=0)
                    >= required_per_column
                ):
                    return PatternSeedShape(
                        row_count=row_count,
                        column_count=column_count,
                    )
        return None

    def generate(
        self,
        context: RunContext,
        generator: npr.Generator,
        shape: PatternSeedShape,
        *,
        joint_column_candidate_pool_size: int,
    ) -> PatternSeed:
        dataset = context.dataset
        eligible = self._eligible_columns(context, shape.row_count)
        if len(eligible) < shape.column_count:
            raise ComponentError(
                f"{self.pattern.value} seeding requires at least {shape.column_count} "
                "support-eligible columns"
            )
        eligible_array = np.asarray(eligible, dtype=np.int64)
        positions = np.asarray(
            [dataset.numeric_positions[column] for column in eligible],
            dtype=np.int64,
        )
        raw = dataset.numeric_matrix()[:, positions]
        support = dataset.support_matrix()[:, eligible_array]
        row_support = np.count_nonzero(support, axis=1)
        strongest_rows = np.argsort(-row_support, kind="stable")[: min(dataset.row_count, 24)]
        pair_candidates = list(combinations((int(value) for value in strongest_rows), 2))
        generator.shuffle(pair_candidates)

        def usable_count(pair: tuple[int, int]) -> int:
            first, second = pair
            usable = support[first] & support[second]
            if self.pattern is PatternKind.MULTIPLICATIVE:
                usable &= np.abs(raw[first]) > NUMERIC_TOLERANCE
            return int(np.count_nonzero(usable))

        pair_candidates.sort(key=usable_count, reverse=True)
        for first, second in pair_candidates[:32]:
            usable = support[first] & support[second]
            if self.pattern is PatternKind.MULTIPLICATIVE:
                usable &= np.abs(raw[first]) > NUMERIC_TOLERANCE
            available = np.flatnonzero(usable)
            if available.size < shape.column_count:
                continue
            pool_size = min(
                len(available),
                max(shape.column_count, joint_column_candidate_pool_size),
            )
            selected_positions = generator.choice(available, pool_size, replace=False)
            candidate_columns = eligible_array[selected_positions]
            candidate_raw = raw[:, selected_positions]
            candidate_scales = robust_column_scales(
                context,
                tuple(int(column) for column in candidate_columns),
            )
            ranking = _joint_relation_ranking(
                self.pattern,
                candidate_raw,
                first,
                second,
                candidate_scales,
            )
            finite = np.flatnonzero(np.isfinite(ranking))
            if finite.size < shape.column_count:
                continue
            order = finite[np.lexsort((candidate_columns[finite], ranking[finite]))]
            chosen = order[: shape.column_count]
            columns = tuple(sorted(int(column) for column in candidate_columns[chosen]))
            matrix = np.column_stack([dataset.numeric_column(column) for column in columns])
            scales = robust_column_scales(context, columns)
            row_scores = _joint_row_scores(context, self.pattern, matrix, first, scales)
            row_scores[first] = -2.0
            row_scores[second] = -1.0
            selected_support = dataset.support_matrix()[:, np.asarray(columns, dtype=np.int64)]
            rows = _support_valid_row_selection(
                context,
                row_scores,
                selected_support,
                shape.row_count,
                (first, second),
            )
            if rows is None:
                continue
            required = context.evaluation_support_policy.required_observations(len(rows))
            candidate_support = selected_support[np.asarray(rows, dtype=np.int64)]
            if np.all(np.count_nonzero(candidate_support, axis=0) >= required):
                return PatternSeed(rows=rows, columns=columns)
        raise ComponentError(
            f"{self.pattern.value} seeding could not construct a support-valid joint anchor"
        )


@dataclass(frozen=True, slots=True)
class AdditivePatternSeedStrategy(_JointPatternSeedStrategy):
    pass


@dataclass(frozen=True, slots=True)
class MultiplicativePatternSeedStrategy(_JointPatternSeedStrategy):
    pass


__all__ = [
    "AdditivePatternSeedStrategy",
    "ConstantPatternSeedStrategy",
    "MultiplicativePatternSeedStrategy",
    "constant_column_errors",
]
