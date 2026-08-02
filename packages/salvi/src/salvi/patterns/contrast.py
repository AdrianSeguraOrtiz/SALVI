"""Pattern-dispatched contrast evaluation with per-column explanations."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from salvi.application.context import RunContext
from salvi.domain.enums import ColumnKind, EvaluationIssueCode, PatternKind
from salvi.domain.models import (
    Bicluster,
    ColumnObjectiveValue,
    ColumnPatternFit,
    EvaluationIssue,
    ObjectiveResult,
    PatternFit,
    PatternGroupFit,
)
from salvi.patterns.joint_models import (
    additive_row_effects,
    multiplicative_column_scales,
    multiplicative_row_effects,
)
from salvi.patterns.math import NUMERIC_TOLERANCE, clamp01, diagnostics

if TYPE_CHECKING:
    from salvi.patterns.catalog import PatternCatalog


def _residual_auc(local: npt.NDArray[np.float64], background: npt.NDArray[np.float64]) -> float:
    """Return the probability that a local residual is below a background residual."""

    ordered = np.sort(local)
    lower = np.searchsorted(ordered, background - NUMERIC_TOLERANCE, side="left")
    upper = np.searchsorted(ordered, background + NUMERIC_TOLERANCE, side="right")
    favorable = float(np.sum(lower + 0.5 * (upper - lower)))
    return clamp01(favorable / (len(local) * len(background)))


def _categorical_separation(local_ratio: float, background_ratio: float) -> float:
    if abs(local_ratio - background_ratio) <= NUMERIC_TOLERANCE:
        return 0.5
    if local_ratio > background_ratio:
        return 0.5 + 0.5 * (local_ratio - background_ratio) / (1.0 - background_ratio)
    return 0.5 - 0.5 * (background_ratio - local_ratio) / background_ratio


def _selected_mask(context: RunContext, bicluster: Bicluster) -> npt.NDArray[np.bool_]:
    selected = np.zeros(context.dataset.row_count, dtype=np.bool_)
    selected[np.asarray(bicluster.row_indices, dtype=np.int64)] = True
    return selected


def _support_is_sufficient(
    context: RunContext,
    source_mask: npt.NDArray[np.bool_],
    region_mask: npt.NDArray[np.bool_],
) -> tuple[bool, int, int]:
    opportunity = int(np.count_nonzero(region_mask))
    observed = int(np.count_nonzero(source_mask & region_mask))
    return (
        context.evaluation_support_policy.is_sufficient(observed, opportunity),
        observed,
        opportunity,
    )


def _invalid_column(
    column_index: int,
    reason: str,
    code: EvaluationIssueCode,
    pattern: PatternKind | None,
) -> tuple[ColumnObjectiveValue, EvaluationIssue]:
    return (
        ColumnObjectiveValue.model_construct(
            column_index=column_index,
            value=0.0,
            valid=False,
            diagnostics=diagnostics(reason=reason),
        ),
        EvaluationIssue(
            code=code,
            message=reason,
            column_index=column_index,
            pattern=pattern,
        ),
    )


def _objective_result(
    values: list[ColumnObjectiveValue],
    issues: list[EvaluationIssue],
) -> ObjectiveResult:
    ordered = tuple(sorted(values, key=lambda value: value.column_index))
    aggregate = 0.0 if not ordered else clamp01(sum(item.value for item in ordered) / len(ordered))
    return ObjectiveResult.model_construct(
        value=aggregate,
        columns=ordered,
        issues=tuple(issues),
    )


def _joint_residual_column(
    context: RunContext,
    selected: npt.NDArray[np.bool_],
    column_fit: ColumnPatternFit,
    *,
    pattern: PatternKind,
    label: str,
    values: npt.NDArray[np.float64],
    prediction: npt.NDArray[np.float64],
    group_identifier: str,
    extra_diagnostics: dict[str, float | int | str | bool | None] | None = None,
) -> tuple[ColumnObjectiveValue, EvaluationIssue | None]:
    column_index = column_fit.column_index
    usable = np.isfinite(values) & np.isfinite(prediction)
    source = context.dataset.support_mask(column_index) & np.isfinite(prediction)
    local_ok, local_source, local_opportunities = _support_is_sufficient(context, source, selected)
    background_ok, background_source, background_opportunities = _support_is_sufficient(
        context, source, ~selected
    )
    residuals = np.abs(values - prediction)
    local = residuals[usable & selected]
    background = residuals[usable & ~selected]
    if not local_ok or local.size == 0:
        return _invalid_column(
            column_index,
            f"insufficient local {label} contrast support",
            EvaluationIssueCode.INSUFFICIENT_LOCAL_SUPPORT,
            pattern,
        )
    if not background_ok or background.size == 0:
        return _invalid_column(
            column_index,
            f"insufficient {label} background support",
            EvaluationIssueCode.INSUFFICIENT_BACKGROUND,
            pattern,
        )
    details: dict[str, float | int | str | bool | None] = {
        "group_identifier": group_identifier,
        "local_available": int(local.size),
        "local_opportunities": local_opportunities,
        "local_source_support": local_source,
        "background_available": int(background.size),
        "background_opportunities": background_opportunities,
        "background_source_support": background_source,
    }
    if extra_diagnostics is not None:
        details.update(extra_diagnostics)
    return (
        ColumnObjectiveValue.model_construct(
            column_index=column_index,
            value=_residual_auc(local, background),
            diagnostics=diagnostics(**details),
        ),
        None,
    )


@dataclass(frozen=True, slots=True)
class ConstantPatternContrastStrategy:
    pattern: PatternKind = PatternKind.CONSTANT

    def evaluate(
        self,
        context: RunContext,
        bicluster: Bicluster,
        fit: PatternFit,
        columns: Sequence[ColumnPatternFit],
    ) -> ObjectiveResult:
        del fit
        selected = _selected_mask(context, bicluster)
        values: list[ColumnObjectiveValue] = []
        issues: list[EvaluationIssue] = []
        for column_fit in columns:
            if context.dataset.column_metadata(column_fit.column_index).kind is ColumnKind.NUMERIC:
                result, issue = self._numeric_column(context, selected, column_fit)
            else:
                result, issue = self._discrete_column(context, selected, column_fit)
            values.append(result)
            if issue is not None:
                issues.append(issue)
        return _objective_result(values, issues)

    def _numeric_column(
        self,
        context: RunContext,
        selected: npt.NDArray[np.bool_],
        column_fit: ColumnPatternFit,
    ) -> tuple[ColumnObjectiveValue, EvaluationIssue | None]:
        column_index = column_fit.column_index
        parameter = column_fit.parameter
        if not isinstance(parameter, float):
            return _invalid_column(
                column_index,
                "numeric prototype is unavailable",
                EvaluationIssueCode.PATTERN_FIT_FAILED,
                self.pattern,
            )
        dataset = context.dataset
        raw = dataset.numeric_column(column_index)
        available = dataset.available_mask(column_index) & np.isfinite(raw)
        source = dataset.support_mask(column_index)
        local_ok, local_source, local_opportunities = _support_is_sufficient(
            context, source, selected
        )
        background_ok, background_source, background_opportunities = _support_is_sufficient(
            context, source, ~selected
        )
        local = np.abs(raw[available & selected] - parameter)
        background = np.abs(raw[available & ~selected] - parameter)
        if not local_ok or local.size == 0:
            return _invalid_column(
                column_index,
                "insufficient local numeric contrast support",
                EvaluationIssueCode.INSUFFICIENT_LOCAL_SUPPORT,
                self.pattern,
            )
        if not background_ok or background.size == 0:
            return _invalid_column(
                column_index,
                "insufficient numeric background support",
                EvaluationIssueCode.INSUFFICIENT_BACKGROUND,
                self.pattern,
            )
        return (
            ColumnObjectiveValue.model_construct(
                column_index=column_index,
                value=_residual_auc(local, background),
                diagnostics=diagnostics(
                    local_available=int(local.size),
                    local_opportunities=local_opportunities,
                    local_source_support=local_source,
                    background_available=int(background.size),
                    background_opportunities=background_opportunities,
                    background_source_support=background_source,
                ),
            ),
            None,
        )

    def _discrete_column(
        self,
        context: RunContext,
        selected: npt.NDArray[np.bool_],
        column_fit: ColumnPatternFit,
    ) -> tuple[ColumnObjectiveValue, EvaluationIssue | None]:
        column_index = column_fit.column_index
        prototype = column_fit.parameter
        if prototype is None or isinstance(prototype, float) or column_fit.available_support == 0:
            return _invalid_column(
                column_index,
                "categorical prototype is unavailable",
                EvaluationIssueCode.PATTERN_FIT_FAILED,
                self.pattern,
            )
        dataset = context.dataset
        background = ~selected
        background_ok, background_source, background_opportunities = _support_is_sufficient(
            context, dataset.support_mask(column_index), background
        )
        codes = dataset.discrete_column(column_index)
        available_background = background & (codes >= 0)
        background_available = int(np.count_nonzero(available_background))
        if not background_ok or background_available == 0:
            return _invalid_column(
                column_index,
                "insufficient categorical background support",
                EvaluationIssueCode.INSUFFICIENT_BACKGROUND,
                self.pattern,
            )
        try:
            prototype_code = dataset.discrete_code(column_index, prototype)
        except ValueError:
            return _invalid_column(
                column_index,
                "categorical prototype is not declared by the prepared column",
                EvaluationIssueCode.PATTERN_FIT_FAILED,
                self.pattern,
            )
        background_prototype = int(np.count_nonzero(codes[available_background] == prototype_code))
        local_ratio = column_fit.prototype_support / column_fit.available_support
        background_ratio = background_prototype / background_available
        return (
            ColumnObjectiveValue.model_construct(
                column_index=column_index,
                value=clamp01(_categorical_separation(local_ratio, background_ratio)),
                diagnostics=diagnostics(
                    local_prototype_ratio=local_ratio,
                    background_prototype_ratio=background_ratio,
                    background_available=background_available,
                    background_opportunities=background_opportunities,
                    background_source_support=background_source,
                ),
            ),
            None,
        )


@dataclass(frozen=True, slots=True)
class AdditivePatternContrastStrategy:
    pattern: PatternKind = PatternKind.ADDITIVE

    def evaluate(
        self,
        context: RunContext,
        bicluster: Bicluster,
        fit: PatternFit,
        columns: Sequence[ColumnPatternFit],
    ) -> ObjectiveResult:
        selected = _selected_mask(context, bicluster)
        groups = {group.identifier: group for group in fit.groups if group.pattern is self.pattern}
        effects = self._group_row_effects(context, fit, groups.values())
        values: list[ColumnObjectiveValue] = []
        issues: list[EvaluationIssue] = []
        for column_fit in columns:
            group = groups.get(column_fit.group_identifier or "")
            result, issue = self._column(context, selected, column_fit, group, effects)
            values.append(result)
            if issue is not None:
                issues.append(issue)
        return _objective_result(values, issues)

    def _group_row_effects(
        self,
        context: RunContext,
        fit: PatternFit,
        groups: Iterable[PatternGroupFit],
    ) -> dict[str, npt.NDArray[np.float64]]:
        result: dict[str, npt.NDArray[np.float64]] = {}
        column_fits = {column.column_index: column for column in fit.columns}
        for group in groups:
            columns = group.column_indices
            parameters = np.asarray(
                [column_fits[column].parameter for column in columns], dtype=np.float64
            )
            positions = tuple(context.dataset.numeric_positions[column] for column in columns)
            values = context.dataset.numeric_matrix(standardized=True)[:, positions]
            source = context.dataset.support_matrix()[:, columns]
            result[group.identifier] = additive_row_effects(
                context,
                values,
                source,
                parameters,
            )
        return result

    def _column(
        self,
        context: RunContext,
        selected: npt.NDArray[np.bool_],
        column_fit: ColumnPatternFit,
        group: PatternGroupFit | None,
        group_effects: dict[str, npt.NDArray[np.float64]],
    ) -> tuple[ColumnObjectiveValue, EvaluationIssue | None]:
        column_index = column_fit.column_index
        if group is None or not isinstance(column_fit.parameter, float):
            return _invalid_column(
                column_index,
                "additive model is unavailable",
                EvaluationIssueCode.PATTERN_FIT_FAILED,
                self.pattern,
            )
        effects = group_effects[group.identifier]
        values = context.dataset.numeric_column(column_index, standardized=True)
        return _joint_residual_column(
            context,
            selected,
            column_fit,
            pattern=self.pattern,
            label="additive",
            values=values,
            prediction=effects + column_fit.parameter,
            group_identifier=group.identifier,
        )


@dataclass(frozen=True, slots=True)
class MultiplicativePatternContrastStrategy:
    pattern: PatternKind = PatternKind.MULTIPLICATIVE

    def evaluate(
        self,
        context: RunContext,
        bicluster: Bicluster,
        fit: PatternFit,
        columns: Sequence[ColumnPatternFit],
    ) -> ObjectiveResult:
        selected = _selected_mask(context, bicluster)
        groups = {group.identifier: group for group in fit.groups if group.pattern is self.pattern}
        effects, scales = self._group_models(context, fit, groups.values())
        values: list[ColumnObjectiveValue] = []
        issues: list[EvaluationIssue] = []
        for column_fit in columns:
            group = groups.get(column_fit.group_identifier or "")
            result, issue = self._column(
                context,
                selected,
                column_fit,
                group,
                effects,
                scales,
            )
            values.append(result)
            if issue is not None:
                issues.append(issue)
        return _objective_result(values, issues)

    def _group_models(
        self,
        context: RunContext,
        fit: PatternFit,
        groups: Iterable[PatternGroupFit],
    ) -> tuple[
        dict[str, npt.NDArray[np.float64]],
        dict[tuple[str, int], float],
    ]:
        effects: dict[str, npt.NDArray[np.float64]] = {}
        scales: dict[tuple[str, int], float] = {}
        column_fits = {column.column_index: column for column in fit.columns}
        for group in groups:
            columns = group.column_indices
            parameters = np.asarray(
                [column_fits[column].parameter for column in columns],
                dtype=np.float64,
            )
            group_scales = multiplicative_column_scales(context, columns)
            positions = tuple(context.dataset.numeric_positions[column] for column in columns)
            matrix = context.dataset.numeric_matrix()[:, positions] / group_scales[np.newaxis, :]
            source = context.dataset.support_matrix()[:, columns]
            effects[group.identifier] = multiplicative_row_effects(
                context,
                matrix,
                source,
                parameters,
            )
            scales.update(
                {
                    (group.identifier, column): float(group_scales[position])
                    for position, column in enumerate(columns)
                }
            )
        return effects, scales

    def _column(
        self,
        context: RunContext,
        selected: npt.NDArray[np.bool_],
        column_fit: ColumnPatternFit,
        group: PatternGroupFit | None,
        group_effects: dict[str, npt.NDArray[np.float64]],
        group_scales: dict[tuple[str, int], float],
    ) -> tuple[ColumnObjectiveValue, EvaluationIssue | None]:
        column_index = column_fit.column_index
        if group is None or not isinstance(column_fit.parameter, float):
            return _invalid_column(
                column_index,
                "multiplicative model is unavailable",
                EvaluationIssueCode.PATTERN_FIT_FAILED,
                self.pattern,
            )
        effects = group_effects[group.identifier]
        scale = group_scales[(group.identifier, column_index)]
        values = context.dataset.numeric_column(column_index) / scale
        return _joint_residual_column(
            context,
            selected,
            column_fit,
            pattern=self.pattern,
            label="multiplicative",
            values=values,
            prediction=effects * column_fit.parameter,
            group_identifier=group.identifier,
            extra_diagnostics={"numeric_scale": scale},
        )


@dataclass(frozen=True, slots=True)
class DefaultPatternContrastEvaluator:
    min_background_ratio: float = 0.10

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_background_ratio < 1.0:
            raise ValueError("min_background_ratio must be in [0, 1)")

    @property
    def cache_key(self) -> str:
        return f"default-contrast:{self.min_background_ratio.hex()}"

    def evaluate(
        self,
        context: RunContext,
        bicluster: Bicluster,
        fit: PatternFit,
        catalog: PatternCatalog,
    ) -> ObjectiveResult:
        background_rows = context.dataset.row_count - len(bicluster.row_indices)
        minimum_background = max(
            2,
            math.ceil(self.min_background_ratio * context.dataset.row_count),
        )
        if background_rows < minimum_background:
            issue = EvaluationIssue(
                code=EvaluationIssueCode.INSUFFICIENT_BACKGROUND,
                message=(
                    f"contrast requires at least {minimum_background} background rows; "
                    f"found {background_rows}"
                ),
            )
            return ObjectiveResult.model_construct(
                value=0.0,
                columns=tuple(
                    ColumnObjectiveValue.model_construct(
                        column_index=column,
                        value=0.0,
                        valid=False,
                        diagnostics=diagnostics(reason="insufficient background rows"),
                    )
                    for column in bicluster.column_indices
                ),
                issues=(issue,),
                diagnostics=diagnostics(
                    background_rows=background_rows,
                    minimum_background_rows=minimum_background,
                ),
            )

        contributions: dict[int, ColumnObjectiveValue] = {}
        issues: list[EvaluationIssue] = []
        by_pattern: dict[PatternKind, list[ColumnPatternFit]] = {}
        for column_fit in fit.columns:
            if column_fit.pattern is None:
                contribution, issue = _invalid_column(
                    column_fit.column_index,
                    "contrast cannot evaluate an unassigned column",
                    EvaluationIssueCode.PATTERN_UNASSIGNED,
                    None,
                )
                contributions[column_fit.column_index] = contribution
                issues.append(issue)
                continue
            by_pattern.setdefault(column_fit.pattern, []).append(column_fit)

        for pattern in sorted(by_pattern, key=lambda kind: kind.value):
            implementation = catalog.implementation(pattern)
            partial = implementation.contrast_strategy.evaluate(
                context,
                bicluster,
                fit,
                by_pattern[pattern],
            )
            contributions.update((column.column_index, column) for column in partial.columns)
            issues.extend(partial.issues)

        values = tuple(contributions[column] for column in bicluster.column_indices)
        aggregate = 0.0 if not values else clamp01(sum(item.value for item in values) / len(values))
        return ObjectiveResult.model_construct(
            value=aggregate,
            columns=values,
            issues=tuple(issues),
            diagnostics=diagnostics(
                background_rows=background_rows,
                minimum_background_rows=minimum_background,
            ),
        )


__all__ = [
    "AdditivePatternContrastStrategy",
    "ConstantPatternContrastStrategy",
    "DefaultPatternContrastEvaluator",
    "MultiplicativePatternContrastStrategy",
]
