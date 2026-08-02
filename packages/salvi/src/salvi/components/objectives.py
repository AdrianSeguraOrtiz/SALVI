"""Built-in scientific objectives with per-column explanations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from salvi.components.protocols import Objective
from salvi.domain.enums import ObjectiveDirection
from salvi.domain.models import Candidate, ColumnObjectiveValue, ObjectiveResult, PatternFit
from salvi.evaluation.workspace import EvaluationWorkspace
from salvi.patterns.contrast import DefaultPatternContrastEvaluator
from salvi.patterns.math import clamp01, diagnostics


def balanced_bicluster_size(
    candidate: Candidate,
    workspace: EvaluationWorkspace,
) -> tuple[float, float, float]:
    """Return balanced size and its row/column coverage without data scans."""

    dataset = workspace.context.dataset
    row_coverage = len(candidate.bicluster.row_indices) / dataset.row_count
    column_coverage = len(candidate.bicluster.column_indices) / dataset.column_count
    denominator = row_coverage + column_coverage
    value = 0.0 if denominator == 0.0 else 2.0 * row_coverage * column_coverage / denominator
    return value, row_coverage, column_coverage


def pattern_fit_internal_coherence(fit: PatternFit) -> float:
    if not fit.columns:
        return 1.0
    return clamp01(
        math.sqrt(sum(column.error * column.error for column in fit.columns) / len(fit.columns))
    )


class ContrastConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    min_background_ratio: Annotated[float, Field(ge=0.0, lt=1.0)] = 0.10


@dataclass(frozen=True, slots=True)
class InternalCoherence:
    normalization_bounds: ClassVar[tuple[float, float]] = (0.0, 1.0)
    component_name: str = "internal_coherence"
    direction: ObjectiveDirection = ObjectiveDirection.MINIMIZE
    provides: frozenset[str] = frozenset({"objective", "objective:internal-coherence"})
    requires: frozenset[str] = frozenset({"evaluation-support", "robust-numeric-data"})

    def evaluate(self, candidate: Candidate, workspace: EvaluationWorkspace) -> ObjectiveResult:
        fit = workspace.infer(candidate)
        columns = tuple(
            ColumnObjectiveValue.model_construct(
                column_index=column.column_index,
                value=column.error,
                valid=column.pattern is not None,
                diagnostics=diagnostics(
                    pattern=column.pattern.value if column.pattern is not None else None,
                    group_identifier=column.group_identifier,
                    source_support=column.source_support,
                    available_support=column.available_support,
                ),
            )
            for column in fit.columns
        )
        value = pattern_fit_internal_coherence(fit)
        return ObjectiveResult.model_construct(
            value=value,
            columns=columns,
            issues=fit.issues,
        )


@dataclass(frozen=True, slots=True)
class Contrast:
    normalization_bounds: ClassVar[tuple[float, float]] = (0.0, 1.0)
    min_background_ratio: float = 0.10
    component_name: str = "contrast"
    direction: ObjectiveDirection = ObjectiveDirection.MAXIMIZE
    provides: frozenset[str] = frozenset({"objective", "objective:contrast"})
    requires: frozenset[str] = frozenset({"evaluation-support", "robust-numeric-data"})
    _evaluator: DefaultPatternContrastEvaluator = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_evaluator",
            DefaultPatternContrastEvaluator(self.min_background_ratio),
        )

    def evaluate(self, candidate: Candidate, workspace: EvaluationWorkspace) -> ObjectiveResult:
        return workspace.contrast(candidate, self._evaluator)


@dataclass(frozen=True, slots=True)
class BalancedBiclusterSize:
    """Maximize the harmonic mean of selected-row and selected-column coverage."""

    normalization_bounds: ClassVar[tuple[float, float]] = (0.0, 1.0)
    component_name: str = "balanced_bicluster_size"
    direction: ObjectiveDirection = ObjectiveDirection.MAXIMIZE
    provides: frozenset[str] = frozenset({"objective"})
    requires: frozenset[str] = frozenset({"prepared-dataset"})

    def evaluate(self, candidate: Candidate, workspace: EvaluationWorkspace) -> ObjectiveResult:
        value, row_coverage, column_coverage = balanced_bicluster_size(candidate, workspace)
        column_count = len(candidate.bicluster.column_indices)
        reduced_column_coverage = (column_count - 1) / workspace.context.dataset.column_count
        reduced_denominator = row_coverage + reduced_column_coverage
        reduced_value = (
            0.0
            if reduced_denominator == 0.0
            else 2.0 * row_coverage * reduced_column_coverage / reduced_denominator
        )
        marginal_value = value - reduced_value
        columns = tuple(
            ColumnObjectiveValue.model_construct(
                column_index=column_index,
                value=marginal_value,
                diagnostics=diagnostics(
                    interpretation="objective loss if this equally weighted column is removed",
                    row_coverage=row_coverage,
                    column_coverage=column_coverage,
                ),
            )
            for column_index in candidate.bicluster.column_indices
        )
        return ObjectiveResult.model_construct(
            value=value,
            columns=columns,
            diagnostics=diagnostics(
                row_coverage=row_coverage,
                column_coverage=column_coverage,
            ),
        )


assert isinstance(InternalCoherence(), Objective)
assert isinstance(Contrast(), Objective)
assert isinstance(BalancedBiclusterSize(), Objective)


__all__ = [
    "BalancedBiclusterSize",
    "Contrast",
    "ContrastConfiguration",
    "InternalCoherence",
    "balanced_bicluster_size",
    "pattern_fit_internal_coherence",
]
