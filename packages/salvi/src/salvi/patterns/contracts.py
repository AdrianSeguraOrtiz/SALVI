"""Extensible contracts for fitting and assigning bicluster patterns."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from salvi.domain.enums import ColumnKind, PatternKind, PatternScope
from salvi.domain.models import (
    Bicluster,
    ColumnPatternFit,
    EvaluationIssue,
    ObjectiveResult,
    PatternCandidateFit,
    PatternFit,
    PatternGroupFit,
)

if TYPE_CHECKING:
    from salvi.application.context import RunContext
    from salvi.patterns.catalog import PatternCatalog


@dataclass(frozen=True, slots=True)
class PatternDefinition:
    """Scientific capabilities of one pattern implementation."""

    kind: PatternKind
    scope: PatternScope
    supported_column_kinds: frozenset[ColumnKind]
    minimum_columns: int = 1
    maximum_groups: int | None = None
    reference_model: bool = False

    def __post_init__(self) -> None:
        if not self.supported_column_kinds:
            raise ValueError("patterns must support at least one column kind")
        if self.minimum_columns < 1:
            raise ValueError("pattern minimum_columns must be positive")
        if self.scope is PatternScope.COLUMN and self.minimum_columns != 1:
            raise ValueError("column-scoped patterns require exactly one column")
        if self.maximum_groups is not None and self.maximum_groups < 1:
            raise ValueError("pattern maximum_groups must be positive when specified")
        if self.reference_model and self.scope is not PatternScope.COLUMN:
            raise ValueError("the reference model must be column-scoped")

    def supports(self, kind: ColumnKind) -> bool:
        return kind in self.supported_column_kinds


@dataclass(frozen=True, slots=True)
class GroupPatternProposal:
    """One joint fit proposed by a subset-scoped pattern implementation."""

    pattern: PatternKind
    columns: tuple[tuple[int, PatternCandidateFit], ...]
    group: PatternGroupFit | None
    rejected_column: int | None = None
    rejection_issue: EvaluationIssue | None = None

    def __post_init__(self) -> None:
        indices = tuple(column for column, _ in self.columns)
        if tuple(sorted(set(indices))) != indices:
            raise ValueError("group proposal columns must be sorted and unique")
        if any(fit.pattern is not self.pattern for _, fit in self.columns):
            raise ValueError("group proposal column fits must use the proposed pattern")
        if self.group is not None:
            if self.rejected_column is not None or self.rejection_issue is not None:
                raise ValueError("a valid group proposal cannot reject a column")
            if self.group.pattern is not self.pattern:
                raise ValueError("group proposal metadata must use the proposed pattern")
            if self.group.column_indices != indices:
                raise ValueError("group proposal metadata must align with its column fits")
        elif (self.rejected_column is None) != (self.rejection_issue is None):
            raise ValueError("rejected group columns require a matching typed issue")
        elif self.rejection_issue is not None:
            if self.rejection_issue.column_index != self.rejected_column:
                raise ValueError("group rejection issue must identify the rejected column")
            if self.rejection_issue.pattern is not self.pattern:
                raise ValueError("group rejection issue must identify the proposed pattern")

    @property
    def valid(self) -> bool:
        return self.group is not None and all(fit.valid for _, fit in self.columns)

    def fit_for(self, column_index: int) -> PatternCandidateFit | None:
        for index, fit in self.columns:
            if index == column_index:
                return fit
        return None


class ColumnPatternFitter(Protocol):
    @property
    def definition(self) -> PatternDefinition: ...

    def fit_column(
        self,
        context: RunContext,
        bicluster: Bicluster,
        column_index: int,
    ) -> PatternCandidateFit: ...


class GroupPatternFitter(Protocol):
    @property
    def definition(self) -> PatternDefinition: ...

    def fit_group(
        self,
        context: RunContext,
        bicluster: Bicluster,
        column_indices: Sequence[int],
    ) -> GroupPatternProposal: ...


class PatternContrastStrategy(Protocol):
    @property
    def pattern(self) -> PatternKind: ...

    def evaluate(
        self,
        context: RunContext,
        bicluster: Bicluster,
        fit: PatternFit,
        columns: Sequence[ColumnPatternFit],
    ) -> ObjectiveResult: ...


@dataclass(frozen=True, slots=True)
class PatternImplementation:
    definition: PatternDefinition
    contrast_strategy: PatternContrastStrategy
    column_fitter: ColumnPatternFitter | None = None
    group_fitter: GroupPatternFitter | None = None

    def __post_init__(self) -> None:
        if self.contrast_strategy.pattern is not self.definition.kind:
            raise ValueError("contrast strategy must match its pattern registration")
        if self.definition.scope is PatternScope.COLUMN:
            if self.column_fitter is None or self.group_fitter is not None:
                raise ValueError("column patterns require one column fitter")
            if self.column_fitter.definition != self.definition:
                raise ValueError("column fitter definition must match its registration")
        elif self.group_fitter is None or self.column_fitter is not None:
            raise ValueError("subset patterns require one group fitter")
        elif self.group_fitter.definition != self.definition:
            raise ValueError("group fitter definition must match its registration")


class MixedPatternAssignmentStrategy(Protocol):
    def assign(
        self,
        context: RunContext,
        bicluster: Bicluster,
        implementations: Sequence[PatternImplementation],
    ) -> PatternFit: ...


class PatternInferenceEngine(Protocol):
    def infer(self, context: RunContext, bicluster: Bicluster) -> PatternFit: ...


class PatternContrastEvaluator(Protocol):
    @property
    def cache_key(self) -> str: ...

    def evaluate(
        self,
        context: RunContext,
        bicluster: Bicluster,
        fit: PatternFit,
        catalog: PatternCatalog,
    ) -> ObjectiveResult: ...


__all__ = [
    "ColumnPatternFitter",
    "GroupPatternFitter",
    "GroupPatternProposal",
    "MixedPatternAssignmentStrategy",
    "PatternContrastEvaluator",
    "PatternContrastStrategy",
    "PatternDefinition",
    "PatternImplementation",
    "PatternInferenceEngine",
]
