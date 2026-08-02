"""Built-in behavioral descriptors."""

from __future__ import annotations

from dataclasses import dataclass

from salvi.application.context import RunContext
from salvi.domain.enums import BinningStrategy, DescriptorValueKind
from salvi.domain.models import Candidate
from salvi.domain.search import DescriptorDomain
from salvi.evaluation.workspace import EvaluationWorkspace

_CARDINALITY_BINNINGS = (
    BinningStrategy.LINEAR,
    BinningStrategy.GEOMETRIC,
    BinningStrategy.EXACT,
    BinningStrategy.CUSTOM,
)


@dataclass(frozen=True, slots=True)
class RowCardinality:
    component_name: str = "row_cardinality"
    provides: frozenset[str] = frozenset({"descriptor", "descriptor:row-cardinality"})
    requires: frozenset[str] = frozenset({"candidate-validity", "prepared-dataset"})

    def domain(self, context: RunContext) -> DescriptorDomain:
        bounds = context.candidate_validity_policy.bounds(context.dataset)
        return DescriptorDomain(
            value_kind=DescriptorValueKind.INTEGER,
            minimum=float(bounds.min_rows),
            maximum=float(bounds.max_rows),
            supported_binnings=_CARDINALITY_BINNINGS,
            recommended_binning=BinningStrategy.GEOMETRIC,
        )

    def describe(self, candidate: Candidate, workspace: EvaluationWorkspace) -> float:
        del workspace
        return float(len(candidate.bicluster.row_indices))


@dataclass(frozen=True, slots=True)
class ColumnCardinality:
    component_name: str = "column_cardinality"
    provides: frozenset[str] = frozenset({"descriptor", "descriptor:column-cardinality"})
    requires: frozenset[str] = frozenset({"candidate-validity", "prepared-dataset"})

    def domain(self, context: RunContext) -> DescriptorDomain:
        bounds = context.candidate_validity_policy.bounds(context.dataset)
        return DescriptorDomain(
            value_kind=DescriptorValueKind.INTEGER,
            minimum=float(bounds.min_columns),
            maximum=float(bounds.max_columns),
            supported_binnings=_CARDINALITY_BINNINGS,
            recommended_binning=BinningStrategy.GEOMETRIC,
        )

    def describe(self, candidate: Candidate, workspace: EvaluationWorkspace) -> float:
        del workspace
        return float(len(candidate.bicluster.column_indices))


__all__ = ["ColumnCardinality", "RowCardinality"]
