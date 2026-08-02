"""Built-in scientific and structural search constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from salvi.components.objectives import (
    balanced_bicluster_size,
    pattern_fit_internal_coherence,
)
from salvi.components.protocols import Constraint
from salvi.domain.models import Candidate, ConstraintResult
from salvi.evaluation.workspace import EvaluationWorkspace
from salvi.patterns.math import diagnostics


class BalancedBiclusterSizeRangeConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    maximum: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.minimum > self.maximum:
            raise ValueError("minimum cannot exceed maximum")
        return self


class MaximumInternalCoherenceConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    maximum_error: Annotated[float, Field(ge=0.0, lt=1.0)] = 0.10


@dataclass(frozen=True, slots=True)
class BalancedBiclusterSizeRange:
    """Require balanced bicluster size to lie inside one inclusive interval."""

    minimum: float = 0.0
    maximum: float = 1.0
    component_name: str = "balanced_bicluster_size_range"
    provides: frozenset[str] = frozenset({"constraint"})
    requires: frozenset[str] = frozenset({"prepared-dataset"})

    def __post_init__(self) -> None:
        BalancedBiclusterSizeRangeConfiguration(
            minimum=self.minimum,
            maximum=self.maximum,
        )

    def evaluate(self, candidate: Candidate, workspace: EvaluationWorkspace) -> ConstraintResult:
        value, row_coverage, column_coverage = balanced_bicluster_size(candidate, workspace)
        return ConstraintResult(
            value=max(self.minimum - value, value - self.maximum),
            diagnostics=diagnostics(
                balanced_size=value,
                minimum=self.minimum,
                maximum=self.maximum,
                row_coverage=row_coverage,
                column_coverage=column_coverage,
            ),
        )


@dataclass(frozen=True, slots=True)
class MaximumInternalCoherence:
    """Require the inferred RMS pattern error not to exceed a configured limit."""

    maximum_error: float = 0.10
    component_name: str = "maximum_internal_coherence"
    provides: frozenset[str] = frozenset({"constraint"})
    requires: frozenset[str] = frozenset({"evaluation-support", "robust-numeric-data"})

    def __post_init__(self) -> None:
        MaximumInternalCoherenceConfiguration(maximum_error=self.maximum_error)

    def evaluate(self, candidate: Candidate, workspace: EvaluationWorkspace) -> ConstraintResult:
        error = pattern_fit_internal_coherence(workspace.infer(candidate))
        return ConstraintResult(
            value=error - self.maximum_error,
            diagnostics=diagnostics(
                internal_coherence_error=error,
                maximum_error=self.maximum_error,
            ),
        )


assert isinstance(BalancedBiclusterSizeRange(), Constraint)
assert isinstance(MaximumInternalCoherence(), Constraint)


__all__ = [
    "BalancedBiclusterSizeRange",
    "BalancedBiclusterSizeRangeConfiguration",
    "MaximumInternalCoherence",
    "MaximumInternalCoherenceConfiguration",
]
