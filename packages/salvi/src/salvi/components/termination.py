"""Search termination components."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import Field

from salvi.components.configuration import EmptyConfiguration
from salvi.domain.search import TerminationProgress


class TerminationConfiguration(EmptyConfiguration):
    max_evaluations: int = Field(default=1, ge=1)


@dataclass(frozen=True, slots=True)
class EvaluationBudget:
    max_evaluations: int = 1
    component_name: str = "evaluation_budget"
    provides: frozenset[str] = frozenset({"termination"})
    requires: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.max_evaluations < 1:
            raise ValueError("max_evaluations must be positive")

    def should_stop(self, evaluations: int) -> bool:
        return evaluations >= self.max_evaluations

    def remaining(self, evaluations: int) -> int | None:
        return max(0, self.max_evaluations - evaluations)

    def progress(self, evaluations: int) -> TerminationProgress:
        return TerminationProgress(
            current=float(evaluations),
            limit=float(self.max_evaluations),
            unit="evaluations",
        )


__all__ = ["EvaluationBudget", "TerminationConfiguration"]
