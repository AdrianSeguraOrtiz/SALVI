"""Deterministic emitter scheduling and archive-derived credit."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from salvi.components.protocols import Emitter
from salvi.domain.search import (
    EmitterAllocation,
    EmitterFeedback,
    SchedulerReport,
)
from salvi.exceptions import ComponentError


class AdaptiveCreditSchedulerConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    exploration_weight: Annotated[float, Field(ge=0.0)] = 0.5
    new_cell_reward: Annotated[float, Field(ge=0.0)] = 1.0
    insertion_reward: Annotated[float, Field(ge=0.0)] = 0.25


class CellBalancedAdaptiveCreditSchedulerConfiguration(AdaptiveCreditSchedulerConfiguration):
    underexplored_cell_weight: Annotated[float, Field(ge=0.0)] = 1.0


class FixedProportionSchedulerConfiguration(BaseModel):
    """Exact long-run allocation shares keyed by configured emitter name."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    shares: dict[str, Annotated[float, Field(gt=0.0, le=1.0)]]

    @model_validator(mode="after")
    def validate_shares(self) -> Self:
        if not self.shares:
            raise ValueError("fixed-proportion scheduler requires at least one emitter share")
        if any(not name.strip() for name in self.shares):
            raise ValueError("fixed-proportion emitter names must not be blank")
        if not math.isclose(sum(self.shares.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("fixed-proportion emitter shares must sum to 1")
        return self


@dataclass(slots=True)
class _EmitterStatistics:
    evaluations: int = 0
    accepted: int = 0
    created_cells: int = 0
    credit: float = 0.0
    allocation_count: int = 0

    def to_report(self, emitter_name: str) -> SchedulerReport:
        return SchedulerReport(
            emitter_name=emitter_name,
            evaluations=self.evaluations,
            accepted=self.accepted,
            created_cells=self.created_cells,
            credit=self.credit,
            allocation_count=self.allocation_count,
        )

    def to_state(self) -> dict[str, int | float]:
        return {
            "evaluations": self.evaluations,
            "accepted": self.accepted,
            "created_cells": self.created_cells,
            "credit": self.credit,
            "allocation_count": self.allocation_count,
        }

    @classmethod
    def from_state(cls, state: object) -> _EmitterStatistics:
        if not isinstance(state, dict):
            raise ComponentError("scheduler emitter state must be a mapping")
        try:
            statistics = cls(
                evaluations=int(state["evaluations"]),
                accepted=int(state["accepted"]),
                created_cells=int(state["created_cells"]),
                credit=float(state["credit"]),
                allocation_count=int(state["allocation_count"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ComponentError("scheduler emitter state is malformed") from error
        if (
            statistics.evaluations < 0
            or statistics.accepted < 0
            or statistics.accepted > statistics.evaluations
            or statistics.created_cells < 0
            or statistics.created_cells > statistics.accepted
            or statistics.credit < 0.0
            or not math.isfinite(statistics.credit)
            or statistics.allocation_count < 0
        ):
            raise ComponentError("scheduler emitter state contains inconsistent values")
        return statistics


def _validate_emitters(emitters: Sequence[Emitter]) -> tuple[str, ...]:
    if not emitters:
        raise ComponentError("scheduler requires at least one emitter")
    names = tuple(emitter.component_name for emitter in emitters)
    if len(set(names)) != len(names):
        raise ComponentError("configured emitter names must be unique")
    return names


def _aggregate_allocations(names: Sequence[str]) -> tuple[EmitterAllocation, ...]:
    counts: dict[str, int] = {}
    order: list[str] = []
    for name in names:
        if name not in counts:
            order.append(name)
            counts[name] = 0
        counts[name] += 1
    return tuple(EmitterAllocation(emitter_name=name, count=counts[name]) for name in order)


@dataclass(slots=True)
class FirstEmitterScheduler:
    """Allocate every request to the first configured emitter."""

    component_name: str = "first"
    provides: frozenset[str] = frozenset({"scheduler"})
    requires: frozenset[str] = frozenset({"emitter"})
    _statistics: dict[str, _EmitterStatistics] = field(default_factory=dict)

    def allocate(
        self,
        emitters: Sequence[Emitter],
        count: int,
    ) -> Sequence[EmitterAllocation]:
        names = _validate_emitters(emitters)
        if count < 1:
            raise ValueError("scheduler allocation count must be positive")
        statistics = self._statistics.setdefault(names[0], _EmitterStatistics())
        statistics.allocation_count += count
        return (EmitterAllocation(emitter_name=names[0], count=count),)

    def update(self, feedback: Sequence[EmitterFeedback]) -> None:
        for item in feedback:
            statistics = self._statistics.setdefault(item.emitter_name, _EmitterStatistics())
            statistics.evaluations += item.evaluated
            statistics.accepted += item.accepted
            statistics.created_cells += item.created_cells
            statistics.credit += float(item.accepted)

    def reports(self, emitters: Sequence[Emitter]) -> Sequence[SchedulerReport]:
        names = _validate_emitters(emitters)
        return tuple(
            self._statistics.get(name, _EmitterStatistics()).to_report(name) for name in names
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "statistics": {
                name: statistics.to_state() for name, statistics in sorted(self._statistics.items())
            }
        }

    def restore(self, state: dict[str, Any], emitters: Sequence[Emitter]) -> None:
        names = _validate_emitters(emitters)
        raw = state.get("statistics", {})
        if not isinstance(raw, dict) or set(raw) - set(names):
            raise ComponentError("scheduler checkpoint references unknown emitters")
        self._statistics = {
            name: _EmitterStatistics.from_state(values) for name, values in raw.items()
        }


@dataclass(slots=True)
class FixedProportionScheduler:
    """Allocate emitters by deterministic cumulative proportions."""

    shares: dict[str, float]
    component_name: str = "fixed_proportion"
    provides: frozenset[str] = frozenset({"scheduler"})
    requires: frozenset[str] = frozenset({"emitter"})
    _statistics: dict[str, _EmitterStatistics] = field(default_factory=dict)

    def __post_init__(self) -> None:
        configuration = FixedProportionSchedulerConfiguration(shares=self.shares)
        self.shares = dict(configuration.shares)

    def _validate_configuration(self, emitters: Sequence[Emitter]) -> tuple[str, ...]:
        names = _validate_emitters(emitters)
        configured = set(self.shares)
        actual = set(names)
        if configured != actual:
            missing = sorted(actual - configured)
            unknown = sorted(configured - actual)
            details = []
            if missing:
                details.append(f"missing shares for {missing}")
            if unknown:
                details.append(f"unknown emitters {unknown}")
            raise ComponentError(
                "fixed-proportion scheduler must define every configured emitter: "
                + "; ".join(details)
            )
        return names

    def allocate(
        self,
        emitters: Sequence[Emitter],
        count: int,
    ) -> Sequence[EmitterAllocation]:
        names = self._validate_configuration(emitters)
        if count < 1:
            raise ValueError("scheduler allocation count must be positive")
        for name in names:
            self._statistics.setdefault(name, _EmitterStatistics())
        selected: list[str] = []
        order = {name: index for index, name in enumerate(names)}
        target_total = sum(item.allocation_count for item in self._statistics.values())
        for _ in range(count):
            target_total += 1
            name = max(
                names,
                key=lambda candidate: (
                    target_total * self.shares[candidate]
                    - self._statistics[candidate].allocation_count,
                    -order[candidate],
                ),
            )
            self._statistics[name].allocation_count += 1
            selected.append(name)
        return _aggregate_allocations(selected)

    def update(self, feedback: Sequence[EmitterFeedback]) -> None:
        for item in feedback:
            try:
                statistics = self._statistics[item.emitter_name]
            except KeyError as error:
                raise ComponentError(
                    f"feedback references unallocated emitter {item.emitter_name!r}"
                ) from error
            statistics.evaluations += item.evaluated
            statistics.accepted += item.accepted
            statistics.created_cells += item.created_cells
            statistics.credit += float(item.accepted)

    def reports(self, emitters: Sequence[Emitter]) -> Sequence[SchedulerReport]:
        names = self._validate_configuration(emitters)
        return tuple(
            self._statistics.get(name, _EmitterStatistics()).to_report(name) for name in names
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "statistics": {
                name: statistics.to_state() for name, statistics in sorted(self._statistics.items())
            }
        }

    def restore(self, state: dict[str, Any], emitters: Sequence[Emitter]) -> None:
        names = self._validate_configuration(emitters)
        raw = state.get("statistics", {})
        if not isinstance(raw, dict) or set(raw) - set(names):
            raise ComponentError("scheduler checkpoint references unknown emitters")
        self._statistics = {
            name: _EmitterStatistics.from_state(values) for name, values in raw.items()
        }


@dataclass(slots=True)
class AdaptiveCreditScheduler:
    """Use deterministic UCB allocation with credit from archive insertions."""

    exploration_weight: float = 0.5
    new_cell_reward: float = 1.0
    insertion_reward: float = 0.25
    component_name: str = "adaptive_credit"
    provides: frozenset[str] = frozenset({"scheduler"})
    requires: frozenset[str] = frozenset({"emitter"})
    _statistics: dict[str, _EmitterStatistics] = field(default_factory=dict)

    def __post_init__(self) -> None:
        configuration = AdaptiveCreditSchedulerConfiguration(
            exploration_weight=self.exploration_weight,
            new_cell_reward=self.new_cell_reward,
            insertion_reward=self.insertion_reward,
        )
        self.exploration_weight = configuration.exploration_weight
        self.new_cell_reward = configuration.new_cell_reward
        self.insertion_reward = configuration.insertion_reward

    def allocate(
        self,
        emitters: Sequence[Emitter],
        count: int,
    ) -> Sequence[EmitterAllocation]:
        names = _validate_emitters(emitters)
        if count < 1:
            raise ValueError("scheduler allocation count must be positive")
        for name in names:
            self._statistics.setdefault(name, _EmitterStatistics())
        pending = {name: 0 for name in names}
        selected: list[str] = []
        for _ in range(count):
            name = max(
                names,
                key=lambda candidate: (
                    self._score(candidate, pending[candidate]),
                    -names.index(candidate),
                ),
            )
            pending[name] += 1
            selected.append(name)
        for name, allocation_count in pending.items():
            self._statistics[name].allocation_count += allocation_count
        return _aggregate_allocations(selected)

    def _score(self, name: str, pending: int) -> float:
        statistics = self._statistics[name]
        effective_evaluations = statistics.evaluations + pending
        if effective_evaluations == 0:
            return math.inf
        mean_credit = statistics.credit / effective_evaluations
        total = sum(item.evaluations for item in self._statistics.values()) + sum(
            item.allocation_count - item.evaluations for item in self._statistics.values()
        )
        exploration = self.exploration_weight * math.sqrt(
            math.log(max(2, total + 1)) / effective_evaluations
        )
        return mean_credit + exploration

    def update(self, feedback: Sequence[EmitterFeedback]) -> None:
        for item in feedback:
            try:
                statistics = self._statistics[item.emitter_name]
            except KeyError as error:
                raise ComponentError(
                    f"feedback references unallocated emitter {item.emitter_name!r}"
                ) from error
            retained_insertions = item.accepted - item.created_cells
            credit = (
                self.new_cell_reward * item.created_cells
                + self.insertion_reward * retained_insertions
            )
            statistics.evaluations += item.evaluated
            statistics.accepted += item.accepted
            statistics.created_cells += item.created_cells
            statistics.credit += credit

    def reports(self, emitters: Sequence[Emitter]) -> Sequence[SchedulerReport]:
        names = _validate_emitters(emitters)
        return tuple(
            self._statistics.get(name, _EmitterStatistics()).to_report(name) for name in names
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "statistics": {
                name: statistics.to_state() for name, statistics in sorted(self._statistics.items())
            }
        }

    def restore(self, state: dict[str, Any], emitters: Sequence[Emitter]) -> None:
        names = _validate_emitters(emitters)
        raw = state.get("statistics", {})
        if not isinstance(raw, dict) or set(raw) - set(names):
            raise ComponentError("scheduler checkpoint references unknown emitters")
        self._statistics = {
            name: _EmitterStatistics.from_state(values) for name, values in raw.items()
        }


@dataclass(slots=True)
class CellBalancedAdaptiveCreditScheduler(AdaptiveCreditScheduler):
    """Reward emitters for useful work in cells receiving less search effort."""

    underexplored_cell_weight: float = 1.0
    component_name: str = "cell_balanced_adaptive_credit"
    _cell_evaluations: dict[tuple[int, ...] | None, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        configuration = CellBalancedAdaptiveCreditSchedulerConfiguration(
            exploration_weight=self.exploration_weight,
            new_cell_reward=self.new_cell_reward,
            insertion_reward=self.insertion_reward,
            underexplored_cell_weight=self.underexplored_cell_weight,
        )
        self.exploration_weight = configuration.exploration_weight
        self.new_cell_reward = configuration.new_cell_reward
        self.insertion_reward = configuration.insertion_reward
        self.underexplored_cell_weight = configuration.underexplored_cell_weight

    def update(self, feedback: Sequence[EmitterFeedback]) -> None:
        for item in feedback:
            try:
                statistics = self._statistics[item.emitter_name]
            except KeyError as error:
                raise ComponentError(
                    f"feedback references unallocated emitter {item.emitter_name!r}"
                ) from error
            credit = 0.0
            if item.cells:
                for cell in item.cells:
                    coordinate = None if cell.coordinate is None else cell.coordinate.indices
                    prior = self._cell_evaluations.get(coordinate, 0)
                    coverage_multiplier = 1.0 + self.underexplored_cell_weight / math.sqrt(
                        prior + 1.0
                    )
                    retained = cell.accepted - cell.created_cells
                    credit += coverage_multiplier * (
                        self.new_cell_reward * cell.created_cells + self.insertion_reward * retained
                    )
                    self._cell_evaluations[coordinate] = prior + cell.evaluated
            else:
                retained = item.accepted - item.created_cells
                credit = (
                    self.new_cell_reward * item.created_cells + self.insertion_reward * retained
                )
            statistics.evaluations += item.evaluated
            statistics.accepted += item.accepted
            statistics.created_cells += item.created_cells
            statistics.credit += credit

    def snapshot(self) -> dict[str, Any]:
        state = super(CellBalancedAdaptiveCreditScheduler, self).snapshot()
        state["cell_evaluations"] = {
            ("__unmapped__" if coordinate is None else ",".join(map(str, coordinate))): count
            for coordinate, count in sorted(
                self._cell_evaluations.items(),
                key=lambda item: () if item[0] is None else item[0],
            )
        }
        return state

    def restore(self, state: dict[str, Any], emitters: Sequence[Emitter]) -> None:
        super(CellBalancedAdaptiveCreditScheduler, self).restore(state, emitters)
        raw = state.get("cell_evaluations", {})
        if not isinstance(raw, dict):
            raise ComponentError("scheduler cell state must be a mapping")
        restored: dict[tuple[int, ...] | None, int] = {}
        try:
            for label, raw_count in raw.items():
                if not isinstance(label, str):
                    raise ValueError
                coordinate = (
                    None
                    if label == "__unmapped__"
                    else tuple(int(value) for value in label.split(","))
                )
                count = int(raw_count)
                if count < 0 or (coordinate is not None and not coordinate):
                    raise ValueError
                restored[coordinate] = count
        except (TypeError, ValueError) as error:
            raise ComponentError("scheduler cell state is malformed") from error
        self._cell_evaluations = restored


__all__ = [
    "AdaptiveCreditScheduler",
    "AdaptiveCreditSchedulerConfiguration",
    "CellBalancedAdaptiveCreditScheduler",
    "CellBalancedAdaptiveCreditSchedulerConfiguration",
    "FirstEmitterScheduler",
    "FixedProportionScheduler",
    "FixedProportionSchedulerConfiguration",
]
