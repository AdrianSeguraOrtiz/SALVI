"""Reusable parent-selection policies for repertoire-based emitters."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.random as npr

from salvi.domain.models import Evaluation, Repertoire


def evaluation_loss(evaluation: Evaluation) -> float:
    """Return a direction-aware scalar used only for local parent preference."""

    if not evaluation.objectives:
        return math.inf
    losses = tuple(
        objective.value if objective.direction.value == "MINIMIZE" else 1.0 - objective.value
        for objective in evaluation.objectives
    )
    return float(np.mean(losses))


def _choose(
    candidates: tuple[Evaluation, ...],
    generator: npr.Generator,
    *,
    pool_size: int,
    guided: bool,
) -> Evaluation | None:
    if not candidates:
        return None
    sample_size = min(pool_size, len(candidates))
    positions = generator.choice(len(candidates), sample_size, replace=False)
    sampled = tuple(candidates[int(position)] for position in positions)
    if guided:
        return min(
            sampled,
            key=lambda evaluation: (
                evaluation_loss(evaluation),
                evaluation.candidate.bicluster.signature,
                evaluation.candidate.identifier,
            ),
        )
    return sampled[int(generator.integers(0, len(sampled)))]


@dataclass(frozen=True, slots=True)
class RepertoireUniformParentSelection:
    """Sample parents uniformly from all eligible repertoire members."""

    component_name: str = "repertoire_uniform"
    provides: frozenset[str] = frozenset({"parent-selection"})
    requires: frozenset[str] = frozenset({"archive", "objective"})

    def select(
        self,
        repertoire: Repertoire,
        generator: npr.Generator,
        *,
        pool_size: int,
        eligible: Callable[[Evaluation], bool],
        guided: bool,
    ) -> Evaluation | None:
        candidates = tuple(
            evaluation for evaluation in repertoire.evaluations if eligible(evaluation)
        )
        return _choose(candidates, generator, pool_size=pool_size, guided=guided)


@dataclass(frozen=True, slots=True)
class CellUniformQualityParentSelection:
    """Sample an occupied cell uniformly, then choose a parent inside it."""

    component_name: str = "cell_uniform_quality"
    provides: frozenset[str] = frozenset({"parent-selection"})
    requires: frozenset[str] = frozenset({"archive", "objective"})

    def select(
        self,
        repertoire: Repertoire,
        generator: npr.Generator,
        *,
        pool_size: int,
        eligible: Callable[[Evaluation], bool],
        guided: bool,
    ) -> Evaluation | None:
        cells: dict[tuple[int, ...], list[Evaluation]] = defaultdict(list)
        for evaluation in repertoire.evaluations:
            if eligible(evaluation):
                coordinate = evaluation.archive_coordinate
                if coordinate is None:
                    raise ValueError("cell-uniform parent selection requires archive coordinates")
                cells[coordinate].append(evaluation)
        if not cells:
            return None
        coordinates = tuple(sorted(cells))
        coordinate = coordinates[int(generator.integers(0, len(coordinates)))]
        candidates = tuple(
            sorted(
                cells[coordinate],
                key=lambda evaluation: (
                    evaluation.candidate.bicluster.signature,
                    evaluation.candidate.identifier,
                ),
            )
        )
        return _choose(candidates, generator, pool_size=pool_size, guided=guided)


__all__ = [
    "CellUniformQualityParentSelection",
    "RepertoireUniformParentSelection",
    "evaluation_loss",
]
