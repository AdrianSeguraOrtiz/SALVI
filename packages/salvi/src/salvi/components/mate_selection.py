"""Pair-selection policies used by crossover emitters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import numpy.random as npr
from pydantic import BaseModel, ConfigDict, Field

from salvi.components.parent_selection import evaluation_loss
from salvi.domain.models import Evaluation, Repertoire


class _EvidenceCompatibilityConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    parent_pool_size: Annotated[int, Field(ge=2)] = 32
    mate_pool_size: Annotated[int, Field(ge=1)] = 4
    minimum_row_jaccard: Annotated[float, Field(gt=0.0, le=1.0)] = 0.1
    minimum_column_jaccard: Annotated[float, Field(gt=0.0, le=1.0)] = 0.1


class CellFirstEvidenceCompatibleMateSelectionConfiguration(_EvidenceCompatibilityConfiguration):
    cell_neighborhood_radius: Annotated[int, Field(ge=0)] = 1


def _jaccard(first: tuple[int, ...], second: tuple[int, ...]) -> float:
    first_set = set(first)
    second_set = set(second)
    union = first_set | second_set
    return 0.0 if not union else len(first_set & second_set) / len(union)


@dataclass(frozen=True, slots=True)
class RepertoireRandomMateSelection:
    """Choose two distinct repertoire members uniformly."""

    component_name: str = "repertoire_random"
    provides: frozenset[str] = frozenset({"mate-selection"})
    requires: frozenset[str] = frozenset({"archive"})

    def select(
        self,
        repertoire: Repertoire,
        generator: npr.Generator,
    ) -> tuple[Evaluation, Evaluation] | None:
        if len(repertoire.evaluations) < 2:
            return None
        positions = generator.choice(len(repertoire.evaluations), 2, replace=False)
        return (
            repertoire.evaluations[int(positions[0])],
            repertoire.evaluations[int(positions[1])],
        )


@dataclass(frozen=True, slots=True)
class CellFirstEvidenceCompatibleMateSelection:
    """Choose a source cell uniformly before evaluating structural compatibility."""

    parent_pool_size: int = 32
    mate_pool_size: int = 4
    minimum_row_jaccard: float = 0.1
    minimum_column_jaccard: float = 0.1
    cell_neighborhood_radius: int = 1
    component_name: str = "cell_first_evidence_compatible"
    provides: frozenset[str] = frozenset({"mate-selection"})
    requires: frozenset[str] = frozenset({"archive", "objective"})

    def __post_init__(self) -> None:
        CellFirstEvidenceCompatibleMateSelectionConfiguration(
            parent_pool_size=self.parent_pool_size,
            mate_pool_size=self.mate_pool_size,
            minimum_row_jaccard=self.minimum_row_jaccard,
            minimum_column_jaccard=self.minimum_column_jaccard,
            cell_neighborhood_radius=self.cell_neighborhood_radius,
        )

    def select(
        self,
        repertoire: Repertoire,
        generator: npr.Generator,
    ) -> tuple[Evaluation, Evaluation] | None:
        grouped: dict[tuple[int, ...], list[Evaluation]] = {}
        for evaluation in repertoire.evaluations:
            coordinate = evaluation.archive_coordinate
            if coordinate is None:
                raise ValueError("cell-first mate selection requires archive coordinates")
            grouped.setdefault(coordinate, []).append(evaluation)
        local_ranks = {
            coordinate: {
                candidate.candidate.identifier: rank
                for rank, candidate in enumerate(
                    sorted(
                        candidates,
                        key=lambda item: (
                            evaluation_loss(item),
                            item.candidate.bicluster.signature,
                        ),
                    )
                )
            }
            for coordinate, candidates in grouped.items()
        }
        coordinates = list(sorted(grouped))
        generator.shuffle(coordinates)
        for coordinate in coordinates:
            members = tuple(grouped[coordinate])
            sample_size = min(self.parent_pool_size, len(members))
            positions = generator.choice(len(members), sample_size, replace=False)
            first = min(
                (members[int(position)] for position in positions),
                key=lambda item: (
                    evaluation_loss(item),
                    item.candidate.bicluster.signature,
                ),
            )
            mates: list[tuple[tuple[float, float, str], Evaluation]] = []
            first_bicluster = first.candidate.bicluster
            for mate_coordinate, candidates in grouped.items():
                if (
                    max(
                        (
                            abs(left - right)
                            for left, right in zip(
                                coordinate,
                                mate_coordinate,
                                strict=True,
                            )
                        ),
                        default=0,
                    )
                    > self.cell_neighborhood_radius
                ):
                    continue
                for second in candidates:
                    if second.candidate.identifier == first.candidate.identifier:
                        continue
                    second_bicluster = second.candidate.bicluster
                    row_similarity = _jaccard(
                        first_bicluster.row_indices,
                        second_bicluster.row_indices,
                    )
                    column_similarity = _jaccard(
                        first_bicluster.column_indices,
                        second_bicluster.column_indices,
                    )
                    if (
                        row_similarity < self.minimum_row_jaccard
                        or column_similarity < self.minimum_column_jaccard
                    ):
                        continue
                    mates.append(
                        (
                            (
                                -(row_similarity + column_similarity),
                                float(local_ranks[mate_coordinate][second.candidate.identifier]),
                                second_bicluster.signature,
                            ),
                            second,
                        )
                    )
            if not mates:
                continue
            pool = tuple(second for _key, second in sorted(mates, key=lambda item: item[0]))[
                : self.mate_pool_size
            ]
            second = pool[int(generator.integers(0, len(pool)))]
            return (first, second) if not bool(generator.integers(0, 2)) else (second, first)
        return None


__all__ = [
    "CellFirstEvidenceCompatibleMateSelection",
    "CellFirstEvidenceCompatibleMateSelectionConfiguration",
    "RepertoireRandomMateSelection",
]
