"""Bounded, pattern-specific discovery of coherent numeric column neighborhoods."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from salvi.application.context import RunContext
from salvi.domain.enums import PatternKind
from salvi.domain.models import Bicluster
from salvi.patterns.joint_models import robust_column_scales
from salvi.patterns.math import NUMERIC_TOLERANCE, nanmedian_2d


def _profile_distances(
    context: RunContext,
    profiles: npt.NDArray[np.float64],
    scales: npt.NDArray[np.float64],
    anchor: int,
    *,
    sign_invariant: bool,
) -> npt.NDArray[np.float64]:
    usable = np.isfinite(profiles) & np.isfinite(profiles[:, anchor])[:, np.newaxis]
    support = np.count_nonzero(usable, axis=0)
    required = context.evaluation_support_policy.required_observations(profiles.shape[0])
    residuals = np.where(
        usable,
        np.minimum(
            1.0,
            np.abs(profiles - profiles[:, anchor, np.newaxis])
            / np.maximum(scales, scales[anchor])[np.newaxis, :],
        ),
        0.0,
    )
    distances = np.full(profiles.shape[1], np.inf, dtype=np.float64)
    np.divide(
        np.sum(residuals, axis=0),
        support,
        out=distances,
        where=support >= required,
    )
    if sign_invariant:
        reversed_residuals = np.where(
            usable,
            np.minimum(
                1.0,
                np.abs(profiles + profiles[:, anchor, np.newaxis])
                / np.maximum(scales, scales[anchor])[np.newaxis, :],
            ),
            0.0,
        )
        reversed_distances = np.full(profiles.shape[1], np.inf, dtype=np.float64)
        np.divide(
            np.sum(reversed_residuals, axis=0),
            support,
            out=reversed_distances,
            where=support >= required,
        )
        distances = np.minimum(distances, reversed_distances)
    distances[anchor] = 0.0
    return distances


def _neighborhoods(
    context: RunContext,
    columns: tuple[int, ...],
    profiles: npt.NDArray[np.float64],
    scales: npt.NDArray[np.float64],
    *,
    minimum_columns: int,
    maximum_anchors: int,
    sign_invariant: bool,
) -> tuple[tuple[int, ...], ...]:
    if len(columns) < minimum_columns:
        return ()
    support = np.count_nonzero(np.isfinite(profiles), axis=0)
    first = min(range(len(columns)), key=lambda index: (-int(support[index]), columns[index]))
    anchors = [first]
    distances = {
        first: _profile_distances(
            context,
            profiles,
            scales,
            first,
            sign_invariant=sign_invariant,
        )
    }
    nearest = distances[first].copy()
    while len(anchors) < min(maximum_anchors, len(columns)):
        candidates = tuple(
            index
            for index in range(len(columns))
            if index not in anchors and np.isfinite(nearest[index])
        )
        if not candidates:
            break
        next_anchor = max(candidates, key=lambda index: (float(nearest[index]), -columns[index]))
        if nearest[next_anchor] <= NUMERIC_TOLERANCE:
            break
        anchors.append(next_anchor)
        distances[next_anchor] = _profile_distances(
            context,
            profiles,
            scales,
            next_anchor,
            sign_invariant=sign_invariant,
        )
        nearest = np.minimum(nearest, distances[next_anchor])

    proposals: set[tuple[int, ...]] = set()
    for anchor in anchors:
        local = distances[anchor]
        ordered = tuple(
            sorted(
                (index for index in range(len(columns)) if np.isfinite(local[index])),
                key=lambda index: (float(local[index]), columns[index]),
            )
        )
        if len(ordered) < minimum_columns:
            continue
        cut = len(ordered)
        if len(ordered) > minimum_columns:
            gaps = tuple(
                (
                    float(local[ordered[position]] - local[ordered[position - 1]]),
                    -position,
                    position,
                )
                for position in range(minimum_columns, len(ordered))
            )
            largest_gap, _, candidate_cut = max(gaps)
            if largest_gap > NUMERIC_TOLERANCE:
                cut = candidate_cut
        proposals.add(tuple(sorted(columns[index] for index in ordered[:cut])))
        proposals.add(tuple(sorted(columns[index] for index in ordered[:minimum_columns])))
    return tuple(sorted(proposals, key=lambda group: (len(group), group)))


@dataclass(frozen=True, slots=True)
class AdditiveNeighborhoodGenerator:
    """Group columns whose raw, locally centered profiles share additive shifts."""

    maximum_anchors: int = 8
    pattern: PatternKind = PatternKind.ADDITIVE

    def propose(
        self,
        context: RunContext,
        bicluster: Bicluster,
        column_indices: Sequence[int],
        *,
        minimum_columns: int,
    ) -> tuple[tuple[int, ...], ...]:
        columns = tuple(sorted(column_indices))
        rows = np.asarray(bicluster.row_indices, dtype=np.int64)
        positions = np.asarray(
            [context.dataset.numeric_positions[column] for column in columns],
            dtype=np.int64,
        )
        matrix = context.dataset.numeric_matrix()[np.ix_(rows, positions)]
        profiles = matrix - nanmedian_2d(matrix, axis=0)[np.newaxis, :]
        return _neighborhoods(
            context,
            columns,
            profiles,
            robust_column_scales(context, columns),
            minimum_columns=minimum_columns,
            maximum_anchors=self.maximum_anchors,
            sign_invariant=False,
        )


@dataclass(frozen=True, slots=True)
class MultiplicativeNeighborhoodGenerator:
    """Group columns with the same robust-scaled proportional row profile."""

    maximum_anchors: int = 8
    pattern: PatternKind = PatternKind.MULTIPLICATIVE

    def propose(
        self,
        context: RunContext,
        bicluster: Bicluster,
        column_indices: Sequence[int],
        *,
        minimum_columns: int,
    ) -> tuple[tuple[int, ...], ...]:
        columns = tuple(sorted(column_indices))
        rows = np.asarray(bicluster.row_indices, dtype=np.int64)
        positions = np.asarray(
            [context.dataset.numeric_positions[column] for column in columns],
            dtype=np.int64,
        )
        column_scales = robust_column_scales(context, columns)
        profiles = context.dataset.numeric_matrix()[np.ix_(rows, positions)] / column_scales
        return _neighborhoods(
            context,
            columns,
            profiles,
            np.ones(len(columns), dtype=np.float64),
            minimum_columns=minimum_columns,
            maximum_anchors=self.maximum_anchors,
            sign_invariant=True,
        )


__all__ = ["AdditiveNeighborhoodGenerator", "MultiplicativeNeighborhoodGenerator"]
