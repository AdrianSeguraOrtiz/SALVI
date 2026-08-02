"""REL, REC, BE, matching, coverage, and uncertainty calculations."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Annotated, Literal, Self

import numpy as np
from pydantic import Field, model_validator
from scipy.optimize import linear_sum_assignment

from salvi_experiments.configuration import FrozenExperimentModel, UncertaintyConfiguration


class BiclusterMembership(FrozenExperimentModel):
    identifier: str = Field(min_length=1)
    row_indices: tuple[int, ...]
    column_indices: tuple[int, ...]

    @model_validator(mode="after")
    def validate_indices(self) -> Self:
        for values, label in (
            (self.row_indices, "row_indices"),
            (self.column_indices, "column_indices"),
        ):
            if not values:
                raise ValueError(f"{label} must not be empty")
            if tuple(sorted(set(values))) != values or values[0] < 0:
                raise ValueError(f"{label} must be sorted, unique, and non-negative")
        return self


class ConfidenceInterval(FrozenExperimentModel):
    estimate: Annotated[float, Field(ge=0.0, le=1.0)]
    lower: Annotated[float, Field(ge=0.0, le=1.0)]
    upper: Annotated[float, Field(ge=0.0, le=1.0)]
    confidence_level: Annotated[float, Field(gt=0.0, lt=1.0)]
    samples: Annotated[int, Field(ge=0)]


class MatchRecord(FrozenExperimentModel):
    perspective: Literal["DETECTED", "GROUND_TRUTH"]
    bicluster_id: str = Field(min_length=1)
    best_match_id: str | None
    row_jaccard: Annotated[float, Field(ge=0.0, le=1.0)]
    column_jaccard: Annotated[float, Field(ge=0.0, le=1.0)]
    structural_similarity: Annotated[float, Field(ge=0.0, le=1.0)]
    cell_jaccard: Annotated[float, Field(ge=0.0, le=1.0)]


class AccuracyResult(FrozenExperimentModel):
    relevance: Annotated[float, Field(ge=0.0, le=1.0)]
    recovery: Annotated[float, Field(ge=0.0, le=1.0)]
    biclustering_error: Annotated[float, Field(ge=0.0, le=1.0)]
    relevance_interval: ConfidenceInterval
    recovery_interval: ConfidenceInterval
    biclustering_error_interval: ConfidenceInterval
    coverage: tuple[tuple[float, float], ...]
    detected_count: Annotated[int, Field(ge=0)]
    ground_truth_count: Annotated[int, Field(ge=1)]
    matches: tuple[MatchRecord, ...]


@dataclass(frozen=True, slots=True)
class _BestSimilarity:
    match_identifier: str | None
    row_jaccard: float
    column_jaccard: float
    structural_similarity: float
    cell_jaccard: float


def _jaccard(first: tuple[int, ...], second: tuple[int, ...]) -> float:
    first_set = set(first)
    second_set = set(second)
    return len(first_set.intersection(second_set)) / len(first_set.union(second_set))


def _similarity(first: BiclusterMembership, second: BiclusterMembership) -> _BestSimilarity:
    row_intersection = len(set(first.row_indices).intersection(second.row_indices))
    column_intersection = len(set(first.column_indices).intersection(second.column_indices))
    row_union = len(first.row_indices) + len(second.row_indices) - row_intersection
    column_union = len(first.column_indices) + len(second.column_indices) - column_intersection
    row_jaccard = row_intersection / row_union
    column_jaccard = column_intersection / column_union
    intersection_cells = row_intersection * column_intersection
    union_cells = (
        len(first.row_indices) * len(first.column_indices)
        + len(second.row_indices) * len(second.column_indices)
        - intersection_cells
    )
    return _BestSimilarity(
        match_identifier=second.identifier,
        row_jaccard=row_jaccard,
        column_jaccard=column_jaccard,
        structural_similarity=math.sqrt(row_jaccard * column_jaccard),
        cell_jaccard=intersection_cells / union_cells,
    )


def _best(
    source: BiclusterMembership,
    targets: tuple[BiclusterMembership, ...],
) -> _BestSimilarity:
    if not targets:
        return _BestSimilarity(None, 0.0, 0.0, 0.0, 0.0)
    similarities = tuple(_similarity(source, target) for target in targets)
    return max(
        similarities,
        key=lambda item: (
            item.structural_similarity,
            item.cell_jaccard,
            item.row_jaccard,
            item.column_jaccard,
            item.match_identifier or "",
        ),
    )


def _prelic_score(
    source: tuple[BiclusterMembership, ...],
    targets: tuple[BiclusterMembership, ...],
) -> float:
    if not source or not targets:
        return 0.0
    best_rows = tuple(
        max(_jaccard(item.row_indices, target.row_indices) for target in targets) for item in source
    )
    best_columns = tuple(
        max(_jaccard(item.column_indices, target.column_indices) for target in targets)
        for item in source
    )
    return math.sqrt((sum(best_rows) / len(best_rows)) * (sum(best_columns) / len(best_columns)))


def _cell_counts(
    biclusters: tuple[BiclusterMembership, ...],
) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for bicluster in biclusters:
        counts.update(
            (row_index, column_index)
            for row_index in bicluster.row_indices
            for column_index in bicluster.column_indices
        )
    return counts


def _biclustering_error(
    detected: tuple[BiclusterMembership, ...],
    ground_truth: tuple[BiclusterMembership, ...],
) -> float:
    detected_counts = _cell_counts(detected)
    ground_truth_counts = _cell_counts(ground_truth)
    union_sum = sum(
        max(detected_counts.get(cell, 0), ground_truth_counts.get(cell, 0))
        for cell in detected_counts.keys() | ground_truth_counts.keys()
    )
    if union_sum == 0:
        return 0.0
    if not detected:
        return 0.0
    intersections = np.zeros((len(detected), len(ground_truth)), dtype=np.float64)
    for detected_index, inferred in enumerate(detected):
        inferred_rows = set(inferred.row_indices)
        inferred_columns = set(inferred.column_indices)
        for truth_index, target in enumerate(ground_truth):
            intersections[detected_index, truth_index] = len(
                inferred_rows.intersection(target.row_indices)
            ) * len(inferred_columns.intersection(target.column_indices))
    row_indices, column_indices = linear_sum_assignment(intersections, maximize=True)
    maximum_matching = float(intersections[row_indices, column_indices].sum())
    return maximum_matching / union_sum


def _bootstrap_prelic(
    source: tuple[BiclusterMembership, ...],
    targets: tuple[BiclusterMembership, ...],
    *,
    estimate: float,
    configuration: UncertaintyConfiguration,
    seed_offset: int,
) -> ConfidenceInterval:
    if configuration.bootstrap_samples == 0 or not source or not targets:
        return ConfidenceInterval(
            estimate=estimate,
            lower=estimate,
            upper=estimate,
            confidence_level=configuration.confidence_level,
            samples=configuration.bootstrap_samples,
        )
    row_scores = np.asarray(
        [
            max(_jaccard(item.row_indices, target.row_indices) for target in targets)
            for item in source
        ],
        dtype=np.float64,
    )
    column_scores = np.asarray(
        [
            max(_jaccard(item.column_indices, target.column_indices) for target in targets)
            for item in source
        ],
        dtype=np.float64,
    )
    random = np.random.default_rng(configuration.seed + seed_offset)
    draws = random.integers(
        0,
        len(source),
        size=(configuration.bootstrap_samples, len(source)),
    )
    samples = np.sqrt(row_scores[draws].mean(axis=1) * column_scores[draws].mean(axis=1))
    alpha = (1.0 - configuration.confidence_level) / 2.0
    lower, upper = np.quantile(samples, (alpha, 1.0 - alpha))
    return ConfidenceInterval(
        estimate=estimate,
        lower=float(lower),
        upper=float(upper),
        confidence_level=configuration.confidence_level,
        samples=configuration.bootstrap_samples,
    )


def _bootstrap_biclustering_error(
    detected: tuple[BiclusterMembership, ...],
    ground_truth: tuple[BiclusterMembership, ...],
    *,
    estimate: float,
    configuration: UncertaintyConfiguration,
) -> ConfidenceInterval:
    if configuration.bootstrap_samples == 0 or not detected or not ground_truth:
        return ConfidenceInterval(
            estimate=estimate,
            lower=estimate,
            upper=estimate,
            confidence_level=configuration.confidence_level,
            samples=configuration.bootstrap_samples,
        )
    random = np.random.default_rng(configuration.seed + 2)
    samples = np.empty(configuration.bootstrap_samples, dtype=np.float64)
    for index in range(configuration.bootstrap_samples):
        detected_sample = tuple(
            detected[item] for item in random.integers(0, len(detected), size=len(detected))
        )
        ground_truth_sample = tuple(
            ground_truth[item]
            for item in random.integers(0, len(ground_truth), size=len(ground_truth))
        )
        samples[index] = _biclustering_error(detected_sample, ground_truth_sample)
    alpha = (1.0 - configuration.confidence_level) / 2.0
    lower, upper = np.quantile(samples, (alpha, 1.0 - alpha))
    return ConfidenceInterval(
        estimate=estimate,
        lower=float(lower),
        upper=float(upper),
        confidence_level=configuration.confidence_level,
        samples=configuration.bootstrap_samples,
    )


def calculate_accuracy(
    detected: tuple[BiclusterMembership, ...],
    ground_truth: tuple[BiclusterMembership, ...],
    *,
    uncertainty: UncertaintyConfiguration,
    coverage_thresholds: tuple[float, ...],
) -> AccuracyResult:
    if not ground_truth:
        raise ValueError("accuracy requires at least one ground-truth bicluster")
    relevance = _prelic_score(detected, ground_truth)
    recovery = _prelic_score(ground_truth, detected)
    biclustering_error = _biclustering_error(detected, ground_truth)
    detected_best = tuple(_best(item, ground_truth) for item in detected)
    truth_best = tuple(_best(item, detected) for item in ground_truth)
    matches = tuple(
        MatchRecord(
            perspective="DETECTED",
            bicluster_id=bicluster.identifier,
            best_match_id=best.match_identifier,
            row_jaccard=best.row_jaccard,
            column_jaccard=best.column_jaccard,
            structural_similarity=best.structural_similarity,
            cell_jaccard=best.cell_jaccard,
        )
        for bicluster, best in zip(detected, detected_best, strict=True)
    ) + tuple(
        MatchRecord(
            perspective="GROUND_TRUTH",
            bicluster_id=bicluster.identifier,
            best_match_id=best.match_identifier,
            row_jaccard=best.row_jaccard,
            column_jaccard=best.column_jaccard,
            structural_similarity=best.structural_similarity,
            cell_jaccard=best.cell_jaccard,
        )
        for bicluster, best in zip(ground_truth, truth_best, strict=True)
    )
    coverage = tuple(
        (
            threshold,
            sum(best.structural_similarity >= threshold for best in truth_best) / len(truth_best),
        )
        for threshold in coverage_thresholds
    )
    return AccuracyResult(
        relevance=relevance,
        recovery=recovery,
        biclustering_error=biclustering_error,
        relevance_interval=_bootstrap_prelic(
            detected,
            ground_truth,
            estimate=relevance,
            configuration=uncertainty,
            seed_offset=0,
        ),
        recovery_interval=_bootstrap_prelic(
            ground_truth,
            detected,
            estimate=recovery,
            configuration=uncertainty,
            seed_offset=1,
        ),
        biclustering_error_interval=_bootstrap_biclustering_error(
            detected,
            ground_truth,
            estimate=biclustering_error,
            configuration=uncertainty,
        ),
        coverage=coverage,
        detected_count=len(detected),
        ground_truth_count=len(ground_truth),
        matches=matches,
    )


__all__ = [
    "AccuracyResult",
    "BiclusterMembership",
    "ConfidenceInterval",
    "MatchRecord",
    "calculate_accuracy",
]
