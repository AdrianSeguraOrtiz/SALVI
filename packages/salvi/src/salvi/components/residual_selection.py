"""Adaptive residual-evidence extraction for compact final repertoires."""

from __future__ import annotations

import heapq
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Literal, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from salvi.application.context import RunContext
from salvi.components.final_selection import (
    _continuous_utilities,
    _deduplicate,
    _group_identifier,
    _selected_objectives,
    _validate_objectives,
)
from salvi.domain.enums import ObjectiveDirection
from salvi.domain.models import Evaluation, FinalSelectionProvenance, Repertoire
from salvi.exceptions import ComponentError

_EPSILON = 1e-12
_QualityScale = Literal["unit_interval", "empirical"]


class AdaptiveResidualEvidenceCoverConfiguration(BaseModel):
    """Configuration for adaptive quality-weighted residual evidence coverage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    objective_names: tuple[str, ...] | None = None
    quality_scale: _QualityScale = "unit_interval"
    overlap_penalty: Annotated[float, Field(ge=0.0)] = 0.50
    low_quality_penalty: Annotated[float, Field(ge=0.0)] = 0.50
    complexity_penalty: Annotated[float, Field(ge=0.0)] = 0.25
    minimum_marginal_evidence: Annotated[float, Field(ge=0.0)] = 1.0
    maximum_dense_cells: Annotated[int, Field(ge=1)] = 10_000_000
    minimum_quality_floor: Annotated[float, Field(ge=0.0, lt=1.0)] = 0.50
    maximum_quality_floor: Annotated[float, Field(ge=0.0, lt=1.0)] = 0.85
    minimum_candidates_for_knee: Annotated[int, Field(ge=3)] = 8
    minimum_knee_prominence: Annotated[float, Field(ge=0.0, le=1.0)] = 0.05
    fallback_quality_quantile: Annotated[float, Field(ge=0.0, le=1.0)] = 0.50

    @model_validator(mode="after")
    def validate_configuration(self) -> Self:
        if self.objective_names is not None:
            if not self.objective_names:
                raise ValueError("objective_names must be null or a non-empty list")
            if any(not name.strip() for name in self.objective_names):
                raise ValueError("objective_names must not contain blank names")
            if len(set(self.objective_names)) != len(self.objective_names):
                raise ValueError("objective_names must be unique")
        if self.minimum_quality_floor > self.maximum_quality_floor:
            raise ValueError("minimum_quality_floor must not exceed maximum_quality_floor")
        return self


@dataclass(frozen=True, slots=True)
class _QualityFloorResolution:
    value: float
    source: str
    candidate_count: int
    quality_span: float
    knee_prominence: float


@dataclass(frozen=True, slots=True)
class _MarginalEvidence:
    new_evidence: float
    redundant_evidence: float
    low_quality_cost: float
    complexity_cost: float
    net_gain: float
    unexplained_fraction: float


@dataclass(frozen=True, slots=True)
class _EvidenceCandidate:
    evaluation: Evaluation
    sources: tuple[Evaluation, ...]
    rows: npt.NDArray[np.intp]
    columns: npt.NDArray[np.intp]
    qualities: npt.NDArray[np.float64]
    strengths: npt.NDArray[np.float64]
    observed_cells: int
    total_evidence: float
    quality_score: float
    complexity_cost: float


@dataclass(frozen=True, slots=True)
class _Decision:
    evidence: _EvidenceCandidate
    score: _MarginalEvidence
    selection_rank: int


class _Coverage:
    def score(self, candidate: _EvidenceCandidate) -> _MarginalEvidence:
        raise NotImplementedError

    def update(self, candidate: _EvidenceCandidate) -> None:
        raise NotImplementedError


class _DenseCoverage(_Coverage):
    def __init__(
        self,
        support: npt.NDArray[np.bool_],
        *,
        overlap_penalty: float,
        low_quality_penalty: float,
        quality_floor: float,
    ) -> None:
        self._support = support
        self._values = np.zeros(support.shape, dtype=np.float32)
        self._overlap_penalty = overlap_penalty
        self._low_quality_penalty = low_quality_penalty
        self._quality_floor = quality_floor

    def score(self, candidate: _EvidenceCandidate) -> _MarginalEvidence:
        indices = np.ix_(candidate.rows, candidate.columns)
        support = self._support[indices]
        current = self._values[indices]
        strengths = candidate.strengths[np.newaxis, :]
        new_evidence = float(np.sum(np.maximum(strengths - current, 0.0), where=support))
        redundant_evidence = float(np.sum(np.minimum(strengths, current), where=support))
        low_quality_cost = _low_quality_cost(
            candidate.qualities,
            support,
            self._quality_floor,
        )
        return _marginal_score(
            candidate,
            new_evidence,
            redundant_evidence,
            low_quality_cost,
            overlap_penalty=self._overlap_penalty,
            low_quality_penalty=self._low_quality_penalty,
        )

    def update(self, candidate: _EvidenceCandidate) -> None:
        indices = np.ix_(candidate.rows, candidate.columns)
        support = self._support[indices]
        current = self._values[indices]
        updated = np.maximum(current, candidate.strengths[np.newaxis, :])
        self._values[indices] = np.where(support, updated, current)


class _SparseCoverage(_Coverage):
    def __init__(
        self,
        support: npt.NDArray[np.bool_],
        *,
        column_count: int,
        overlap_penalty: float,
        low_quality_penalty: float,
        quality_floor: float,
    ) -> None:
        self._support = support
        self._column_count = column_count
        self._values: dict[int, float] = {}
        self._overlap_penalty = overlap_penalty
        self._low_quality_penalty = low_quality_penalty
        self._quality_floor = quality_floor

    def score(self, candidate: _EvidenceCandidate) -> _MarginalEvidence:
        new_evidence = 0.0
        redundant_evidence = 0.0
        low_quality_cost = 0.0
        for row in candidate.rows:
            row_index = int(row)
            offset = row_index * self._column_count
            for position, column in enumerate(candidate.columns):
                column_index = int(column)
                if not self._support[row_index, column_index]:
                    continue
                strength = float(candidate.strengths[position])
                current = self._values.get(offset + column_index, 0.0)
                new_evidence += max(0.0, strength - current)
                redundant_evidence += min(strength, current)
                low_quality_cost += _quality_deficit(
                    float(candidate.qualities[position]),
                    self._quality_floor,
                )
        return _marginal_score(
            candidate,
            new_evidence,
            redundant_evidence,
            low_quality_cost,
            overlap_penalty=self._overlap_penalty,
            low_quality_penalty=self._low_quality_penalty,
        )

    def update(self, candidate: _EvidenceCandidate) -> None:
        for row in candidate.rows:
            row_index = int(row)
            offset = row_index * self._column_count
            for position, column in enumerate(candidate.columns):
                column_index = int(column)
                if not self._support[row_index, column_index]:
                    continue
                key = offset + column_index
                self._values[key] = max(
                    self._values.get(key, 0.0),
                    float(candidate.strengths[position]),
                )


def _quality_deficit(quality: float, quality_floor: float) -> float:
    if quality_floor <= _EPSILON:
        return 0.0
    return max(0.0, quality_floor - quality) / quality_floor


def _low_quality_cost(
    qualities: npt.NDArray[np.float64],
    support: npt.NDArray[np.bool_],
    quality_floor: float,
) -> float:
    if quality_floor <= _EPSILON:
        return 0.0
    deficits = np.maximum(quality_floor - qualities, 0.0) / quality_floor
    return float(np.sum(np.broadcast_to(deficits, support.shape), where=support))


def _marginal_score(
    candidate: _EvidenceCandidate,
    new_evidence: float,
    redundant_evidence: float,
    low_quality_cost: float,
    *,
    overlap_penalty: float,
    low_quality_penalty: float,
) -> _MarginalEvidence:
    net_gain = (
        new_evidence
        - overlap_penalty * redundant_evidence
        - low_quality_penalty * low_quality_cost
        - candidate.complexity_cost
    )
    unexplained_fraction = (
        new_evidence / candidate.total_evidence if candidate.total_evidence > _EPSILON else 0.0
    )
    return _MarginalEvidence(
        new_evidence=new_evidence,
        redundant_evidence=redundant_evidence,
        low_quality_cost=low_quality_cost,
        complexity_cost=candidate.complexity_cost,
        net_gain=net_gain,
        unexplained_fraction=min(1.0, max(0.0, unexplained_fraction)),
    )


def _log2_choose(total: int, selected: int) -> float:
    if selected < 0 or selected > total:
        raise ComponentError("candidate membership lies outside the prepared dataset")
    return (
        math.lgamma(total + 1) - math.lgamma(selected + 1) - math.lgamma(total - selected + 1)
    ) / math.log(2.0)


def _resolve_quality_knee(
    qualities: Sequence[float],
    *,
    minimum_floor: float,
    maximum_floor: float,
    minimum_candidates: int,
    minimum_prominence: float,
    fallback_quantile: float,
) -> _QualityFloorResolution:
    """Resolve a two-sided quality transition or a robust quantile fallback."""

    ordered = np.asarray(sorted(qualities, reverse=True), dtype=np.float64)
    candidate_count = int(ordered.size)
    if candidate_count and not bool(np.all(np.isfinite(ordered))):
        raise ComponentError("residual evidence quality values must be finite")
    quality_span = float(ordered[0] - ordered[-1]) if candidate_count else 0.0
    minimum_fallback = _QualityFloorResolution(
        value=minimum_floor,
        source="minimum_floor_fallback",
        candidate_count=candidate_count,
        quality_span=quality_span,
        knee_prominence=0.0,
    )
    if candidate_count < minimum_candidates:
        return minimum_fallback
    quantile_floor = min(
        maximum_floor,
        max(minimum_floor, float(np.quantile(ordered, fallback_quantile))),
    )
    quantile_fallback = _QualityFloorResolution(
        value=quantile_floor,
        source="quality_quantile_fallback",
        candidate_count=candidate_count,
        quality_span=quality_span,
        knee_prominence=0.0,
    )
    if quality_span <= _EPSILON:
        return quantile_fallback

    normalized_quality = (ordered - ordered[-1]) / quality_span
    rank = np.linspace(0.0, 1.0, candidate_count, dtype=np.float64)
    prominence_by_rank = normalized_quality - (1.0 - rank)
    interior = prominence_by_rank[1:-1]
    if interior.size == 0:
        return quantile_fallback
    upper_knee_index = int(np.argmax(interior)) + 1
    lower_knee_index = int(np.argmin(interior)) + 1
    upper_prominence = max(0.0, float(prominence_by_rank[upper_knee_index]))
    lower_prominence = max(0.0, -float(prominence_by_rank[lower_knee_index]))
    prominence = min(upper_prominence, lower_prominence)
    if upper_knee_index >= lower_knee_index or prominence + _EPSILON < minimum_prominence:
        return quantile_fallback

    transition_midpoint = float((ordered[upper_knee_index] + ordered[lower_knee_index]) / 2.0)
    return _QualityFloorResolution(
        value=min(maximum_floor, max(minimum_floor, transition_midpoint)),
        source="quality_knee",
        candidate_count=candidate_count,
        quality_span=quality_span,
        knee_prominence=prominence,
    )


@dataclass(frozen=True, slots=True)
class AdaptiveResidualEvidenceCoverSelector:
    """Select compact, complementary biclusters using residual matrix evidence."""

    objective_names: tuple[str, ...] | None = None
    quality_scale: _QualityScale = "unit_interval"
    overlap_penalty: float = 0.50
    low_quality_penalty: float = 0.50
    complexity_penalty: float = 0.25
    minimum_marginal_evidence: float = 1.0
    maximum_dense_cells: int = 10_000_000
    minimum_quality_floor: float = 0.50
    maximum_quality_floor: float = 0.85
    minimum_candidates_for_knee: int = 8
    minimum_knee_prominence: float = 0.05
    fallback_quality_quantile: float = 0.50
    component_name: str = "adaptive_residual_evidence_cover"
    provides: frozenset[str] = frozenset({"final-selection"})
    requires: frozenset[str] = frozenset({"search-result", "objective"})

    def __post_init__(self) -> None:
        AdaptiveResidualEvidenceCoverConfiguration(
            objective_names=self.objective_names,
            quality_scale=self.quality_scale,
            overlap_penalty=self.overlap_penalty,
            low_quality_penalty=self.low_quality_penalty,
            complexity_penalty=self.complexity_penalty,
            minimum_marginal_evidence=self.minimum_marginal_evidence,
            maximum_dense_cells=self.maximum_dense_cells,
            minimum_quality_floor=self.minimum_quality_floor,
            maximum_quality_floor=self.maximum_quality_floor,
            minimum_candidates_for_knee=self.minimum_candidates_for_knee,
            minimum_knee_prominence=self.minimum_knee_prominence,
            fallback_quality_quantile=self.fallback_quality_quantile,
        )

    def select(self, context: RunContext, repertoire: Repertoire) -> Repertoire:
        identifiers = tuple(
            evaluation.candidate.identifier for evaluation in repertoire.evaluations
        )
        if len(set(identifiers)) != len(identifiers):
            raise ComponentError(
                "residual evidence final selection requires unique candidate identifiers"
            )
        feasible = tuple(evaluation for evaluation in repertoire.evaluations if evaluation.feasible)
        if not feasible:
            return Repertoire()

        _validate_objectives(feasible, self.objective_names)
        representatives, duplicates = _deduplicate(feasible, self.objective_names)
        scalar_quality, column_quality = self._quality(representatives)
        resolution = _resolve_quality_knee(
            tuple(scalar_quality.values()),
            minimum_floor=self.minimum_quality_floor,
            maximum_floor=self.maximum_quality_floor,
            minimum_candidates=self.minimum_candidates_for_knee,
            minimum_prominence=self.minimum_knee_prominence,
            fallback_quantile=self.fallback_quality_quantile,
        )
        evidence = tuple(
            self._prepare_candidate(
                context,
                evaluation,
                duplicates[evaluation.candidate.bicluster.signature],
                scalar_quality[evaluation.candidate.identifier],
                column_quality[evaluation.candidate.identifier],
                quality_floor=resolution.value,
            )
            for evaluation in representatives
        )
        decisions = self._select(
            evidence,
            self._coverage(context, quality_floor=resolution.value),
        )

        selected: list[Evaluation] = []
        for decision in decisions:
            item = decision.evidence
            source_identifiers = tuple(
                sorted(source.candidate.identifier for source in item.sources)
            )
            source_coordinates = tuple(
                sorted(
                    {
                        source.archive_coordinate
                        for source in item.sources
                        if source.archive_coordinate is not None
                    }
                )
            )
            denominator = (
                decision.score.new_evidence
                + self.overlap_penalty * decision.score.redundant_evidence
                + self.low_quality_penalty * decision.score.low_quality_cost
                + decision.score.complexity_cost
            )
            marginal_gain = (
                max(0.0, decision.score.net_gain) / denominator if denominator > _EPSILON else 0.0
            )
            provenance = FinalSelectionProvenance(
                selector=self.component_name,
                group_identifier=_group_identifier(source_identifiers),
                selection_rank=decision.selection_rank,
                quality_score=item.quality_score,
                novelty_score=decision.score.unexplained_fraction,
                marginal_gain=min(1.0, marginal_gain),
                source_candidate_identifiers=source_identifiers,
                source_archive_coordinates=source_coordinates,
            )
            selected.append(item.evaluation.model_copy(update={"final_selection": provenance}))
        return Repertoire(evaluations=tuple(selected))

    def _quality(
        self,
        evaluations: Sequence[Evaluation],
    ) -> tuple[dict[str, float], dict[str, npt.NDArray[np.float64]]]:
        if self.quality_scale == "empirical":
            empirical_scalar = {
                identifier: min(values)
                for identifier, values in _continuous_utilities(
                    evaluations,
                    self.objective_names,
                ).items()
            }
            return (
                empirical_scalar,
                {
                    evaluation.candidate.identifier: np.full(
                        len(evaluation.candidate.bicluster.column_indices),
                        empirical_scalar[evaluation.candidate.identifier],
                        dtype=np.float64,
                    )
                    for evaluation in evaluations
                },
            )

        scalar: dict[str, float] = {}
        columns: dict[str, npt.NDArray[np.float64]] = {}
        for evaluation in evaluations:
            selected = _selected_objectives(evaluation, self.objective_names)
            utilities = tuple(
                self._objective_utility(item.value, item.direction) for item in selected
            )
            scalar[evaluation.candidate.identifier] = min(utilities)
            if all(item.columns for item in selected):
                per_objective = tuple(
                    {
                        column.column_index: self._objective_utility(
                            column.value,
                            objective.direction,
                        )
                        for column in objective.columns
                    }
                    for objective in selected
                )
                columns[evaluation.candidate.identifier] = np.asarray(
                    tuple(
                        min(values[column] for values in per_objective)
                        for column in evaluation.candidate.bicluster.column_indices
                    ),
                    dtype=np.float64,
                )
            else:
                columns[evaluation.candidate.identifier] = np.full(
                    len(evaluation.candidate.bicluster.column_indices),
                    scalar[evaluation.candidate.identifier],
                    dtype=np.float64,
                )
        return scalar, columns

    @staticmethod
    def _objective_utility(value: float, direction: ObjectiveDirection) -> float:
        if value < 0.0 or value > 1.0:
            raise ComponentError(
                "adaptive residual evidence requires unit-interval objective values; "
                "configure quality_scale='empirical' for objectives on arbitrary scales"
            )
        return value if direction is ObjectiveDirection.MAXIMIZE else 1.0 - value

    def _prepare_candidate(
        self,
        context: RunContext,
        evaluation: Evaluation,
        sources: tuple[Evaluation, ...],
        quality_score: float,
        qualities: npt.NDArray[np.float64],
        *,
        quality_floor: float,
    ) -> _EvidenceCandidate:
        rows = np.asarray(evaluation.candidate.bicluster.row_indices, dtype=np.intp)
        columns = np.asarray(
            evaluation.candidate.bicluster.column_indices,
            dtype=np.intp,
        )
        if (
            rows.size == 0
            or columns.size == 0
            or rows[-1] >= context.dataset.row_count
            or columns[-1] >= context.dataset.column_count
        ):
            raise ComponentError("candidate membership lies outside the prepared dataset")
        strengths = np.maximum(qualities - quality_floor, 0.0) / (1.0 - quality_floor)
        support = context.dataset.support_matrix()[np.ix_(rows, columns)]
        observed_by_column = np.count_nonzero(support, axis=0)
        observed_cells = int(np.sum(observed_by_column))
        total_evidence = float(np.dot(strengths, observed_by_column))
        complexity_cost = self.complexity_penalty * (
            _log2_choose(context.dataset.row_count, len(rows))
            + _log2_choose(context.dataset.column_count, len(columns))
        )
        return _EvidenceCandidate(
            evaluation=evaluation,
            sources=sources,
            rows=rows,
            columns=columns,
            qualities=qualities,
            strengths=strengths,
            observed_cells=observed_cells,
            total_evidence=total_evidence,
            quality_score=quality_score,
            complexity_cost=complexity_cost,
        )

    def _coverage(self, context: RunContext, *, quality_floor: float) -> _Coverage:
        support = context.dataset.support_matrix()
        if support.size <= self.maximum_dense_cells:
            return _DenseCoverage(
                support,
                overlap_penalty=self.overlap_penalty,
                low_quality_penalty=self.low_quality_penalty,
                quality_floor=quality_floor,
            )
        return _SparseCoverage(
            support,
            column_count=context.dataset.column_count,
            overlap_penalty=self.overlap_penalty,
            low_quality_penalty=self.low_quality_penalty,
            quality_floor=quality_floor,
        )

    def _select(
        self,
        candidates: Sequence[_EvidenceCandidate],
        coverage: _Coverage,
    ) -> tuple[_Decision, ...]:
        by_identifier = {
            candidate.evaluation.candidate.identifier: candidate for candidate in candidates
        }
        heap: list[tuple[float, float, int, str, str, int, _MarginalEvidence]] = []
        version = 0
        for candidate in candidates:
            score = coverage.score(candidate)
            heapq.heappush(heap, self._heap_entry(candidate, score, version))

        decisions: list[_Decision] = []
        selected_rank = 0
        while heap:
            (
                _negative_gain,
                _negative_quality,
                _negative_support,
                _signature,
                identifier,
                scored_version,
                score,
            ) = heapq.heappop(heap)
            candidate = by_identifier[identifier]
            if scored_version != version:
                refreshed = coverage.score(candidate)
                heapq.heappush(heap, self._heap_entry(candidate, refreshed, version))
                continue
            if score.net_gain <= self.minimum_marginal_evidence + _EPSILON:
                break
            decisions.append(
                _Decision(
                    evidence=candidate,
                    score=score,
                    selection_rank=selected_rank,
                )
            )
            selected_rank += 1
            coverage.update(candidate)
            version += 1
        return tuple(decisions)

    @staticmethod
    def _heap_entry(
        candidate: _EvidenceCandidate,
        score: _MarginalEvidence,
        version: int,
    ) -> tuple[float, float, int, str, str, int, _MarginalEvidence]:
        evaluation = candidate.evaluation
        return (
            -score.net_gain,
            -candidate.quality_score,
            -candidate.observed_cells,
            evaluation.candidate.bicluster.signature,
            evaluation.candidate.identifier,
            version,
            score,
        )


__all__ = [
    "AdaptiveResidualEvidenceCoverConfiguration",
    "AdaptiveResidualEvidenceCoverSelector",
]
