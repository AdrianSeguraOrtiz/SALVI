"""Immutable contracts for descriptors, archives, and serial search state."""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal, Self, overload

from pydantic import Field, field_validator, model_validator

from salvi.domain.enums import (
    ArchiveInsertionStatus,
    BinningStrategy,
    DescriptorValueKind,
    EvaluationIntegrationMode,
    PatternKind,
)
from salvi.domain.models import Candidate, Evaluation, FrozenModel, Repertoire


class DescriptorDomain(FrozenModel):
    """Semantic numeric domain declared by a descriptor."""

    value_kind: DescriptorValueKind
    minimum: float
    maximum: float
    supported_binnings: tuple[BinningStrategy, ...]
    recommended_binning: BinningStrategy

    @model_validator(mode="after")
    def validate_domain(self) -> Self:
        if not math.isfinite(self.minimum) or not math.isfinite(self.maximum):
            raise ValueError("descriptor bounds must be finite")
        if self.minimum > self.maximum:
            raise ValueError("descriptor minimum cannot exceed its maximum")
        if not self.supported_binnings:
            raise ValueError("a descriptor must support at least one binning strategy")
        if len(set(self.supported_binnings)) != len(self.supported_binnings):
            raise ValueError("supported descriptor binning strategies must be unique")
        if self.recommended_binning not in self.supported_binnings:
            raise ValueError("recommended binning must be supported by the descriptor")
        if self.value_kind is DescriptorValueKind.INTEGER and (
            not self.minimum.is_integer() or not self.maximum.is_integer()
        ):
            raise ValueError("integer descriptor bounds must be integral")
        return self


class CandidateBounds(FrozenModel):
    min_rows: Annotated[int, Field(ge=1)]
    max_rows: Annotated[int, Field(ge=1)]
    min_columns: Annotated[int, Field(ge=1)]
    max_columns: Annotated[int, Field(ge=1)]

    @model_validator(mode="after")
    def validate_bounds(self) -> CandidateBounds:
        if self.min_rows > self.max_rows or self.min_columns > self.max_columns:
            raise ValueError("candidate minimum cardinality cannot exceed its maximum")
        return self


class TerminationProgress(FrozenModel):
    """Engine-independent progress reported by a termination criterion."""

    current: Annotated[float, Field(ge=0.0)]
    limit: Annotated[float, Field(gt=0.0)] | None = None
    unit: str = Field(min_length=1)

    @property
    def fraction(self) -> float | None:
        if self.limit is None:
            return None
        return min(1.0, self.current / self.limit)


class ArchiveCellCoordinate(FrozenModel):
    indices: tuple[Annotated[int, Field(ge=0)], ...]

    @field_validator("indices")
    @classmethod
    def require_axis(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("archive coordinates require at least one axis")
        return value


class ArchiveCellTarget(FrozenModel):
    """One reachable cardinality cell and a representative candidate shape."""

    coordinate: ArchiveCellCoordinate
    row_count: Annotated[int, Field(ge=1)]
    column_count: Annotated[int, Field(ge=1)]


class BootstrapCellState(FrozenModel):
    """Checkpointable coverage progress for one archive cell."""

    target: ArchiveCellTarget
    required_patterns: tuple[PatternKind, ...]
    accepted_patterns: tuple[PatternKind, ...] = ()
    attempts: Annotated[int, Field(ge=0)] = 0
    maximum_attempts: Annotated[int, Field(ge=1)]

    @model_validator(mode="after")
    def validate_progress(self) -> BootstrapCellState:
        if not self.required_patterns:
            raise ValueError("bootstrap cells require at least one requested pattern")
        if self.maximum_attempts < len(self.required_patterns):
            raise ValueError("bootstrap maximum attempts cannot be smaller than required seeds")
        remaining = list(self.required_patterns)
        for pattern in self.accepted_patterns:
            try:
                remaining.remove(pattern)
            except ValueError as error:
                raise ValueError(
                    "accepted bootstrap patterns must be a sub-multiset of required patterns"
                ) from error
        if self.attempts < len(self.accepted_patterns):
            raise ValueError("bootstrap attempts cannot be smaller than accepted seeds")
        return self

    @property
    def complete(self) -> bool:
        return len(self.accepted_patterns) >= len(self.required_patterns)

    @property
    def exhausted(self) -> bool:
        return self.attempts >= self.maximum_attempts

    @property
    def pending_pattern(self) -> PatternKind | None:
        remaining = list(self.required_patterns)
        for pattern in self.accepted_patterns:
            remaining.remove(pattern)
        return None if not remaining else remaining[0]


class ArchiveCell(FrozenModel):
    coordinate: ArchiveCellCoordinate
    evaluations: tuple[Evaluation, ...]

    @field_validator("evaluations")
    @classmethod
    def require_members(cls, value: tuple[Evaluation, ...]) -> tuple[Evaluation, ...]:
        if not value:
            raise ValueError("archive cells cannot be empty")
        signatures = tuple(item.candidate.bicluster.signature for item in value)
        if len(set(signatures)) != len(signatures):
            raise ValueError("archive cells cannot contain duplicate biclusters")
        return value


class ArchiveInsertionOutcome(FrozenModel):
    candidate_identifier: str = Field(min_length=1)
    status: ArchiveInsertionStatus
    coordinate: ArchiveCellCoordinate | None = None
    evicted_candidate_identifiers: tuple[str, ...] = ()
    created_cell: bool = False

    @property
    def accepted(self) -> bool:
        return self.status in {
            ArchiveInsertionStatus.INSERTED,
            ArchiveInsertionStatus.INSERTED_WITH_EVICTIONS,
        }


class EmitterAllocation(FrozenModel):
    emitter_name: str = Field(min_length=1)
    count: Annotated[int, Field(ge=1)]


class EmitterCellFeedback(FrozenModel):
    coordinate: ArchiveCellCoordinate | None
    evaluated: Annotated[int, Field(ge=1)]
    accepted: Annotated[int, Field(ge=0)]
    created_cells: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def validate_counts(self) -> EmitterCellFeedback:
        if self.accepted > self.evaluated or self.created_cells > self.accepted:
            raise ValueError("emitter cell feedback counts are inconsistent")
        return self


class EmitterFeedback(FrozenModel):
    """Raw archive outcomes attributed to one emitter allocation."""

    emitter_name: str = Field(min_length=1)
    evaluated: Annotated[int, Field(ge=1)]
    accepted: Annotated[int, Field(ge=0)]
    created_cells: Annotated[int, Field(ge=0)]
    evictions: Annotated[int, Field(ge=0)]
    statuses: tuple[tuple[ArchiveInsertionStatus, Annotated[int, Field(ge=1)]], ...]
    cells: tuple[EmitterCellFeedback, ...] = ()

    @model_validator(mode="after")
    def validate_counts(self) -> EmitterFeedback:
        if self.accepted > self.evaluated or self.created_cells > self.accepted:
            raise ValueError("emitter feedback counts are inconsistent")
        if sum(count for _, count in self.statuses) != self.evaluated:
            raise ValueError("emitter feedback statuses must account for every evaluation")
        if len({status for status, _ in self.statuses}) != len(self.statuses):
            raise ValueError("emitter feedback statuses must be unique")
        if self.cells and (
            sum(cell.evaluated for cell in self.cells) != self.evaluated
            or sum(cell.accepted for cell in self.cells) != self.accepted
            or sum(cell.created_cells for cell in self.cells) != self.created_cells
        ):
            raise ValueError("emitter cell feedback must account for aggregate counts exactly")
        return self


class SchedulerReport(FrozenModel):
    emitter_name: str = Field(min_length=1)
    evaluations: Annotated[int, Field(ge=0)]
    accepted: Annotated[int, Field(ge=0)]
    created_cells: Annotated[int, Field(ge=0)]
    credit: Annotated[float, Field(ge=0.0)]
    allocation_count: Annotated[int, Field(ge=0)]


class SearchUpdate(FrozenModel):
    outcomes: tuple[ArchiveInsertionOutcome, ...]
    emitter_feedback: tuple[EmitterFeedback, ...] = ()
    scheduler_reports: tuple[SchedulerReport, ...] = ()

    @model_validator(mode="after")
    def validate_outcomes(self) -> SearchUpdate:
        if sum(feedback.evaluated for feedback in self.emitter_feedback) > len(self.outcomes):
            raise ValueError("emitter feedback cannot exceed search update outcomes")
        return self


class EvaluationBatch(FrozenModel):
    """One bounded executor result plus its observable runtime properties."""

    evaluations: tuple[Evaluation, ...]
    completion_order: tuple[str, ...]
    duration_seconds: Annotated[float, Field(ge=0.0)]
    worker_count: Annotated[int, Field(ge=1)]
    peak_in_flight: Annotated[int, Field(ge=0)]
    integration_mode: EvaluationIntegrationMode
    candidate_duration_seconds: tuple[Annotated[float, Field(ge=0.0)], ...] = ()
    component_duration_seconds: tuple[
        tuple[str, Annotated[float, Field(ge=0.0)]],
        ...,
    ] = ()

    @model_validator(mode="after")
    def validate_batch(self) -> EvaluationBatch:
        identifiers = tuple(item.candidate.identifier for item in self.evaluations)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("evaluation batches require unique candidate identifiers")
        if len(self.completion_order) != len(identifiers) or set(self.completion_order) != set(
            identifiers
        ):
            raise ValueError("completion order must contain every evaluated candidate exactly once")
        if self.peak_in_flight > len(self.evaluations):
            raise ValueError("peak in-flight work cannot exceed the batch size")
        if self.candidate_duration_seconds and len(self.candidate_duration_seconds) != len(
            self.evaluations
        ):
            raise ValueError("candidate timings must align with evaluated candidates")
        component_names = tuple(name for name, _duration in self.component_duration_seconds)
        if any(not name.strip() for name in component_names):
            raise ValueError("timed component names must not be blank")
        if len(set(component_names)) != len(component_names):
            raise ValueError("timed component names must be unique")
        return self

    def __len__(self) -> int:
        return len(self.evaluations)

    @overload
    def __getitem__(self, index: int) -> Evaluation: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[Evaluation, ...]: ...

    def __getitem__(self, index: int | slice) -> Evaluation | tuple[Evaluation, ...]:
        return self.evaluations[index]


class SearchProgress(FrozenModel):
    evaluations: Annotated[int, Field(ge=0)]
    accepted: Annotated[int, Field(ge=0)]
    rejected: Annotated[int, Field(ge=0)]
    occupied_cells: Annotated[int, Field(ge=0)]
    repertoire_size: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def validate_counts(self) -> SearchProgress:
        if self.accepted + self.rejected != self.evaluations:
            raise ValueError("accepted and rejected counts must equal evaluated candidates")
        return self


class SearchCheckpoint(FrozenModel):
    """Versioned state sufficient for exact deterministic serial continuation."""

    schema_version: Literal[4] = 4
    engine_name: str = Field(min_length=1)
    run_identifier: str = Field(min_length=1)
    dataset_identifier: str = Field(min_length=1)
    search_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_count: Annotated[int, Field(ge=0)]
    accepted_count: Annotated[int, Field(ge=0)]
    rejected_count: Annotated[int, Field(ge=0)]
    next_candidate_sequence: Annotated[int, Field(ge=0)]
    integration_mode: EvaluationIntegrationMode
    initial_candidates: tuple[Candidate, ...]
    pending_candidates: tuple[Candidate, ...] = ()
    pending_emitter_names: tuple[str | None, ...] = ()
    repertoire: Repertoire
    random_stream_states: dict[str, dict[str, Any]]
    scheduler_state: dict[str, Any]
    bootstrap_cells: tuple[BootstrapCellState, ...] = ()

    @model_validator(mode="after")
    def validate_pending_work(self) -> SearchCheckpoint:
        if len(self.pending_candidates) != len(self.pending_emitter_names):
            raise ValueError("pending candidates and emitter attribution must have equal lengths")
        identifiers = tuple(candidate.identifier for candidate in self.pending_candidates)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("pending checkpoint candidates must be unique")
        return self


__all__ = [
    "ArchiveCell",
    "ArchiveCellCoordinate",
    "ArchiveCellTarget",
    "ArchiveInsertionOutcome",
    "BootstrapCellState",
    "CandidateBounds",
    "DescriptorDomain",
    "EmitterAllocation",
    "EmitterCellFeedback",
    "EmitterFeedback",
    "EvaluationBatch",
    "SchedulerReport",
    "SearchCheckpoint",
    "SearchProgress",
    "SearchUpdate",
]
