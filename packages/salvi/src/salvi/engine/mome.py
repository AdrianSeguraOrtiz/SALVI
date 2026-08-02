"""Deterministic serial MOME search engine."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from salvi.api.run import RunSpecification
from salvi.application.context import QdRunContext, RunContext
from salvi.components.contracts import EngineCompositionContract, qd_engine_contract
from salvi.components.protocols import (
    Archive,
    CellCoverageInitializer,
    CellTargetArchive,
    ComponentTimingSource,
)
from salvi.domain.enums import PatternKind
from salvi.domain.models import Candidate, Evaluation, Repertoire
from salvi.domain.search import (
    ArchiveCellCoordinate,
    ArchiveCellTarget,
    ArchiveInsertionOutcome,
    BootstrapCellState,
    EmitterCellFeedback,
    EmitterFeedback,
    SearchCheckpoint,
    SearchProgress,
    SearchUpdate,
)
from salvi.exceptions import ComponentError


class SerialMomeConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    initial_population_size: Annotated[int, Field(ge=1)] = 64
    batch_size: Annotated[int, Field(ge=1)] = 16


@dataclass(slots=True)
class SerialMomeSearchEngine:
    initial_population_size: int = 64
    configured_batch_size: int = 16
    component_name: str = "serial_mome"
    composition_contract: EngineCompositionContract = field(
        default_factory=lambda: qd_engine_contract("serial_mome")
    )
    provides: frozenset[str] = frozenset({"search-engine", "search-result", "checkpoint-resume"})
    requires: frozenset[str] = frozenset(
        {"initialization", "evaluation", "archive", "emitter", "scheduler", "termination"}
    )
    _specification: RunSpecification | None = None
    _context: QdRunContext | None = None
    _archive: Archive | None = None
    _initial_candidates: tuple[Candidate, ...] = ()
    _bootstrap_cells: tuple[BootstrapCellState, ...] = ()
    _awaiting_candidates: tuple[Candidate, ...] = ()
    _awaiting_emitter_names: tuple[str | None, ...] = ()
    _replay_pending: bool = False
    _evaluation_count: int = 0
    _accepted_count: int = 0
    _next_candidate_sequence: int = 0
    _component_timings: dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        configuration = SerialMomeConfiguration(
            initial_population_size=self.initial_population_size,
            batch_size=self.configured_batch_size,
        )
        self.initial_population_size = configuration.initial_population_size
        self.configured_batch_size = configuration.batch_size

    @property
    def batch_size(self) -> int:
        return self.configured_batch_size

    @property
    def pending_candidates(self) -> Sequence[Candidate]:
        return self._awaiting_candidates

    def initialize(self, specification: RunSpecification, context: RunContext) -> None:
        archive = specification.require_archive()
        self._component_timings.clear()

        started = perf_counter()
        archive.initialize(
            context,
            specification.objectives,
            specification.descriptors,
            specification.constraints,
        )
        self._record_timing(
            f"archive.{archive.component_name}",
            perf_counter() - started,
        )
        needs_cell_targets = isinstance(
            specification.initializer,
            CellCoverageInitializer,
        ) or any("archive-cell-targets" in emitter.requires for emitter in specification.emitters)
        archive_cell_targets: tuple[ArchiveCellTarget, ...] = ()
        if needs_cell_targets:
            if not isinstance(archive, CellTargetArchive):
                raise ComponentError(
                    "configured cell-target consumers require an archive exposing cell targets"
                )
            archive_cell_targets = tuple(archive.cell_targets())
        context = QdRunContext.from_run_context(
            context,
            parent_selection_policy=specification.parent_selection_policy,
            mate_selection_policy=specification.mate_selection_policy,
            crossover_operator=specification.crossover_operator,
            mutation_operator=specification.mutation_operator,
            archive_cell_targets=archive_cell_targets,
        )

        started = perf_counter()
        bootstrap_cells: tuple[BootstrapCellState, ...] = ()
        initial_candidates: tuple[Candidate, ...] = ()
        if isinstance(specification.initializer, CellCoverageInitializer):
            if not isinstance(archive, CellTargetArchive):
                raise ComponentError(
                    "cell-coverage initialization requires an archive exposing cell targets"
                )
            bootstrap_cells = tuple(
                specification.initializer.bootstrap_plan(
                    context,
                    context.archive_cell_targets,
                )
            )
        else:
            initial_candidates = tuple(
                specification.initializer.initialize(
                    context,
                    self.initial_population_size,
                    start_sequence=0,
                )
            )
            if len(initial_candidates) != self.initial_population_size:
                raise ComponentError(
                    "initializer did not produce the number of candidates requested by serial MOME"
                )
        self._record_timing(
            f"initializer.{specification.initializer.component_name}",
            perf_counter() - started,
        )

        self._specification = specification
        self._context = context
        self._archive = archive
        self._initial_candidates = initial_candidates
        self._bootstrap_cells = bootstrap_cells
        self._awaiting_candidates = ()
        self._awaiting_emitter_names = ()
        self._replay_pending = False
        self._evaluation_count = 0
        self._accepted_count = 0
        self._next_candidate_sequence = len(initial_candidates)

    def ask(self, count: int) -> Sequence[Candidate]:
        specification, context, archive = self._require_initialized()
        if count < 1:
            raise ValueError("ask count must be positive")
        if self._awaiting_candidates:
            if self._replay_pending:
                if count < len(self._awaiting_candidates):
                    raise ComponentError(
                        "ask count cannot be smaller than the restored pending batch"
                    )
                self._replay_pending = False
                return self._awaiting_candidates
            raise ComponentError("tell must be called before asking for another batch")

        remaining = specification.termination.remaining(self._evaluation_count)
        if remaining == 0:
            return ()
        requested = min(count, self.configured_batch_size)
        if remaining is not None:
            requested = min(requested, remaining)

        candidates = list(self._bootstrap_candidates(requested))
        emitter_names: list[str | None] = [None] * len(candidates)
        while self._initial_candidates and len(candidates) < requested:
            candidates.append(self._initial_candidates[0])
            self._initial_candidates = self._initial_candidates[1:]
            emitter_names.append(None)

        missing = requested - len(candidates)
        if missing:
            emitter_by_name = {
                emitter.component_name: emitter for emitter in specification.emitters
            }
            scheduler = specification.require_scheduler()
            started = perf_counter()
            allocations = tuple(scheduler.allocate(specification.emitters, missing))
            self._record_timing(
                f"scheduler.{scheduler.component_name}",
                perf_counter() - started,
            )
            if sum(allocation.count for allocation in allocations) != missing:
                raise ComponentError("scheduler allocations must exactly match the requested count")
            repertoire = archive.repertoire()
            for allocation in allocations:
                try:
                    emitter = emitter_by_name[allocation.emitter_name]
                except KeyError as error:
                    raise ComponentError(
                        f"scheduler selected unknown emitter {allocation.emitter_name!r}"
                    ) from error
                started = perf_counter()
                emitted = tuple(
                    emitter.emit(
                        context,
                        repertoire,
                        allocation.count,
                        start_sequence=self._next_candidate_sequence,
                    )
                )
                if len(emitted) != allocation.count:
                    raise ComponentError(
                        f"emitter {emitter.component_name!r} produced {len(emitted)} "
                        f"candidates; {allocation.count} were requested"
                    )
                for candidate in emitted:
                    if (
                        candidate.provenance is None
                        or candidate.provenance.producer != allocation.emitter_name
                    ):
                        raise ComponentError(
                            f"emitter {allocation.emitter_name!r} returned a candidate "
                            "without matching provenance"
                        )
                self._next_candidate_sequence += len(emitted)
                self._record_timing(
                    f"emitter.{emitter.component_name}",
                    perf_counter() - started,
                )
                if isinstance(emitter, ComponentTimingSource):
                    for name, duration in emitter.drain_component_timings():
                        self._record_timing(name, duration)
                candidates.extend(emitted)
                emitter_names.extend([allocation.emitter_name] * len(emitted))

        asked = tuple(candidates)
        identifiers = tuple(candidate.identifier for candidate in asked)
        if len(asked) != requested:
            raise ComponentError("search did not produce the requested candidate count")
        if len(set(identifiers)) != len(identifiers):
            raise ComponentError("candidate identifiers must be unique within an asked batch")
        started = perf_counter()
        for candidate in asked:
            context.candidate_validity_policy.validate(candidate, context.dataset)
        self._record_timing(
            f"validity.{context.candidate_validity_policy.component_name}",
            perf_counter() - started,
        )
        self._awaiting_candidates = asked
        self._awaiting_emitter_names = tuple(emitter_names)
        return asked

    def _bootstrap_candidates(self, count: int) -> tuple[Candidate, ...]:
        specification, context, _archive = self._require_initialized()
        initializer = specification.initializer
        if not isinstance(initializer, CellCoverageInitializer):
            return ()
        available = tuple(
            item
            for item in sorted(
                self._bootstrap_cells,
                key=lambda value: (value.attempts, value.target.coordinate.indices),
            )
            if not item.complete and not item.exhausted
        )[:count]
        if not available:
            return ()
        started = perf_counter()
        generated = tuple(
            initializer.initialize_bootstrap(
                context,
                available,
                start_sequence=self._next_candidate_sequence,
            )
        )
        self._record_timing(
            f"initializer.{initializer.component_name}",
            perf_counter() - started,
        )
        if len(generated) != len(available):
            raise ComponentError(
                "cell-coverage initializer must return one candidate per requested cell"
            )
        attempted = {item.target.coordinate.indices for item in available}
        self._bootstrap_cells = tuple(
            item.model_copy(update={"attempts": item.attempts + 1})
            if item.target.coordinate.indices in attempted
            else item
            for item in self._bootstrap_cells
        )
        self._next_candidate_sequence += len(generated)
        return generated

    def tell(self, evaluations: Sequence[Evaluation]) -> SearchUpdate:
        specification, _context, archive = self._require_initialized()
        if not self._awaiting_candidates:
            raise ComponentError("ask must be called before tell")
        awaiting_by_identifier = {
            candidate.identifier: candidate for candidate in self._awaiting_candidates
        }
        evaluated_by_identifier = {
            evaluation.candidate.identifier: evaluation.candidate for evaluation in evaluations
        }
        if (
            len(evaluated_by_identifier) != len(evaluations)
            or evaluated_by_identifier != awaiting_by_identifier
        ):
            raise ComponentError("tell evaluations must match the preceding asked batch exactly")

        started = perf_counter()
        outcomes = tuple(archive.add(evaluations))
        self._record_timing(
            f"archive.{archive.component_name}",
            perf_counter() - started,
        )
        if len(outcomes) != len(evaluations):
            raise ComponentError("archive did not return one insertion outcome per evaluation")
        accepted = sum(outcome.accepted for outcome in outcomes)
        self._evaluation_count += len(evaluations)
        self._accepted_count += accepted
        self._update_bootstrap(evaluations, outcomes)

        feedback = self._emitter_feedback(outcomes)
        scheduler = specification.require_scheduler()
        started = perf_counter()
        scheduler.update(feedback)
        reports = tuple(scheduler.reports(specification.emitters))
        self._record_timing(
            f"scheduler.{scheduler.component_name}",
            perf_counter() - started,
        )
        self._awaiting_candidates = ()
        self._awaiting_emitter_names = ()
        self._replay_pending = False
        return SearchUpdate(
            outcomes=outcomes,
            emitter_feedback=feedback,
            scheduler_reports=reports,
        )

    def _update_bootstrap(
        self,
        evaluations: Sequence[Evaluation],
        outcomes: Sequence[ArchiveInsertionOutcome],
    ) -> None:
        if not self._bootstrap_cells:
            return
        accepted: list[tuple[tuple[int, ...], PatternKind]] = []
        for evaluation, outcome in zip(evaluations, outcomes, strict=True):
            if not outcome.accepted:
                continue
            provenance = evaluation.candidate.provenance
            if (
                provenance is None
                or provenance.target_archive_coordinate is None
                or provenance.pattern_hint is None
                or outcome.coordinate is None
                or outcome.coordinate.indices != provenance.target_archive_coordinate
            ):
                continue
            accepted.append((provenance.target_archive_coordinate, provenance.pattern_hint))
        if not accepted:
            return
        updated: list[BootstrapCellState] = []
        remaining = list(accepted)
        for item in self._bootstrap_cells:
            match = next(
                (
                    value
                    for value in remaining
                    if value[0] == item.target.coordinate.indices
                    and value[1] == item.pending_pattern
                ),
                None,
            )
            if match is not None:
                remaining.remove(match)
                item = item.model_copy(
                    update={"accepted_patterns": (*item.accepted_patterns, match[1])}
                )
            updated.append(item)
        self._bootstrap_cells = tuple(updated)

    def _emitter_feedback(
        self,
        outcomes: Sequence[ArchiveInsertionOutcome],
    ) -> tuple[EmitterFeedback, ...]:
        grouped: dict[str, list[ArchiveInsertionOutcome]] = {}
        order: list[str] = []
        emitter_by_identifier = {
            candidate.identifier: emitter_name
            for candidate, emitter_name in zip(
                self._awaiting_candidates,
                self._awaiting_emitter_names,
                strict=True,
            )
        }
        for outcome in outcomes:
            emitter_name = emitter_by_identifier[outcome.candidate_identifier]
            if emitter_name is None:
                continue
            if emitter_name not in grouped:
                grouped[emitter_name] = []
                order.append(emitter_name)
            grouped[emitter_name].append(outcome)
        return tuple(
            EmitterFeedback(
                emitter_name=emitter_name,
                evaluated=len(items),
                accepted=sum(item.accepted for item in items),
                created_cells=sum(item.created_cell for item in items),
                evictions=sum(len(item.evicted_candidate_identifiers) for item in items),
                statuses=tuple(
                    sorted(
                        Counter(item.status for item in items).items(),
                        key=lambda item: item[0].value,
                    )
                ),
                cells=self._cell_feedback(items),
            )
            for emitter_name in order
            for items in (grouped[emitter_name],)
        )

    @staticmethod
    def _cell_feedback(
        outcomes: Sequence[ArchiveInsertionOutcome],
    ) -> tuple[EmitterCellFeedback, ...]:
        grouped: dict[tuple[int, ...] | None, list[ArchiveInsertionOutcome]] = {}
        for outcome in outcomes:
            coordinate = None if outcome.coordinate is None else outcome.coordinate.indices
            grouped.setdefault(coordinate, []).append(outcome)
        return tuple(
            EmitterCellFeedback(
                coordinate=(
                    None if coordinate is None else ArchiveCellCoordinate(indices=coordinate)
                ),
                evaluated=len(items),
                accepted=sum(item.accepted for item in items),
                created_cells=sum(item.created_cell for item in items),
            )
            for coordinate, items in sorted(
                grouped.items(),
                key=lambda item: () if item[0] is None else item[0],
            )
        )

    def finished(self) -> bool:
        if self._specification is None:
            return False
        return self._specification.termination.should_stop(self._evaluation_count)

    def result(self) -> Repertoire:
        _specification, _context, archive = self._require_initialized()
        if self._awaiting_candidates:
            raise ComponentError("cannot obtain a result while a batch is awaiting evaluation")
        return archive.repertoire()

    def progress(self) -> SearchProgress:
        _specification, _context, archive = self._require_initialized()
        return SearchProgress(
            evaluations=self._evaluation_count,
            accepted=self._accepted_count,
            rejected=self._evaluation_count - self._accepted_count,
            occupied_cells=archive.occupied_cell_count,
            repertoire_size=archive.repertoire_size,
        )

    def checkpoint(self) -> SearchCheckpoint:
        specification, context, archive = self._require_initialized()
        return SearchCheckpoint(
            engine_name=self.component_name,
            run_identifier=specification.run_identifier,
            dataset_identifier=context.dataset.metadata.identifier,
            search_fingerprint=specification.search_fingerprint,
            evaluation_count=self._evaluation_count,
            accepted_count=self._accepted_count,
            rejected_count=self._evaluation_count - self._accepted_count,
            next_candidate_sequence=self._next_candidate_sequence,
            integration_mode=specification.executor.integration_mode,
            initial_candidates=self._initial_candidates,
            pending_candidates=self._awaiting_candidates,
            pending_emitter_names=self._awaiting_emitter_names,
            repertoire=archive.repertoire(),
            random_stream_states=context.random_streams.snapshot(),
            scheduler_state=specification.require_scheduler().snapshot(),
            bootstrap_cells=self._bootstrap_cells,
        )

    def restore(self, checkpoint: SearchCheckpoint) -> None:
        specification, context, archive = self._require_initialized()
        if self._awaiting_candidates:
            raise ComponentError("a checkpoint cannot replace a batch awaiting evaluation")
        if (
            checkpoint.engine_name != self.component_name
            or checkpoint.run_identifier != specification.run_identifier
            or checkpoint.dataset_identifier != context.dataset.metadata.identifier
            or checkpoint.search_fingerprint != specification.search_fingerprint
            or checkpoint.integration_mode is not specification.executor.integration_mode
        ):
            raise ComponentError("checkpoint does not match the initialized SALVI run")
        if checkpoint.accepted_count + checkpoint.rejected_count != checkpoint.evaluation_count:
            raise ComponentError("checkpoint search counters are inconsistent")
        for candidate in (*checkpoint.initial_candidates, *checkpoint.pending_candidates):
            context.candidate_validity_policy.validate(candidate, context.dataset)
        for evaluation in checkpoint.repertoire.evaluations:
            context.candidate_validity_policy.validate(evaluation.candidate, context.dataset)

        archive.restore(checkpoint.repertoire)
        context.random_streams.restore(checkpoint.random_stream_states)
        specification.require_scheduler().restore(
            checkpoint.scheduler_state,
            specification.emitters,
        )
        self._initial_candidates = checkpoint.initial_candidates
        self._bootstrap_cells = checkpoint.bootstrap_cells
        self._awaiting_candidates = checkpoint.pending_candidates
        self._awaiting_emitter_names = checkpoint.pending_emitter_names
        self._replay_pending = bool(checkpoint.pending_candidates)
        self._evaluation_count = checkpoint.evaluation_count
        self._accepted_count = checkpoint.accepted_count
        self._next_candidate_sequence = checkpoint.next_candidate_sequence

    def drain_component_timings(self) -> tuple[tuple[str, float], ...]:
        timings = tuple(sorted(self._component_timings.items()))
        self._component_timings.clear()
        return timings

    def _record_timing(self, name: str, duration: float) -> None:
        self._component_timings[name] = self._component_timings.get(name, 0.0) + duration

    def _require_initialized(self) -> tuple[RunSpecification, QdRunContext, Archive]:
        if self._specification is None or self._context is None or self._archive is None:
            raise ComponentError("search engine is not initialized")
        return self._specification, self._context, self._archive


__all__ = ["SerialMomeConfiguration", "SerialMomeSearchEngine"]
