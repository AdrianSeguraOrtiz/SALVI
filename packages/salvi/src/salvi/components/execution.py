"""Bounded serial, shared-memory, and multi-process evaluation executors."""

from __future__ import annotations

import multiprocessing
import queue
from collections.abc import Callable, Sequence
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass
from time import perf_counter
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from salvi.application.context import NamedRandomStreams, RunContext
from salvi.components.protocols import (
    CancellationSignal,
    CandidateValidityPolicy,
    Constraint,
    Descriptor,
    EvaluationSupportPolicy,
    Objective,
)
from salvi.domain.enums import EvaluationIntegrationMode, EvaluationIssueCode
from salvi.domain.models import (
    Candidate,
    ConstraintValue,
    Evaluation,
    EvaluationIssue,
    NamedValue,
    ObjectiveValue,
)
from salvi.domain.prepared import PreparedDataset
from salvi.domain.search import EvaluationBatch
from salvi.evaluation.workspace import EvaluationWorkspace
from salvi.exceptions import ComponentError, EvaluationWorkerError, RunCancelledError
from salvi.patterns.configuration import PatternConfiguration


class ThreadPoolEvaluationConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    integration_mode: EvaluationIntegrationMode = EvaluationIntegrationMode.DETERMINISTIC
    max_in_flight: Annotated[int, Field(ge=1)] | None = None


class ProcessPoolEvaluationConfiguration(ThreadPoolEvaluationConfiguration):
    """Configuration shared by bounded CPU-oriented process workers."""


def _evaluate_candidate(
    candidate: Candidate,
    objectives: Sequence[Objective],
    descriptors: Sequence[Descriptor],
    workspace: EvaluationWorkspace,
    constraints: Sequence[Constraint] = (),
    cancellation: CancellationSignal | None = None,
    component_durations: dict[str, float] | None = None,
) -> Evaluation:
    if cancellation is not None:
        cancellation.raise_if_cancelled()
    validity_started = perf_counter() if component_durations is not None else None
    try:
        workspace.context.candidate_validity_policy.validate(candidate, workspace.context.dataset)
    except ComponentError as error:
        if component_durations is not None:
            assert validity_started is not None
            name = f"validity.{workspace.context.candidate_validity_policy.component_name}"
            component_durations[name] = component_durations.get(name, 0.0) + (
                perf_counter() - validity_started
            )
        return Evaluation(
            candidate=candidate,
            objectives=(),
            descriptors=(),
            issues=(
                EvaluationIssue(
                    code=EvaluationIssueCode.INVALID_STRUCTURE,
                    message=str(error),
                ),
            ),
        )
    if component_durations is not None:
        assert validity_started is not None
        name = f"validity.{workspace.context.candidate_validity_policy.component_name}"
        component_durations[name] = component_durations.get(name, 0.0) + (
            perf_counter() - validity_started
        )
    results = []
    for objective in objectives:
        started = perf_counter() if component_durations is not None else None
        results.append(workspace.objective(candidate, objective))
        if component_durations is not None:
            assert started is not None
            name = f"objective.{objective.component_name}"
            component_durations[name] = component_durations.get(name, 0.0) + (
                perf_counter() - started
            )
    constraint_results = []
    for constraint in constraints:
        started = perf_counter() if component_durations is not None else None
        constraint_results.append(constraint.evaluate(candidate, workspace))
        if component_durations is not None:
            assert started is not None
            name = f"constraint.{constraint.component_name}"
            component_durations[name] = component_durations.get(name, 0.0) + (
                perf_counter() - started
            )
    descriptor_results = []
    for descriptor in descriptors:
        started = perf_counter() if component_durations is not None else None
        descriptor_results.append(descriptor.describe(candidate, workspace))
        if component_durations is not None:
            assert started is not None
            name = f"descriptor.{descriptor.component_name}"
            component_durations[name] = component_durations.get(name, 0.0) + (
                perf_counter() - started
            )
    issues = tuple(issue for result in results for issue in result.issues)
    evaluation = Evaluation(
        candidate=candidate,
        objectives=tuple(
            ObjectiveValue(
                name=objective.component_name,
                value=result.value,
                direction=objective.direction,
                columns=result.columns,
                diagnostics=result.diagnostics,
            )
            for objective, result in zip(objectives, results, strict=True)
        ),
        descriptors=tuple(
            NamedValue(
                name=descriptor.component_name,
                value=value,
            )
            for descriptor, value in zip(descriptors, descriptor_results, strict=True)
        ),
        constraints=tuple(
            ConstraintValue(
                name=constraint.component_name,
                value=result.value,
                diagnostics=result.diagnostics,
            )
            for constraint, result in zip(constraints, constraint_results, strict=True)
        ),
        pattern_fit=workspace.cached_pattern_fit(candidate),
        issues=tuple(dict.fromkeys(issues)),
    )
    if cancellation is not None:
        cancellation.raise_if_cancelled()
    return evaluation


@dataclass(frozen=True, slots=True)
class _TimedEvaluation:
    evaluation: Evaluation
    duration_seconds: float
    component_duration_seconds: tuple[tuple[str, float], ...]


def _evaluate_timed_candidate(
    candidate: Candidate,
    objectives: Sequence[Objective],
    descriptors: Sequence[Descriptor],
    workspace: EvaluationWorkspace,
    constraints: Sequence[Constraint] = (),
    cancellation: CancellationSignal | None = None,
    collect_timings: bool = True,
) -> _TimedEvaluation:
    started = perf_counter() if collect_timings else None
    component_durations: dict[str, float] | None = {} if collect_timings else None
    evaluation = _evaluate_candidate(
        candidate,
        objectives,
        descriptors,
        workspace,
        constraints,
        cancellation,
        component_durations,
    )
    return _TimedEvaluation(
        evaluation=evaluation,
        duration_seconds=0.0 if started is None else perf_counter() - started,
        component_duration_seconds=(
            () if component_durations is None else tuple(sorted(component_durations.items()))
        ),
    )


@dataclass(frozen=True, slots=True)
class _BoundedEvaluationResult:
    completed: tuple[tuple[int, _TimedEvaluation], ...]
    completion_order: tuple[str, ...]
    peak_in_flight: int


def _collect_bounded_evaluations(
    candidates: Sequence[Candidate],
    *,
    limit: int,
    submit: Callable[[int, Candidate], Future[_TimedEvaluation]],
    cancellation: CancellationSignal | None,
) -> _BoundedEvaluationResult:
    completed: list[tuple[int, _TimedEvaluation]] = []
    completion_order: list[str] = []
    futures: dict[int, Future[_TimedEvaluation]] = {}
    notifications: queue.Queue[int] = queue.Queue(maxsize=limit)
    next_index = 0
    peak_in_flight = 0

    def submit_available() -> None:
        nonlocal next_index, peak_in_flight
        while next_index < len(candidates) and len(futures) < limit:
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            index = next_index
            next_index += 1
            future = submit(index, candidates[index])

            def notify_completion(
                _future: Future[_TimedEvaluation],
                completed_index: int = index,
            ) -> None:
                notifications.put(completed_index)

            future.add_done_callback(notify_completion)
            futures[index] = future
            peak_in_flight = max(peak_in_flight, len(futures))

    try:
        submit_available()
        while futures:
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            try:
                index = notifications.get(timeout=0.05)
            except queue.Empty:
                continue
            future = futures.pop(index)
            try:
                evaluation = future.result()
            except RunCancelledError:
                for pending in futures.values():
                    pending.cancel()
                raise
            except Exception as error:
                for pending in futures.values():
                    pending.cancel()
                candidate = candidates[index]
                raise EvaluationWorkerError(
                    f"evaluation worker failed for candidate {candidate.identifier!r}: {error}"
                ) from error
            completed.append((index, evaluation))
            completion_order.append(evaluation.evaluation.candidate.identifier)
            submit_available()
    except BaseException:
        for pending in futures.values():
            pending.cancel()
        raise

    return _BoundedEvaluationResult(
        completed=tuple(completed),
        completion_order=tuple(completion_order),
        peak_in_flight=peak_in_flight,
    )


def _timing_payload(
    timed: Sequence[_TimedEvaluation],
    *,
    collect_timings: bool,
) -> tuple[tuple[float, ...], tuple[tuple[str, float], ...]]:
    if not collect_timings:
        return (), ()
    components: dict[str, float] = {}
    for item in timed:
        for name, duration in item.component_duration_seconds:
            components[name] = components.get(name, 0.0) + duration
    return (
        tuple(item.duration_seconds for item in timed),
        tuple(sorted(components.items())),
    )


@dataclass(frozen=True, slots=True)
class SerialEvaluationExecutor:
    component_name: str = "serial"
    provides: frozenset[str] = frozenset({"evaluation"})
    requires: frozenset[str] = frozenset({"objective"})

    @property
    def integration_mode(self) -> EvaluationIntegrationMode:
        return EvaluationIntegrationMode.DETERMINISTIC

    @property
    def max_in_flight(self) -> int:
        return 1

    @property
    def uses_child_processes(self) -> bool:
        return False

    def validate_worker_count(self, worker_count: int) -> None:
        if worker_count != 1:
            raise ComponentError("serial evaluation requires exactly one worker")

    def evaluate(
        self,
        candidates: Sequence[Candidate],
        objectives: Sequence[Objective],
        descriptors: Sequence[Descriptor],
        workspace: EvaluationWorkspace,
        *,
        constraints: Sequence[Constraint] = (),
        worker_count: int = 1,
        cancellation: CancellationSignal | None = None,
        collect_timings: bool = True,
    ) -> EvaluationBatch:
        self.validate_worker_count(worker_count)
        started = perf_counter()
        timed: list[_TimedEvaluation] = []
        for candidate in candidates:
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            timed.append(
                _evaluate_timed_candidate(
                    candidate,
                    objectives,
                    descriptors,
                    workspace,
                    constraints,
                    cancellation,
                    collect_timings,
                )
            )
        evaluations = [item.evaluation for item in timed]
        candidate_timings, component_timings = _timing_payload(
            timed,
            collect_timings=collect_timings,
        )
        identifiers = tuple(item.candidate.identifier for item in evaluations)
        return EvaluationBatch(
            evaluations=tuple(evaluations),
            completion_order=identifiers,
            duration_seconds=perf_counter() - started,
            worker_count=1,
            peak_in_flight=1 if evaluations else 0,
            integration_mode=self.integration_mode,
            candidate_duration_seconds=candidate_timings,
            component_duration_seconds=component_timings,
        )

    def close(self) -> None:
        return None


@dataclass(slots=True)
class ThreadPoolEvaluationExecutor:
    """Evaluate a bounded batch against one shared, read-only prepared dataset."""

    configured_integration_mode: EvaluationIntegrationMode = EvaluationIntegrationMode.DETERMINISTIC
    configured_max_in_flight: int | None = None
    component_name: str = "thread_pool"
    provides: frozenset[str] = frozenset({"evaluation"})
    requires: frozenset[str] = frozenset({"objective"})
    _pool: ThreadPoolExecutor | None = None
    _worker_count: int | None = None

    def __post_init__(self) -> None:
        configuration = ThreadPoolEvaluationConfiguration(
            integration_mode=self.configured_integration_mode,
            max_in_flight=self.configured_max_in_flight,
        )
        self.configured_integration_mode = configuration.integration_mode
        self.configured_max_in_flight = configuration.max_in_flight

    @property
    def integration_mode(self) -> EvaluationIntegrationMode:
        return self.configured_integration_mode

    @property
    def max_in_flight(self) -> int | None:
        return self.configured_max_in_flight

    @property
    def uses_child_processes(self) -> bool:
        return False

    def validate_worker_count(self, worker_count: int) -> None:
        if worker_count < 1:
            raise ComponentError("thread-pool evaluation requires at least one worker")

    def evaluate(
        self,
        candidates: Sequence[Candidate],
        objectives: Sequence[Objective],
        descriptors: Sequence[Descriptor],
        workspace: EvaluationWorkspace,
        *,
        constraints: Sequence[Constraint] = (),
        worker_count: int = 1,
        cancellation: CancellationSignal | None = None,
        collect_timings: bool = True,
    ) -> EvaluationBatch:
        self.validate_worker_count(worker_count)
        started = perf_counter()
        if not candidates:
            return EvaluationBatch(
                evaluations=(),
                completion_order=(),
                duration_seconds=perf_counter() - started,
                worker_count=worker_count,
                peak_in_flight=0,
                integration_mode=self.integration_mode,
            )

        limit = min(
            len(candidates),
            self.configured_max_in_flight or worker_count,
        )
        active_worker_capacity = min(worker_count, limit)
        pool = self._ensure_pool(worker_count)
        bounded = _collect_bounded_evaluations(
            candidates,
            limit=limit,
            submit=lambda _index, candidate: pool.submit(
                _evaluate_timed_candidate,
                candidate,
                objectives,
                descriptors,
                workspace,
                constraints,
                cancellation,
                collect_timings,
            ),
            cancellation=cancellation,
        )

        ordered = (
            tuple(evaluation for _, evaluation in sorted(bounded.completed))
            if self.integration_mode is EvaluationIntegrationMode.DETERMINISTIC
            else tuple(evaluation for _, evaluation in bounded.completed)
        )
        candidate_timings, component_timings = _timing_payload(
            ordered,
            collect_timings=collect_timings,
        )
        return EvaluationBatch(
            evaluations=tuple(item.evaluation for item in ordered),
            completion_order=bounded.completion_order,
            duration_seconds=perf_counter() - started,
            worker_count=active_worker_capacity,
            peak_in_flight=bounded.peak_in_flight,
            integration_mode=self.integration_mode,
            candidate_duration_seconds=candidate_timings,
            component_duration_seconds=component_timings,
        )

    def _ensure_pool(self, worker_count: int) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="salvi-evaluation",
            )
            self._worker_count = worker_count
        elif self._worker_count != worker_count:
            raise ComponentError("thread-pool worker count cannot change during a run")
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
            self._worker_count = None


@dataclass(slots=True)
class _ProcessWorkerState:
    context: RunContext
    objectives: tuple[Objective, ...]
    descriptors: tuple[Descriptor, ...]
    constraints: tuple[Constraint, ...]
    batch_identifier: int = -1
    workspace: EvaluationWorkspace | None = None


_PROCESS_WORKER_STATE: _ProcessWorkerState | None = None


def _initialize_process_worker(
    dataset: PreparedDataset,
    patterns: PatternConfiguration,
    candidate_validity_policy: CandidateValidityPolicy,
    evaluation_support_policy: EvaluationSupportPolicy,
    objectives: tuple[Objective, ...],
    descriptors: tuple[Descriptor, ...],
    constraints: tuple[Constraint, ...],
) -> None:  # pragma: no cover - executed and verified in spawned worker processes
    global _PROCESS_WORKER_STATE
    context = RunContext(
        dataset=dataset,
        patterns=patterns,
        random_streams=NamedRandomStreams(0),
        candidate_validity_policy=candidate_validity_policy,
        evaluation_support_policy=evaluation_support_policy,
    )
    _PROCESS_WORKER_STATE = _ProcessWorkerState(
        context=context,
        objectives=objectives,
        descriptors=descriptors,
        constraints=constraints,
    )


def _evaluate_process_candidate(  # pragma: no cover - spawned worker entry point
    batch_identifier: int,
    candidate: Candidate,
    collect_timings: bool,
) -> _TimedEvaluation:
    state = _PROCESS_WORKER_STATE
    if state is None:
        raise RuntimeError("process evaluation worker is not initialized")
    if state.workspace is None or state.batch_identifier != batch_identifier:
        state.workspace = EvaluationWorkspace(state.context)
        state.batch_identifier = batch_identifier
    return _evaluate_timed_candidate(
        candidate,
        state.objectives,
        state.descriptors,
        state.workspace,
        state.constraints,
        collect_timings=collect_timings,
    )


def _evaluate_process_chunk(  # pragma: no cover - spawned worker entry point
    batch_identifier: int,
    indexed_candidates: tuple[tuple[int, Candidate], ...],
    collect_timings: bool,
) -> tuple[tuple[int, _TimedEvaluation], ...]:
    return tuple(
        (
            index,
            _evaluate_process_candidate(batch_identifier, candidate, collect_timings),
        )
        for index, candidate in indexed_candidates
    )


@dataclass(slots=True)
class ProcessPoolEvaluationExecutor:
    """Evaluate CPU-bound candidates in bounded persistent worker processes."""

    configured_integration_mode: EvaluationIntegrationMode = EvaluationIntegrationMode.DETERMINISTIC
    configured_max_in_flight: int | None = None
    component_name: str = "process_pool"
    provides: frozenset[str] = frozenset({"evaluation"})
    requires: frozenset[str] = frozenset({"objective"})
    _pool: ProcessPoolExecutor | None = None
    _worker_count: int | None = None
    _runtime_key: tuple[int, tuple[int, ...], tuple[int, ...], tuple[int, ...]] | None = None
    _batch_sequence: int = 0

    def __post_init__(self) -> None:
        configuration = ProcessPoolEvaluationConfiguration(
            integration_mode=self.configured_integration_mode,
            max_in_flight=self.configured_max_in_flight,
        )
        self.configured_integration_mode = configuration.integration_mode
        self.configured_max_in_flight = configuration.max_in_flight

    @property
    def integration_mode(self) -> EvaluationIntegrationMode:
        return self.configured_integration_mode

    @property
    def max_in_flight(self) -> int | None:
        return self.configured_max_in_flight

    @property
    def uses_child_processes(self) -> bool:
        return True

    def validate_worker_count(self, worker_count: int) -> None:
        if worker_count < 1:
            raise ComponentError("process-pool evaluation requires at least one worker")

    def evaluate(
        self,
        candidates: Sequence[Candidate],
        objectives: Sequence[Objective],
        descriptors: Sequence[Descriptor],
        workspace: EvaluationWorkspace,
        *,
        constraints: Sequence[Constraint] = (),
        worker_count: int = 1,
        cancellation: CancellationSignal | None = None,
        collect_timings: bool = True,
    ) -> EvaluationBatch:
        self.validate_worker_count(worker_count)
        started = perf_counter()
        if not candidates:
            return EvaluationBatch(
                evaluations=(),
                completion_order=(),
                duration_seconds=perf_counter() - started,
                worker_count=worker_count,
                peak_in_flight=0,
                integration_mode=self.integration_mode,
            )
        limit = min(
            len(candidates),
            self.configured_max_in_flight or worker_count,
        )
        active_worker_capacity = min(worker_count, limit)
        pool = self._ensure_pool(
            worker_count,
            workspace.context,
            tuple(objectives),
            tuple(descriptors),
            tuple(constraints),
        )
        batch_identifier = self._batch_sequence
        self._batch_sequence += 1
        if self.integration_mode is EvaluationIntegrationMode.DETERMINISTIC:
            completed, completion_order, peak_in_flight = self._evaluate_chunks(
                pool,
                batch_identifier,
                candidates,
                active_worker_capacity,
                limit,
                cancellation,
                collect_timings,
            )
            ordered = tuple(evaluation for _, evaluation in sorted(completed))
        else:
            bounded = _collect_bounded_evaluations(
                candidates,
                limit=limit,
                submit=lambda _index, candidate: pool.submit(
                    _evaluate_process_candidate,
                    batch_identifier,
                    candidate,
                    collect_timings,
                ),
                cancellation=cancellation,
            )
            ordered = tuple(evaluation for _, evaluation in bounded.completed)
            completion_order = bounded.completion_order
            peak_in_flight = bounded.peak_in_flight
        candidate_timings, component_timings = _timing_payload(
            ordered,
            collect_timings=collect_timings,
        )
        return EvaluationBatch(
            evaluations=tuple(item.evaluation for item in ordered),
            completion_order=completion_order,
            duration_seconds=perf_counter() - started,
            worker_count=active_worker_capacity,
            peak_in_flight=peak_in_flight,
            integration_mode=self.integration_mode,
            candidate_duration_seconds=candidate_timings,
            component_duration_seconds=component_timings,
        )

    @staticmethod
    def _evaluate_chunks(
        pool: ProcessPoolExecutor,
        batch_identifier: int,
        candidates: Sequence[Candidate],
        worker_count: int,
        max_candidates_in_flight: int,
        cancellation: CancellationSignal | None,
        collect_timings: bool,
    ) -> tuple[
        tuple[tuple[int, _TimedEvaluation], ...],
        tuple[str, ...],
        int,
    ]:
        target_tasks = max(1, min(len(candidates), worker_count * 2))
        chunk_size = min(
            max(1, (len(candidates) + target_tasks - 1) // target_tasks),
            max(1, max_candidates_in_flight // worker_count),
        )
        chunks = tuple(
            tuple(enumerate(candidates[start : start + chunk_size], start=start))
            for start in range(0, len(candidates), chunk_size)
        )
        maximum_pending_chunks = max(1, max_candidates_in_flight // chunk_size)
        futures: dict[
            Future[tuple[tuple[int, _TimedEvaluation], ...]],
            tuple[tuple[int, Candidate], ...],
        ] = {}
        completed: list[tuple[int, _TimedEvaluation]] = []
        completion_order: list[str] = []
        next_chunk = 0
        peak_in_flight = 0

        def submit_available() -> None:
            nonlocal next_chunk, peak_in_flight
            while next_chunk < len(chunks) and len(futures) < maximum_pending_chunks:
                if cancellation is not None:
                    cancellation.raise_if_cancelled()
                chunk = chunks[next_chunk]
                next_chunk += 1
                futures[
                    pool.submit(
                        _evaluate_process_chunk,
                        batch_identifier,
                        chunk,
                        collect_timings,
                    )
                ] = chunk
                peak_in_flight = max(
                    peak_in_flight,
                    sum(len(pending_chunk) for pending_chunk in futures.values()),
                )

        try:
            submit_available()
            while futures:
                if cancellation is not None:
                    cancellation.raise_if_cancelled()
                done, _pending = wait(
                    futures,
                    timeout=0.05,
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    chunk = futures.pop(future)
                    try:
                        results = future.result()
                    except Exception as error:
                        first = chunk[0][1]
                        raise EvaluationWorkerError(
                            f"evaluation worker failed for candidate {first.identifier!r}: {error}"
                        ) from error
                    completed.extend(results)
                    completion_order.extend(
                        result.evaluation.candidate.identifier for _, result in results
                    )
                submit_available()
        except BaseException:
            for future in futures:
                future.cancel()
            raise
        return (
            tuple(completed),
            tuple(completion_order),
            peak_in_flight,
        )

    def _ensure_pool(
        self,
        worker_count: int,
        context: RunContext,
        objectives: tuple[Objective, ...],
        descriptors: tuple[Descriptor, ...],
        constraints: tuple[Constraint, ...],
    ) -> ProcessPoolExecutor:
        runtime_key = (
            id(context),
            tuple(id(objective) for objective in objectives),
            tuple(id(descriptor) for descriptor in descriptors),
            tuple(id(constraint) for constraint in constraints),
        )
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_process_worker,
                initargs=(
                    context.dataset,
                    context.patterns,
                    context.candidate_validity_policy,
                    context.evaluation_support_policy,
                    objectives,
                    descriptors,
                    constraints,
                ),
            )
            self._worker_count = worker_count
            self._runtime_key = runtime_key
        elif self._worker_count != worker_count or self._runtime_key != runtime_key:
            raise ComponentError("process-pool runtime cannot change during a run")
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
            self._worker_count = None
            self._runtime_key = None


__all__ = [
    "ProcessPoolEvaluationConfiguration",
    "ProcessPoolEvaluationExecutor",
    "SerialEvaluationExecutor",
    "ThreadPoolEvaluationConfiguration",
    "ThreadPoolEvaluationExecutor",
]
