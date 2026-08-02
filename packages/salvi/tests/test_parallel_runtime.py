from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from salvi.application.run_service import CancellationToken, RunService
from salvi.components import diagnostic_observers as observer_module
from salvi.components.descriptors import RowCardinality
from salvi.components.execution import (
    ProcessPoolEvaluationExecutor,
    SerialEvaluationExecutor,
    ThreadPoolEvaluationExecutor,
)
from salvi.components.objectives import Contrast, InternalCoherence
from salvi.components.observers import ResourceUsageObserver, RuntimeThroughputObserver
from salvi.domain import (
    Bicluster,
    BinningStrategy,
    Candidate,
    EvaluationBatch,
    EvaluationIntegrationMode,
    EventType,
    ObjectiveDirection,
    ObjectiveResult,
    RunEvent,
    SearchCheckpoint,
)
from salvi.evaluation.workspace import EvaluationWorkspace
from salvi.exceptions import (
    ArtifactError,
    ComponentError,
    EvaluationWorkerError,
    RunCancelledError,
)
from salvi.infrastructure.events import SQLiteRunEventSource
from salvi.infrastructure.files import atomic_write_text

from .conftest import configuration_mapping, create_dataset_bundle, write_configuration


@dataclass(frozen=True, slots=True)
class DelayedObjective:
    delays: dict[str, float]
    failures: frozenset[str] = frozenset()
    component_name: str = "delayed"
    direction: ObjectiveDirection = ObjectiveDirection.MINIMIZE
    provides: frozenset[str] = frozenset({"objective"})
    requires: frozenset[str] = frozenset()

    def evaluate(
        self,
        candidate: Candidate,
        workspace: EvaluationWorkspace,
    ) -> ObjectiveResult:
        del workspace
        time.sleep(self.delays.get(candidate.identifier, 0.0))
        if candidate.identifier in self.failures:
            raise RuntimeError("deliberate worker failure")
        return ObjectiveResult(
            value=float(candidate.bicluster.row_indices[0]),
            columns=(),
        )


def _candidates() -> tuple[Candidate, ...]:
    return (
        Candidate(
            identifier="first",
            bicluster=Bicluster(row_indices=(0, 1), column_indices=(0, 1)),
        ),
        Candidate(
            identifier="second",
            bicluster=Bicluster(row_indices=(1, 2), column_indices=(0, 1)),
        ),
        Candidate(
            identifier="third",
            bicluster=Bicluster(row_indices=(2, 3), column_indices=(0, 1)),
        ),
    )


def _parallel_mapping(
    dataset: Path,
    output: Path,
    *,
    integration_mode: EvaluationIntegrationMode = EvaluationIntegrationMode.DETERMINISTIC,
) -> dict[str, object]:
    mapping = configuration_mapping(dataset, output, overwrite=True)
    mapping["search"] = {
        "engine": {
            "name": "serial_mome",
            "parameters": {"initial_population_size": 4, "batch_size": 2},
        },
        "objectives": [
            {"name": "internal_coherence", "parameters": {}},
            {"name": "contrast", "parameters": {"min_background_ratio": 0.1}},
        ],
        "descriptors": [
            {"name": "row_cardinality", "parameters": {}},
            {"name": "column_cardinality", "parameters": {}},
        ],
        "archive": {
            "name": "deep_grid_mome",
            "parameters": {
                "axes": [
                    {
                        "descriptor": "row_cardinality",
                        "binning": BinningStrategy.EXACT.value,
                    },
                    {
                        "descriptor": "column_cardinality",
                        "binning": BinningStrategy.EXACT.value,
                    },
                ],
                "cell_capacity": 3,
            },
        },
        "parent_selection": {"name": "repertoire_uniform", "parameters": {}},
        "initialization": {"name": "uniform_random", "parameters": {}},
        "emitters": [{"name": "random_move", "parameters": {}}],
        "scheduler": {"name": "first", "parameters": {}},
        "termination": {
            "name": "evaluation_budget",
            "parameters": {"max_evaluations": 8},
        },
    }
    mapping["execution"] = {
        "executor": {
            "name": "thread_pool",
            "parameters": {
                "integration_mode": integration_mode.value,
                "max_in_flight": 2,
            },
        },
        "workers": 2,
        "cancellation_grace_seconds": 1,
    }
    mapping["monitoring"] = {
        "queue_capacity": 64,
        "checkpoint_interval_evaluations": 2,
        "observers": [
            {"name": "runtime_throughput", "parameters": {}},
            {"name": "resource_usage", "parameters": {"every_evaluations": 2}},
            {"name": "component_timing", "parameters": {"every_evaluations": 2}},
        ],
    }
    mapping["final_selection"] = None
    return mapping


def _scientific_state(result: object) -> tuple[tuple[object, ...], ...]:
    from salvi.domain import RunResult

    assert isinstance(result, RunResult)
    return tuple(
        (
            evaluation.candidate.identifier,
            evaluation.candidate.bicluster.signature,
            evaluation.objectives,
            evaluation.descriptors,
        )
        for evaluation in result.repertoire.evaluations
    )


def test_deterministic_parallel_evaluation_matches_serial_science(run_context) -> None:
    candidates = _candidates()
    objectives = (InternalCoherence(), Contrast(min_background_ratio=0.1))
    descriptors = (RowCardinality(),)
    serial = SerialEvaluationExecutor().evaluate(
        candidates,
        objectives,
        descriptors,
        EvaluationWorkspace(run_context),
    )
    parallel_executor = ThreadPoolEvaluationExecutor(
        configured_integration_mode=EvaluationIntegrationMode.DETERMINISTIC,
        configured_max_in_flight=2,
    )
    try:
        parallel = parallel_executor.evaluate(
            candidates,
            objectives,
            descriptors,
            EvaluationWorkspace(run_context),
            worker_count=3,
        )
    finally:
        parallel_executor.close()

    assert parallel.evaluations == serial.evaluations
    assert parallel.peak_in_flight == 2
    assert parallel.integration_mode is EvaluationIntegrationMode.DETERMINISTIC
    assert len(parallel.candidate_duration_seconds) == len(candidates)
    assert {name for name, _duration in parallel.component_duration_seconds} >= {
        "objective.internal_coherence",
        "objective.contrast",
        "descriptor.row_cardinality",
    }


def test_executor_skips_optional_component_instrumentation(run_context) -> None:
    batch = SerialEvaluationExecutor().evaluate(
        _candidates(),
        (InternalCoherence(), Contrast(min_background_ratio=0.1)),
        (RowCardinality(),),
        EvaluationWorkspace(run_context),
        collect_timings=False,
    )

    assert batch.evaluations
    assert batch.duration_seconds >= 0.0
    assert batch.candidate_duration_seconds == ()
    assert batch.component_duration_seconds == ()


def test_deterministic_process_evaluation_matches_serial_science(run_context) -> None:
    candidates = _candidates()
    objectives = (InternalCoherence(), Contrast(min_background_ratio=0.1))
    descriptors = (RowCardinality(),)
    serial = SerialEvaluationExecutor().evaluate(
        candidates,
        objectives,
        descriptors,
        EvaluationWorkspace(run_context),
    )
    process_executor = ProcessPoolEvaluationExecutor(configured_max_in_flight=2)
    try:
        process = process_executor.evaluate(
            candidates,
            objectives,
            descriptors,
            EvaluationWorkspace(run_context),
            worker_count=2,
        )
        repeated = process_executor.evaluate(
            candidates,
            objectives,
            descriptors,
            EvaluationWorkspace(run_context),
            worker_count=2,
        )
        with pytest.raises(ComponentError, match="runtime cannot change"):
            process_executor.evaluate(
                candidates,
                (InternalCoherence(), Contrast(min_background_ratio=0.1)),
                descriptors,
                EvaluationWorkspace(run_context),
                worker_count=2,
            )
    finally:
        process_executor.close()

    assert process.evaluations == serial.evaluations
    assert repeated.evaluations == serial.evaluations
    assert process.peak_in_flight == 2
    assert process.worker_count == 2


def test_parallel_modes_expose_completion_order_and_bound_in_flight_work(run_context) -> None:
    candidates = _candidates()
    objective = DelayedObjective({"first": 0.08, "second": 0.005, "third": 0.005})
    deterministic_executor = ThreadPoolEvaluationExecutor(
        configured_integration_mode=EvaluationIntegrationMode.DETERMINISTIC,
        configured_max_in_flight=2,
    )
    throughput_executor = ThreadPoolEvaluationExecutor(
        configured_integration_mode=EvaluationIntegrationMode.THROUGHPUT,
        configured_max_in_flight=2,
    )
    try:
        deterministic = deterministic_executor.evaluate(
            candidates,
            (objective,),
            (RowCardinality(),),
            EvaluationWorkspace(run_context),
            worker_count=3,
        )
        throughput = throughput_executor.evaluate(
            candidates,
            (objective,),
            (RowCardinality(),),
            EvaluationWorkspace(run_context),
            worker_count=3,
        )
    finally:
        deterministic_executor.close()
        throughput_executor.close()

    submitted = tuple(candidate.identifier for candidate in candidates)
    assert tuple(item.candidate.identifier for item in deterministic.evaluations) == submitted
    assert deterministic.completion_order != submitted
    assert (
        tuple(item.candidate.identifier for item in throughput.evaluations)
        == throughput.completion_order
    )
    assert throughput.peak_in_flight == 2


def test_parallel_worker_failure_and_cancellation_are_typed(run_context) -> None:
    candidates = _candidates()
    failing = ThreadPoolEvaluationExecutor(configured_max_in_flight=2)
    with pytest.raises(EvaluationWorkerError, match="second"):
        failing.evaluate(
            candidates,
            (DelayedObjective({}, frozenset({"second"})),),
            (RowCardinality(),),
            EvaluationWorkspace(run_context),
            worker_count=2,
        )
    failing.close()

    token = CancellationToken()
    cancelling = ThreadPoolEvaluationExecutor(configured_max_in_flight=2)
    errors: list[BaseException] = []

    def evaluate() -> None:
        try:
            cancelling.evaluate(
                candidates,
                (DelayedObjective({candidate.identifier: 0.15 for candidate in candidates}),),
                (RowCardinality(),),
                EvaluationWorkspace(run_context),
                worker_count=2,
                cancellation=token,
            )
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=evaluate)
    thread.start()
    time.sleep(0.02)
    token.cancel()
    thread.join(timeout=2)
    cancelling.close()

    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], RunCancelledError)

    process = ProcessPoolEvaluationExecutor(configured_max_in_flight=2)
    try:
        with pytest.raises(EvaluationWorkerError, match="second"):
            process.evaluate(
                candidates,
                (DelayedObjective({}, frozenset({"second"})),),
                (RowCardinality(),),
                EvaluationWorkspace(run_context),
                worker_count=2,
            )
    finally:
        process.close()


def test_executor_contract_guards_empty_batches_and_pool_reuse(run_context) -> None:
    workspace = EvaluationWorkspace(run_context)
    invalid_candidate = Candidate(
        identifier="out-of-bounds",
        bicluster=Bicluster(row_indices=(0, 100), column_indices=(0, 1)),
    )
    invalid = SerialEvaluationExecutor().evaluate(
        (invalid_candidate,),
        (DelayedObjective({}),),
        (RowCardinality(),),
        workspace,
    )
    assert not invalid[0].valid
    assert invalid[0].issues[0].code.value == "INVALID_STRUCTURE"

    with pytest.raises(ComponentError, match="exactly one worker"):
        SerialEvaluationExecutor().evaluate(
            (),
            (DelayedObjective({}),),
            (RowCardinality(),),
            workspace,
            worker_count=2,
        )

    executor = ThreadPoolEvaluationExecutor(configured_max_in_flight=2)
    with pytest.raises(ComponentError, match="at least one worker"):
        executor.evaluate(
            (),
            (DelayedObjective({}),),
            (RowCardinality(),),
            workspace,
            worker_count=0,
        )
    empty = executor.evaluate(
        (),
        (DelayedObjective({}),),
        (RowCardinality(),),
        workspace,
        worker_count=2,
    )
    assert len(empty) == 0
    assert empty[:] == ()
    executor.evaluate(
        _candidates()[:1],
        (DelayedObjective({}),),
        (RowCardinality(),),
        workspace,
        worker_count=2,
    )
    with pytest.raises(ComponentError, match="cannot change"):
        executor.evaluate(
            _candidates()[:1],
            (DelayedObjective({}),),
            (RowCardinality(),),
            workspace,
            worker_count=1,
        )
    executor.close()
    executor.close()

    process = ProcessPoolEvaluationExecutor(configured_max_in_flight=2)
    with pytest.raises(ComponentError, match="at least one worker"):
        process.evaluate(
            (),
            (DelayedObjective({}),),
            (RowCardinality(),),
            workspace,
            worker_count=0,
        )
    process_empty = process.evaluate(
        (),
        (DelayedObjective({}),),
        (RowCardinality(),),
        workspace,
        worker_count=2,
    )
    assert len(process_empty) == 0
    process.close()


def test_evaluation_batch_rejects_ambiguous_runtime_reports(run_context) -> None:
    evaluation = SerialEvaluationExecutor().evaluate(
        _candidates()[:1],
        (DelayedObjective({}),),
        (RowCardinality(),),
        EvaluationWorkspace(run_context),
    )[0]
    common = {
        "duration_seconds": 0.0,
        "worker_count": 1,
        "integration_mode": EvaluationIntegrationMode.DETERMINISTIC,
    }
    with pytest.raises(ValueError, match="unique candidate identifiers"):
        EvaluationBatch(
            evaluations=(evaluation, evaluation),
            completion_order=("first", "first"),
            peak_in_flight=1,
            **common,
        )
    with pytest.raises(ValueError, match="every evaluated candidate"):
        EvaluationBatch(
            evaluations=(evaluation,),
            completion_order=("unknown",),
            peak_in_flight=1,
            **common,
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        EvaluationBatch(
            evaluations=(evaluation,),
            completion_order=("first",),
            peak_in_flight=2,
            **common,
        )


def test_runtime_and_resource_observers_handle_sparse_runtime_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    throughput = RuntimeThroughputObserver()
    assert throughput.on_event(RunEvent(event_type=EventType.PROGRESS, payload={})) == ()
    assert (
        throughput.on_event(
            RunEvent(event_type=EventType.CANDIDATES_EVALUATED, payload={"evaluations": 1})
        )
        == ()
    )
    samples = throughput.on_event(
        RunEvent(
            event_type=EventType.CANDIDATES_EVALUATED,
            payload={
                "evaluations": 2,
                "count": 2,
                "runtime": {
                    "duration_seconds": 0.0,
                    "worker_count": 2,
                    "peak_in_flight": 2,
                },
            },
        )
    )
    assert {sample.name: sample.value for sample in samples}[
        "runtime.evaluations_per_second"
    ] == 0.0

    resources = ResourceUsageObserver(every_evaluations=2)
    assert resources.on_event(RunEvent(event_type=EventType.RUN_STARTED, payload={})) == ()
    assert (
        resources.on_event(RunEvent(event_type=EventType.PROGRESS, payload={"evaluations": 1}))
        == ()
    )
    monkeypatch.setattr(observer_module, "_resident_memory_bytes", lambda: None)
    without_memory = resources.on_event(
        RunEvent(event_type=EventType.PROGRESS, payload={"evaluations": 2})
    )
    assert {sample.name for sample in without_memory} == {
        "resource.process_cpu_seconds",
        "resource.interval_cpu_percent",
        "resource.active_threads",
    }
    monkeypatch.setattr(observer_module, "_resident_memory_bytes", lambda: 4096)
    with_memory = resources.on_event(
        RunEvent(event_type=EventType.PROGRESS, payload={"evaluations": 4})
    )
    assert {sample.name: sample.value for sample in with_memory}[
        "resource.resident_memory_bytes"
    ] == 4096


def test_atomic_text_write_never_leaves_a_temporary_file_on_failure(tmp_path: Path) -> None:
    destination = tmp_path / "checkpoint.json"
    atomic_write_text(destination, "first")
    atomic_write_text(destination, "second")
    assert destination.read_text() == "second"

    invalid_destination = tmp_path / "directory"
    invalid_destination.mkdir()
    with pytest.raises(ArtifactError, match="cannot atomically write"):
        atomic_write_text(invalid_destination, "content")
    assert not (tmp_path / ".directory.tmp").exists()


def test_failed_batch_writes_resumable_pending_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = create_dataset_bundle(tmp_path / "dataset", rows=8, columns=3)
    failed_path = write_configuration(
        tmp_path / "failed.yaml",
        _parallel_mapping(dataset, tmp_path / "failed-output"),
    )
    original_evaluate = ThreadPoolEvaluationExecutor.evaluate

    def fail_batch(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        del self, args, kwargs
        raise EvaluationWorkerError("deliberate batch failure")

    monkeypatch.setattr(ThreadPoolEvaluationExecutor, "evaluate", fail_batch)
    with pytest.raises(EvaluationWorkerError, match="deliberate"):
        RunService().run(failed_path)
    monkeypatch.setattr(ThreadPoolEvaluationExecutor, "evaluate", original_evaluate)

    metadata = json.loads((tmp_path / "failed-output" / "run-metadata.json").read_text())
    recovery_path = Path(metadata["recovery_checkpoint"])
    checkpoint = SearchCheckpoint.model_validate_json(recovery_path.read_text())
    assert checkpoint.pending_candidates
    assert checkpoint.integration_mode is EvaluationIntegrationMode.DETERMINISTIC

    resumed_mapping = _parallel_mapping(dataset, tmp_path / "resumed-output")
    resumed_mapping["run"]["resume_from_checkpoint"] = str(recovery_path)
    resumed = RunService().run(write_configuration(tmp_path / "resumed.yaml", resumed_mapping))
    uninterrupted = RunService().run(
        write_configuration(
            tmp_path / "uninterrupted.yaml",
            _parallel_mapping(dataset, tmp_path / "uninterrupted-output"),
        )
    )

    assert _scientific_state(resumed) == _scientific_state(uninterrupted)
    source = SQLiteRunEventSource(resumed.event_store)
    metrics = source.poll_metrics(limit=10_000)
    names = {metric.name for metric in metrics}
    assert "runtime.evaluations_per_second" in names
    assert "resource.process_cpu_seconds" in names
    assert "timing.component.initializer.uniform_random.seconds" in names
    assert "timing.component.archive.deep_grid_mome.seconds" in names
    assert "timing.scientific.objective.internal_coherence.seconds" in names
    assert "timing.observer.component_timing.seconds" in names


def test_full_deterministic_parallel_run_matches_serial_and_records_runtime(
    tmp_path: Path,
) -> None:
    dataset = create_dataset_bundle(tmp_path / "dataset", rows=8, columns=3)
    parallel_mapping = _parallel_mapping(dataset, tmp_path / "parallel-output")
    serial_mapping = _parallel_mapping(dataset, tmp_path / "serial-output")
    serial_mapping["execution"] = {
        "executor": {"name": "serial", "parameters": {}},
        "workers": 1,
        "cancellation_grace_seconds": 1,
    }

    parallel = RunService().run(write_configuration(tmp_path / "parallel.yaml", parallel_mapping))
    serial = RunService().run(write_configuration(tmp_path / "serial.yaml", serial_mapping))

    assert _scientific_state(parallel) == _scientific_state(serial)
    metadata = json.loads((parallel.output_directory / "run-metadata.json").read_text())
    assert metadata["runtime"] == {
        "executor": "thread_pool",
        "workers": 2,
        "integration_mode": "DETERMINISTIC",
        "max_in_flight": 2,
        "reproducibility": "submission-order deterministic",
    }


def test_full_deterministic_process_run_matches_serial(tmp_path: Path) -> None:
    dataset = create_dataset_bundle(tmp_path / "dataset", rows=8, columns=3)
    process_mapping = _parallel_mapping(dataset, tmp_path / "process-output")
    process_mapping["execution"]["executor"]["name"] = "process_pool"
    serial_mapping = _parallel_mapping(dataset, tmp_path / "serial-output")
    serial_mapping["execution"] = {
        "executor": {"name": "serial", "parameters": {}},
        "workers": 1,
        "cancellation_grace_seconds": 1,
    }

    process = RunService().run(write_configuration(tmp_path / "process.yaml", process_mapping))
    serial = RunService().run(write_configuration(tmp_path / "serial.yaml", serial_mapping))

    assert _scientific_state(process) == _scientific_state(serial)
    metadata = json.loads((process.output_directory / "run-metadata.json").read_text())
    assert metadata["runtime"]["executor"] == "process_pool"
    assert metadata["runtime"]["integration_mode"] == "DETERMINISTIC"


def test_throughput_integration_is_explicit_in_metadata_and_checkpoint(
    tmp_path: Path,
) -> None:
    dataset = create_dataset_bundle(tmp_path / "dataset", rows=8, columns=3)
    mapping = _parallel_mapping(
        dataset,
        tmp_path / "output",
        integration_mode=EvaluationIntegrationMode.THROUGHPUT,
    )
    result = RunService().run(write_configuration(tmp_path / "throughput.yaml", mapping))

    metadata = json.loads((result.output_directory / "run-metadata.json").read_text())
    assert metadata["runtime"]["integration_mode"] == "THROUGHPUT"
    assert metadata["runtime"]["reproducibility"] == "completion-order dependent"
    checkpoint = SearchCheckpoint.model_validate_json(
        (result.output_directory / "checkpoints" / "checkpoint-000000000008.json").read_text()
    )
    assert checkpoint.integration_mode is EvaluationIntegrationMode.THROUGHPUT
