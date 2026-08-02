"""Single execution gateway shared by the API, CLI, and GUI."""

from __future__ import annotations

import json
import shutil
import threading
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from salvi.api.run import RunSpecification
from salvi.application.configuration import (
    LoadedConfiguration,
    LoadedRunConfiguration,
    PipelineConfiguration,
    RunBinding,
    SalviConfiguration,
    bind_pipeline,
    load_bound_configuration,
    load_configuration,
    write_effective_configuration,
)
from salvi.application.event_payloads import (
    collect_payload_requirements,
    evaluation_batch_payload,
    runtime_payload,
)
from salvi.application.factory import (
    build_specification,
    prepare_run,
    resolve_component_defaults,
)
from salvi.components.defaults import default_component_registry
from salvi.components.protocols import (
    CancellationSignal,
    ComponentTimingSource,
    EventPayloadRequirement,
    SearchEngine,
)
from salvi.components.registry import ComponentRegistry
from salvi.domain.enums import EventType, RunStatus
from salvi.domain.models import RunEvent, RunResult
from salvi.domain.search import SearchCheckpoint
from salvi.evaluation.workspace import EvaluationWorkspace
from salvi.exceptions import (
    RunCancelledError,
    RunError,
)
from salvi.infrastructure.bicluster_set import BiclusterSetWriter
from salvi.infrastructure.events import EventPublisher, SQLiteEventStore
from salvi.infrastructure.files import atomic_write_text, sha256_file
from salvi.versioning import package_version


class CancellationToken:
    """Cooperative cancellation shared across application adapters."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise RunCancelledError("run cancelled")


class RunService:
    """Validates, composes, and executes SALVI runs."""

    def __init__(self, registry: ComponentRegistry | None = None) -> None:
        self._registry = registry

    def validate(self, path: str | Path) -> LoadedConfiguration:
        """Validate a fully bound effective configuration artifact."""

        loaded = load_configuration(path)
        self._validate_configuration(loaded.configuration)
        return loaded

    def validate_pipeline(
        self,
        path: str | Path,
        binding: RunBinding,
    ) -> LoadedRunConfiguration:
        """Validate a reusable pipeline after binding it to one concrete run."""

        loaded = load_bound_configuration(path, binding)
        self._validate_configuration(loaded.configuration)
        return loaded

    def validate_pipeline_configuration(
        self,
        configuration: PipelineConfiguration,
        binding: RunBinding,
    ) -> SalviConfiguration:
        """Validate an in-memory reusable pipeline against one concrete binding."""

        effective = bind_pipeline(configuration, binding)
        self._validate_configuration(effective)
        return effective

    def _validate_configuration(self, configuration: SalviConfiguration) -> None:
        specification = build_specification(configuration, self._registry)
        try:
            prepared = prepare_run(specification)
            if specification.archive is not None:
                specification.archive.initialize(
                    prepared.context,
                    specification.objectives,
                    specification.descriptors,
                    specification.constraints,
                )
        finally:
            specification.executor.close()

    def run(
        self,
        path: str | Path,
        *,
        cancellation: CancellationSignal | None = None,
    ) -> RunResult:
        return self._execute_loaded(
            load_configuration(path),
            cancellation=cancellation,
        )

    def run_pipeline(
        self,
        path: str | Path,
        binding: RunBinding,
        *,
        cancellation: CancellationSignal | None = None,
    ) -> RunResult:
        return self._execute_loaded(
            load_bound_configuration(path, binding),
            cancellation=cancellation,
        )

    def _execute_loaded(
        self,
        loaded: LoadedConfiguration | LoadedRunConfiguration,
        *,
        cancellation: CancellationSignal | None,
    ) -> RunResult:
        token = cancellation or CancellationToken()
        active_registry = self._registry or default_component_registry()
        configuration = resolve_component_defaults(loaded.configuration, active_registry)
        resume_path = configuration.run.resume_from_checkpoint
        if (
            resume_path is not None
            and configuration.output.overwrite
            and self._is_within(resume_path, configuration.output.directory)
        ):
            raise RunError("a checkpoint cannot be resumed from an output directory being replaced")
        output = self._prepare_output(
            configuration.output.directory, configuration.output.overwrite
        )
        effective_path = output / "effective-configuration.yaml"
        metadata_path = output / "run-metadata.json"
        event_store_path = output / "run.sqlite"
        write_effective_configuration(configuration, effective_path)
        started_at = datetime.now(UTC)
        run_started_clock = perf_counter()
        metadata: dict[str, Any] = {
            "schema_version": 2,
            "salvi_version": package_version(),
            "run_identifier": configuration.run.identifier,
            "status": RunStatus.RUNNING.value,
            "seed": configuration.run.seed,
            "started_at": started_at.isoformat(),
            "finished_at": None,
            "configuration": configuration.model_dump(mode="json"),
        }
        self._write_metadata(metadata_path, metadata)
        store = SQLiteEventStore(event_store_path)
        publisher: EventPublisher | None = None
        engine: SearchEngine | None = None
        specification: RunSpecification | None = None

        try:
            specification = build_specification(configuration, active_registry)
            metadata["runtime"] = {
                "executor": specification.executor.component_name,
                "workers": specification.worker_count,
                "integration_mode": specification.executor.integration_mode.value,
                "max_in_flight": specification.executor.max_in_flight,
                "reproducibility": (
                    "submission-order deterministic"
                    if specification.executor.integration_mode.value == "DETERMINISTIC"
                    else "completion-order dependent"
                ),
            }
            metadata["monitoring"] = {
                "observers": [observer.component_name for observer in specification.observers],
                "termination": specification.termination.progress(0).model_dump(mode="json"),
                "archive_axes": [
                    descriptor.component_name for descriptor in specification.descriptors
                ],
            }
            self._write_metadata(metadata_path, metadata)
            payload_requirements = collect_payload_requirements(specification.observers)
            publisher = EventPublisher(
                store,
                specification.observers,
                capacity=configuration.monitoring.queue_capacity,
            )
            self._publish(
                publisher,
                EventType.RUN_STARTED,
                {
                    "run_identifier": configuration.run.identifier,
                    "termination": specification.termination.progress(0).model_dump(mode="json"),
                },
            )
            self._publish(
                publisher,
                EventType.CONFIGURATION_VALIDATED,
                {"source": str(loaded.source)},
            )
            token.raise_if_cancelled()
            self._publish(
                publisher,
                EventType.DATASET_VALIDATED,
                {
                    "identifier": specification.dataset.identifier,
                    "rows": specification.dataset.row_count,
                    "columns": specification.dataset.column_count,
                },
            )
            self._publish(
                publisher,
                EventType.COMPONENTS_BUILT,
                {
                    "objectives": len(specification.objectives),
                    "constraints": len(specification.constraints),
                    "descriptors": len(specification.descriptors),
                    "emitters": len(specification.emitters),
                },
            )
            token.raise_if_cancelled()

            prepared = prepare_run(specification)
            metadata["preprocessing"] = asdict(prepared.preprocessing)
            self._write_metadata(metadata_path, metadata)
            prepared_payload: dict[str, Any] = {
                "identifier": prepared.context.dataset.metadata.identifier,
                "stages": len(prepared.preprocessing.steps),
                "source_columns": prepared.context.dataset.source_column_count,
                "prepared_columns": prepared.context.dataset.column_count,
                "source_missing_values": prepared.context.dataset.missing_count,
                "unavailable_values": prepared.context.dataset.unavailable_count,
                "imputed_values": prepared.context.dataset.imputed_count,
                "memory_bytes": prepared.preprocessing.final_memory_bytes,
            }
            if EventPayloadRequirement.COMPONENT_TIMINGS in payload_requirements:
                prepared_payload.update(
                    loading_seconds=prepared.preprocessing.loading_seconds,
                    preprocessing_steps=[
                        {
                            "component_name": step.component_name,
                            "duration_seconds": step.duration_seconds,
                        }
                        for step in prepared.preprocessing.steps
                    ],
                )
            self._publish(
                publisher,
                EventType.DATASET_PREPARED,
                prepared_payload,
            )
            token.raise_if_cancelled()

            engine = specification.search_engine
            resumable = "checkpoint-resume" in engine.provides
            search_started = perf_counter()
            initialization_started = perf_counter()
            engine.initialize(specification, prepared.context)
            if resume_path is not None:
                engine.restore(self._read_checkpoint(resume_path))
            initialization_duration = perf_counter() - initialization_started
            initialization_components = self._drain_component_timings(engine)
            initial_progress = engine.progress()
            self._publish(
                publisher,
                EventType.ENGINE_INITIALIZED,
                {
                    "resumed": resume_path is not None,
                    "evaluations": initial_progress.evaluations,
                    "pending_candidates": len(engine.pending_candidates),
                    "integration_mode": specification.executor.integration_mode.value,
                    "workers": specification.worker_count,
                    "runtime": runtime_payload(
                        initialization_duration,
                        initialization_components,
                        payload_requirements,
                    ),
                },
            )
            checkpoint_interval = configuration.monitoring.checkpoint_interval_evaluations
            next_checkpoint = (
                None
                if checkpoint_interval is None
                else (initial_progress.evaluations // checkpoint_interval + 1) * checkpoint_interval
            )
            last_checkpoint_evaluations: int | None = None
            pending_checkpoint_path: Path | None = None
            while not engine.finished():
                token.raise_if_cancelled()
                before_batch = engine.progress()
                ask_started = perf_counter()
                candidates = tuple(engine.ask(engine.batch_size))
                ask_duration = perf_counter() - ask_started
                ask_components = self._drain_component_timings(engine)
                if not candidates:
                    raise RunError("search engine returned no candidates before termination")
                self._publish(
                    publisher,
                    EventType.CANDIDATES_ASKED,
                    {
                        "count": len(candidates),
                        "evaluations": before_batch.evaluations,
                        "runtime": runtime_payload(
                            ask_duration,
                            ask_components,
                            payload_requirements,
                        ),
                    },
                )
                emitter_names = {emitter.component_name for emitter in specification.emitters}
                allocation_counts: dict[str, int] = {}
                for candidate in candidates:
                    provenance = candidate.provenance
                    if provenance is not None and provenance.producer in emitter_names:
                        allocation_counts[provenance.producer] = (
                            allocation_counts.get(provenance.producer, 0) + 1
                        )
                if allocation_counts:
                    self._publish(
                        publisher,
                        EventType.SCHEDULER_ALLOCATION_UPDATED,
                        {
                            "evaluations": engine.progress().evaluations,
                            "allocations": [
                                {"emitter_name": name, "count": count}
                                for name, count in allocation_counts.items()
                            ],
                        },
                    )
                checkpoint_due = (
                    next_checkpoint is not None
                    and before_batch.evaluations + len(candidates) >= next_checkpoint
                )
                if checkpoint_due:
                    pending_checkpoint_path = self._write_checkpoint(
                        output / "checkpoints",
                        engine.checkpoint(),
                        state="pending",
                    )
                    self._publish(
                        publisher,
                        EventType.CHECKPOINT_WRITTEN,
                        {
                            "path": str(pending_checkpoint_path),
                            "evaluations": before_batch.evaluations,
                            "pending_candidates": len(candidates),
                            "state": "pending",
                        },
                    )
                self._publish(
                    publisher,
                    EventType.EVALUATION_BATCH_STARTED,
                    {
                        "evaluations": before_batch.evaluations,
                        "count": len(candidates),
                        "workers": specification.worker_count,
                        "integration_mode": specification.executor.integration_mode.value,
                        "max_in_flight": specification.executor.max_in_flight,
                    },
                )
                workspace = EvaluationWorkspace(prepared.context)
                batch = specification.executor.evaluate(
                    candidates,
                    specification.objectives,
                    specification.descriptors,
                    workspace,
                    constraints=specification.constraints,
                    worker_count=specification.worker_count,
                    cancellation=token,
                    collect_timings=(
                        EventPayloadRequirement.COMPONENT_TIMINGS in payload_requirements
                    ),
                )
                evaluations = batch.evaluations
                self._publish(
                    publisher,
                    EventType.CANDIDATES_EVALUATED,
                    evaluation_batch_payload(
                        batch,
                        evaluations=engine.progress().evaluations + len(evaluations),
                        requirements=payload_requirements,
                    ),
                )
                update_started = perf_counter()
                update = engine.tell(evaluations)
                progress = engine.progress()
                update_duration = perf_counter() - update_started
                update_components = self._drain_component_timings(engine)
                update_events: list[RunEvent] = []
                if update.outcomes:
                    update_events.append(
                        RunEvent(
                            event_type=EventType.ARCHIVE_UPDATED,
                            payload={
                                "evaluations": progress.evaluations,
                                "occupied_cells": progress.occupied_cells,
                                "repertoire_size": progress.repertoire_size,
                                "outcomes": [
                                    outcome.model_dump(mode="json") for outcome in update.outcomes
                                ],
                            },
                        )
                    )
                if update.emitter_feedback:
                    update_events.append(
                        RunEvent(
                            event_type=EventType.EMITTER_CREDIT_UPDATED,
                            payload={
                                "evaluations": progress.evaluations,
                                "feedback": [
                                    feedback.model_dump(mode="json")
                                    for feedback in update.emitter_feedback
                                ],
                                "reports": [
                                    report.model_dump(mode="json")
                                    for report in update.scheduler_reports
                                ],
                            },
                        )
                    )
                update_events.extend(
                    (
                        RunEvent(
                            event_type=EventType.ENGINE_UPDATED,
                            payload={
                                "finished": engine.finished(),
                                "runtime": runtime_payload(
                                    update_duration,
                                    update_components,
                                    payload_requirements,
                                ),
                                **progress.model_dump(mode="json"),
                            },
                        ),
                        RunEvent(
                            event_type=EventType.PROGRESS,
                            payload={
                                **progress.model_dump(mode="json"),
                                "termination": specification.termination.progress(
                                    progress.evaluations
                                ).model_dump(mode="json"),
                            },
                        ),
                    )
                )
                publisher.publish_many(update_events)
                if next_checkpoint is not None and progress.evaluations >= next_checkpoint:
                    assert checkpoint_interval is not None
                    checkpoint_path = self._write_checkpoint(
                        output / "checkpoints",
                        engine.checkpoint(),
                    )
                    last_checkpoint_evaluations = progress.evaluations
                    self._publish(
                        publisher,
                        EventType.CHECKPOINT_WRITTEN,
                        {
                            "path": str(checkpoint_path),
                            "evaluations": progress.evaluations,
                            "pending_candidates": 0,
                            "state": "complete",
                        },
                    )
                    if pending_checkpoint_path is not None:
                        pending_checkpoint_path.unlink(missing_ok=True)
                        pending_checkpoint_path = None
                    while next_checkpoint <= progress.evaluations:
                        next_checkpoint += checkpoint_interval

            progress = engine.progress()
            if (
                checkpoint_interval is not None
                and last_checkpoint_evaluations != progress.evaluations
            ):
                checkpoint_path = self._write_checkpoint(
                    output / "checkpoints",
                    engine.checkpoint(),
                )
                self._publish(
                    publisher,
                    EventType.CHECKPOINT_WRITTEN,
                    {
                        "path": str(checkpoint_path),
                        "evaluations": progress.evaluations,
                        "pending_candidates": 0,
                        "state": "complete",
                    },
                )
            source_checkpoint_path: Path | None = None
            source_checkpoint_evaluations: int | None = None
            if resumable:
                final_checkpoint = engine.checkpoint()
                source_checkpoint_path = self._write_checkpoint(
                    output / "checkpoints",
                    final_checkpoint,
                    state="final",
                )
                source_checkpoint_evaluations = final_checkpoint.evaluation_count
                self._publish(
                    publisher,
                    EventType.CHECKPOINT_WRITTEN,
                    {
                        "path": str(source_checkpoint_path),
                        "evaluations": final_checkpoint.evaluation_count,
                        "pending_candidates": 0,
                        "state": "final",
                    },
                )

            search_repertoire = engine.result()
            search_duration = perf_counter() - search_started
            selection_started = perf_counter()
            if specification.final_selector is None:
                repertoire = search_repertoire
            else:
                repertoire = specification.final_selector.select(
                    prepared.context,
                    search_repertoire,
                )
            selection_duration = perf_counter() - selection_started
            if specification.final_selector is not None:
                search_artifact_directory = output / "artifacts" / "search-repertoire"
                search_artifact_started = perf_counter()
                BiclusterSetWriter().write(
                    search_artifact_directory,
                    identifier=f"{configuration.run.identifier}-search-repertoire",
                    dataset_identifier=prepared.context.dataset.metadata.identifier,
                    row_count=prepared.context.dataset.row_count,
                    source_column_count=prepared.context.dataset.source_column_count,
                    columns=prepared.context.dataset.columns,
                    repertoire=search_repertoire,
                    source_run=configuration.run.identifier,
                    source_checkpoint=(
                        None
                        if source_checkpoint_path is None
                        else str(source_checkpoint_path.relative_to(output))
                    ),
                    source_checkpoint_sha256=(
                        None
                        if source_checkpoint_path is None
                        else sha256_file(source_checkpoint_path)
                    ),
                    source_checkpoint_evaluations=source_checkpoint_evaluations,
                )
                search_artifact_duration = perf_counter() - search_artifact_started
                search_manifest_path = search_artifact_directory / "manifest.json"
                search_artifact = store.record_artifact(
                    "search-repertoire",
                    search_manifest_path,
                    "application/vnd.salvi.bicluster-set+json",
                    sha256_file(search_manifest_path),
                )
                self._publish(
                    publisher,
                    EventType.ARTIFACT_WRITTEN,
                    {
                        "identifier": search_artifact.identifier,
                        "path": str(search_artifact.path),
                        "repertoire_size": len(search_repertoire.evaluations),
                        "artifact_kind": "search_repertoire",
                        "runtime": {"duration_seconds": search_artifact_duration},
                    },
                )
            if specification.final_selector is not None:
                self._publish(
                    publisher,
                    EventType.FINAL_SELECTION_COMPLETED,
                    {
                        "selector": specification.final_selector.component_name,
                        "evaluations": progress.evaluations,
                        "input_count": len(search_repertoire.evaluations),
                        "output_count": len(repertoire.evaluations),
                        "runtime": {"duration_seconds": selection_duration},
                    },
                )
            artifact_directory = output / "artifacts" / "repertoire"
            final_artifact_started = perf_counter()
            BiclusterSetWriter().write(
                artifact_directory,
                identifier=f"{configuration.run.identifier}-repertoire",
                dataset_identifier=prepared.context.dataset.metadata.identifier,
                row_count=prepared.context.dataset.row_count,
                source_column_count=prepared.context.dataset.source_column_count,
                columns=prepared.context.dataset.columns,
                repertoire=repertoire,
                source_run=configuration.run.identifier,
                source_checkpoint=(
                    None
                    if source_checkpoint_path is None
                    else str(source_checkpoint_path.relative_to(output))
                ),
                source_checkpoint_sha256=(
                    None if source_checkpoint_path is None else sha256_file(source_checkpoint_path)
                ),
                source_checkpoint_evaluations=source_checkpoint_evaluations,
            )
            final_artifact_duration = perf_counter() - final_artifact_started
            manifest_path = artifact_directory / "manifest.json"
            artifact = store.record_artifact(
                "final-repertoire",
                manifest_path,
                "application/vnd.salvi.bicluster-set+json",
                sha256_file(manifest_path),
            )
            self._publish(
                publisher,
                EventType.ARTIFACT_WRITTEN,
                {
                    "identifier": artifact.identifier,
                    "path": str(artifact.path),
                    "repertoire_size": len(repertoire.evaluations),
                    "artifact_kind": "final_repertoire",
                    "runtime": {"duration_seconds": final_artifact_duration},
                },
            )
            total_duration = perf_counter() - run_started_clock
            self._publish(
                publisher,
                EventType.RUN_COMPLETED,
                {
                    "status": RunStatus.COMPLETED.value,
                    "evaluations": progress.evaluations,
                    "runtime": {
                        "search_seconds": search_duration,
                        "selection_seconds": selection_duration,
                        "total_seconds": total_duration,
                    },
                },
            )
            self._close_publisher(publisher, store)
            publisher = None
            metadata.update(
                status=RunStatus.COMPLETED.value,
                finished_at=datetime.now(UTC).isoformat(),
                search=progress.model_dump(mode="json"),
                result_count=len(repertoire.evaluations),
                final_selection=(
                    None
                    if specification.final_selector is None
                    else {
                        "selector": specification.final_selector.component_name,
                        "input_count": len(search_repertoire.evaluations),
                        "output_count": len(repertoire.evaluations),
                        "duration_seconds": selection_duration,
                        "source_checkpoint": (
                            None
                            if source_checkpoint_path is None
                            else str(source_checkpoint_path.relative_to(output))
                        ),
                    }
                ),
                timing_seconds={
                    "search": search_duration,
                    "selection": selection_duration,
                    "total": total_duration,
                },
            )
            self._write_metadata(metadata_path, metadata)
            return RunResult(
                status=RunStatus.COMPLETED,
                output_directory=output,
                event_store=event_store_path,
                repertoire=repertoire,
                message="scientific search completed",
            )
        except RunCancelledError:
            recovery = self._write_recovery_checkpoint(
                engine,
                output / "checkpoints",
                reason="cancelled",
            )
            if publisher is not None:
                if recovery is not None:
                    self._publish(
                        publisher,
                        EventType.CHECKPOINT_WRITTEN,
                        {
                            "path": str(recovery),
                            "state": "recovery",
                            "reason": "cancelled",
                        },
                    )
                self._publish(
                    publisher,
                    EventType.RUN_CANCELLED,
                    {"status": RunStatus.CANCELLED.value},
                )
                self._close_publisher(publisher, store)
                publisher = None
            metadata.update(
                status=RunStatus.CANCELLED.value,
                finished_at=datetime.now(UTC).isoformat(),
                recovery_checkpoint=None if recovery is None else str(recovery),
            )
            self._write_metadata(metadata_path, metadata)
            raise
        except Exception as error:
            recovery = self._write_recovery_checkpoint(
                engine,
                output / "checkpoints",
                reason="failed",
            )
            if publisher is not None:
                if recovery is not None:
                    self._publish(
                        publisher,
                        EventType.CHECKPOINT_WRITTEN,
                        {
                            "path": str(recovery),
                            "state": "recovery",
                            "reason": "failed",
                        },
                    )
                self._publish(
                    publisher,
                    EventType.RUN_FAILED,
                    {"status": RunStatus.FAILED.value, "error": str(error)},
                )
                self._close_publisher(publisher, store)
                publisher = None
            metadata.update(
                status=RunStatus.FAILED.value,
                finished_at=datetime.now(UTC).isoformat(),
                error=str(error),
                recovery_checkpoint=None if recovery is None else str(recovery),
            )
            self._write_metadata(metadata_path, metadata)
            raise
        finally:
            if specification is not None:
                specification.executor.close()
            store.close()

    @staticmethod
    def _read_checkpoint(path: Path) -> SearchCheckpoint:
        try:
            return SearchCheckpoint.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise RunError(f"cannot read checkpoint {path}: {error}") from error

    @staticmethod
    def _write_checkpoint(
        directory: Path,
        checkpoint: SearchCheckpoint,
        *,
        state: str = "complete",
    ) -> Path:
        suffix = "" if state == "complete" else f"-{state}"
        path = directory / f"checkpoint-{checkpoint.evaluation_count:012d}{suffix}.json"
        atomic_write_text(path, checkpoint.model_dump_json() + "\n")
        return path

    @classmethod
    def _write_recovery_checkpoint(
        cls,
        engine: SearchEngine | None,
        directory: Path,
        *,
        reason: str,
    ) -> Path | None:
        if engine is None or not engine.pending_candidates:
            return None
        try:
            return cls._write_checkpoint(directory, engine.checkpoint(), state=f"recovery-{reason}")
        except Exception:
            return None

    @staticmethod
    def _is_within(path: Path, directory: Path) -> bool:
        try:
            path.resolve().relative_to(directory.resolve())
        except ValueError:
            return False
        return True

    @staticmethod
    def _publish(publisher: EventPublisher, event_type: EventType, payload: dict[str, Any]) -> None:
        publisher.publish(RunEvent(event_type=event_type, payload=payload))

    @staticmethod
    def _drain_component_timings(engine: SearchEngine) -> dict[str, float]:
        if not isinstance(engine, ComponentTimingSource):
            return {}
        timings: dict[str, float] = {}
        for name, duration in engine.drain_component_timings():
            if not name.strip() or duration < 0.0:
                raise RunError("search engine returned invalid component timing data")
            timings[name] = timings.get(name, 0.0) + duration
        return timings

    @staticmethod
    def _close_publisher(publisher: EventPublisher, store: SQLiteEventStore) -> None:
        try:
            publisher.close()
        except RunError as error:
            store.append(
                RunEvent(
                    event_type=EventType.WARNING,
                    payload={"message": "observer dispatcher shutdown failed", "error": str(error)},
                )
            )

    @staticmethod
    def _prepare_output(path: Path, overwrite: bool) -> Path:
        output = path.resolve()
        if output == Path(output.anchor):
            raise RunError("refusing to use a filesystem root as the output directory")
        if output.exists():
            if not overwrite:
                raise RunError(f"output directory already exists: {output}")
            shutil.rmtree(output)
        output.mkdir(parents=True)
        for child in ("logs", "checkpoints", "artifacts"):
            (output / child).mkdir()
        return output

    @staticmethod
    def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
        atomic_write_text(path, json.dumps(metadata, indent=2, sort_keys=True) + "\n")
