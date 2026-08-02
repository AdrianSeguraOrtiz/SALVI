"""Spawn-isolated SALVI execution for the local web application."""

from __future__ import annotations

import json
import multiprocessing
import threading
from datetime import UTC, datetime
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Event as MultiprocessingEvent
from pathlib import Path
from time import monotonic, sleep
from typing import Any

from salvi.application.configuration import (
    RunBinding,
    parse_pipeline_configuration,
    serialize_pipeline_configuration,
)
from salvi.application.run_service import RunService
from salvi.domain.enums import RunStatus
from salvi.exceptions import ArtifactError, RunCancelledError, RunError
from salvi.web.adapters import normalized_identifier
from salvi.web.models import WebRunRecord
from salvi.web.storage import WebStateStore


class ProcessCancellationSignal:
    def __init__(self, event: MultiprocessingEvent) -> None:
        self._event = event

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise RunCancelledError("run cancelled")


def _run_child(
    pipeline_path: str,
    binding_payload: dict[str, Any],
    cancellation_event: MultiprocessingEvent,
) -> None:
    try:
        RunService().run_pipeline(
            Path(pipeline_path),
            RunBinding.model_validate(binding_payload),
            cancellation=ProcessCancellationSignal(cancellation_event),
        )
    except (Exception, KeyboardInterrupt):
        raise SystemExit(1) from None


class WebRunManager:
    """Own at most one active subprocess while retaining completed run history."""

    def __init__(self, store: WebStateStore) -> None:
        self._store = store
        self._context = multiprocessing.get_context("spawn")
        self._lock = threading.Lock()
        self._process: BaseProcess | None = None
        self._cancellation_event: MultiprocessingEvent | None = None
        self._active_identifier: str | None = None
        self._monitor: threading.Thread | None = None
        self._store.mark_interrupted_runs()

    @property
    def active_identifier(self) -> str | None:
        with self._lock:
            if self._process is None or not self._process.is_alive():
                return None
            return self._active_identifier

    def start(
        self,
        *,
        pipeline_text: str,
        dataset_identifier: str,
        run_identifier: str,
        seed: int,
        analyses: tuple[str, ...] = (),
    ) -> WebRunRecord:
        pipeline = parse_pipeline_configuration(pipeline_text, source="web editor")
        dataset = self._store.get_dataset(dataset_identifier)
        if dataset is None:
            raise ArtifactError(f"unknown dataset: {dataset_identifier}")
        identifier = normalized_identifier(run_identifier)
        run_directory = self._store.paths.runs / identifier
        output = run_directory / "output"
        pipeline_path = run_directory / "pipeline.yaml"
        if self._store.get_run(identifier) is not None or run_directory.exists():
            raise ArtifactError(f"a run named {identifier!r} already exists")
        binding = RunBinding(
            identifier=identifier,
            dataset_bundle=dataset.bundle_path,
            output_directory=output,
            seed=seed,
        )
        RunService().validate_pipeline_configuration(pipeline, binding)

        with self._lock:
            if self._process is not None and self._process.is_alive():
                raise RunError("another SALVI run is already active")
            run_directory.mkdir(parents=True)
            pipeline_path.write_text(
                serialize_pipeline_configuration(pipeline),
                encoding="utf-8",
            )
            record = WebRunRecord(
                identifier=identifier,
                dataset_identifier=dataset_identifier,
                pipeline_path=pipeline_path,
                output_directory=output,
                seed=seed,
                analyses=analyses,
                status=RunStatus.RUNNING,
                started_at=datetime.now(UTC),
            )
            cancellation = self._context.Event()
            process = self._context.Process(
                target=_run_child,
                args=(
                    str(pipeline_path),
                    binding.model_dump(mode="json"),
                    cancellation,
                ),
                name=f"salvi-{identifier}",
                daemon=False,
            )
            self._store.put_run(record)
            try:
                process.start()
            except Exception:
                self._store.put_run(
                    record.model_copy(
                        update={
                            "status": RunStatus.FAILED,
                            "finished_at": datetime.now(UTC),
                            "error": "Could not start the SALVI worker process.",
                        }
                    )
                )
                raise
            self._process = process
            self._cancellation_event = cancellation
            self._active_identifier = identifier
            monitor = threading.Thread(
                target=self._monitor_process,
                args=(identifier, process),
                name=f"salvi-monitor-{identifier}",
                daemon=True,
            )
            self._monitor = monitor
            monitor.start()
            return record

    def cancel(self, identifier: str, *, grace_seconds: float = 5.0) -> WebRunRecord:
        with self._lock:
            if (
                self._active_identifier != identifier
                or self._process is None
                or not self._process.is_alive()
                or self._cancellation_event is None
            ):
                raise RunError(f"run {identifier!r} is not active")
            self._cancellation_event.set()
            process = self._process
        deadline = monotonic() + grace_seconds
        while process.is_alive() and monotonic() < deadline:
            sleep(0.05)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
        monitor = self._monitor
        if monitor is not None and monitor.is_alive():
            monitor.join(timeout=2.0)
        record = self._store.get_run(identifier)
        if record is None:
            raise ArtifactError(f"unknown run: {identifier}")
        return record

    def shutdown(self) -> None:
        identifier = self.active_identifier
        if identifier is not None:
            self.cancel(identifier)
        monitor = self._monitor
        if monitor is not None and monitor.is_alive():
            monitor.join(timeout=2.0)

    def _monitor_process(
        self,
        identifier: str,
        process: multiprocessing.Process,
    ) -> None:
        process.join()
        record = self._store.get_run(identifier)
        if record is None:
            return
        status, error, started_at, finished_at = self._read_outcome(record, process.exitcode)
        self._store.put_run(
            record.model_copy(
                update={
                    "status": status,
                    "error": error,
                    "started_at": started_at or record.started_at,
                    "finished_at": finished_at or datetime.now(UTC),
                }
            )
        )
        with self._lock:
            if self._active_identifier == identifier:
                self._process = None
                self._cancellation_event = None
                self._active_identifier = None

    @staticmethod
    def _read_outcome(
        record: WebRunRecord,
        exit_code: int | None,
    ) -> tuple[RunStatus, str | None, datetime | None, datetime | None]:
        metadata_path = record.output_directory / "run-metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            status = RunStatus(str(metadata["status"]))
            error = None if metadata.get("error") is None else str(metadata["error"])
            started_at = datetime.fromisoformat(str(metadata["started_at"]))
            finished_at = (
                None
                if metadata.get("finished_at") is None
                else datetime.fromisoformat(str(metadata["finished_at"]))
            )
            return status, error, started_at, finished_at
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            status = RunStatus.CANCELLED if exit_code == -15 else RunStatus.FAILED
            return (
                status,
                f"SALVI worker exited with code {exit_code} before writing final metadata.",
                None,
                datetime.now(UTC),
            )


__all__ = ["ProcessCancellationSignal", "WebRunManager"]
