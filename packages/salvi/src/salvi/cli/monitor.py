"""Console progress monitor backed by the canonical run event store."""

from __future__ import annotations

import sqlite3
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from salvi.domain.enums import EventType
from salvi.infrastructure.events import SQLiteRunEventSource

_STAGE_LABELS = {
    EventType.RUN_STARTED: "Starting",
    EventType.CONFIGURATION_VALIDATED: "Configuration",
    EventType.DATASET_VALIDATED: "Dataset",
    EventType.COMPONENTS_BUILT: "Components",
    EventType.DATASET_PREPARED: "Preprocessing",
    EventType.ENGINE_INITIALIZED: "Initialization",
    EventType.EVALUATION_BATCH_STARTED: "Search",
    EventType.FINAL_SELECTION_COMPLETED: "Final selection",
    EventType.ARTIFACT_WRITTEN: "Artifacts",
    EventType.RUN_COMPLETED: "Completed",
    EventType.RUN_FAILED: "Failed",
    EventType.RUN_CANCELLED: "Cancelled",
}


@dataclass(slots=True)
class _ProgressState:
    stage: str = "Starting"
    evaluations: int = 0
    occupied_cells: int = 0
    repertoire_size: int = 0
    rate: float | None = None
    termination_current: float = 0.0
    termination_limit: float | None = None
    termination_unit: str = "evaluations"


class ConsoleRunMonitor:
    """Poll a run SQLite store and render concise CLI progress on stderr."""

    def __init__(
        self,
        event_store: Path,
        *,
        interval_seconds: float,
        minimum_mtime_ns: int | None = None,
        stream: TextIO | None = None,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self._event_store = event_store.resolve()
        self._interval_seconds = interval_seconds
        self._minimum_mtime_ns = minimum_mtime_ns
        self._stream = stream or sys.stderr
        self._interactive = self._stream.isatty()
        self._state = _ProgressState()
        self._source: SQLiteRunEventSource | None = None
        self._last_event_sequence = 0
        self._last_metric_sequence = 0
        self._last_line = ""
        self._started_at = time.monotonic()
        self._rendered = False
        self._dirty = True

    def monitor_until_finished(self, run_finished: Callable[[], bool]) -> None:
        """Render progress until the supplied callable reports completion."""

        while not run_finished():
            self._poll_once()
            self._render()
            time.sleep(self._interval_seconds)
        self.drain()

    def drain(self) -> None:
        """Consume the final event batch and terminate the interactive line cleanly."""

        for _ in range(3):
            before_event = self._last_event_sequence
            before_metric = self._last_metric_sequence
            self._poll_once()
            if (
                self._last_event_sequence == before_event
                and self._last_metric_sequence == before_metric
            ):
                break
        self._render()
        if self._interactive and self._rendered:
            self._stream.write("\n")
            self._stream.flush()

    def _poll_once(self) -> None:
        try:
            self._poll_ready_store()
        except sqlite3.OperationalError:
            # The writer may have created the file but not committed its schema yet.
            self._source = None

    def _poll_ready_store(self) -> None:
        if self._source is None:
            if not self._event_store.exists():
                return
            if (
                self._minimum_mtime_ns is not None
                and self._event_store.stat().st_mtime_ns < self._minimum_mtime_ns
            ):
                return
            self._source = SQLiteRunEventSource(self._event_store)
        events = self._source.poll(self._last_event_sequence, limit=2000)
        for event in events:
            if event.sequence is not None:
                self._last_event_sequence = event.sequence
            self._dirty = True
            label = _STAGE_LABELS.get(event.event_type)
            if label is not None:
                self._state.stage = label
            if event.event_type is EventType.PROGRESS:
                self._state.evaluations = int(event.payload.get("evaluations", 0))
                self._state.occupied_cells = int(event.payload.get("occupied_cells", 0))
                self._state.repertoire_size = int(event.payload.get("repertoire_size", 0))
                termination = event.payload.get("termination")
                if isinstance(termination, dict):
                    current = termination.get("current")
                    limit = termination.get("limit")
                    unit = termination.get("unit")
                    if isinstance(current, int | float):
                        self._state.termination_current = float(current)
                    if isinstance(limit, int | float):
                        self._state.termination_limit = float(limit)
                    if isinstance(unit, str) and unit:
                        self._state.termination_unit = unit

        metrics = self._source.poll_metrics(self._last_metric_sequence, limit=2000)
        for metric in metrics:
            if metric.sequence is not None:
                self._last_metric_sequence = metric.sequence
            if metric.name == "runtime.evaluations_per_second":
                self._state.rate = metric.value
                self._dirty = True

    def _render(self, *, force: bool = False) -> None:
        if not self._interactive and not force and not self._dirty:
            return
        line = self._format_line()
        if not force and line == self._last_line:
            self._dirty = False
            return
        if self._interactive:
            self._stream.write(f"\r\x1b[K{line}")
        else:
            self._stream.write(f"{line}\n")
        self._stream.flush()
        self._last_line = line
        self._rendered = True
        self._dirty = False

    def _format_line(self) -> str:
        current = self._state.termination_current or float(self._state.evaluations)
        limit = self._state.termination_limit
        unit = self._state.termination_unit
        if unit == "evaluations":
            current_text = f"{int(current):,} evals"
            bounded_text = None if limit is None else f"{int(current):,}/{int(limit):,} evals"
        else:
            current_text = f"{current:,.1f} {unit}"
            bounded_text = None if limit is None else f"{current:,.1f}/{limit:,.1f} {unit}"
        if limit is None:
            progress_text = current_text
            percent_text = ""
        else:
            percent = min(100.0, current * 100.0 / limit)
            progress_text = bounded_text or current_text
            percent_text = f" {percent:5.1f}%"

        rate = self._effective_rate()
        rate_text = "" if rate is None else f" | {rate:,.1f} eval/s"
        eta_text = self._eta_text(rate)
        return (
            f"[{self._state.stage}] {progress_text}{percent_text}"
            f" | cells {self._state.occupied_cells:,}"
            f" | repertoire {self._state.repertoire_size:,}"
            f"{rate_text}{eta_text}"
        )

    def _effective_rate(self) -> float | None:
        if self._state.rate is not None and self._state.rate > 0:
            return self._state.rate
        elapsed = time.monotonic() - self._started_at
        if elapsed <= 0 or self._state.evaluations <= 0:
            return None
        return self._state.evaluations / elapsed

    def _eta_text(self, rate: float | None) -> str:
        if (
            self._state.termination_limit is None
            or self._state.termination_unit != "evaluations"
            or rate is None
            or rate <= 0
            or self._state.termination_current >= self._state.termination_limit
        ):
            return ""
        remaining = max(0.0, self._state.termination_limit - self._state.termination_current)
        return f" | eta {_format_duration(remaining / rate)}"


def _format_duration(seconds: float) -> str:
    rounded = max(0, round(seconds))
    minutes, second = divmod(rounded, 60)
    hours, minute = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minute:02d}m"
    if minute:
        return f"{minute}m{second:02d}s"
    return f"{second}s"
