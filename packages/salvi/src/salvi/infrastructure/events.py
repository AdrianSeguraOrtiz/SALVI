"""SQLite-backed event stream and non-blocking publisher."""

from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path
from types import TracebackType
from typing import Self, TypeVar

from salvi.components.protocols import Observer
from salvi.domain.enums import EventType
from salvi.domain.models import MetricSample, RunArtifact, RunEvent, RunMetric
from salvi.exceptions import RunError

_ReadResult = TypeVar("_ReadResult")


class SQLiteEventStore:
    """Append-only run event store with concurrent WAL readers."""

    def __init__(self, path: Path, *, initialize: bool = True) -> None:
        self.path = path.resolve()
        self._read_only = not initialize
        if initialize:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        self._write_connection: sqlite3.Connection | None = None
        if initialize:
            self._initialize()

    def _connect(self, *, shared_writer: bool = False) -> sqlite3.Connection:
        database = f"{self.path.as_uri()}?mode=ro" if self._read_only else str(self.path)
        connection = sqlite3.connect(
            database,
            timeout=30.0,
            check_same_thread=not shared_writer,
            uri=self._read_only,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
            if not self._read_only:
                connection.commit()
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS metrics (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_sequence INTEGER,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    step INTEGER,
                    FOREIGN KEY(event_sequence) REFERENCES events(sequence)
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    identifier TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    event_sequence INTEGER,
                    FOREIGN KEY(event_sequence) REFERENCES events(sequence)
                );
                CREATE INDEX IF NOT EXISTS idx_events_type_sequence
                    ON events(event_type, sequence);
                CREATE INDEX IF NOT EXISTS idx_metrics_name_sequence
                    ON metrics(name, sequence);
                """
            )

    @contextmanager
    def _writer(self) -> Iterator[sqlite3.Connection]:
        with self._write_lock:
            if self._write_connection is None:
                self._write_connection = self._connect(shared_writer=True)
            try:
                yield self._write_connection
                self._write_connection.commit()
            except Exception:
                self._write_connection.rollback()
                raise

    def close(self) -> None:
        with self._write_lock:
            if self._write_connection is not None:
                self._write_connection.close()
                self._write_connection = None

    def append(self, event: RunEvent) -> RunEvent:
        return self.append_many((event,))[0]

    def append_many(self, events: Sequence[RunEvent]) -> tuple[RunEvent, ...]:
        """Persist an ordered event group in one transaction."""

        if not events:
            return ()
        persisted: list[RunEvent] = []
        with self._writer() as connection:
            for event in events:
                cursor = connection.execute(
                    "INSERT INTO events(event_type, timestamp, payload_json) VALUES (?, ?, ?)",
                    (
                        event.event_type.value,
                        event.timestamp.isoformat(),
                        json.dumps(event.payload, sort_keys=True, separators=(",", ":")),
                    ),
                )
                if cursor.lastrowid is None:
                    raise RunError("SQLite did not return an event sequence")
                persisted.append(event.model_copy(update={"sequence": cursor.lastrowid}))
        return tuple(persisted)

    def read_after(self, sequence: int = 0, *, limit: int = 1000) -> tuple[RunEvent, ...]:
        if sequence < 0 or limit < 1:
            raise ValueError("sequence must be non-negative and limit must be positive")
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT sequence, event_type, timestamp, payload_json
                FROM events
                WHERE sequence > ?
                ORDER BY sequence
                LIMIT ?
                """,
                (sequence, limit),
            ).fetchall()
        return tuple(
            RunEvent(
                sequence=int(row["sequence"]),
                event_type=EventType(row["event_type"]),
                timestamp=row["timestamp"],
                payload=json.loads(row["payload_json"]),
            )
            for row in rows
        )

    def event_count(self) -> int:
        with self._connection() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM events").fetchone()
        return int(row["count"])

    def read_event_page(self, offset: int, *, limit: int = 256) -> tuple[RunEvent, ...]:
        if offset < 0 or limit < 1:
            raise ValueError("offset must be non-negative and limit must be positive")
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT sequence, event_type, timestamp, payload_json
                FROM events
                ORDER BY sequence
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return tuple(
            RunEvent(
                sequence=int(row["sequence"]),
                event_type=EventType(row["event_type"]),
                timestamp=row["timestamp"],
                payload=json.loads(row["payload_json"]),
            )
            for row in rows
        )

    def record_metric(
        self,
        name: str,
        value: float,
        *,
        step: int | None = None,
        event_sequence: int | None = None,
    ) -> RunMetric:
        return self.record_metrics(
            (MetricSample(name=name, value=value, step=step),),
            event_sequence=event_sequence,
        )[0]

    def record_metrics(
        self,
        samples: Sequence[MetricSample],
        *,
        event_sequence: int | None = None,
    ) -> tuple[RunMetric, ...]:
        """Persist one observer-event metric batch in a single transaction."""

        return self._record_metric_batch(tuple((event_sequence, sample) for sample in samples))

    def _record_metric_batch(
        self,
        records: Sequence[tuple[int | None, MetricSample]],
    ) -> tuple[RunMetric, ...]:
        if not records:
            return ()
        persisted: list[RunMetric] = []
        with self._writer() as connection:
            for event_sequence, sample in records:
                cursor = connection.execute(
                    "INSERT INTO metrics(event_sequence, name, value, step) VALUES (?, ?, ?, ?)",
                    (event_sequence, sample.name, sample.value, sample.step),
                )
                if cursor.lastrowid is None:
                    raise RunError("SQLite did not return a metric sequence")
                persisted.append(
                    RunMetric(
                        sequence=cursor.lastrowid,
                        event_sequence=event_sequence,
                        name=sample.name,
                        value=sample.value,
                        step=sample.step,
                    )
                )
        return tuple(persisted)

    def read_metrics(
        self,
        after_sequence: int = 0,
        *,
        name: str | None = None,
        limit: int = 1000,
    ) -> tuple[RunMetric, ...]:
        if after_sequence < 0 or limit < 1:
            raise ValueError("sequence must be non-negative and limit must be positive")
        query = "SELECT sequence, event_sequence, name, value, step FROM metrics WHERE sequence > ?"
        parameters: list[object] = [after_sequence]
        if name is not None:
            query += " AND name = ?"
            parameters.append(name)
        query += " ORDER BY sequence LIMIT ?"
        parameters.append(limit)
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return tuple(
            RunMetric(
                sequence=row["sequence"],
                event_sequence=row["event_sequence"],
                name=row["name"],
                value=row["value"],
                step=row["step"],
            )
            for row in rows
        )

    def metric_names(self) -> tuple[str, ...]:
        with self._connection() as connection:
            rows = connection.execute("SELECT DISTINCT name FROM metrics ORDER BY name").fetchall()
        return tuple(str(row["name"]) for row in rows)

    def read_metric_series(
        self,
        name: str,
        *,
        limit: int = 2000,
    ) -> tuple[RunMetric, ...]:
        if not name or limit < 1:
            raise ValueError("name must not be empty and limit must be positive")
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT sequence, event_sequence, name, value, step
                FROM (
                    SELECT sequence, event_sequence, name, value, step
                    FROM metrics
                    WHERE name = ?
                    ORDER BY sequence DESC
                    LIMIT ?
                )
                ORDER BY sequence
                """,
                (name, limit),
            ).fetchall()
        return tuple(
            RunMetric(
                sequence=int(row["sequence"]),
                event_sequence=row["event_sequence"],
                name=row["name"],
                value=float(row["value"]),
                step=row["step"],
            )
            for row in rows
        )

    def record_artifact(
        self,
        identifier: str,
        path: Path,
        media_type: str,
        checksum: str,
        *,
        event_sequence: int | None = None,
    ) -> RunArtifact:
        with self._writer() as connection:
            connection.execute(
                """
                INSERT INTO artifacts(identifier, path, media_type, checksum, event_sequence)
                VALUES (?, ?, ?, ?, ?)
                """,
                (identifier, str(path), media_type, checksum, event_sequence),
            )
        return RunArtifact(
            identifier=identifier,
            path=path,
            media_type=media_type,
            checksum=checksum,
            event_sequence=event_sequence,
        )

    def read_artifacts(self) -> tuple[RunArtifact, ...]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT identifier, path, media_type, checksum, event_sequence
                FROM artifacts
                ORDER BY identifier
                """
            ).fetchall()
        return tuple(
            RunArtifact(
                identifier=row["identifier"],
                path=Path(row["path"]),
                media_type=row["media_type"],
                checksum=row["checksum"],
                event_sequence=row["event_sequence"],
            )
            for row in rows
        )


class SQLiteRunEventSource:
    """Replayable event source used by monitors and future visualizations."""

    def __init__(self, path: Path) -> None:
        self._store = SQLiteEventStore(path, initialize=False)

    def poll(self, after_sequence: int = 0, *, limit: int = 1000) -> tuple[RunEvent, ...]:
        return self._while_ready(
            lambda: self._store.read_after(after_sequence, limit=limit),
            (),
        )

    def event_count(self) -> int:
        return self._while_ready(self._store.event_count, 0)

    def event_page(self, offset: int, *, limit: int = 256) -> tuple[RunEvent, ...]:
        return self._while_ready(
            lambda: self._store.read_event_page(offset, limit=limit),
            (),
        )

    def poll_metrics(
        self,
        after_sequence: int = 0,
        *,
        name: str | None = None,
        limit: int = 1000,
    ) -> tuple[RunMetric, ...]:
        return self._while_ready(
            lambda: self._store.read_metrics(after_sequence, name=name, limit=limit),
            (),
        )

    def artifacts(self) -> tuple[RunArtifact, ...]:
        return self._while_ready(self._store.read_artifacts, ())

    def metric_names(self) -> tuple[str, ...]:
        return self._while_ready(self._store.metric_names, ())

    def metric_series(self, name: str, *, limit: int = 2000) -> tuple[RunMetric, ...]:
        return self._while_ready(
            lambda: self._store.read_metric_series(name, limit=limit),
            (),
        )

    def _while_ready(
        self,
        operation: Callable[[], _ReadResult],
        empty: _ReadResult,
    ) -> _ReadResult:
        if not self._store.path.is_file():
            return empty
        try:
            return operation()
        except sqlite3.OperationalError as error:
            message = str(error).lower()
            if (
                "no such table" in message
                or "locked" in message
                or "unable to open database file" in message
            ):
                return empty
            raise


class EventPublisher:
    """Durable publisher with bounded, best-effort observer notification."""

    _STOP = object()
    _METRIC_BATCH_SIZE = 256
    _METRIC_FLUSH_SECONDS = 0.05

    def __init__(
        self,
        store: SQLiteEventStore,
        observers: Sequence[Observer] = (),
        *,
        capacity: int = 1024,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self._store = store
        self._observers = tuple(observers)
        self._queue: queue.Queue[RunEvent | object] = queue.Queue(maxsize=capacity)
        self._dropped_notifications = 0
        self._failed_observers: set[int] = set()
        self._metric_persistence_failed = False
        self._closed = False
        self._thread: threading.Thread | None = None
        if self._observers:
            self._thread = threading.Thread(
                target=self._consume,
                name="salvi-observer-dispatcher",
                daemon=True,
            )
            self._thread.start()

    @property
    def dropped_notifications(self) -> int:
        return self._dropped_notifications

    def publish(self, event: RunEvent) -> RunEvent:
        return self.publish_many((event,))[0]

    def publish_many(self, events: Sequence[RunEvent]) -> tuple[RunEvent, ...]:
        """Persist and dispatch an ordered event group with one SQLite commit."""

        if self._closed:
            raise RunError("cannot publish to a closed event publisher")
        persisted = self._store.append_many(events)
        if self._thread is None:
            return persisted
        for event in persisted:
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                self._dropped_notifications += 1
        return persisted

    def flush(self) -> None:
        if self._thread is None:
            return
        self._queue.join()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._thread is None:
            return
        try:
            self._queue.put(self._STOP, timeout=5.0)
        except queue.Full as error:
            raise RunError("observer queue did not accept its shutdown marker") from error
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise RunError("observer dispatcher did not stop")
        if self._dropped_notifications:
            self._store.append(
                RunEvent(
                    event_type=EventType.WARNING,
                    payload={
                        "message": "observer notifications were dropped",
                        "count": self._dropped_notifications,
                    },
                )
            )

    def _consume(self) -> None:
        pending: list[tuple[int | None, MetricSample]] = []
        pending_tasks = 0
        last_flush = time.monotonic()
        timing_sinks: tuple[Callable[[str, float], None], ...] = tuple(
            sink
            for observer in self._observers
            if callable(sink := getattr(observer, "record_observer_duration", None))
        )

        def flush_metrics() -> None:
            nonlocal pending_tasks, last_flush
            try:
                if pending and not self._metric_persistence_failed:
                    self._store._record_metric_batch(pending)
            except Exception as error:
                self._metric_persistence_failed = True
                with suppress(Exception):
                    self._store.append(
                        RunEvent(
                            event_type=EventType.WARNING,
                            payload={
                                "message": "observer metrics disabled after persistence failure",
                                "error": str(error),
                            },
                        )
                    )
            finally:
                pending.clear()
                for _ in range(pending_tasks):
                    self._queue.task_done()
                pending_tasks = 0
                last_flush = time.monotonic()

        while True:
            timeout = max(
                0.0,
                self._METRIC_FLUSH_SECONDS - (time.monotonic() - last_flush),
            )
            try:
                item = self._queue.get(timeout=timeout)
            except queue.Empty:
                flush_metrics()
                continue
            pending_tasks += 1
            try:
                if item is self._STOP:
                    flush_metrics()
                    return
                if not isinstance(item, RunEvent):
                    raise TypeError("event publisher received an invalid queue item")
                samples: list[MetricSample] = []
                for index, observer in enumerate(self._observers):
                    if index in self._failed_observers:
                        continue
                    observer_started = time.perf_counter()
                    try:
                        samples.extend(observer.on_event(item))
                    except Exception as error:
                        self._failed_observers.add(index)
                        self._store.append(
                            RunEvent(
                                event_type=EventType.WARNING,
                                payload={
                                    "message": "observer disabled after failure",
                                    "observer": observer.component_name,
                                    "error": str(error),
                                },
                            )
                        )
                    finally:
                        observer_duration = time.perf_counter() - observer_started
                        for sink in timing_sinks:
                            sink(observer.component_name, observer_duration)
                pending.extend((item.sequence, sample) for sample in samples)
                if (
                    len(pending) >= self._METRIC_BATCH_SIZE
                    or pending_tasks >= self._METRIC_BATCH_SIZE
                ):
                    flush_metrics()
            except BaseException:
                flush_metrics()
                raise

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
