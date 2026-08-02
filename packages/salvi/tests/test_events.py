from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from salvi.components.observers import ComponentTimingObserver
from salvi.domain import EventType, MetricSample, RunEvent
from salvi.exceptions import RunError
from salvi.infrastructure.events import EventPublisher, SQLiteEventStore, SQLiteRunEventSource


@dataclass
class RecordingObserver:
    component_name: str = "recording"
    provides: frozenset[str] = frozenset({"observer"})
    requires: frozenset[str] = frozenset()
    events: list[RunEvent] = field(default_factory=list)

    def on_event(self, event: RunEvent) -> tuple[()]:
        self.events.append(event)
        return ()


@dataclass
class BlockingObserver(RecordingObserver):
    entered: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)

    def on_event(self, event: RunEvent) -> tuple[()]:
        self.entered.set()
        self.release.wait(timeout=5)
        return super().on_event(event)


@dataclass
class FailingObserver(RecordingObserver):
    def on_event(self, event: RunEvent) -> tuple[()]:
        self.events.append(event)
        raise RuntimeError("observer failure")


def test_event_store_round_trip_metrics_and_artifacts(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "run.sqlite")
    persisted = store.append(RunEvent(event_type=EventType.RUN_STARTED, payload={"seed": 7}))
    assert persisted.sequence == 1
    assert store.read_after() == (persisted,)
    assert store.read_after(1) == ()
    metric = store.record_metric("quality", 0.5, step=1, event_sequence=1)
    artifact = store.record_artifact("result", tmp_path / "result", "application/json", "0" * 64)
    assert store.read_metrics() == (metric,)
    assert store.read_metrics(name="missing") == ()
    assert store.read_artifacts() == (artifact,)
    source = SQLiteRunEventSource(tmp_path / "run.sqlite")
    assert source.poll_metrics() == (metric,)
    assert source.artifacts() == (artifact,)
    assert source.event_count() == 1
    assert source.event_page(0, limit=1) == (persisted,)
    assert source.metric_names() == ("quality",)
    assert source.metric_series("quality") == (metric,)
    batch = store.record_metrics(
        (
            MetricSample(name="quality", value=0.6, step=2),
            MetricSample(name="coverage", value=0.7, step=2),
        ),
        event_sequence=1,
    )
    assert tuple(item.sequence for item in batch) == (2, 3)
    assert store.record_metrics(()) == ()
    with pytest.raises(ValueError):
        store.read_after(-1)
    with pytest.raises(ValueError):
        source.event_page(-1)
    with pytest.raises(ValueError):
        source.metric_series("")
    store.close()


def test_event_source_does_not_initialize_a_writer_schema(tmp_path: Path) -> None:
    path = tmp_path / "run.sqlite"

    source = SQLiteRunEventSource(path)

    assert not path.exists()
    assert source.poll() == ()
    assert source.event_count() == 0
    assert source.event_page(0) == ()
    assert source.poll_metrics() == ()
    assert source.artifacts() == ()
    assert source.metric_names() == ()
    assert source.metric_series("metric") == ()


def test_event_pages_and_metric_series_are_bounded_and_ordered(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "run.sqlite")
    for index in range(7):
        store.append(RunEvent(event_type=EventType.PROGRESS, payload={"index": index}))
        store.record_metric("coverage", index / 10, step=index)
    source = SQLiteRunEventSource(tmp_path / "run.sqlite")
    assert [item.payload["index"] for item in source.event_page(2, limit=3)] == [2, 3, 4]
    assert [item.step for item in source.metric_series("coverage", limit=3)] == [4, 5, 6]
    store.close()


def test_publisher_persists_before_notifying_observers(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "run.sqlite")
    observer = RecordingObserver()
    with EventPublisher(store, (observer,), capacity=8) as publisher:
        persisted = publisher.publish(RunEvent(event_type=EventType.RUN_STARTED))
        assert persisted.sequence == 1
        publisher.flush()
    assert observer.events[0].sequence == 1
    assert store.read_after() == tuple(observer.events)
    with pytest.raises(RunError, match="closed"):
        publisher.publish(RunEvent(event_type=EventType.PROGRESS))
    store.close()


def test_publisher_reports_observer_callback_cost_to_timing_observer(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "run.sqlite")
    timing = ComponentTimingObserver(every_evaluations=100)
    with EventPublisher(store, (timing,), capacity=8) as publisher:
        publisher.publish(RunEvent(event_type=EventType.PROGRESS, payload={"evaluations": 1}))
        publisher.flush()
        publisher.publish(RunEvent(event_type=EventType.RUN_COMPLETED, payload={"evaluations": 1}))
        publisher.flush()

    metrics = {metric.name: metric.value for metric in store.read_metrics(limit=100)}
    assert metrics["timing.observer.component_timing.seconds"] > 0.0
    store.close()


def test_wal_allows_concurrent_readers_during_writes(tmp_path: Path) -> None:
    path = tmp_path / "run.sqlite"
    store = SQLiteEventStore(path)
    source = SQLiteRunEventSource(path)
    started = threading.Event()

    def write_events() -> None:
        started.set()
        for index in range(50):
            store.append(RunEvent(event_type=EventType.PROGRESS, payload={"index": index}))

    thread = threading.Thread(target=write_events)
    thread.start()
    started.wait(timeout=2)
    observed: dict[int, RunEvent] = {}
    while thread.is_alive():
        for event in source.poll(max(observed, default=0)):
            assert event.sequence is not None
            observed[event.sequence] = event
    thread.join()
    for event in source.poll(max(observed, default=0)):
        assert event.sequence is not None
        observed[event.sequence] = event
    assert len(observed) == 50
    assert [event.payload["index"] for event in observed.values()] == list(range(50))
    store.close()


def test_slow_observer_never_drops_durable_events(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "run.sqlite")
    observer = BlockingObserver()
    publisher = EventPublisher(store, (observer,), capacity=1)
    publisher.publish(RunEvent(event_type=EventType.PROGRESS, payload={"index": 0}))
    assert observer.entered.wait(timeout=2)
    publisher.publish(RunEvent(event_type=EventType.PROGRESS, payload={"index": 1}))
    publisher.publish(RunEvent(event_type=EventType.PROGRESS, payload={"index": 2}))
    observer.release.set()
    publisher.close()
    events = store.read_after()
    assert [event.payload.get("index") for event in events[:3]] == [0, 1, 2]
    assert events[-1].event_type is EventType.WARNING
    assert publisher.dropped_notifications == 1
    store.close()


def test_failing_observer_is_disabled_without_failing_the_run(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "run.sqlite")
    observer = FailingObserver()
    with EventPublisher(store, (observer,), capacity=8) as publisher:
        publisher.publish(RunEvent(event_type=EventType.RUN_STARTED))
        publisher.flush()
        publisher.publish(RunEvent(event_type=EventType.RUN_COMPLETED))
        publisher.flush()
    assert len(observer.events) == 1
    events = store.read_after()
    assert [event.event_type for event in events].count(EventType.WARNING) == 1
    assert EventType.RUN_COMPLETED in (event.event_type for event in events)
    store.close()


def test_metric_persistence_failure_does_not_block_the_publisher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from salvi.domain import MetricSample

    @dataclass
    class MetricObserver(RecordingObserver):
        def on_event(self, event: RunEvent) -> tuple[MetricSample, ...]:
            self.events.append(event)
            return (MetricSample(name="metric", value=1.0),)

    store = SQLiteEventStore(tmp_path / "run.sqlite")
    original = store._record_metric_batch
    attempts = 0

    def fail_once(records):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RunError("metric write failed")
        return original(records)

    monkeypatch.setattr(store, "_record_metric_batch", fail_once)
    observer = MetricObserver()
    with EventPublisher(store, (observer,), capacity=8) as publisher:
        publisher.publish(RunEvent(event_type=EventType.PROGRESS))
        publisher.flush()
        publisher.publish(RunEvent(event_type=EventType.RUN_COMPLETED))
        publisher.flush()

    assert attempts == 1
    assert store.read_metrics() == ()
    warnings = tuple(event for event in store.read_after() if event.event_type is EventType.WARNING)
    assert len(warnings) == 1
    assert "metrics disabled" in warnings[0].payload["message"]
    store.close()
