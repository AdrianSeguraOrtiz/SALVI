from __future__ import annotations

from pathlib import Path

import pytest

from salvi.components import diagnostic_observers as observer_module
from salvi.components.observers import (
    ArchiveCoverageObserver,
    ArchiveDescriptorDistributionObserver,
    CandidateDiversityObserver,
    CandidateOutcomesObserver,
    ComponentTimingObserver,
    DescriptorDistributionObserver,
    EmitterCreditObserver,
    EvaluationIssuesObserver,
    ObjectiveDistributionObserver,
    QDArchiveDiagnosticsObserver,
    RuntimeThroughputObserver,
    SearchProgressObserver,
)
from salvi.domain import EventType, RunEvent
from salvi.infrastructure.events import EventPublisher, SQLiteEventStore


def test_observers_persist_search_archive_and_distribution_metrics(
    tmp_path: Path,
) -> None:
    store = SQLiteEventStore(tmp_path / "run.sqlite")
    observers = (
        SearchProgressObserver(),
        ArchiveCoverageObserver(),
        CandidateOutcomesObserver(every_evaluations=2),
        DescriptorDistributionObserver(every_evaluations=2),
        ObjectiveDistributionObserver(every_evaluations=2),
        EmitterCreditObserver(every_evaluations=2),
        CandidateDiversityObserver(window_size=8, every_evaluations=2),
    )
    with EventPublisher(store, observers, capacity=32) as publisher:
        publisher.publish(
            RunEvent(
                event_type=EventType.CANDIDATES_EVALUATED,
                payload={
                    "evaluations": 2,
                    "items": [
                        {
                            "signature": "a" * 64,
                            "rows": [0, 1],
                            "columns": [0, 1],
                            "objectives": [
                                {"name": "internal_coherence", "value": 0.1},
                                {"name": "contrast", "value": 0.9},
                            ],
                            "descriptors": [
                                {"name": "row_cardinality", "value": 2.0},
                                {"name": "column_cardinality", "value": 2.0},
                            ],
                        },
                        {
                            "signature": "b" * 64,
                            "rows": [1, 2],
                            "columns": [0, 2],
                            "objectives": [
                                {"name": "internal_coherence", "value": 0.2},
                                {"name": "contrast", "value": 0.8},
                            ],
                            "descriptors": [
                                {"name": "row_cardinality", "value": 2.0},
                                {"name": "column_cardinality", "value": 2.0},
                            ],
                        },
                    ],
                },
            )
        )
        publisher.publish(
            RunEvent(
                event_type=EventType.ARCHIVE_UPDATED,
                payload={
                    "evaluations": 2,
                    "occupied_cells": 1,
                    "repertoire_size": 2,
                    "outcomes": [
                        {"status": "INSERTED", "created_cell": True},
                        {"status": "INSERTED", "created_cell": False},
                    ],
                },
            )
        )
        publisher.publish(
            RunEvent(
                event_type=EventType.EMITTER_CREDIT_UPDATED,
                payload={
                    "evaluations": 2,
                    "reports": [
                        {
                            "emitter_name": "add_row",
                            "evaluations": 2,
                            "accepted": 1,
                            "created_cells": 1,
                            "credit": 1.0,
                            "allocation_count": 2,
                        }
                    ],
                },
            )
        )
        publisher.publish(
            RunEvent(
                event_type=EventType.PROGRESS,
                payload={
                    "evaluations": 2,
                    "accepted": 2,
                    "rejected": 0,
                    "occupied_cells": 1,
                    "repertoire_size": 2,
                },
            )
        )
        publisher.flush()

    metrics = store.read_metrics(limit=1000)
    by_name = {metric.name: metric.value for metric in metrics}
    assert by_name["search.evaluations"] == 2
    assert by_name["archive.occupied_cells"] == 1
    assert by_name["archive.repertoire_size"] == 2
    assert by_name["outcomes.retained_rate"] == 1
    assert by_name["descriptor.row_cardinality.mean"] == 2
    assert by_name["objective.internal_coherence.mean"] == pytest.approx(0.15)
    assert by_name["objective.internal_coherence.first_quartile"] == pytest.approx(0.125)
    assert by_name["objective.internal_coherence.third_quartile"] == pytest.approx(0.175)
    assert by_name["emitter.add_row.credit"] == 1
    assert by_name["emitter.add_row.retention_rate"] == 0.5
    assert by_name["emitter.add_row.new_cell_rate"] == 0.5
    assert by_name["diversity.cumulative_unique"] == 2
    assert by_name["diversity.window_duplicate_ratio"] == 0
    assert by_name["diversity.nearest_distance.mean"] > 0
    store.close()


def test_candidate_diversity_observer_counts_exact_duplicates() -> None:
    observer = CandidateDiversityObserver(window_size=4, every_evaluations=2)
    samples = observer.on_event(
        RunEvent(
            event_type=EventType.CANDIDATES_EVALUATED,
            payload={
                "evaluations": 2,
                "items": [
                    {
                        "signature": "a" * 64,
                        "rows": [0, 1],
                        "columns": [0, 1],
                    },
                    {
                        "signature": "a" * 64,
                        "rows": [0, 1],
                        "columns": [0, 1],
                    },
                ],
            },
        )
    )
    by_name = {sample.name: sample.value for sample in samples}
    assert by_name["diversity.window_unique"] == 1
    assert by_name["diversity.window_duplicate_ratio"] == 0.5
    assert by_name["diversity.nearest_distance.mean"] == 0


def test_candidate_diversity_bounds_pairwise_distance_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def counted_distance(
        left_rows: frozenset[int],
        left_columns: frozenset[int],
        right_rows: frozenset[int],
        right_columns: frozenset[int],
        *,
        row_weight: float,
    ) -> float:
        nonlocal calls
        calls += 1
        return float(left_rows != right_rows or left_columns != right_columns or row_weight != 0.5)

    monkeypatch.setattr(observer_module, "structural_distance", counted_distance)
    observer = CandidateDiversityObserver(
        window_size=10,
        distance_sample_size=4,
        every_evaluations=10,
    )
    samples = observer.on_event(
        RunEvent(
            event_type=EventType.CANDIDATES_EVALUATED,
            payload={
                "evaluations": 10,
                "items": [
                    {
                        "signature": f"{index:064x}",
                        "rows": [index, index + 1],
                        "columns": [0, 1],
                    }
                    for index in range(10)
                ],
            },
        )
    )

    by_name = {sample.name: sample.value for sample in samples}
    assert by_name["diversity.window_unique"] == 10
    assert by_name["diversity.distance_sample_size"] == 4
    assert calls == 6


def test_archive_descriptor_distribution_tracks_retained_members_after_evictions() -> None:
    observer = ArchiveDescriptorDistributionObserver(every_evaluations=2)
    observer.on_event(
        RunEvent(
            event_type=EventType.CANDIDATES_EVALUATED,
            payload={
                "evaluations": 2,
                "items": [
                    {
                        "identifier": "first",
                        "descriptors": [
                            {"name": "row_cardinality", "value": 2.0},
                            {"name": "column_cardinality", "value": 3.0},
                        ],
                    },
                    {
                        "identifier": "second",
                        "descriptors": [
                            {"name": "row_cardinality", "value": 4.0},
                            {"name": "column_cardinality", "value": 5.0},
                        ],
                    },
                ],
            },
        )
    )
    first_samples = observer.on_event(
        RunEvent(
            event_type=EventType.ARCHIVE_UPDATED,
            payload={
                "evaluations": 2,
                "outcomes": [
                    {
                        "candidate_identifier": "first",
                        "status": "INSERTED",
                        "coordinate": {"indices": [0, 0]},
                    },
                    {
                        "candidate_identifier": "second",
                        "status": "INSERTED",
                        "coordinate": {"indices": [0, 0]},
                    },
                ],
            },
        )
    )
    first_by_name = {sample.name: sample.value for sample in first_samples}
    assert first_by_name["archive_descriptor.row_cardinality.mean"] == 3
    assert first_by_name["archive_cell.members.mean"] == 2

    observer.on_event(
        RunEvent(
            event_type=EventType.CANDIDATES_EVALUATED,
            payload={
                "evaluations": 3,
                "items": [
                    {
                        "identifier": "larger",
                        "descriptors": [
                            {"name": "row_cardinality", "value": 8.0},
                            {"name": "column_cardinality", "value": 9.0},
                        ],
                    }
                ],
            },
        )
    )
    observer.on_event(
        RunEvent(
            event_type=EventType.ARCHIVE_UPDATED,
            payload={
                "evaluations": 3,
                "outcomes": [
                    {
                        "candidate_identifier": "larger",
                        "status": "INSERTED_WITH_EVICTIONS",
                        "coordinate": {"indices": [1, 1]},
                        "evicted_candidate_identifiers": ["first"],
                    }
                ],
            },
        )
    )
    final_samples = observer.on_event(
        RunEvent(
            event_type=EventType.RUN_COMPLETED,
            payload={"evaluations": 3},
        )
    )
    final_by_name = {sample.name: sample.value for sample in final_samples}
    assert final_by_name["archive_descriptor.row_cardinality.mean"] == 6
    assert final_by_name["archive_cell.members.mean"] == 1


def test_candidate_outcomes_are_mutually_exclusive_window_rates() -> None:
    observer = CandidateOutcomesObserver(every_evaluations=6)
    samples = observer.on_event(
        RunEvent(
            event_type=EventType.ARCHIVE_UPDATED,
            payload={
                "evaluations": 6,
                "outcomes": [
                    {"status": "INSERTED"},
                    {"status": "INSERTED_WITH_EVICTIONS"},
                    {"status": "REJECTED_INVALID"},
                    {"status": "REJECTED_DUPLICATE"},
                    {"status": "REJECTED_DOMINATED"},
                    {"status": "REJECTED_CAPACITY"},
                ],
            },
        )
    )
    by_name = {sample.name: sample.value for sample in samples}
    assert by_name["outcomes.retained_rate"] == pytest.approx(2 / 6)
    assert by_name["outcomes.rejected_invalid_rate"] == pytest.approx(1 / 6)
    assert by_name["outcomes.rejected_duplicate_rate"] == pytest.approx(1 / 6)
    assert by_name["outcomes.rejected_dominated_rate"] == pytest.approx(1 / 6)
    assert by_name["outcomes.rejected_capacity_rate"] == pytest.approx(1 / 6)
    assert by_name["outcomes.rejected_out_of_bounds_rate"] == 0
    assert sum(sample.value for sample in samples) == pytest.approx(1)


def test_progress_archive_and_runtime_observers_do_not_duplicate_concepts() -> None:
    progress = SearchProgressObserver().on_event(
        RunEvent(
            event_type=EventType.PROGRESS,
            payload={
                "evaluations": 10,
                "accepted": 2,
                "rejected": 8,
                "occupied_cells": 3,
                "repertoire_size": 7,
            },
        )
    )
    assert tuple(sample.name for sample in progress) == ("search.evaluations",)

    archive = ArchiveCoverageObserver()
    update = RunEvent(
        event_type=EventType.ARCHIVE_UPDATED,
        payload={"evaluations": 10, "occupied_cells": 3, "repertoire_size": 7},
    )
    assert {sample.name for sample in archive.on_event(update)} == {
        "archive.occupied_cells",
        "archive.repertoire_size",
    }
    assert archive.on_event(update) == ()

    runtime = RuntimeThroughputObserver().on_event(
        RunEvent(
            event_type=EventType.CANDIDATES_EVALUATED,
            payload={
                "evaluations": 10,
                "count": 2,
                "runtime": {
                    "duration_seconds": 0.5,
                    "worker_count": 2,
                    "peak_in_flight": 2,
                },
            },
        )
    )
    assert "runtime.batch_seconds" not in {sample.name for sample in runtime}


def test_qd_issue_and_pipeline_observers_expose_search_diagnostics() -> None:
    issue_observer = EvaluationIssuesObserver(every_evaluations=2)
    issue_samples = issue_observer.on_event(
        RunEvent(
            event_type=EventType.CANDIDATES_EVALUATED,
            payload={
                "evaluations": 2,
                "items": [
                    {"issues": ["INSUFFICIENT_LOCAL_SUPPORT"]},
                    {"issues": []},
                ],
            },
        )
    )
    assert {sample.name: sample.value for sample in issue_samples}[
        "evaluation.issue.insufficient_local_support.candidate_rate"
    ] == 0.5
    assert {sample.name: sample.value for sample in issue_samples}["evaluation.invalid_rate"] == 0.5

    qd_observer = QDArchiveDiagnosticsObserver(
        every_evaluations=2,
        include_cell_metrics=True,
    )
    qd_samples = qd_observer.on_event(
        RunEvent(
            event_type=EventType.ARCHIVE_UPDATED,
            payload={
                "evaluations": 2,
                "outcomes": [
                    {
                        "status": "INSERTED",
                        "coordinate": {"indices": [1, 2]},
                        "evicted_candidate_identifiers": [],
                    },
                    {
                        "status": "REJECTED_CAPACITY",
                        "coordinate": {"indices": [1, 2]},
                        "evicted_candidate_identifiers": [],
                    },
                ],
            },
        )
    )
    by_name = {sample.name: sample.value for sample in qd_samples}
    assert by_name["qd.visited_cells"] == 1
    assert by_name["qd.cell.1_2.attempts"] == 2
    assert by_name["qd.cell.1_2.acceptance_ratio"] == 0.5

    timing_observer = ComponentTimingObserver(every_evaluations=10)
    timing_observer.record_observer_duration("candidate_outcomes", 0.05)
    setup = timing_observer.on_event(
        RunEvent(
            event_type=EventType.ENGINE_INITIALIZED,
            payload={
                "runtime": {
                    "duration_seconds": 0.4,
                    "component_duration_seconds": {
                        "initializer.pattern_aware": 0.3,
                        "archive.deep_grid_mome": 0.1,
                    },
                }
            },
        )
    )
    assert {sample.name: sample.value for sample in setup} == {
        "timing.setup.search_engine_initialization.seconds": 0.4
    }
    assert not timing_observer.on_event(
        RunEvent(
            event_type=EventType.CANDIDATES_ASKED,
            payload={
                "evaluations": 10,
                "runtime": {
                    "duration_seconds": 0.25,
                    "component_duration_seconds": {"emitter.add_row": 0.2},
                },
            },
        )
    )
    assert not timing_observer.on_event(
        RunEvent(
            event_type=EventType.CANDIDATES_EVALUATED,
            payload={
                "evaluations": 10,
                "runtime": {
                    "duration_seconds": 0.5,
                    "candidate_duration_seconds": [0.1, 0.2],
                    "component_duration_seconds": {
                        "objective.internal_coherence": 0.15,
                    },
                },
            },
        )
    )
    timing = timing_observer.on_event(
        RunEvent(event_type=EventType.PROGRESS, payload={"evaluations": 10})
    )
    by_name = {sample.name: sample.value for sample in timing}
    assert by_name["timing.search.candidate_generation.seconds"] == 0.25
    assert by_name["timing.search.evaluation_batch.seconds"] == 0.5
    assert by_name["timing.component.initializer.pattern_aware.seconds"] == 0.3
    assert by_name["timing.component.archive.deep_grid_mome.seconds"] == 0.1
    assert by_name["timing.component.emitter.add_row.seconds"] == 0.2
    assert by_name["timing.scientific.objective.internal_coherence.seconds"] == 0.15
    assert by_name["timing.observer.candidate_outcomes.seconds"] == 0.05
    assert by_name["timing.candidate_evaluation.mean_seconds"] == pytest.approx(0.15)
    assert by_name["timing.candidate_evaluation.p95_seconds"] == pytest.approx(0.195)


def test_observer_cadence_samples_when_batches_cross_thresholds() -> None:
    observer = ObjectiveDistributionObserver(every_evaluations=9)

    def event(step: int) -> RunEvent:
        return RunEvent(
            event_type=EventType.CANDIDATES_EVALUATED,
            payload={
                "evaluations": step,
                "items": [
                    {
                        "objectives": [
                            {"name": "internal_coherence", "value": step / 80},
                        ]
                    }
                ],
            },
        )

    assert observer.on_event(event(8)) == ()
    first_window = observer.on_event(event(16))
    assert {sample.step for sample in first_window} == {16}
    assert {sample.name: sample.value for sample in first_window}[
        "objective.internal_coherence.mean"
    ] == pytest.approx(0.15)
    second_window = observer.on_event(event(24))
    assert {sample.step for sample in second_window} == {24}
    assert {sample.name: sample.value for sample in second_window}[
        "objective.internal_coherence.mean"
    ] == pytest.approx(0.3)
