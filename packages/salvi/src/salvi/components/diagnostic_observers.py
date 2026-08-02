"""Scientific, QD, resource, and stability diagnostic observers."""

from __future__ import annotations

import os
import sys
import threading
from collections import Counter, deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter, process_time
from typing import ClassVar

import numpy as np

from salvi.components.protocols import EventPayloadRequirement
from salvi.components.runtime_observers import (
    _RETAINED_ARCHIVE_STATUSES,
    CandidateDiversityObserverConfiguration,
    DistributionObserverConfiguration,
    QDArchiveDiagnosticsObserverConfiguration,
    ResourceUsageObserverConfiguration,
    _evaluations,
    _numeric_summary,
    _sample_due,
)
from salvi.domain.enums import EventType
from salvi.domain.models import MetricSample, RunEvent
from salvi.evaluation.structure import structural_distance


@dataclass(slots=True)
class EvaluationIssuesObserver:
    """Count scientific invalidity causes and the shapes on which they occur."""

    every_evaluations: int = 1_000
    component_name: str = "evaluation_issues"
    provides: frozenset[str] = frozenset({"observer"})
    requires: frozenset[str] = frozenset()
    event_payload_requirements: ClassVar[frozenset[EventPayloadRequirement]] = frozenset(
        {EventPayloadRequirement.EVALUATION_ISSUES}
    )
    _issues: Counter[str] = field(default_factory=Counter, init=False)
    _evaluated: int = field(default=0, init=False)
    _invalid: int = field(default=0, init=False)
    _next_sample: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        DistributionObserverConfiguration(every_evaluations=self.every_evaluations)
        self._next_sample = self.every_evaluations

    def on_event(self, event: RunEvent) -> Sequence[MetricSample]:
        if event.event_type is EventType.CANDIDATES_EVALUATED:
            for evaluation in _evaluations(event):
                self._evaluated += 1
                issues = frozenset(str(value) for value in evaluation.get("issues", ()))
                if issues:
                    self._invalid += 1
                    self._issues.update(issues)
            step = int(event.payload.get("evaluations", 0))
            due, self._next_sample = _sample_due(
                step,
                self._next_sample,
                self.every_evaluations,
            )
        elif event.event_type is EventType.RUN_COMPLETED:
            step = int(event.payload.get("evaluations", 0))
            due = self._evaluated > 0
        else:
            return ()
        if not due:
            return ()
        if self._evaluated <= 0:
            return ()
        samples = [
            MetricSample(
                name="evaluation.valid_rate",
                value=(self._evaluated - self._invalid) / self._evaluated,
                step=step,
            ),
            MetricSample(
                name="evaluation.invalid_rate",
                value=self._invalid / self._evaluated,
                step=step,
            ),
        ]
        samples.extend(
            MetricSample(
                name=f"evaluation.issue.{code.lower()}.candidate_rate",
                value=count / self._evaluated,
                step=step,
            )
            for code, count in sorted(self._issues.items())
        )
        self._evaluated = 0
        self._invalid = 0
        self._issues.clear()
        return tuple(samples)


@dataclass(slots=True)
class QDArchiveDiagnosticsObserver:
    """Track attempts, acceptance, turnover, and stagnation for every visited QD cell."""

    every_evaluations: int = 1_000
    include_cell_metrics: bool = False
    component_name: str = "qd_archive_diagnostics"
    provides: frozenset[str] = frozenset({"observer"})
    requires: frozenset[str] = frozenset({"archive"})
    _cell_attempts: Counter[tuple[int, ...]] = field(default_factory=Counter, init=False)
    _cell_accepted: Counter[tuple[int, ...]] = field(default_factory=Counter, init=False)
    _cell_evictions: Counter[tuple[int, ...]] = field(default_factory=Counter, init=False)
    _last_improvement: dict[tuple[int, ...], int] = field(default_factory=dict, init=False)
    _unmapped_statuses: Counter[str] = field(default_factory=Counter, init=False)
    _next_sample: int = field(default=0, init=False)
    _last_sample_step: int = field(default=-1, init=False)

    def __post_init__(self) -> None:
        QDArchiveDiagnosticsObserverConfiguration(
            every_evaluations=self.every_evaluations,
            include_cell_metrics=self.include_cell_metrics,
        )
        self._next_sample = self.every_evaluations

    def on_event(self, event: RunEvent) -> Sequence[MetricSample]:
        if event.event_type is EventType.ARCHIVE_UPDATED:
            step = int(event.payload.get("evaluations", 0))
            for raw in event.payload.get("outcomes", ()):
                if not isinstance(raw, dict):
                    continue
                status = str(raw.get("status", "UNKNOWN"))
                coordinate_raw = raw.get("coordinate")
                if isinstance(coordinate_raw, dict):
                    coordinate_raw = coordinate_raw.get("indices")
                if not isinstance(coordinate_raw, list | tuple):
                    self._unmapped_statuses[status] += 1
                    continue
                coordinate = tuple(int(index) for index in coordinate_raw)
                self._cell_attempts[coordinate] += 1
                if status in _RETAINED_ARCHIVE_STATUSES:
                    self._cell_accepted[coordinate] += 1
                    self._cell_evictions[coordinate] += len(
                        raw.get("evicted_candidate_identifiers", ())
                    )
                    self._last_improvement[coordinate] = step
            due, self._next_sample = _sample_due(
                step,
                self._next_sample,
                self.every_evaluations,
            )
        elif event.event_type is EventType.RUN_COMPLETED:
            step = int(event.payload.get("evaluations", 0))
            due = bool(self._cell_attempts) and step > self._last_sample_step
        else:
            return ()
        if not due:
            return ()
        attempts = tuple(float(value) for value in self._cell_attempts.values())
        acceptance = tuple(
            self._cell_accepted[coordinate] / count
            for coordinate, count in self._cell_attempts.items()
        )
        stagnation = tuple(
            float(step - self._last_improvement.get(coordinate, 0))
            for coordinate in self._cell_attempts
        )
        samples: list[MetricSample] = [
            MetricSample(
                name="qd.visited_cells",
                value=float(len(self._cell_attempts)),
                step=step,
            ),
            MetricSample(
                name="qd.unmapped_attempts",
                value=float(sum(self._unmapped_statuses.values())),
                step=step,
            ),
        ]
        if self.include_cell_metrics:
            for coordinate in sorted(self._cell_attempts):
                label = "_".join(str(index) for index in coordinate)
                attempts_count = self._cell_attempts[coordinate]
                samples.extend(
                    (
                        MetricSample(
                            name=f"qd.cell.{label}.attempts",
                            value=float(attempts_count),
                            step=step,
                        ),
                        MetricSample(
                            name=f"qd.cell.{label}.accepted",
                            value=float(self._cell_accepted[coordinate]),
                            step=step,
                        ),
                        MetricSample(
                            name=f"qd.cell.{label}.acceptance_ratio",
                            value=self._cell_accepted[coordinate] / attempts_count,
                            step=step,
                        ),
                        MetricSample(
                            name=f"qd.cell.{label}.evictions",
                            value=float(self._cell_evictions[coordinate]),
                            step=step,
                        ),
                        MetricSample(
                            name=f"qd.cell.{label}.stagnation_evaluations",
                            value=float(step - self._last_improvement.get(coordinate, 0)),
                            step=step,
                        ),
                    )
                )
        else:
            samples.extend(_numeric_summary("qd.cell_attempts", attempts, step))
            samples.extend(_numeric_summary("qd.cell_acceptance_ratio", acceptance, step))
            samples.extend(_numeric_summary("qd.cell_stagnation_evaluations", stagnation, step))
        self._last_sample_step = step
        return tuple(samples)


def _resident_memory_bytes() -> int | None:
    """Return current RSS using standard-library facilities when available."""

    statm = Path("/proc/self/statm")
    if statm.is_file():
        try:
            resident_pages = int(statm.read_text(encoding="ascii").split()[1])
            return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError, IndexError):
            return None
    try:
        import resource

        maximum_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, OSError, ValueError):
        return None
    return maximum_rss if sys.platform == "darwin" else maximum_rss * 1024


@dataclass(slots=True)
class ResourceUsageObserver:
    """Sample process CPU, memory and thread use at configured progress steps."""

    every_evaluations: int = 1_000
    component_name: str = "resource_usage"
    provides: frozenset[str] = frozenset({"observer"})
    requires: frozenset[str] = frozenset()
    _last_wall_seconds: float = field(default_factory=perf_counter, init=False)
    _last_cpu_seconds: float = field(default_factory=process_time, init=False)
    _next_sample: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        ResourceUsageObserverConfiguration(every_evaluations=self.every_evaluations)
        self._next_sample = self.every_evaluations

    def on_event(self, event: RunEvent) -> Sequence[MetricSample]:
        if event.event_type is not EventType.PROGRESS:
            return ()
        step = int(event.payload.get("evaluations", 0))
        due, self._next_sample = _sample_due(
            step,
            self._next_sample,
            self.every_evaluations,
        )
        if not due:
            return ()
        wall = perf_counter()
        cpu = process_time()
        wall_delta = wall - self._last_wall_seconds
        cpu_delta = cpu - self._last_cpu_seconds
        self._last_wall_seconds = wall
        self._last_cpu_seconds = cpu
        samples = [
            MetricSample(name="resource.process_cpu_seconds", value=cpu, step=step),
            MetricSample(
                name="resource.interval_cpu_percent",
                value=0.0 if wall_delta <= 0.0 else 100.0 * cpu_delta / wall_delta,
                step=step,
            ),
            MetricSample(
                name="resource.active_threads",
                value=float(threading.active_count()),
                step=step,
            ),
        ]
        resident = _resident_memory_bytes()
        if resident is not None:
            samples.append(
                MetricSample(
                    name="resource.resident_memory_bytes",
                    value=float(resident),
                    step=step,
                )
            )
        return tuple(samples)


@dataclass(slots=True)
class CandidateDiversityObserver:
    window_size: int = 128
    distance_sample_size: int = 128
    row_weight: float = 0.5
    every_evaluations: int = 1_000
    component_name: str = "candidate_diversity"
    provides: frozenset[str] = frozenset({"observer"})
    requires: frozenset[str] = frozenset()
    event_payload_requirements: ClassVar[frozenset[EventPayloadRequirement]] = frozenset(
        {EventPayloadRequirement.CANDIDATE_STRUCTURE}
    )
    _window: deque[tuple[str, frozenset[int], frozenset[int]]] = field(init=False)
    _all_signatures: set[str] = field(default_factory=set, init=False)
    _evaluated: int = field(default=0, init=False)
    _next_sample: int = field(default=0, init=False)
    _last_sample_step: int = field(default=-1, init=False)

    def __post_init__(self) -> None:
        CandidateDiversityObserverConfiguration(
            window_size=self.window_size,
            distance_sample_size=self.distance_sample_size,
            row_weight=self.row_weight,
            every_evaluations=self.every_evaluations,
        )
        self._window = deque(maxlen=self.window_size)
        self._next_sample = self.every_evaluations

    def _append(
        self,
        signature: str,
        rows: frozenset[int],
        columns: frozenset[int],
    ) -> None:
        self._window.append((signature, rows, columns))

    def _nearest_distances(self) -> tuple[float, ...]:
        if len(self._window) < 2:
            return ()
        structures = tuple(self._window)
        if len(structures) > self.distance_sample_size:
            indices = np.linspace(
                0,
                len(structures) - 1,
                num=self.distance_sample_size,
                dtype=np.int64,
            )
            structures = tuple(structures[int(index)] for index in indices)
        nearest = [1.0] * len(structures)
        for left_index, (_, left_rows, left_columns) in enumerate(structures):
            for right_index in range(left_index + 1, len(structures)):
                _, right_rows, right_columns = structures[right_index]
                distance = structural_distance(
                    left_rows,
                    left_columns,
                    right_rows,
                    right_columns,
                    row_weight=self.row_weight,
                )
                nearest[left_index] = min(nearest[left_index], distance)
                nearest[right_index] = min(nearest[right_index], distance)
        return tuple(nearest)

    def on_event(self, event: RunEvent) -> Sequence[MetricSample]:
        if event.event_type is EventType.CANDIDATES_EVALUATED:
            for evaluation in _evaluations(event):
                signature = str(evaluation["signature"])
                rows = frozenset(int(value) for value in evaluation.get("rows", ()))
                columns = frozenset(int(value) for value in evaluation.get("columns", ()))
                self._append(signature, rows, columns)
                self._all_signatures.add(signature)
                self._evaluated += 1
            step = int(event.payload.get("evaluations", self._evaluated))
            due, self._next_sample = _sample_due(
                step,
                self._next_sample,
                self.every_evaluations,
            )
        elif event.event_type is EventType.RUN_COMPLETED:
            step = int(event.payload.get("evaluations", self._evaluated))
            due = bool(self._window) and step > self._last_sample_step
        else:
            return ()
        if not due:
            return ()
        signatures = tuple(signature for signature, _, _ in self._window)
        unique = len(set(signatures))
        distances = self._nearest_distances()
        samples: list[MetricSample] = [
            MetricSample(name="diversity.window_size", value=float(len(self._window)), step=step),
            MetricSample(name="diversity.window_unique", value=float(unique), step=step),
            MetricSample(
                name="diversity.distance_sample_size",
                value=float(min(len(self._window), self.distance_sample_size)),
                step=step,
            ),
            MetricSample(
                name="diversity.window_duplicate_ratio",
                value=0.0 if not signatures else 1.0 - unique / len(signatures),
                step=step,
            ),
            MetricSample(
                name="diversity.cumulative_unique",
                value=float(len(self._all_signatures)),
                step=step,
            ),
            MetricSample(
                name="diversity.cumulative_duplicate_ratio",
                value=(
                    0.0
                    if self._evaluated == 0
                    else 1.0 - len(self._all_signatures) / self._evaluated
                ),
                step=step,
            ),
        ]
        samples.extend(_numeric_summary("diversity.nearest_distance", distances, step))
        self._last_sample_step = step
        return tuple(samples)


__all__ = [
    "CandidateDiversityObserver",
    "EvaluationIssuesObserver",
    "QDArchiveDiagnosticsObserver",
    "ResourceUsageObserver",
]
