"""Passive observers that derive lightweight metrics from durable run events."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from salvi.components.protocols import EventPayloadRequirement
from salvi.domain.enums import EventType
from salvi.domain.models import MetricSample, RunEvent


class DistributionObserverConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    every_evaluations: Annotated[int, Field(ge=1)] = 1_000


class CandidateDiversityObserverConfiguration(DistributionObserverConfiguration):
    window_size: Annotated[int, Field(ge=2)] = 128
    distance_sample_size: Annotated[int, Field(ge=2)] = 128
    row_weight: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5


class ResourceUsageObserverConfiguration(DistributionObserverConfiguration):
    pass


class QDArchiveDiagnosticsObserverConfiguration(DistributionObserverConfiguration):
    include_cell_metrics: bool = Field(
        default=False,
        description=(
            "Persist exact metrics for every visited cell so a two-dimensional archive can "
            "be rendered as a heatmap; otherwise persist bounded distribution summaries."
        ),
    )


_RETAINED_ARCHIVE_STATUSES = frozenset({"INSERTED", "INSERTED_WITH_EVICTIONS"})
_ARCHIVE_OUTCOME_LABELS = {
    "REJECTED_INVALID": "invalid",
    "REJECTED_DUPLICATE": "duplicate",
    "REJECTED_DOMINATED": "dominated",
    "REJECTED_CAPACITY": "capacity",
    "REJECTED_OUT_OF_BOUNDS": "out_of_bounds",
}


def _numeric_summary(prefix: str, values: Sequence[float], step: int) -> tuple[MetricSample, ...]:
    if not values:
        return ()
    array = np.asarray(values, dtype=np.float64)
    return (
        MetricSample(name=f"{prefix}.minimum", value=float(np.min(array)), step=step),
        MetricSample(
            name=f"{prefix}.first_quartile",
            value=float(np.quantile(array, 0.25)),
            step=step,
        ),
        MetricSample(name=f"{prefix}.median", value=float(np.median(array)), step=step),
        MetricSample(name=f"{prefix}.mean", value=float(np.mean(array)), step=step),
        MetricSample(
            name=f"{prefix}.third_quartile",
            value=float(np.quantile(array, 0.75)),
            step=step,
        ),
        MetricSample(name=f"{prefix}.maximum", value=float(np.max(array)), step=step),
    )


def _evaluations(event: RunEvent) -> tuple[dict[str, Any], ...]:
    raw = event.payload.get("items", ())
    if not isinstance(raw, list):
        return ()
    return tuple(item for item in raw if isinstance(item, dict))


def _sample_due(
    step: int,
    next_sample: int,
    interval: int,
) -> tuple[bool, int]:
    """Advance a cadence after a batch crosses one or more sample thresholds."""

    if step < next_sample:
        return False, next_sample
    while next_sample <= step:
        next_sample += interval
    return True, next_sample


@dataclass(frozen=True, slots=True)
class SearchProgressObserver:
    component_name: str = "search_progress"
    provides: frozenset[str] = frozenset({"observer"})
    requires: frozenset[str] = frozenset()

    def on_event(self, event: RunEvent) -> Sequence[MetricSample]:
        if event.event_type is not EventType.PROGRESS:
            return ()
        step = int(event.payload.get("evaluations", 0))
        return (
            MetricSample(
                name="search.evaluations",
                value=float(step),
                step=step,
            ),
        )


@dataclass(slots=True)
class ArchiveCoverageObserver:
    component_name: str = "archive_coverage"
    provides: frozenset[str] = frozenset({"observer"})
    requires: frozenset[str] = frozenset({"archive"})
    _last_state: tuple[int, int] | None = field(default=None, init=False)

    def on_event(self, event: RunEvent) -> Sequence[MetricSample]:
        if event.event_type is not EventType.ARCHIVE_UPDATED:
            return ()
        step = int(event.payload.get("evaluations", 0))
        state = (
            int(event.payload.get("occupied_cells", 0)),
            int(event.payload.get("repertoire_size", 0)),
        )
        if state == self._last_state:
            return ()
        self._last_state = state
        return (
            MetricSample(
                name="archive.occupied_cells",
                value=float(state[0]),
                step=step,
            ),
            MetricSample(
                name="archive.repertoire_size",
                value=float(state[1]),
                step=step,
            ),
        )


@dataclass(slots=True)
class CandidateOutcomesObserver:
    """Report mutually exclusive archive decisions over bounded evaluation windows."""

    every_evaluations: int = 1_000
    component_name: str = "candidate_outcomes"
    provides: frozenset[str] = frozenset({"observer"})
    requires: frozenset[str] = frozenset({"archive"})
    _statuses: Counter[str] = field(default_factory=Counter, init=False)
    _next_sample: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        DistributionObserverConfiguration(every_evaluations=self.every_evaluations)
        self._next_sample = self.every_evaluations

    def on_event(self, event: RunEvent) -> Sequence[MetricSample]:
        if event.event_type is EventType.ARCHIVE_UPDATED:
            for raw in event.payload.get("outcomes", ()):
                if isinstance(raw, dict):
                    self._statuses[str(raw.get("status", "UNKNOWN")).upper()] += 1
            step = int(event.payload.get("evaluations", 0))
            due, self._next_sample = _sample_due(
                step,
                self._next_sample,
                self.every_evaluations,
            )
        elif event.event_type is EventType.RUN_COMPLETED:
            step = int(event.payload.get("evaluations", 0))
            due = bool(self._statuses)
        else:
            return ()
        if not due or not self._statuses:
            return ()
        total = sum(self._statuses.values())
        retained = sum(self._statuses[status] for status in _RETAINED_ARCHIVE_STATUSES)
        samples = [
            MetricSample(
                name="outcomes.retained_rate",
                value=retained / total,
                step=step,
            )
        ]
        samples.extend(
            MetricSample(
                name=f"outcomes.rejected_{label}_rate",
                value=self._statuses[status] / total,
                step=step,
            )
            for status, label in _ARCHIVE_OUTCOME_LABELS.items()
        )
        self._statuses.clear()
        return tuple(samples)


@dataclass(slots=True)
class DescriptorDistributionObserver:
    every_evaluations: int = 1_000
    component_name: str = "descriptor_distribution"
    provides: frozenset[str] = frozenset({"observer"})
    requires: frozenset[str] = frozenset({"descriptor"})
    event_payload_requirements: ClassVar[frozenset[EventPayloadRequirement]] = frozenset(
        {EventPayloadRequirement.EVALUATION_DESCRIPTORS}
    )
    _next_sample: int = field(default=0, init=False)
    _values: dict[str, list[float]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        DistributionObserverConfiguration(every_evaluations=self.every_evaluations)
        self._next_sample = self.every_evaluations

    def on_event(self, event: RunEvent) -> Sequence[MetricSample]:
        if event.event_type is EventType.CANDIDATES_EVALUATED:
            for evaluation in _evaluations(event):
                for descriptor in evaluation.get("descriptors", ()):
                    if isinstance(descriptor, dict):
                        self._values.setdefault(str(descriptor["name"]), []).append(
                            float(descriptor["value"])
                        )
            step = int(event.payload.get("evaluations", 0))
            due, self._next_sample = _sample_due(
                step,
                self._next_sample,
                self.every_evaluations,
            )
        elif event.event_type is EventType.RUN_COMPLETED:
            step = int(event.payload.get("evaluations", 0))
            due = bool(self._values)
        else:
            return ()
        if not due:
            return ()
        samples = tuple(
            sample
            for name, values in sorted(self._values.items())
            for sample in _numeric_summary(f"descriptor.{name}", values, step)
        )
        self._values.clear()
        return samples


@dataclass(slots=True)
class ArchiveDescriptorDistributionObserver:
    """Summarize the current repertoire instead of recently evaluated candidates."""

    every_evaluations: int = 1_000
    component_name: str = "archive_descriptor_distribution"
    provides: frozenset[str] = frozenset({"observer"})
    requires: frozenset[str] = frozenset({"archive", "descriptor"})
    event_payload_requirements: ClassVar[frozenset[EventPayloadRequirement]] = frozenset(
        {EventPayloadRequirement.EVALUATION_DESCRIPTORS}
    )
    _next_sample: int = field(default=0, init=False)
    _pending: dict[str, dict[str, float]] = field(default_factory=dict, init=False)
    _retained: dict[str, tuple[tuple[int, ...], dict[str, float]]] = field(
        default_factory=dict,
        init=False,
    )
    _last_sample_step: int = field(default=-1, init=False)

    def __post_init__(self) -> None:
        DistributionObserverConfiguration(every_evaluations=self.every_evaluations)
        self._next_sample = self.every_evaluations

    def on_event(self, event: RunEvent) -> Sequence[MetricSample]:
        if event.event_type is EventType.CANDIDATES_EVALUATED:
            for evaluation in _evaluations(event):
                identifier = evaluation.get("identifier")
                if not isinstance(identifier, str):
                    continue
                descriptors = {
                    str(descriptor["name"]): float(descriptor["value"])
                    for descriptor in evaluation.get("descriptors", ())
                    if isinstance(descriptor, dict)
                    and "name" in descriptor
                    and "value" in descriptor
                }
                self._pending[identifier] = descriptors
            return ()
        if event.event_type is EventType.ARCHIVE_UPDATED:
            step = int(event.payload.get("evaluations", 0))
            self._integrate_outcomes(event.payload.get("outcomes", ()))
            due, self._next_sample = _sample_due(
                step,
                self._next_sample,
                self.every_evaluations,
            )
        elif event.event_type is EventType.RUN_COMPLETED:
            step = int(event.payload.get("evaluations", 0))
            due = bool(self._retained) and step > self._last_sample_step
        else:
            return ()
        if not due or not self._retained:
            return ()
        descriptor_values: dict[str, list[float]] = {}
        cell_members: Counter[tuple[int, ...]] = Counter()
        for coordinate, descriptors in self._retained.values():
            cell_members[coordinate] += 1
            for name, value in descriptors.items():
                descriptor_values.setdefault(name, []).append(value)
        samples = [
            sample
            for name, values in sorted(descriptor_values.items())
            for sample in _numeric_summary(f"archive_descriptor.{name}", values, step)
        ]
        samples.extend(
            _numeric_summary(
                "archive_cell.members",
                tuple(float(value) for value in cell_members.values()),
                step,
            )
        )
        self._last_sample_step = step
        return tuple(samples)

    def _integrate_outcomes(self, raw_outcomes: object) -> None:
        if not isinstance(raw_outcomes, list | tuple):
            return
        for raw in raw_outcomes:
            if not isinstance(raw, dict):
                continue
            identifier = raw.get("candidate_identifier")
            if not isinstance(identifier, str):
                continue
            descriptors = self._pending.pop(identifier, None)
            status = str(raw.get("status", "")).upper()
            if status not in _RETAINED_ARCHIVE_STATUSES:
                continue
            for evicted_identifier in raw.get("evicted_candidate_identifiers", ()):
                if isinstance(evicted_identifier, str):
                    self._retained.pop(evicted_identifier, None)
            coordinate_raw = raw.get("coordinate")
            if isinstance(coordinate_raw, dict):
                coordinate_raw = coordinate_raw.get("indices")
            if descriptors is None or not isinstance(coordinate_raw, list | tuple):
                continue
            coordinate = tuple(int(index) for index in coordinate_raw)
            self._retained[identifier] = (coordinate, descriptors)


@dataclass(slots=True)
class ObjectiveDistributionObserver:
    every_evaluations: int = 1_000
    component_name: str = "objective_distribution"
    provides: frozenset[str] = frozenset({"observer"})
    requires: frozenset[str] = frozenset()
    event_payload_requirements: ClassVar[frozenset[EventPayloadRequirement]] = frozenset(
        {EventPayloadRequirement.EVALUATION_OBJECTIVES}
    )
    _next_sample: int = field(default=0, init=False)
    _values: dict[str, list[float]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        DistributionObserverConfiguration(every_evaluations=self.every_evaluations)
        self._next_sample = self.every_evaluations

    def on_event(self, event: RunEvent) -> Sequence[MetricSample]:
        if event.event_type is EventType.CANDIDATES_EVALUATED:
            for evaluation in _evaluations(event):
                for objective in evaluation.get("objectives", ()):
                    if isinstance(objective, dict):
                        self._values.setdefault(str(objective["name"]), []).append(
                            float(objective["value"])
                        )
            step = int(event.payload.get("evaluations", 0))
            due, self._next_sample = _sample_due(
                step,
                self._next_sample,
                self.every_evaluations,
            )
        elif event.event_type is EventType.RUN_COMPLETED:
            step = int(event.payload.get("evaluations", 0))
            due = bool(self._values)
        else:
            return ()
        if not due:
            return ()
        samples = tuple(
            sample
            for name, values in sorted(self._values.items())
            for sample in _numeric_summary(f"objective.{name}", values, step)
        )
        self._values.clear()
        return samples


@dataclass(slots=True)
class EmitterCreditObserver:
    every_evaluations: int = 1_000
    component_name: str = "emitter_credit"
    provides: frozenset[str] = frozenset({"observer"})
    requires: frozenset[str] = frozenset({"scheduler"})
    _previous: dict[str, tuple[int, int, int]] = field(default_factory=dict, init=False)
    _latest_reports: tuple[dict[str, Any], ...] = field(default=(), init=False)
    _next_sample: int = field(default=0, init=False)
    _last_sample_step: int = field(default=-1, init=False)

    def __post_init__(self) -> None:
        DistributionObserverConfiguration(every_evaluations=self.every_evaluations)
        self._next_sample = self.every_evaluations

    def on_event(self, event: RunEvent) -> Sequence[MetricSample]:
        if event.event_type is EventType.EMITTER_CREDIT_UPDATED:
            self._latest_reports = tuple(
                dict(report)
                for report in event.payload.get("reports", ())
                if isinstance(report, dict)
            )
            step = int(event.payload.get("evaluations", 0))
            due, self._next_sample = _sample_due(
                step,
                self._next_sample,
                self.every_evaluations,
            )
        elif event.event_type is EventType.RUN_COMPLETED:
            step = int(event.payload.get("evaluations", 0))
            due = bool(self._latest_reports) and step > self._last_sample_step
        else:
            return ()
        if not due:
            return ()
        return self._samples(step)

    def _samples(self, step: int) -> tuple[MetricSample, ...]:
        samples: list[MetricSample] = []
        reports = self._latest_reports
        total_allocations = sum(int(report.get("allocation_count", 0)) for report in reports)
        for report in reports:
            emitter_name = str(report["emitter_name"])
            prefix = f"emitter.{emitter_name}"
            evaluations = int(report["evaluations"])
            accepted = int(report["accepted"])
            created_cells = int(report["created_cells"])
            previous = self._previous.get(emitter_name, (0, 0, 0))
            window_evaluations = evaluations - previous[0]
            window_accepted = accepted - previous[1]
            window_created_cells = created_cells - previous[2]
            if window_evaluations > 0:
                samples.extend(
                    (
                        MetricSample(
                            name=f"{prefix}.retention_rate",
                            value=window_accepted / window_evaluations,
                            step=step,
                        ),
                        MetricSample(
                            name=f"{prefix}.new_cell_rate",
                            value=window_created_cells / window_evaluations,
                            step=step,
                        ),
                    )
                )
            samples.extend(
                (
                    MetricSample(
                        name=f"{prefix}.credit",
                        value=float(report["credit"]),
                        step=step,
                    ),
                    MetricSample(
                        name=f"{prefix}.allocation_share",
                        value=(
                            0.0
                            if total_allocations <= 0
                            else int(report["allocation_count"]) / total_allocations
                        ),
                        step=step,
                    ),
                )
            )
            self._previous[emitter_name] = (evaluations, accepted, created_cells)
        self._last_sample_step = step
        return tuple(samples)


@dataclass(frozen=True, slots=True)
class RuntimeThroughputObserver:
    """Persist bounded-executor throughput without inspecting worker internals."""

    component_name: str = "runtime_throughput"
    provides: frozenset[str] = frozenset({"observer"})
    requires: frozenset[str] = frozenset()

    def on_event(self, event: RunEvent) -> Sequence[MetricSample]:
        if event.event_type is not EventType.CANDIDATES_EVALUATED:
            return ()
        runtime = event.payload.get("runtime")
        if not isinstance(runtime, dict):
            return ()
        step = int(event.payload.get("evaluations", 0))
        count = int(event.payload.get("count", 0))
        duration = float(runtime.get("duration_seconds", 0.0))
        samples = [
            MetricSample(
                name="runtime.evaluations_per_second",
                value=0.0 if duration <= 0.0 else count / duration,
                step=step,
            ),
        ]
        for name in ("worker_count", "peak_in_flight"):
            value = runtime.get(name)
            if isinstance(value, int | float):
                samples.append(MetricSample(name=f"runtime.{name}", value=float(value), step=step))
        return tuple(samples)


@dataclass(slots=True)
class ComponentTimingObserver:
    """Attribute wall-clock cost to setup, search, scientific, output, and observer work."""

    every_evaluations: int = 1_000
    component_name: str = "component_timing"
    provides: frozenset[str] = frozenset({"observer"})
    requires: frozenset[str] = frozenset()
    event_payload_requirements: ClassVar[frozenset[EventPayloadRequirement]] = frozenset(
        {EventPayloadRequirement.COMPONENT_TIMINGS}
    )
    _next_sample: int = field(init=False)
    _last_step: int = 0
    _window_seconds: dict[str, float] = field(default_factory=dict, init=False)
    _observer_seconds: dict[str, float] = field(default_factory=dict, init=False)
    _candidate_seconds: list[float] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        configuration = DistributionObserverConfiguration(every_evaluations=self.every_evaluations)
        self.every_evaluations = configuration.every_evaluations
        self._next_sample = self.every_evaluations

    def record_observer_duration(self, observer_name: str, duration_seconds: float) -> None:
        """Receive dispatcher-measured observer cost without timing observers recursively."""

        if duration_seconds >= 0.0:
            self._observer_seconds[observer_name] = (
                self._observer_seconds.get(observer_name, 0.0) + duration_seconds
            )

    def on_event(self, event: RunEvent) -> Sequence[MetricSample]:
        step = int(event.payload.get("evaluations", self._last_step))
        self._last_step = max(self._last_step, step)
        if event.event_type is EventType.DATASET_PREPARED:
            return self._setup_samples(event)
        if event.event_type is EventType.ENGINE_INITIALIZED:
            runtime = event.payload.get("runtime")
            if isinstance(runtime, dict):
                self._collect_component_timings(runtime)
            return self._single_runtime_sample(
                event,
                "timing.setup.search_engine_initialization.seconds",
                step,
            )
        mapping = {
            EventType.CANDIDATES_ASKED: "timing.search.candidate_generation.seconds",
            EventType.CANDIDATES_EVALUATED: "timing.search.evaluation_batch.seconds",
            EventType.ENGINE_UPDATED: "timing.search.search_update.seconds",
        }
        metric_name = mapping.get(event.event_type)
        if metric_name is not None:
            runtime = event.payload.get("runtime")
            if isinstance(runtime, dict):
                duration = runtime.get("duration_seconds")
                if isinstance(duration, int | float):
                    self._window_seconds[metric_name] = self._window_seconds.get(
                        metric_name, 0.0
                    ) + float(duration)
                if event.event_type is EventType.CANDIDATES_EVALUATED:
                    self._collect_evaluation_timings(runtime)
                else:
                    self._collect_component_timings(runtime)
            return ()
        if event.event_type is EventType.FINAL_SELECTION_COMPLETED:
            return self._single_runtime_sample(
                event,
                "timing.output.final_selection.seconds",
                step,
            )
        if event.event_type is EventType.ARTIFACT_WRITTEN:
            artifact_kind = str(event.payload.get("artifact_kind", "artifact"))
            return self._single_runtime_sample(
                event,
                f"timing.output.{artifact_kind}.seconds",
                step,
            )
        if event.event_type is EventType.PROGRESS:
            due, self._next_sample = _sample_due(
                step,
                self._next_sample,
                self.every_evaluations,
            )
            return self._flush(step) if due else ()
        if event.event_type is EventType.RUN_COMPLETED:
            return self._flush(step)
        return ()

    def _setup_samples(self, event: RunEvent) -> tuple[MetricSample, ...]:
        samples: list[MetricSample] = []
        loading = event.payload.get("loading_seconds")
        if isinstance(loading, int | float):
            samples.append(
                MetricSample(
                    name="timing.setup.dataset_loading.seconds",
                    value=float(loading),
                    step=0,
                )
            )
        raw_steps = event.payload.get("preprocessing_steps")
        if isinstance(raw_steps, list):
            for raw in raw_steps:
                if not isinstance(raw, dict):
                    continue
                component = raw.get("component_name")
                duration = raw.get("duration_seconds")
                if isinstance(component, str) and isinstance(duration, int | float):
                    samples.append(
                        MetricSample(
                            name=f"timing.setup.preprocessing.{component}.seconds",
                            value=float(duration),
                            step=0,
                        )
                    )
        return tuple(samples)

    @staticmethod
    def _single_runtime_sample(
        event: RunEvent,
        name: str,
        step: int,
    ) -> tuple[MetricSample, ...]:
        runtime = event.payload.get("runtime")
        if not isinstance(runtime, dict):
            return ()
        duration = runtime.get("duration_seconds")
        if not isinstance(duration, int | float):
            return ()
        return (MetricSample(name=name, value=float(duration), step=step),)

    def _collect_evaluation_timings(self, runtime: dict[str, Any]) -> None:
        raw_candidates = runtime.get("candidate_duration_seconds")
        if isinstance(raw_candidates, list):
            self._candidate_seconds.extend(
                float(value) for value in raw_candidates if isinstance(value, int | float)
            )
        raw_components = runtime.get("component_duration_seconds")
        if isinstance(raw_components, dict):
            for component, duration in raw_components.items():
                if isinstance(component, str) and isinstance(duration, int | float):
                    name = f"timing.scientific.{component}.seconds"
                    self._window_seconds[name] = self._window_seconds.get(name, 0.0) + float(
                        duration
                    )

    def _collect_component_timings(self, runtime: dict[str, Any]) -> None:
        raw_components = runtime.get("component_duration_seconds")
        if not isinstance(raw_components, dict):
            return
        for component, duration in raw_components.items():
            if isinstance(component, str) and isinstance(duration, int | float):
                name = f"timing.component.{component}.seconds"
                self._window_seconds[name] = self._window_seconds.get(name, 0.0) + float(duration)

    def _flush(self, step: int) -> tuple[MetricSample, ...]:
        samples = [
            MetricSample(name=name, value=value, step=step)
            for name, value in sorted(self._window_seconds.items())
        ]
        samples.extend(
            MetricSample(
                name=f"timing.observer.{name}.seconds",
                value=value,
                step=step,
            )
            for name, value in sorted(self._observer_seconds.items())
        )
        if self._candidate_seconds:
            values = np.asarray(self._candidate_seconds, dtype=np.float64)
            samples.extend(
                (
                    MetricSample(
                        name="timing.candidate_evaluation.mean_seconds",
                        value=float(np.mean(values)),
                        step=step,
                    ),
                    MetricSample(
                        name="timing.candidate_evaluation.p95_seconds",
                        value=float(np.quantile(values, 0.95)),
                        step=step,
                    ),
                )
            )
        self._window_seconds.clear()
        self._observer_seconds.clear()
        self._candidate_seconds.clear()
        return tuple(samples)


__all__ = [
    "ArchiveCoverageObserver",
    "ArchiveDescriptorDistributionObserver",
    "CandidateDiversityObserverConfiguration",
    "CandidateOutcomesObserver",
    "ComponentTimingObserver",
    "DescriptorDistributionObserver",
    "DistributionObserverConfiguration",
    "EmitterCreditObserver",
    "ObjectiveDistributionObserver",
    "QDArchiveDiagnosticsObserverConfiguration",
    "ResourceUsageObserverConfiguration",
    "RuntimeThroughputObserver",
    "SearchProgressObserver",
]
