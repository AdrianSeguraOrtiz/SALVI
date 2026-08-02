"""Immutable public models for the component presentation catalog."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from salvi.components.protocols import ComponentKind
from salvi.domain.enums import PatternKind
from salvi.domain.models import FrozenModel


class ComponentMaturity(StrEnum):
    """Publication status of one component implementation."""

    STABLE = "STABLE"
    EXPERIMENTAL = "EXPERIMENTAL"


class WorkflowStage(StrEnum):
    INPUT = "INPUT"
    PREPARATION = "PREPARATION"
    EVALUATION = "EVALUATION"
    SEARCH = "SEARCH"
    OUTPUT = "OUTPUT"
    ANALYSIS = "ANALYSIS"


class WorkflowConnectionKind(StrEnum):
    PRIMARY = "PRIMARY"
    SUPPORT = "SUPPORT"
    CONTROL = "CONTROL"
    FEEDBACK = "FEEDBACK"


class ParameterWidget(StrEnum):
    BOOLEAN = "BOOLEAN"
    NUMBER = "NUMBER"
    SELECT = "SELECT"
    TEXT = "TEXT"
    STRUCTURED = "STRUCTURED"


class ObserverViewKind(StrEnum):
    PROGRESS = "PROGRESS"
    KPI_SERIES = "KPI_SERIES"
    SERIES = "SERIES"
    DISTRIBUTION = "DISTRIBUTION"
    GROUPED_SERIES = "GROUPED_SERIES"
    STACKED_SERIES = "STACKED_SERIES"
    HEATMAP = "HEATMAP"
    TABLE = "TABLE"


class MetricValueKind(StrEnum):
    COUNTER = "COUNTER"
    GAUGE = "GAUGE"
    DELTA = "DELTA"
    RATE = "RATE"
    DISTRIBUTION = "DISTRIBUTION"


class MetricTemporalScope(StrEnum):
    CUMULATIVE = "CUMULATIVE"
    CURRENT = "CURRENT"
    BATCH = "BATCH"
    WINDOW = "WINDOW"


class MetricPopulation(StrEnum):
    RUN = "RUN"
    EVALUATED_CANDIDATES = "EVALUATED_CANDIDATES"
    ARCHIVE_DECISIONS = "ARCHIVE_DECISIONS"
    REPERTOIRE = "REPERTOIRE"
    QD_CELLS = "QD_CELLS"
    EMITTERS = "EMITTERS"
    PROCESS = "PROCESS"


class WorkflowConnection(FrozenModel):
    """One catalog-owned connection entering a workflow role."""

    source: ComponentKind
    kind: WorkflowConnectionKind = WorkflowConnectionKind.SUPPORT


class WorkflowStagePresentation(FrozenModel):
    """Catalog-owned stage metadata rendered by visual clients."""

    stage: WorkflowStage
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    order: int = Field(ge=0)
    icon: str = Field(min_length=1)
    theme: str = Field(min_length=1)
    preferred_columns: int = Field(default=1, ge=1)


class RolePresentation(FrozenModel):
    """Stable workflow placement for one component role."""

    kind: ComponentKind
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    stage: WorkflowStage
    order: int = Field(ge=0)
    icon: str = Field(default="component", min_length=1)
    repeatable: bool = False
    configuration_path: tuple[str, ...] = ()
    incoming: tuple[WorkflowConnection, ...] = ()
    accepts_pipeline_input: bool = False
    emits_pipeline_output: bool = False


class ObserverMetricPresentation(FrozenModel):
    """Semantic contract for one metric family emitted by an observer."""

    pattern: str = Field(min_length=1)
    label: str = Field(min_length=1)
    description: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    value_kind: MetricValueKind
    temporal_scope: MetricTemporalScope
    population: MetricPopulation
    display_group: str = Field(min_length=1)


class ObserverMetricGroupPresentation(FrozenModel):
    """User-facing explanation for one selectable metric group."""

    name: str = Field(min_length=1)
    label: str = Field(min_length=1)
    description: str = Field(min_length=1)


class ObserverPresentation(FrozenModel):
    """Declarative rendering contract for one observer."""

    view_kind: ObserverViewKind
    title: str = Field(min_length=1)
    metric_patterns: tuple[str, ...]
    empty_message: str = Field(min_length=1)
    x_axis_label: str | None = None
    y_axis_label: str | None = None
    metrics: tuple[ObserverMetricPresentation, ...] = ()
    groups: tuple[ObserverMetricGroupPresentation, ...] = ()


class ComponentReference(FrozenModel):
    """Catalog reference to another component."""

    kind: ComponentKind
    name: str = Field(min_length=1)


class ComponentParameterDescription(FrozenModel):
    """One editable parameter exposed by a component configuration model."""

    name: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    required: bool
    default: Any = None
    value_schema: dict[str, Any]
    applicable_patterns: tuple[PatternKind, ...] = ()
    widget: ParameterWidget
    unit: str | None = None
    advanced: bool = False


class ComponentDescription(FrozenModel):
    """Stable, toolkit-independent metadata consumed by CLI and GUI adapters."""

    kind: ComponentKind
    name: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    provides: tuple[str, ...]
    requires: tuple[str, ...]
    supported_patterns: tuple[PatternKind, ...]
    conflicts: tuple[ComponentReference, ...] = ()
    compatibility_notes: tuple[str, ...] = ()
    maturity: ComponentMaturity = ComponentMaturity.STABLE
    parameters: tuple[ComponentParameterDescription, ...]
    stage: WorkflowStage
    order: int = Field(ge=0)
    observer_view: ObserverPresentation | None = None


__all__ = [
    "ComponentDescription",
    "ComponentMaturity",
    "ComponentParameterDescription",
    "ComponentReference",
    "MetricPopulation",
    "MetricTemporalScope",
    "MetricValueKind",
    "ObserverMetricGroupPresentation",
    "ObserverMetricPresentation",
    "ObserverPresentation",
    "ObserverViewKind",
    "ParameterWidget",
    "RolePresentation",
    "WorkflowConnection",
    "WorkflowConnectionKind",
    "WorkflowStage",
    "WorkflowStagePresentation",
]
