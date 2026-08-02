"""Behavioral contracts for composable SALVI components."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from salvi.domain.enums import EvaluationIntegrationMode, ObjectiveDirection
from salvi.domain.models import (
    Bicluster,
    Candidate,
    ConstraintResult,
    Evaluation,
    MetricSample,
    ObjectiveResult,
    Repertoire,
    RunEvent,
)
from salvi.domain.search import (
    ArchiveCellTarget,
    ArchiveInsertionOutcome,
    BootstrapCellState,
    CandidateBounds,
    DescriptorDomain,
    EmitterAllocation,
    EmitterFeedback,
    EvaluationBatch,
    SchedulerReport,
    SearchCheckpoint,
    SearchProgress,
    SearchUpdate,
    TerminationProgress,
)

if TYPE_CHECKING:
    from salvi.api.run import RunSpecification
    from salvi.application.context import RunContext
    from salvi.components.contracts import EngineCompositionContract
    from salvi.domain.prepared import PreparedDataset
    from salvi.evaluation.workspace import EvaluationWorkspace


class ComponentKind(StrEnum):
    SOURCE_COLUMN_FILTER = "source_column_filter"
    MISSING_VALUES_POLICY = "missing_values_policy"
    COLUMN_AUGMENTATION = "column_augmentation"
    NUMERIC_TRANSFORMATION = "numeric_transformation"
    CANDIDATE_VALIDITY_POLICY = "candidate_validity_policy"
    EVALUATION_SUPPORT_POLICY = "evaluation_support_policy"
    INITIALIZER = "initializer"
    OBJECTIVE = "objective"
    CONSTRAINT = "constraint"
    DESCRIPTOR = "descriptor"
    ARCHIVE = "archive"
    PARENT_SELECTION_POLICY = "parent_selection_policy"
    MATE_SELECTION_POLICY = "mate_selection_policy"
    CROSSOVER_OPERATOR = "crossover_operator"
    MUTATION_OPERATOR = "mutation_operator"
    EMITTER = "emitter"
    SCHEDULER = "scheduler"
    SEARCH_ENGINE = "search_engine"
    EVALUATION_EXECUTOR = "evaluation_executor"
    OBSERVER = "observer"
    TERMINATION = "termination"
    FINAL_SELECTOR = "final_selector"


class EventPayloadRequirement(StrEnum):
    """Optional durable-event fields requested by configured observers."""

    CANDIDATE_STRUCTURE = "candidate-structure"
    COMPONENT_TIMINGS = "component-timings"
    EVALUATION_CONSTRAINTS = "evaluation-constraints"
    EVALUATION_DESCRIPTORS = "evaluation-descriptors"
    EVALUATION_ISSUES = "evaluation-issues"
    EVALUATION_OBJECTIVES = "evaluation-objectives"


@runtime_checkable
class Component(Protocol):
    @property
    def component_name(self) -> str: ...

    @property
    def provides(self) -> frozenset[str]: ...

    @property
    def requires(self) -> frozenset[str]: ...


@runtime_checkable
class CompositionAwareComponent(Protocol):
    """Optional component-owned validation against a complete composition."""

    def composition_issues(
        self,
        components: Sequence[tuple[ComponentKind, Component]],
    ) -> Sequence[str]: ...


@runtime_checkable
class SourceColumnFilteringStage(Component, Protocol):
    @property
    def stage_kind(self) -> ComponentKind: ...

    def transform(self, dataset: PreparedDataset) -> PreparedDataset: ...


@runtime_checkable
class MissingValuesPolicy(Component, Protocol):
    def apply(self, dataset: PreparedDataset) -> PreparedDataset: ...


@runtime_checkable
class ColumnAugmentationStage(Component, Protocol):
    @property
    def stage_kind(self) -> ComponentKind: ...

    def transform(self, dataset: PreparedDataset) -> PreparedDataset: ...


@runtime_checkable
class NumericTransformationStage(Component, Protocol):
    @property
    def stage_kind(self) -> ComponentKind: ...

    def transform(self, dataset: PreparedDataset) -> PreparedDataset: ...


@runtime_checkable
class CandidateValidityPolicy(Component, Protocol):
    def validate_dataset(self, dataset: PreparedDataset) -> None: ...

    def validate(self, candidate: Candidate, dataset: PreparedDataset) -> None: ...

    def bounds(self, dataset: PreparedDataset) -> CandidateBounds: ...


@runtime_checkable
class EvaluationSupportPolicy(Component, Protocol):
    def validate_dataset(self, dataset: PreparedDataset) -> None: ...

    def required_observations(self, opportunity_count: int) -> int: ...

    def is_sufficient(self, observed_count: int, opportunity_count: int) -> bool: ...


@runtime_checkable
class Initializer(Component, Protocol):
    def initialize(
        self,
        context: RunContext,
        count: int,
        *,
        start_sequence: int = 0,
    ) -> Sequence[Candidate]: ...


@runtime_checkable
class CellCoverageInitializer(Initializer, Protocol):
    """Initializer capable of filling explicit QD cardinality cells."""

    def bootstrap_plan(
        self,
        context: RunContext,
        targets: Sequence[ArchiveCellTarget],
    ) -> Sequence[BootstrapCellState]: ...

    def initialize_bootstrap(
        self,
        context: RunContext,
        states: Sequence[BootstrapCellState],
        *,
        start_sequence: int = 0,
    ) -> Sequence[Candidate]: ...


@runtime_checkable
class Objective(Component, Protocol):
    @property
    def direction(self) -> ObjectiveDirection: ...

    def evaluate(self, candidate: Candidate, workspace: EvaluationWorkspace) -> ObjectiveResult: ...


@runtime_checkable
class Constraint(Component, Protocol):
    def evaluate(
        self,
        candidate: Candidate,
        workspace: EvaluationWorkspace,
    ) -> ConstraintResult: ...


@runtime_checkable
class Descriptor(Component, Protocol):
    def domain(self, context: RunContext) -> DescriptorDomain: ...

    def describe(self, candidate: Candidate, workspace: EvaluationWorkspace) -> float: ...


@runtime_checkable
class Archive(Component, Protocol):
    @property
    def occupied_cell_count(self) -> int: ...

    @property
    def repertoire_size(self) -> int: ...

    def initialize(
        self,
        context: RunContext,
        objectives: Sequence[Objective],
        descriptors: Sequence[Descriptor],
        constraints: Sequence[Constraint] = (),
    ) -> None: ...

    def add(self, evaluations: Sequence[Evaluation]) -> Sequence[ArchiveInsertionOutcome]: ...

    def repertoire(self) -> Repertoire: ...

    def restore(self, repertoire: Repertoire) -> None: ...


@runtime_checkable
class CellTargetArchive(Archive, Protocol):
    def cell_targets(self) -> Sequence[ArchiveCellTarget]: ...


@runtime_checkable
class ParentSelectionPolicy(Component, Protocol):
    def select(
        self,
        repertoire: Repertoire,
        generator: Any,
        *,
        pool_size: int,
        eligible: Callable[[Evaluation], bool],
        guided: bool,
    ) -> Evaluation | None: ...


@runtime_checkable
class MateSelectionPolicy(Component, Protocol):
    def select(
        self,
        repertoire: Repertoire,
        generator: Any,
    ) -> tuple[Evaluation, Evaluation] | None: ...


@runtime_checkable
class CrossoverOperator(Component, Protocol):
    def cross(
        self,
        context: RunContext,
        first: Evaluation,
        second: Evaluation,
        generator: Any,
    ) -> Bicluster: ...


@runtime_checkable
class MutationOperator(Component, Protocol):
    def mutate(
        self,
        context: RunContext,
        parent: Evaluation,
        generator: Any,
    ) -> Bicluster: ...


@runtime_checkable
class Emitter(Component, Protocol):
    def emit(
        self,
        context: RunContext,
        repertoire: Repertoire,
        count: int,
        *,
        start_sequence: int = 0,
    ) -> Sequence[Candidate]: ...


@runtime_checkable
class Scheduler(Component, Protocol):
    def allocate(
        self,
        emitters: Sequence[Emitter],
        count: int,
    ) -> Sequence[EmitterAllocation]: ...

    def update(self, feedback: Sequence[EmitterFeedback]) -> None: ...

    def reports(self, emitters: Sequence[Emitter]) -> Sequence[SchedulerReport]: ...

    def snapshot(self) -> dict[str, Any]: ...

    def restore(self, state: dict[str, Any], emitters: Sequence[Emitter]) -> None: ...


@runtime_checkable
class SearchEngine(Component, Protocol):
    @property
    def composition_contract(self) -> EngineCompositionContract: ...

    @property
    def batch_size(self) -> int: ...

    @property
    def pending_candidates(self) -> Sequence[Candidate]: ...

    def initialize(self, specification: RunSpecification, context: RunContext) -> None: ...

    def ask(self, count: int) -> Sequence[Candidate]: ...

    def tell(self, evaluations: Sequence[Evaluation]) -> SearchUpdate: ...

    def finished(self) -> bool: ...

    def result(self) -> Repertoire: ...

    def progress(self) -> SearchProgress: ...

    def checkpoint(self) -> SearchCheckpoint: ...

    def restore(self, checkpoint: SearchCheckpoint) -> None: ...


@runtime_checkable
class ComponentTimingSource(Protocol):
    """Optional runtime contract for engines that expose component-level timings."""

    def drain_component_timings(self) -> Sequence[tuple[str, float]]: ...


@runtime_checkable
class EvaluationExecutor(Component, Protocol):
    @property
    def integration_mode(self) -> EvaluationIntegrationMode: ...

    @property
    def max_in_flight(self) -> int | None: ...

    @property
    def uses_child_processes(self) -> bool: ...

    def validate_worker_count(self, worker_count: int) -> None: ...

    def evaluate(
        self,
        candidates: Sequence[Candidate],
        objectives: Sequence[Objective],
        descriptors: Sequence[Descriptor],
        workspace: EvaluationWorkspace,
        *,
        constraints: Sequence[Constraint] = (),
        worker_count: int = 1,
        cancellation: CancellationSignal | None = None,
        collect_timings: bool = True,
    ) -> EvaluationBatch: ...

    def close(self) -> None: ...


@runtime_checkable
class CancellationSignal(Protocol):
    @property
    def cancelled(self) -> bool: ...

    def raise_if_cancelled(self) -> None: ...


@runtime_checkable
class Observer(Component, Protocol):
    def on_event(self, event: RunEvent) -> Sequence[MetricSample]: ...


@runtime_checkable
class ObserverPayloadRequirements(Protocol):
    """Optional observer contract for requesting expensive event payloads."""

    @property
    def event_payload_requirements(self) -> frozenset[EventPayloadRequirement]: ...


@runtime_checkable
class TerminationCriterion(Component, Protocol):
    def should_stop(self, evaluations: int) -> bool: ...

    def remaining(self, evaluations: int) -> int | None: ...

    def progress(self, evaluations: int) -> TerminationProgress: ...


@runtime_checkable
class FinalSelector(Component, Protocol):
    def select(
        self,
        context: RunContext,
        repertoire: Repertoire,
    ) -> Repertoire: ...
