"""Programmatic run composition API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self, cast

from salvi.components.contracts import (
    validate_component_capabilities,
    validate_component_composition,
    validate_optional_strategy_consumers,
)
from salvi.components.protocols import (
    Archive,
    CandidateValidityPolicy,
    ColumnAugmentationStage,
    Component,
    ComponentKind,
    Constraint,
    CrossoverOperator,
    Descriptor,
    Emitter,
    EvaluationExecutor,
    EvaluationSupportPolicy,
    FinalSelector,
    Initializer,
    MateSelectionPolicy,
    MissingValuesPolicy,
    MutationOperator,
    NumericTransformationStage,
    Objective,
    Observer,
    ParentSelectionPolicy,
    Scheduler,
    SearchEngine,
    SourceColumnFilteringStage,
    TerminationCriterion,
)
from salvi.domain.models import Dataset
from salvi.exceptions import ComponentError
from salvi.patterns.configuration import PatternConfiguration


@dataclass(frozen=True, slots=True)
class RunSpecification:
    run_identifier: str
    seed: int
    dataset: Dataset
    patterns: PatternConfiguration
    worker_count: int
    search_fingerprint: str
    source_column_filters: tuple[SourceColumnFilteringStage, ...]
    missing_values_policy: MissingValuesPolicy
    column_augmentations: tuple[ColumnAugmentationStage, ...]
    numeric_transformations: tuple[NumericTransformationStage, ...]
    candidate_validity_policy: CandidateValidityPolicy
    evaluation_support_policy: EvaluationSupportPolicy
    search_engine: SearchEngine
    objectives: tuple[Objective, ...]
    constraints: tuple[Constraint, ...]
    descriptors: tuple[Descriptor, ...]
    archive: Archive | None
    parent_selection_policy: ParentSelectionPolicy | None
    mate_selection_policy: MateSelectionPolicy | None
    crossover_operator: CrossoverOperator | None
    mutation_operator: MutationOperator | None
    emitters: tuple[Emitter, ...]
    scheduler: Scheduler | None
    initializer: Initializer
    executor: EvaluationExecutor
    observers: tuple[Observer, ...]
    termination: TerminationCriterion
    final_selector: FinalSelector | None

    def require_archive(self) -> Archive:
        if self.archive is None:
            raise ComponentError("the active search engine requires an archive")
        return self.archive

    def require_scheduler(self) -> Scheduler:
        if self.scheduler is None:
            raise ComponentError("the active search engine requires a scheduler")
        return self.scheduler

    def require_crossover_operator(self) -> CrossoverOperator:
        if self.crossover_operator is None:
            raise ComponentError("the active search engine requires a crossover operator")
        return self.crossover_operator

    def require_mutation_operator(self) -> MutationOperator:
        if self.mutation_operator is None:
            raise ComponentError("the active search engine requires a mutation operator")
        return self.mutation_operator


class SalviRunBuilder:
    """Mutable builder that emits one validated immutable specification."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        run_identifier: str = "programmatic-run",
        seed: int = 0,
        patterns: PatternConfiguration | None = None,
        worker_count: int = 1,
        search_fingerprint: str = "0" * 64,
    ) -> None:
        if not run_identifier.strip():
            raise ComponentError("run_identifier must not be blank")
        if seed < 0:
            raise ComponentError("seed must be non-negative")
        if worker_count < 1:
            raise ComponentError("worker_count must be positive")
        if len(search_fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in search_fingerprint
        ):
            raise ComponentError("search_fingerprint must be a lowercase SHA-256 value")
        self._run_identifier = run_identifier
        self._seed = seed
        self._dataset = dataset
        self._patterns = patterns or PatternConfiguration()
        self._worker_count = worker_count
        self._search_fingerprint = search_fingerprint
        self._source_column_filters: list[SourceColumnFilteringStage] = []
        self._missing_values_policy: MissingValuesPolicy | None = None
        self._column_augmentations: list[ColumnAugmentationStage] = []
        self._numeric_transformations: list[NumericTransformationStage] = []
        self._candidate_validity_policy: CandidateValidityPolicy | None = None
        self._evaluation_support_policy: EvaluationSupportPolicy | None = None
        self._search_engine: SearchEngine | None = None
        self._objectives: list[Objective] = []
        self._constraints: list[Constraint] = []
        self._descriptors: list[Descriptor] = []
        self._archive: Archive | None = None
        self._parent_selection_policy: ParentSelectionPolicy | None = None
        self._mate_selection_policy: MateSelectionPolicy | None = None
        self._crossover_operator: CrossoverOperator | None = None
        self._mutation_operator: MutationOperator | None = None
        self._emitters: list[Emitter] = []
        self._scheduler: Scheduler | None = None
        self._initializer: Initializer | None = None
        self._executor: EvaluationExecutor | None = None
        self._observers: list[Observer] = []
        self._termination: TerminationCriterion | None = None
        self._final_selector: FinalSelector | None = None

    def _assign_unique(self, attribute: str, component: object) -> None:
        if getattr(self, attribute) is not None:
            raise ComponentError(f"{attribute.removeprefix('_')} is already configured")
        setattr(self, attribute, component)

    def with_missing_values_policy(self, policy: MissingValuesPolicy) -> Self:
        self._assign_unique("_missing_values_policy", policy)
        return self

    def add_column_augmentation(self, stage: ColumnAugmentationStage) -> Self:
        self._column_augmentations.append(stage)
        return self

    def add_source_column_filter(self, stage: SourceColumnFilteringStage) -> Self:
        self._source_column_filters.append(stage)
        return self

    def add_numeric_transformation(self, stage: NumericTransformationStage) -> Self:
        self._numeric_transformations.append(stage)
        return self

    def with_candidate_validity_policy(self, policy: CandidateValidityPolicy) -> Self:
        self._assign_unique("_candidate_validity_policy", policy)
        return self

    def with_evaluation_support_policy(self, policy: EvaluationSupportPolicy) -> Self:
        self._assign_unique("_evaluation_support_policy", policy)
        return self

    def with_search_engine(self, engine: SearchEngine) -> Self:
        self._assign_unique("_search_engine", engine)
        return self

    def add_objective(self, objective: Objective) -> Self:
        self._objectives.append(objective)
        return self

    def add_constraint(self, constraint: Constraint) -> Self:
        self._constraints.append(constraint)
        return self

    def add_descriptor(self, descriptor: Descriptor) -> Self:
        self._descriptors.append(descriptor)
        return self

    def with_archive(self, archive: Archive) -> Self:
        self._assign_unique("_archive", archive)
        return self

    def with_parent_selection_policy(self, policy: ParentSelectionPolicy) -> Self:
        self._assign_unique("_parent_selection_policy", policy)
        return self

    def with_mate_selection_policy(self, policy: MateSelectionPolicy) -> Self:
        self._assign_unique("_mate_selection_policy", policy)
        return self

    def with_crossover_operator(self, operator: CrossoverOperator) -> Self:
        self._assign_unique("_crossover_operator", operator)
        return self

    def with_mutation_operator(self, operator: MutationOperator) -> Self:
        self._assign_unique("_mutation_operator", operator)
        return self

    def add_emitter(self, emitter: Emitter) -> Self:
        self._emitters.append(emitter)
        return self

    def with_scheduler(self, scheduler: Scheduler) -> Self:
        self._assign_unique("_scheduler", scheduler)
        return self

    def with_initializer(self, initializer: Initializer) -> Self:
        self._assign_unique("_initializer", initializer)
        return self

    def with_executor(self, executor: EvaluationExecutor) -> Self:
        self._assign_unique("_executor", executor)
        return self

    def add_observer(self, observer: Observer) -> Self:
        self._observers.append(observer)
        return self

    def with_termination(self, termination: TerminationCriterion) -> Self:
        self._assign_unique("_termination", termination)
        return self

    def with_final_selector(self, selector: FinalSelector) -> Self:
        self._assign_unique("_final_selector", selector)
        return self

    def build(self) -> RunSpecification:
        required = {
            "missing_values_policy": self._missing_values_policy,
            "candidate_validity_policy": self._candidate_validity_policy,
            "evaluation_support_policy": self._evaluation_support_policy,
            "search_engine": self._search_engine,
            "initializer": self._initializer,
            "executor": self._executor,
            "termination": self._termination,
        }
        missing = tuple(name for name, component in required.items() if component is None)
        if missing:
            raise ComponentError(f"missing required components: {', '.join(missing)}")
        if not self._objectives:
            raise ComponentError("at least one objective is required")
        for name, collection in (
            ("objectives", self._objectives),
            ("constraints", self._constraints),
            ("descriptors", self._descriptors),
            ("emitters", self._emitters),
        ):
            component_names = tuple(component.component_name for component in collection)
            if len(set(component_names)) != len(component_names):
                raise ComponentError(f"{name} must have unique component names")

        assert self._missing_values_policy is not None
        assert self._candidate_validity_policy is not None
        assert self._evaluation_support_policy is not None
        assert self._search_engine is not None
        assert self._initializer is not None
        assert self._executor is not None
        assert self._termination is not None
        self._executor.validate_worker_count(self._worker_count)
        role_components: list[tuple[ComponentKind, object]] = [
            *(
                (ComponentKind.SOURCE_COLUMN_FILTER, component)
                for component in self._source_column_filters
            ),
            (ComponentKind.MISSING_VALUES_POLICY, self._missing_values_policy),
            *(
                (ComponentKind.COLUMN_AUGMENTATION, component)
                for component in self._column_augmentations
            ),
            *(
                (ComponentKind.NUMERIC_TRANSFORMATION, component)
                for component in self._numeric_transformations
            ),
            (ComponentKind.CANDIDATE_VALIDITY_POLICY, self._candidate_validity_policy),
            (ComponentKind.EVALUATION_SUPPORT_POLICY, self._evaluation_support_policy),
            (ComponentKind.INITIALIZER, self._initializer),
            *((ComponentKind.OBJECTIVE, component) for component in self._objectives),
            *((ComponentKind.CONSTRAINT, component) for component in self._constraints),
            *((ComponentKind.DESCRIPTOR, component) for component in self._descriptors),
            *((ComponentKind.EMITTER, component) for component in self._emitters),
            (ComponentKind.EVALUATION_EXECUTOR, self._executor),
            (ComponentKind.TERMINATION, self._termination),
            *((ComponentKind.OBSERVER, component) for component in self._observers),
            (ComponentKind.SEARCH_ENGINE, self._search_engine),
        ]
        optional_roles = (
            (ComponentKind.ARCHIVE, self._archive),
            (ComponentKind.PARENT_SELECTION_POLICY, self._parent_selection_policy),
            (ComponentKind.MATE_SELECTION_POLICY, self._mate_selection_policy),
            (ComponentKind.CROSSOVER_OPERATOR, self._crossover_operator),
            (ComponentKind.MUTATION_OPERATOR, self._mutation_operator),
            (ComponentKind.SCHEDULER, self._scheduler),
            (ComponentKind.FINAL_SELECTOR, self._final_selector),
        )
        role_components.extend(
            (kind, component) for kind, component in optional_roles if component is not None
        )
        typed_components = tuple(
            (kind, cast(Component, component)) for kind, component in role_components
        )
        self._search_engine.composition_contract.validate(
            tuple((kind, component.component_name) for kind, component in typed_components)
        )
        validate_component_capabilities(
            tuple(
                (
                    kind,
                    component.component_name,
                    component.provides,
                    component.requires,
                )
                for kind, component in typed_components
            )
        )
        validate_optional_strategy_consumers(
            tuple(
                (kind, component.component_name, component.requires)
                for kind, component in typed_components
            )
        )
        validate_component_composition(typed_components)

        return RunSpecification(
            run_identifier=self._run_identifier,
            seed=self._seed,
            dataset=self._dataset,
            patterns=self._patterns,
            worker_count=self._worker_count,
            search_fingerprint=self._search_fingerprint,
            source_column_filters=tuple(self._source_column_filters),
            missing_values_policy=self._missing_values_policy,
            column_augmentations=tuple(self._column_augmentations),
            numeric_transformations=tuple(self._numeric_transformations),
            candidate_validity_policy=self._candidate_validity_policy,
            evaluation_support_policy=self._evaluation_support_policy,
            search_engine=self._search_engine,
            objectives=tuple(self._objectives),
            constraints=tuple(self._constraints),
            descriptors=tuple(self._descriptors),
            archive=self._archive,
            parent_selection_policy=self._parent_selection_policy,
            mate_selection_policy=self._mate_selection_policy,
            crossover_operator=self._crossover_operator,
            mutation_operator=self._mutation_operator,
            emitters=tuple(self._emitters),
            scheduler=self._scheduler,
            initializer=self._initializer,
            executor=self._executor,
            observers=tuple(self._observers),
            termination=self._termination,
            final_selector=self._final_selector,
        )


class SalviRun:
    """Entry point for programmatic SALVI composition."""

    @staticmethod
    def builder(
        dataset: Dataset,
        *,
        run_identifier: str = "programmatic-run",
        seed: int = 0,
        patterns: PatternConfiguration | None = None,
        worker_count: int = 1,
        search_fingerprint: str = "0" * 64,
    ) -> SalviRunBuilder:
        return SalviRunBuilder(
            dataset,
            run_identifier=run_identifier,
            seed=seed,
            patterns=patterns,
            worker_count=worker_count,
            search_fingerprint=search_fingerprint,
        )
