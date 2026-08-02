"""Configuration-to-component composition."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from time import perf_counter
from typing import cast

from salvi.api.run import RunSpecification, SalviRun
from salvi.application.configuration import ComponentSpec, SalviConfiguration
from salvi.application.context import (
    NamedRandomStreams,
    PreparedRun,
    PreprocessingReport,
    PreprocessingStepReport,
    RunContext,
)
from salvi.components.contracts import runtime_capability_errors
from salvi.components.defaults import default_component_registry
from salvi.components.protocols import (
    Archive,
    CandidateValidityPolicy,
    ColumnAugmentationStage,
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
from salvi.components.registry import ComponentRegistry
from salvi.domain.prepared import PreparedDataset
from salvi.exceptions import ComponentError
from salvi.infrastructure.dataset_bundle import DatasetBundleReader
from salvi.infrastructure.files import sha256_file


def _create(
    registry: ComponentRegistry,
    kind: ComponentKind,
    specification: ComponentSpec,
) -> object:
    return registry.create(kind, specification.name, specification.parameters)


def _configured_components(
    configuration: SalviConfiguration,
) -> tuple[tuple[ComponentKind, str], ...]:
    """Return every configured registry component in execution order."""

    components: list[tuple[ComponentKind, str]] = [
        *tuple(
            (ComponentKind.SOURCE_COLUMN_FILTER, stage.name)
            for stage in configuration.preprocessing.source_column_filters
        ),
        (
            ComponentKind.MISSING_VALUES_POLICY,
            configuration.preprocessing.missing_values.name,
        ),
        *tuple(
            (ComponentKind.COLUMN_AUGMENTATION, stage.name)
            for stage in configuration.preprocessing.column_augmentations
        ),
        *tuple(
            (ComponentKind.NUMERIC_TRANSFORMATION, stage.name)
            for stage in configuration.preprocessing.numeric_transformations
        ),
        (
            ComponentKind.CANDIDATE_VALIDITY_POLICY,
            configuration.evaluation.candidate_validity.name,
        ),
        (
            ComponentKind.EVALUATION_SUPPORT_POLICY,
            configuration.evaluation.observed_support.name,
        ),
        (ComponentKind.SEARCH_ENGINE, configuration.search.engine.name),
        *tuple(
            (ComponentKind.OBJECTIVE, objective.name)
            for objective in configuration.search.objectives
        ),
        *tuple(
            (ComponentKind.CONSTRAINT, constraint.name)
            for constraint in configuration.search.constraints
        ),
        *tuple(
            (ComponentKind.DESCRIPTOR, descriptor.name)
            for descriptor in configuration.search.descriptors
        ),
        (ComponentKind.INITIALIZER, configuration.search.initialization.name),
        *tuple((ComponentKind.EMITTER, emitter.name) for emitter in configuration.search.emitters),
        (ComponentKind.TERMINATION, configuration.search.termination.name),
        (ComponentKind.EVALUATION_EXECUTOR, configuration.execution.executor.name),
        *tuple(
            (ComponentKind.OBSERVER, observer.name)
            for observer in configuration.monitoring.observers
        ),
    ]
    optional_roles = (
        (ComponentKind.ARCHIVE, configuration.search.archive),
        (ComponentKind.PARENT_SELECTION_POLICY, configuration.search.parent_selection),
        (ComponentKind.MATE_SELECTION_POLICY, configuration.search.mate_selection),
        (ComponentKind.CROSSOVER_OPERATOR, configuration.search.crossover),
        (ComponentKind.MUTATION_OPERATOR, configuration.search.mutation),
        (ComponentKind.SCHEDULER, configuration.search.scheduler),
        (ComponentKind.FINAL_SELECTOR, configuration.final_selection),
    )
    components.extend(
        (kind, specification.name)
        for kind, specification in optional_roles
        if specification is not None
    )
    return tuple(components)


def resolve_component_defaults(
    configuration: SalviConfiguration,
    registry: ComponentRegistry | None = None,
) -> SalviConfiguration:
    """Return the effective configuration with every component default materialized."""

    active_registry = registry or default_component_registry()

    def resolved(kind: ComponentKind, specification: ComponentSpec) -> ComponentSpec:
        return specification.model_copy(
            update={
                "parameters": active_registry.resolve_parameters(
                    kind,
                    specification.name,
                    specification.parameters,
                )
            }
        )

    def optional(
        kind: ComponentKind,
        specification: ComponentSpec | None,
    ) -> ComponentSpec | None:
        return None if specification is None else resolved(kind, specification)

    preprocessing = configuration.preprocessing.model_copy(
        update={
            "source_column_filters": tuple(
                resolved(ComponentKind.SOURCE_COLUMN_FILTER, item)
                for item in configuration.preprocessing.source_column_filters
            ),
            "missing_values": resolved(
                ComponentKind.MISSING_VALUES_POLICY,
                configuration.preprocessing.missing_values,
            ),
            "column_augmentations": tuple(
                resolved(ComponentKind.COLUMN_AUGMENTATION, item)
                for item in configuration.preprocessing.column_augmentations
            ),
            "numeric_transformations": tuple(
                resolved(ComponentKind.NUMERIC_TRANSFORMATION, item)
                for item in configuration.preprocessing.numeric_transformations
            ),
        }
    )
    evaluation = configuration.evaluation.model_copy(
        update={
            "candidate_validity": resolved(
                ComponentKind.CANDIDATE_VALIDITY_POLICY,
                configuration.evaluation.candidate_validity,
            ),
            "observed_support": resolved(
                ComponentKind.EVALUATION_SUPPORT_POLICY,
                configuration.evaluation.observed_support,
            ),
        }
    )
    search = configuration.search.model_copy(
        update={
            "engine": resolved(ComponentKind.SEARCH_ENGINE, configuration.search.engine),
            "objectives": tuple(
                resolved(ComponentKind.OBJECTIVE, item) for item in configuration.search.objectives
            ),
            "constraints": tuple(
                resolved(ComponentKind.CONSTRAINT, item)
                for item in configuration.search.constraints
            ),
            "descriptors": tuple(
                resolved(ComponentKind.DESCRIPTOR, item)
                for item in configuration.search.descriptors
            ),
            "archive": optional(ComponentKind.ARCHIVE, configuration.search.archive),
            "parent_selection": optional(
                ComponentKind.PARENT_SELECTION_POLICY,
                configuration.search.parent_selection,
            ),
            "mate_selection": optional(
                ComponentKind.MATE_SELECTION_POLICY,
                configuration.search.mate_selection,
            ),
            "crossover": optional(
                ComponentKind.CROSSOVER_OPERATOR,
                configuration.search.crossover,
            ),
            "mutation": optional(
                ComponentKind.MUTATION_OPERATOR,
                configuration.search.mutation,
            ),
            "initialization": resolved(
                ComponentKind.INITIALIZER,
                configuration.search.initialization,
            ),
            "emitters": tuple(
                resolved(ComponentKind.EMITTER, item) for item in configuration.search.emitters
            ),
            "scheduler": optional(ComponentKind.SCHEDULER, configuration.search.scheduler),
            "termination": resolved(
                ComponentKind.TERMINATION,
                configuration.search.termination,
            ),
        }
    )
    execution = configuration.execution.model_copy(
        update={
            "executor": resolved(
                ComponentKind.EVALUATION_EXECUTOR,
                configuration.execution.executor,
            )
        }
    )
    monitoring = configuration.monitoring.model_copy(
        update={
            "observers": tuple(
                resolved(ComponentKind.OBSERVER, item)
                for item in configuration.monitoring.observers
            )
        }
    )
    return configuration.model_copy(
        update={
            "preprocessing": preprocessing,
            "evaluation": evaluation,
            "search": search,
            "execution": execution,
            "monitoring": monitoring,
            "final_selection": optional(
                ComponentKind.FINAL_SELECTOR, configuration.final_selection
            ),
        }
    )


def build_specification(
    configuration: SalviConfiguration,
    registry: ComponentRegistry | None = None,
) -> RunSpecification:
    active_registry = registry or default_component_registry()
    configuration = resolve_component_defaults(configuration, active_registry)
    engine_registration = active_registry.get(
        ComponentKind.SEARCH_ENGINE,
        configuration.search.engine.name,
    )
    runtime_errors = runtime_capability_errors(
        engine_registration.name,
        engine_registration.provides,
        resume_requested=configuration.run.resume_from_checkpoint is not None,
        periodic_checkpoints_requested=(
            configuration.monitoring.checkpoint_interval_evaluations is not None
        ),
    )
    if runtime_errors:
        raise ComponentError("; ".join(runtime_errors))
    active_registry.validate_composition(
        _configured_components(configuration),
        configuration.patterns.allowed,
    )
    dataset = DatasetBundleReader().inspect(configuration.dataset.bundle)
    builder = SalviRun.builder(
        dataset,
        run_identifier=configuration.run.identifier,
        seed=configuration.run.seed,
        patterns=configuration.patterns,
        worker_count=configuration.execution.workers,
        search_fingerprint=_search_fingerprint(configuration, active_registry),
    )

    for stage in configuration.preprocessing.source_column_filters:
        builder.add_source_column_filter(
            cast(
                SourceColumnFilteringStage,
                _create(active_registry, ComponentKind.SOURCE_COLUMN_FILTER, stage),
            )
        )
    builder.with_missing_values_policy(
        cast(
            MissingValuesPolicy,
            _create(
                active_registry,
                ComponentKind.MISSING_VALUES_POLICY,
                configuration.preprocessing.missing_values,
            ),
        )
    )
    for stage in configuration.preprocessing.column_augmentations:
        builder.add_column_augmentation(
            cast(
                ColumnAugmentationStage,
                _create(active_registry, ComponentKind.COLUMN_AUGMENTATION, stage),
            )
        )
    for stage in configuration.preprocessing.numeric_transformations:
        builder.add_numeric_transformation(
            cast(
                NumericTransformationStage,
                _create(active_registry, ComponentKind.NUMERIC_TRANSFORMATION, stage),
            )
        )
    builder.with_candidate_validity_policy(
        cast(
            CandidateValidityPolicy,
            _create(
                active_registry,
                ComponentKind.CANDIDATE_VALIDITY_POLICY,
                configuration.evaluation.candidate_validity,
            ),
        )
    )
    builder.with_evaluation_support_policy(
        cast(
            EvaluationSupportPolicy,
            _create(
                active_registry,
                ComponentKind.EVALUATION_SUPPORT_POLICY,
                configuration.evaluation.observed_support,
            ),
        )
    )
    builder.with_search_engine(
        cast(
            SearchEngine,
            _create(active_registry, ComponentKind.SEARCH_ENGINE, configuration.search.engine),
        )
    )
    for objective in configuration.search.objectives:
        builder.add_objective(
            cast(Objective, _create(active_registry, ComponentKind.OBJECTIVE, objective))
        )
    for constraint in configuration.search.constraints:
        builder.add_constraint(
            cast(Constraint, _create(active_registry, ComponentKind.CONSTRAINT, constraint))
        )
    for descriptor in configuration.search.descriptors:
        builder.add_descriptor(
            cast(Descriptor, _create(active_registry, ComponentKind.DESCRIPTOR, descriptor))
        )
    if configuration.search.archive is not None:
        builder.with_archive(
            cast(
                Archive,
                _create(
                    active_registry,
                    ComponentKind.ARCHIVE,
                    configuration.search.archive,
                ),
            )
        )
    if configuration.search.parent_selection is not None:
        builder.with_parent_selection_policy(
            cast(
                ParentSelectionPolicy,
                _create(
                    active_registry,
                    ComponentKind.PARENT_SELECTION_POLICY,
                    configuration.search.parent_selection,
                ),
            )
        )
    if configuration.search.mate_selection is not None:
        builder.with_mate_selection_policy(
            cast(
                MateSelectionPolicy,
                _create(
                    active_registry,
                    ComponentKind.MATE_SELECTION_POLICY,
                    configuration.search.mate_selection,
                ),
            )
        )
    if configuration.search.crossover is not None:
        builder.with_crossover_operator(
            cast(
                CrossoverOperator,
                _create(
                    active_registry,
                    ComponentKind.CROSSOVER_OPERATOR,
                    configuration.search.crossover,
                ),
            )
        )
    if configuration.search.mutation is not None:
        builder.with_mutation_operator(
            cast(
                MutationOperator,
                _create(
                    active_registry,
                    ComponentKind.MUTATION_OPERATOR,
                    configuration.search.mutation,
                ),
            )
        )
    for emitter in configuration.search.emitters:
        builder.add_emitter(cast(Emitter, _create(active_registry, ComponentKind.EMITTER, emitter)))
    if configuration.search.scheduler is not None:
        builder.with_scheduler(
            cast(
                Scheduler,
                _create(
                    active_registry,
                    ComponentKind.SCHEDULER,
                    configuration.search.scheduler,
                ),
            )
        )
    builder.with_initializer(
        cast(
            Initializer,
            _create(
                active_registry,
                ComponentKind.INITIALIZER,
                configuration.search.initialization,
            ),
        )
    )
    builder.with_executor(
        cast(
            EvaluationExecutor,
            _create(
                active_registry,
                ComponentKind.EVALUATION_EXECUTOR,
                configuration.execution.executor,
            ),
        )
    )
    for observer in configuration.monitoring.observers:
        builder.add_observer(
            cast(Observer, _create(active_registry, ComponentKind.OBSERVER, observer))
        )
    builder.with_termination(
        cast(
            TerminationCriterion,
            _create(
                active_registry,
                ComponentKind.TERMINATION,
                configuration.search.termination,
            ),
        )
    )
    if configuration.final_selection is not None:
        builder.with_final_selector(
            cast(
                FinalSelector,
                _create(
                    active_registry,
                    ComponentKind.FINAL_SELECTOR,
                    configuration.final_selection,
                ),
            )
        )
    return builder.build()


def _search_fingerprint(
    configuration: SalviConfiguration,
    registry: ComponentRegistry,
) -> str:
    """Fingerprint all state that can change scientific continuation."""

    payload = configuration.model_dump(mode="json")
    run = payload["run"]
    assert isinstance(run, dict)
    run.pop("resume_from_checkpoint", None)
    payload.pop("output", None)
    payload.pop("monitoring", None)
    payload.pop("final_selection", None)
    search = payload["search"]
    assert isinstance(search, dict)
    termination = search["termination"]
    assert isinstance(termination, dict)
    parameters = termination["parameters"]
    assert isinstance(parameters, dict)
    registration = registry.get(
        ComponentKind.TERMINATION,
        configuration.search.termination.name,
    )
    for parameter in registration.continuation_fingerprint_exclusions:
        parameters.pop(parameter, None)
    dataset = payload["dataset"]
    assert isinstance(dataset, dict)
    manifest = configuration.dataset.bundle / "dataset.yaml"
    dataset["bundle"] = {
        "manifest_sha256": sha256_file(manifest),
    }
    material = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def prepare_run(specification: RunSpecification) -> PreparedRun:
    """Load once and apply the configured component pipeline exactly once."""

    loading_started = perf_counter()
    loaded = DatasetBundleReader().load(specification.dataset.bundle_path)
    if loaded.dataset != specification.dataset:
        raise ComponentError("DatasetBundle metadata changed after component composition")
    dataset = PreparedDataset.from_arrow(
        loaded.dataset,
        loaded.table,
        loaded.row_identifiers,
    )
    loading_seconds = perf_counter() - loading_started
    initial_memory = dataset.memory_bytes
    reports: list[PreprocessingStepReport] = []

    transforms: tuple[tuple[str, Callable[[PreparedDataset], PreparedDataset]], ...] = (
        *tuple(
            (stage.component_name, stage.transform) for stage in specification.source_column_filters
        ),
        (
            specification.missing_values_policy.component_name,
            specification.missing_values_policy.apply,
        ),
        *tuple(
            (stage.component_name, stage.transform)
            for stage in (
                *specification.column_augmentations,
                *specification.numeric_transformations,
            )
        ),
    )
    for component_name, transform in transforms:
        before = dataset.memory_bytes
        started = perf_counter()
        transformed = transform(dataset)
        duration = perf_counter() - started
        if not isinstance(transformed, PreparedDataset):
            raise ComponentError(
                f"preprocessing component {component_name!r} did not return a PreparedDataset"
            )
        dataset = transformed
        reports.append(
            PreprocessingStepReport(
                component_name=component_name,
                duration_seconds=duration,
                memory_before_bytes=before,
                memory_after_bytes=dataset.memory_bytes,
            )
        )

    specification.candidate_validity_policy.validate_dataset(dataset)
    specification.evaluation_support_policy.validate_dataset(dataset)

    context = RunContext(
        dataset=dataset,
        patterns=specification.patterns,
        random_streams=NamedRandomStreams(specification.seed),
        candidate_validity_policy=specification.candidate_validity_policy,
        evaluation_support_policy=specification.evaluation_support_policy,
    )
    return PreparedRun(
        specification=specification,
        context=context,
        preprocessing=PreprocessingReport(
            loading_seconds=loading_seconds,
            initial_memory_bytes=initial_memory,
            final_memory_bytes=dataset.memory_bytes,
            steps=tuple(reports),
        ),
    )
