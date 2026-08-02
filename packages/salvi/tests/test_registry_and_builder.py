from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from salvi.api import SalviRun, execute_in_memory
from salvi.application.configuration import ComponentSpec
from salvi.application.factory import build_specification, prepare_run
from salvi.components.candidate_initialization import UniformRandomInitializer
from salvi.components.catalog import ComponentMaturity
from salvi.components.configuration import EmptyConfiguration
from salvi.components.contracts import (
    EngineCompositionContract,
    RoleCardinality,
    validate_optional_strategy_consumers,
)
from salvi.components.defaults import default_component_registry
from salvi.components.descriptors import ColumnCardinality, RowCardinality
from salvi.components.evaluation_policies import MinimumCardinality, MinimumObservedSupport
from salvi.components.execution import SerialEvaluationExecutor
from salvi.components.membership_emitters import RandomMoveEmitter
from salvi.components.objectives import InternalCoherence
from salvi.components.observers import SearchProgressObserver
from salvi.components.parent_selection import RepertoireUniformParentSelection
from salvi.components.preprocessing import PreserveMissingValues, RobustNumericScaling
from salvi.components.protocols import ComponentKind
from salvi.components.registry import ComponentRegistration, ComponentRegistry
from salvi.components.schedulers import FirstEmitterScheduler
from salvi.components.termination import EvaluationBudget
from salvi.domain import (
    ColumnKind,
    ColumnMetadata,
    Dataset,
    PatternKind,
    PreparedDataset,
    SearchFamily,
)
from salvi.engine.archive import DeepGridMomeArchive, DeepGridMomeConfiguration
from salvi.engine.mome import SerialMomeSearchEngine
from salvi.exceptions import ComponentError
from salvi.patterns import PatternConfiguration


def test_engine_composition_contracts_reject_invalid_declarations() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        RoleCardinality(minimum=-1)
    with pytest.raises(ValueError, match="smaller"):
        RoleCardinality(minimum=2, maximum=1)
    with pytest.raises(ValueError, match="must not be blank"):
        EngineCompositionContract(
            engine_name=" ",
            search_family=SearchFamily.QUALITY_DIVERSITY,
            rules=(),
        )
    with pytest.raises(ValueError, match="must be unique"):
        EngineCompositionContract(
            engine_name="duplicate",
            search_family=SearchFamily.QUALITY_DIVERSITY,
            rules=(
                (ComponentKind.OBJECTIVE, RoleCardinality()),
                (ComponentKind.OBJECTIVE, RoleCardinality()),
            ),
        )

    assert RoleCardinality(minimum=2, maximum=None).describe() == "at least 2"
    assert RoleCardinality(minimum=2, maximum=2).describe() == "exactly 2"
    assert RoleCardinality(minimum=1, maximum=3).describe() == "between 1 and 3"


def test_optional_strategy_cannot_consume_its_own_capability() -> None:
    with pytest.raises(ComponentError, match="no active component consumes"):
        validate_optional_strategy_consumers(
            (
                (
                    ComponentKind.CROSSOVER_OPERATOR,
                    "self-referential",
                    frozenset({"crossover-operator"}),
                ),
            )
        )


def test_registry_is_deterministic_and_validates_parameters() -> None:
    registry = default_component_registry()
    descriptions = registry.describe(ComponentKind.DESCRIPTOR)
    assert tuple(item.name for item in descriptions) == (
        "column_cardinality",
        "row_cardinality",
    )
    with pytest.raises(ComponentError, match="unknown objective"):
        registry.create(ComponentKind.OBJECTIVE, "missing", {})
    with pytest.raises(ComponentError, match="invalid configuration"):
        registry.create(ComponentKind.INITIALIZER, "stratified", {"cardinality_levels": 0})

    registration = registry.get(ComponentKind.OBJECTIVE, "internal_coherence")
    with pytest.raises(ComponentError, match="already registered"):
        registry.register(registration)


def test_public_component_catalog_exposes_editable_metadata() -> None:
    registry = default_component_registry()
    public = registry.catalog()
    assert public
    assert all(item.title and item.description for item in public)

    contrast = next(
        item for item in registry.catalog(ComponentKind.OBJECTIVE) if item.name == "contrast"
    )
    parameter = contrast.parameters[0]
    assert parameter.name == "min_background_ratio"
    assert parameter.default == 0.1
    assert parameter.value_schema["minimum"] == 0.0

    recombination = next(
        item
        for item in registry.catalog(ComponentKind.CROSSOVER_OPERATOR)
        if item.name == "evidence_weighted_recombination"
    )
    assert recombination.maturity is ComponentMaturity.STABLE
    assert set(recombination.supported_patterns) == set(PatternKind)

    initializer = next(
        item for item in registry.catalog(ComponentKind.INITIALIZER) if item.name == "pattern_aware"
    )
    pool_size = next(
        item for item in initializer.parameters if item.name == "joint_column_candidate_pool_size"
    )
    assert pool_size.applicable_patterns == (
        PatternKind.ADDITIVE,
        PatternKind.MULTIPLICATIVE,
    )


def test_registry_validates_declared_pattern_support_and_conflicts() -> None:
    registry = ComponentRegistry()
    registry.register(
        ComponentRegistration(
            kind=ComponentKind.OBJECTIVE,
            name="constant-only",
            configuration_model=EmptyConfiguration,
            factory=lambda _: InternalCoherence(),
            provides=frozenset({"objective"}),
            requires=frozenset(),
            supported_patterns=frozenset({PatternKind.CONSTANT}),
            conflicts=frozenset({(ComponentKind.OBJECTIVE, "other")}),
        )
    )
    registry.register(
        ComponentRegistration(
            kind=ComponentKind.OBJECTIVE,
            name="other",
            configuration_model=EmptyConfiguration,
            factory=lambda _: InternalCoherence(),
            provides=frozenset({"objective"}),
            requires=frozenset(),
        )
    )

    with pytest.raises(ComponentError, match="does not support configured patterns"):
        registry.validate_composition(
            ((ComponentKind.OBJECTIVE, "constant-only"),),
            (PatternKind.ADDITIVE,),
        )
    with pytest.raises(ComponentError, match="conflicts with configured components"):
        registry.validate_composition(
            (
                (ComponentKind.OBJECTIVE, "constant-only"),
                (ComponentKind.OBJECTIVE, "other"),
            ),
            (PatternKind.CONSTANT,),
        )


def test_registry_rejects_unknown_continuation_fingerprint_exclusions() -> None:
    registry = ComponentRegistry()
    with pytest.raises(ComponentError, match="excludes unknown"):
        registry.register(
            ComponentRegistration(
                kind=ComponentKind.OBJECTIVE,
                name="invalid-fingerprint-policy",
                configuration_model=EmptyConfiguration,
                factory=lambda _: InternalCoherence(),
                provides=frozenset({"objective"}),
                requires=frozenset(),
                continuation_fingerprint_exclusions=frozenset({"missing"}),
            )
        )


def test_registry_rejects_incoherent_registration_metadata() -> None:
    def registration(
        *,
        kind: ComponentKind = ComponentKind.OBJECTIVE,
        name: str = "component",
        provides: frozenset[str] = frozenset({"objective"}),
        supported_patterns: frozenset[PatternKind] = frozenset(PatternKind),
        conflicts: frozenset[tuple[ComponentKind, str]] = frozenset(),
        composition_contract: EngineCompositionContract | None = None,
        parameter_patterns: tuple[tuple[str, frozenset[PatternKind]], ...] = (),
    ) -> ComponentRegistration:
        return ComponentRegistration(
            kind=kind,
            name=name,
            configuration_model=EmptyConfiguration,
            factory=(
                (lambda _: SerialMomeSearchEngine())
                if kind is ComponentKind.SEARCH_ENGINE
                else (lambda _: InternalCoherence())
            ),
            provides=provides,
            requires=frozenset(),
            supported_patterns=supported_patterns,
            conflicts=conflicts,
            composition_contract=composition_contract,
            parameter_patterns=parameter_patterns,
        )

    invalid_registrations = (
        (
            registration(supported_patterns=frozenset()),
            "at least one pattern",
        ),
        (
            registration(conflicts=frozenset({(ComponentKind.OBJECTIVE, "component")})),
            "cannot conflict with itself",
        ),
        (
            registration(kind=ComponentKind.SEARCH_ENGINE, provides=frozenset()),
            "must declare a composition contract",
        ),
        (
            registration(
                kind=ComponentKind.SEARCH_ENGINE,
                provides=frozenset({"search-engine"}),
                composition_contract=EngineCompositionContract(
                    engine_name="other",
                    search_family=SearchFamily.QUALITY_DIVERSITY,
                    rules=(),
                ),
            ),
            "mismatched composition contract",
        ),
        (
            registration(
                composition_contract=EngineCompositionContract(
                    engine_name="component",
                    search_family=SearchFamily.QUALITY_DIVERSITY,
                    rules=(),
                )
            ),
            "only search-engine registrations",
        ),
        (
            registration(parameter_patterns=(("unknown", frozenset({PatternKind.CONSTANT})),)),
            "unknown parameters",
        ),
    )

    for invalid, message in invalid_registrations:
        with pytest.raises(ComponentError, match=message):
            ComponentRegistry().register(invalid)


def test_registry_requires_exactly_one_search_engine_per_composition() -> None:
    registry = ComponentRegistry()
    registry.register(
        ComponentRegistration(
            kind=ComponentKind.OBJECTIVE,
            name="objective",
            configuration_model=EmptyConfiguration,
            factory=lambda _: InternalCoherence(),
            provides=frozenset({"objective"}),
            requires=frozenset(),
        )
    )
    with pytest.raises(ComponentError, match="exactly one search engine, found 0"):
        registry.validate_composition(
            ((ComponentKind.OBJECTIVE, "objective"),),
            (PatternKind.CONSTANT,),
        )


def test_registry_contains_every_runtime_component() -> None:
    registry = default_component_registry()
    expected = {
        ComponentKind.CANDIDATE_VALIDITY_POLICY: {"minimum_cardinality"},
        ComponentKind.EVALUATION_SUPPORT_POLICY: {"minimum_observed_support"},
        ComponentKind.INITIALIZER: {
            "uniform_random",
            "stratified",
            "pattern_aware",
            "cell_coverage_pattern_aware",
        },
        ComponentKind.OBJECTIVE: {
            "internal_coherence",
            "contrast",
            "balanced_bicluster_size",
        },
        ComponentKind.CONSTRAINT: {
            "balanced_bicluster_size_range",
            "maximum_internal_coherence",
        },
        ComponentKind.DESCRIPTOR: {"row_cardinality", "column_cardinality"},
        ComponentKind.ARCHIVE: {"deep_grid_mome"},
        ComponentKind.MATE_SELECTION_POLICY: {
            "repertoire_random",
            "cell_first_evidence_compatible",
        },
        ComponentKind.CROSSOVER_OPERATOR: {
            "membership_recombination",
            "evidence_weighted_recombination",
            "half_uniform_membership",
        },
        ComponentKind.MUTATION_OPERATOR: {"bit_flip_membership"},
        ComponentKind.EMITTER: {
            "random_move",
            "add_row",
            "remove_row",
            "swap_row",
            "add_column",
            "remove_column",
            "swap_column",
            "shape_move",
            "crossover",
            "mutation",
            "restart",
            "cell_coverage_restart",
            "alternating_pattern_local_search",
        },
        ComponentKind.SCHEDULER: {
            "first",
            "adaptive_credit",
            "cell_balanced_adaptive_credit",
            "fixed_proportion",
        },
        ComponentKind.SEARCH_ENGINE: {"serial_mome", "pymoo_nsga2"},
        ComponentKind.EVALUATION_EXECUTOR: {"serial", "thread_pool", "process_pool"},
        ComponentKind.OBSERVER: {
            "search_progress",
            "archive_coverage",
            "candidate_outcomes",
            "descriptor_distribution",
            "archive_descriptor_distribution",
            "objective_distribution",
            "emitter_credit",
            "candidate_diversity",
            "runtime_throughput",
            "resource_usage",
            "component_timing",
            "evaluation_issues",
            "qd_archive_diagnostics",
        },
        ComponentKind.TERMINATION: {"evaluation_budget"},
        ComponentKind.FINAL_SELECTOR: {
            "adaptive_residual_evidence_cover",
            "containment_marginal_quality",
        },
    }

    for kind, required_names in expected.items():
        available = {registration.name for registration in registry.describe(kind)}
        assert required_names <= available
    assert registry.default_search_engine(SearchFamily.QUALITY_DIVERSITY).name == "serial_mome"
    assert (
        registry.default_search_engine(SearchFamily.CONVENTIONAL_MULTI_OBJECTIVE).name
        == "pymoo_nsga2"
    )


def test_registry_detects_factory_capability_mismatch() -> None:
    registry = ComponentRegistry()
    registry.register(
        ComponentRegistration(
            kind=ComponentKind.OBJECTIVE,
            name="broken",
            configuration_model=EmptyConfiguration,
            factory=lambda _: InternalCoherence(),
            provides=frozenset({"wrong"}),
            requires=frozenset(),
        )
    )
    with pytest.raises(ComponentError, match="does not match"):
        registry.create(ComponentKind.OBJECTIVE, "broken", {})

    failing = ComponentRegistry()
    failing.register(
        ComponentRegistration(
            kind=ComponentKind.OBJECTIVE,
            name="failing",
            configuration_model=EmptyConfiguration,
            factory=lambda _: (_ for _ in ()).throw(RuntimeError("factory failure")),
            provides=frozenset(),
            requires=frozenset(),
        )
    )
    with pytest.raises(ComponentError, match="failed to create"):
        failing.create(ComponentKind.OBJECTIVE, "failing", {})

    wrong_kind = ComponentRegistry()
    wrong_kind.register(
        ComponentRegistration(
            kind=ComponentKind.OBJECTIVE,
            name="observer-as-objective",
            configuration_model=EmptyConfiguration,
            factory=lambda _: SearchProgressObserver(),
            provides=SearchProgressObserver().provides,
            requires=SearchProgressObserver().requires,
        )
    )
    with pytest.raises(ComponentError, match="valid objective"):
        wrong_kind.create(ComponentKind.OBJECTIVE, "observer-as-objective", {})

    wrong_stage_kind = ComponentRegistry()
    scaling = RobustNumericScaling()
    wrong_stage_kind.register(
        ComponentRegistration(
            kind=ComponentKind.COLUMN_AUGMENTATION,
            name="numeric-as-augmentation",
            configuration_model=EmptyConfiguration,
            factory=lambda _: RobustNumericScaling(),
            provides=scaling.provides,
            requires=scaling.requires,
        )
    )
    with pytest.raises(ComponentError, match="declares stage kind"):
        wrong_stage_kind.create(ComponentKind.COLUMN_AUGMENTATION, "numeric-as-augmentation", {})


def test_builder_rejects_missing_and_duplicate_unique_components() -> None:
    dataset = Dataset(
        identifier="dataset",
        bundle_path=Path("dataset"),
        row_count=2,
        column_count=2,
        columns=(
            ColumnMetadata(index=0, name="a", kind=ColumnKind.NUMERIC),
            ColumnMetadata(index=1, name="b", kind=ColumnKind.NUMERIC),
        ),
    )
    builder = SalviRun.builder(dataset)
    with pytest.raises(ComponentError, match="missing required components"):
        builder.build()
    builder.with_missing_values_policy(PreserveMissingValues())
    with pytest.raises(ComponentError, match="already configured"):
        builder.with_missing_values_policy(PreserveMissingValues())
    builder.with_candidate_validity_policy(MinimumCardinality())
    with pytest.raises(ComponentError, match="already configured"):
        builder.with_candidate_validity_policy(MinimumCardinality())

    with pytest.raises(ComponentError, match="run_identifier"):
        SalviRun.builder(dataset, run_identifier=" ")
    with pytest.raises(ComponentError, match="seed"):
        SalviRun.builder(dataset, seed=-1)
    with pytest.raises(ComponentError, match="worker_count"):
        SalviRun.builder(dataset, worker_count=0)


@dataclass(frozen=True)
class ImpossibleObjective(InternalCoherence):
    component_name: str = "impossible"
    requires: frozenset[str] = frozenset({"capability-that-does-not-exist"})


@dataclass(frozen=True)
class SelfDependentObjective(InternalCoherence):
    component_name: str = "self-dependent"
    provides: frozenset[str] = frozenset({"objective", "self-capability"})
    requires: frozenset[str] = frozenset({"initialization", "self-capability"})


def test_builder_validates_capabilities() -> None:
    dataset = Dataset(
        identifier="dataset",
        bundle_path=Path("dataset"),
        row_count=2,
        column_count=2,
        columns=(
            ColumnMetadata(index=0, name="a", kind=ColumnKind.NUMERIC),
            ColumnMetadata(index=1, name="b", kind=ColumnKind.NUMERIC),
        ),
    )
    builder = _scientific_builder(dataset).add_objective(ImpossibleObjective())
    with pytest.raises(ComponentError, match="unavailable capabilities"):
        builder.build()

    self_dependent = _scientific_builder(dataset).add_objective(SelfDependentObjective())
    with pytest.raises(ComponentError, match="self-capability"):
        self_dependent.build()


def test_builder_rejects_duplicate_objective_names() -> None:
    dataset = Dataset(
        identifier="dataset",
        bundle_path=Path("dataset"),
        row_count=2,
        column_count=2,
        columns=(
            ColumnMetadata(index=0, name="a", kind=ColumnKind.NUMERIC),
            ColumnMetadata(index=1, name="b", kind=ColumnKind.NUMERIC),
        ),
    )
    builder = (
        _scientific_builder(dataset)
        .add_objective(InternalCoherence())
        .add_objective(InternalCoherence())
    )
    with pytest.raises(ComponentError, match="unique component names"):
        builder.build()


def _scientific_builder(dataset: Dataset):
    archive_configuration = DeepGridMomeConfiguration()
    return (
        SalviRun.builder(dataset)
        .with_missing_values_policy(PreserveMissingValues())
        .add_numeric_transformation(RobustNumericScaling())
        .with_candidate_validity_policy(MinimumCardinality())
        .with_evaluation_support_policy(MinimumObservedSupport())
        .with_search_engine(SerialMomeSearchEngine())
        .add_descriptor(RowCardinality())
        .add_descriptor(ColumnCardinality())
        .with_archive(
            DeepGridMomeArchive(
                axes=archive_configuration.axes,
                cell_capacity=archive_configuration.cell_capacity,
            )
        )
        .with_parent_selection_policy(RepertoireUniformParentSelection())
        .add_emitter(RandomMoveEmitter())
        .with_scheduler(FirstEmitterScheduler())
        .with_initializer(UniformRandomInitializer())
        .with_executor(SerialEvaluationExecutor())
        .with_termination(EvaluationBudget())
    )


def test_configuration_factory_builds_complete_specification(configuration_path: Path) -> None:
    from salvi.application.configuration import load_configuration

    configuration = load_configuration(configuration_path).configuration
    specification = build_specification(configuration)
    assert specification.dataset.identifier == "test-dataset"
    assert specification.run_identifier == "test-run"
    assert specification.seed == 7
    assert specification.patterns == PatternConfiguration()
    assert specification.worker_count == 1
    assert specification.candidate_validity_policy.component_name == "minimum_cardinality"
    assert specification.evaluation_support_policy.component_name == "minimum_observed_support"
    assert specification.parent_selection_policy is not None
    assert specification.parent_selection_policy.component_name == "repertoire_uniform"
    assert len(specification.numeric_transformations) == 1
    assert len(specification.objectives) == 2
    assert len(specification.descriptors) == 2
    assert len(specification.emitters) == 1
    assert len(specification.observers) == 1


def test_programmatic_specification_executes_in_memory(configuration_path: Path) -> None:
    from salvi.application.configuration import load_configuration

    configuration = load_configuration(configuration_path).configuration
    result = execute_in_memory(build_specification(configuration))

    assert result.evaluations == 2
    assert result.search_repertoire == result.repertoire
    assert result.repertoire.evaluations


def test_scientific_objectives_reject_missing_robust_numeric_scaling(
    configuration_path: Path,
) -> None:
    from salvi.application.configuration import load_configuration

    configuration = load_configuration(configuration_path).configuration
    without_scaling = configuration.model_copy(
        update={
            "preprocessing": configuration.preprocessing.model_copy(
                update={"numeric_transformations": ()}
            ),
            "search": configuration.search.model_copy(
                update={"objectives": (ComponentSpec(name="internal_coherence", parameters={}),)}
            ),
        }
    )

    with pytest.raises(ComponentError, match="robust-numeric-data"):
        build_specification(without_scaling)


@dataclass
class CountingMissingValuesPolicy:
    component_name: str = "counting-policy"
    provides: frozenset[str] = frozenset({"missing-values-handled"})
    requires: frozenset[str] = frozenset({"prepared-dataset"})
    calls: int = 0
    seen: PreparedDataset | None = None

    def apply(self, dataset: PreparedDataset) -> PreparedDataset:
        self.calls += 1
        self.seen = dataset
        return dataset


@dataclass
class TrackingPreprocessor:
    component_name: str = "tracking-stage"
    stage_kind: ComponentKind = ComponentKind.NUMERIC_TRANSFORMATION
    provides: frozenset[str] = frozenset({"tracked-transformation"})
    requires: frozenset[str] = frozenset({"prepared-dataset", "missing-values-handled"})
    calls: int = 0
    seen: PreparedDataset | None = None

    def transform(self, dataset: PreparedDataset) -> PreparedDataset:
        self.calls += 1
        self.seen = dataset
        return dataset


@dataclass
class TrackingSourceFilter:
    component_name: str = "tracking-source-filter"
    stage_kind: ComponentKind = ComponentKind.SOURCE_COLUMN_FILTER
    provides: frozenset[str] = frozenset({"tracked-filter"})
    requires: frozenset[str] = frozenset({"prepared-dataset"})
    calls: int = 0

    def transform(self, dataset: PreparedDataset) -> PreparedDataset:
        self.calls += 1
        return dataset.select_columns((0, 1))


def test_preprocessing_is_application_owned_and_runs_once(configuration_path: Path) -> None:
    from salvi.application.configuration import load_configuration

    raw = build_specification(load_configuration(configuration_path).configuration)
    policy = CountingMissingValuesPolicy()
    source_filter = TrackingSourceFilter()
    stage = TrackingPreprocessor()
    configured = replace(
        raw,
        source_column_filters=(source_filter,),
        missing_values_policy=policy,
        numeric_transformations=(stage,),
    )

    prepared = prepare_run(configured)
    assert raw.dataset.identifier == "test-dataset"
    assert prepared.context.dataset.metadata is not raw.dataset
    assert (source_filter.calls, policy.calls, stage.calls) == (1, 1, 1)
    assert policy.seen is stage.seen is prepared.context.dataset
    assert prepared.context.dataset.column_count == 2

    prepared.specification.search_engine.initialize(
        prepared.specification,
        prepared.context,
    )
    assert (source_filter.calls, policy.calls, stage.calls) == (1, 1, 1)


def test_component_spec_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ComponentSpec.model_validate({"name": "component", "unexpected": True})
