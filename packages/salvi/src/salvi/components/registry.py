"""Explicit component registry used by configuration-driven composition."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

from salvi.components.catalog import (
    ComponentDescription,
    ComponentMaturity,
    describe_configuration,
    humanize_identifier,
)
from salvi.components.contracts import (
    EngineCompositionContract,
    validate_component_capabilities,
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
from salvi.domain.enums import PatternKind, SearchFamily
from salvi.exceptions import ComponentError

ComponentFactory = Callable[[BaseModel], Component]

_COMPONENT_PROTOCOLS: dict[ComponentKind, type[object]] = {
    ComponentKind.MISSING_VALUES_POLICY: MissingValuesPolicy,
    ComponentKind.COLUMN_AUGMENTATION: ColumnAugmentationStage,
    ComponentKind.SOURCE_COLUMN_FILTER: SourceColumnFilteringStage,
    ComponentKind.NUMERIC_TRANSFORMATION: NumericTransformationStage,
    ComponentKind.CANDIDATE_VALIDITY_POLICY: CandidateValidityPolicy,
    ComponentKind.EVALUATION_SUPPORT_POLICY: EvaluationSupportPolicy,
    ComponentKind.INITIALIZER: Initializer,
    ComponentKind.OBJECTIVE: Objective,
    ComponentKind.CONSTRAINT: Constraint,
    ComponentKind.DESCRIPTOR: Descriptor,
    ComponentKind.ARCHIVE: Archive,
    ComponentKind.PARENT_SELECTION_POLICY: ParentSelectionPolicy,
    ComponentKind.MATE_SELECTION_POLICY: MateSelectionPolicy,
    ComponentKind.CROSSOVER_OPERATOR: CrossoverOperator,
    ComponentKind.MUTATION_OPERATOR: MutationOperator,
    ComponentKind.EMITTER: Emitter,
    ComponentKind.SCHEDULER: Scheduler,
    ComponentKind.SEARCH_ENGINE: SearchEngine,
    ComponentKind.EVALUATION_EXECUTOR: EvaluationExecutor,
    ComponentKind.OBSERVER: Observer,
    ComponentKind.TERMINATION: TerminationCriterion,
    ComponentKind.FINAL_SELECTOR: FinalSelector,
}


@dataclass(frozen=True, slots=True)
class ComponentRegistration:
    kind: ComponentKind
    name: str
    configuration_model: type[BaseModel]
    factory: ComponentFactory
    provides: frozenset[str]
    requires: frozenset[str]
    title: str = ""
    description: str = ""
    supported_patterns: frozenset[PatternKind] = field(
        default_factory=lambda: frozenset(PatternKind)
    )
    conflicts: frozenset[tuple[ComponentKind, str]] = frozenset()
    compatibility_notes: tuple[str, ...] = ()
    maturity: ComponentMaturity = ComponentMaturity.STABLE
    parameter_patterns: tuple[tuple[str, frozenset[PatternKind]], ...] = ()
    continuation_fingerprint_exclusions: frozenset[str] = frozenset()
    composition_contract: EngineCompositionContract | None = None
    default_for_search_family: bool = False

    def describe(self) -> ComponentDescription:
        title = self.title or humanize_identifier(self.name)
        description = self.description or f"{title} {humanize_identifier(self.kind.value).lower()}."
        return describe_configuration(
            kind=self.kind,
            name=self.name,
            title=title,
            description=description,
            provides=self.provides,
            requires=self.requires,
            supported_patterns=self.supported_patterns,
            conflicts=self.conflicts,
            compatibility_notes=self.compatibility_notes,
            maturity=self.maturity,
            parameter_patterns=dict(self.parameter_patterns),
            schema=self.configuration_model.model_json_schema(),
            search_family=(
                None
                if self.composition_contract is None
                else self.composition_contract.search_family
            ),
            default_for_search_family=self.default_for_search_family,
        )


class ComponentRegistry:
    """Registry with deterministic lookup and no implicit plugin discovery."""

    def __init__(self) -> None:
        self._registrations: dict[tuple[ComponentKind, str], ComponentRegistration] = {}

    def register(self, registration: ComponentRegistration) -> None:
        key = (registration.kind, registration.name)
        if key in self._registrations:
            raise ComponentError(
                f"component {registration.kind.value}:{registration.name} is already registered"
            )
        if not registration.supported_patterns:
            raise ComponentError(
                f"component {registration.kind.value}:{registration.name} must support "
                "at least one pattern"
            )
        if key in registration.conflicts:
            raise ComponentError(
                f"component {registration.kind.value}:{registration.name} cannot conflict "
                "with itself"
            )
        if registration.kind is ComponentKind.SEARCH_ENGINE:
            if registration.composition_contract is None:
                raise ComponentError(
                    f"search engine {registration.name!r} must declare a composition contract"
                )
            if registration.composition_contract.engine_name != registration.name:
                raise ComponentError(
                    f"search engine {registration.name!r} has a mismatched composition contract"
                )
        elif registration.composition_contract is not None:
            raise ComponentError(
                "only search-engine registrations may declare composition contracts"
            )
        if (
            registration.default_for_search_family
            and registration.kind is not ComponentKind.SEARCH_ENGINE
        ):
            raise ComponentError(
                "only search-engine registrations may be defaults for a search family"
            )
        parameter_names = set(registration.configuration_model.model_fields)
        unknown_parameters = {
            name for name, _ in registration.parameter_patterns if name not in parameter_names
        }
        if unknown_parameters:
            raise ComponentError(
                f"component {registration.kind.value}:{registration.name} declares pattern "
                f"applicability for unknown parameters: {', '.join(sorted(unknown_parameters))}"
            )
        unknown_fingerprint_exclusions = (
            registration.continuation_fingerprint_exclusions - parameter_names
        )
        if unknown_fingerprint_exclusions:
            raise ComponentError(
                f"component {registration.kind.value}:{registration.name} excludes unknown "
                "continuation-fingerprint parameters: "
                f"{', '.join(sorted(unknown_fingerprint_exclusions))}"
            )
        self._registrations[key] = registration

    def validate_composition(
        self,
        components: Sequence[tuple[ComponentKind, str]],
        allowed_patterns: Sequence[PatternKind],
    ) -> None:
        """Validate registry-level compatibility before component construction."""

        selected = frozenset(components)
        patterns = frozenset(allowed_patterns)
        registrations = tuple(self.get(*key) for key in components)
        for key in components:
            registration = self.get(*key)
            unsupported = patterns - registration.supported_patterns
            if unsupported:
                values = ", ".join(sorted(pattern.value for pattern in unsupported))
                raise ComponentError(
                    f"component {registration.kind.value}:{registration.name} does not "
                    f"support configured patterns: {values}"
                )
            active_conflicts = registration.conflicts & selected
            if active_conflicts:
                values = ", ".join(
                    f"{kind.value}:{name}"
                    for kind, name in sorted(
                        active_conflicts,
                        key=lambda item: (item[0].value, item[1]),
                    )
                )
                raise ComponentError(
                    f"component {registration.kind.value}:{registration.name} conflicts "
                    f"with configured components: {values}"
                )
        engines = tuple(
            registration
            for registration in registrations
            if registration.kind is ComponentKind.SEARCH_ENGINE
        )
        if len(engines) != 1:
            raise ComponentError(
                f"a composition requires exactly one search engine, found {len(engines)}"
            )
        contract = engines[0].composition_contract
        assert contract is not None
        contract.validate(components)
        validate_component_capabilities(
            tuple(
                (
                    registration.kind,
                    registration.name,
                    registration.provides,
                    registration.requires,
                )
                for registration in registrations
            )
        )
        validate_optional_strategy_consumers(
            tuple(
                (registration.kind, registration.name, registration.requires)
                for registration in registrations
            )
        )

    def create(
        self,
        kind: ComponentKind,
        name: str,
        parameters: dict[str, object],
    ) -> Component:
        registration = self.get(kind, name)
        configuration = self._validate_parameters(registration, parameters)
        try:
            component = registration.factory(configuration)
        except Exception as error:
            raise ComponentError(f"failed to create {kind.value}:{name}: {error}") from error
        expected_protocol = _COMPONENT_PROTOCOLS[kind]
        if not isinstance(component, expected_protocol):
            raise ComponentError(
                f"factory for {kind.value}:{name} did not return a valid {kind.value}"
            )
        if kind in {
            ComponentKind.COLUMN_AUGMENTATION,
            ComponentKind.SOURCE_COLUMN_FILTER,
            ComponentKind.NUMERIC_TRANSFORMATION,
        }:
            declared_stage_kind = getattr(component, "stage_kind", None)
            if declared_stage_kind is not kind:
                label = (
                    declared_stage_kind.value
                    if isinstance(declared_stage_kind, ComponentKind)
                    else repr(declared_stage_kind)
                )
                raise ComponentError(f"component {kind.value}:{name} declares stage kind {label!r}")
        if (
            component.provides != registration.provides
            or component.requires != registration.requires
        ):
            raise ComponentError(
                f"component {kind.value}:{name} does not match its registered capabilities"
            )
        return component

    def resolve_parameters(
        self,
        kind: ComponentKind,
        name: str,
        parameters: dict[str, object],
    ) -> dict[str, object]:
        """Validate parameters and materialize every registered default."""

        registration = self.get(kind, name)
        return self._validate_parameters(registration, parameters).model_dump(mode="json")

    @staticmethod
    def _validate_parameters(
        registration: ComponentRegistration,
        parameters: dict[str, object],
    ) -> BaseModel:
        try:
            return registration.configuration_model.model_validate(parameters)
        except ValidationError as error:
            raise ComponentError(
                f"invalid configuration for {registration.kind.value}:{registration.name}: {error}"
            ) from error

    def get(self, kind: ComponentKind, name: str) -> ComponentRegistration:
        try:
            return self._registrations[(kind, name)]
        except KeyError as error:
            available = (
                ", ".join(registration.name for registration in self.describe(kind)) or "none"
            )
            raise ComponentError(
                f"unknown {kind.value} component {name!r}; available: {available}"
            ) from error

    def describe(self, kind: ComponentKind | None = None) -> tuple[ComponentRegistration, ...]:
        registrations = (
            registration
            for registration in self._registrations.values()
            if kind is None or registration.kind is kind
        )
        return tuple(sorted(registrations, key=lambda item: (item.kind.value, item.name)))

    def search_engines(
        self,
        family: SearchFamily,
    ) -> tuple[ComponentRegistration, ...]:
        """Return search engines registered for one architecture family."""

        return tuple(
            registration
            for registration in self.describe(ComponentKind.SEARCH_ENGINE)
            if registration.composition_contract is not None
            and registration.composition_contract.search_family is family
        )

    def default_search_engine(self, family: SearchFamily) -> ComponentRegistration:
        """Return the single explicit default engine for an architecture family."""

        defaults = tuple(
            registration
            for registration in self.search_engines(family)
            if registration.default_for_search_family
        )
        if len(defaults) != 1:
            raise ComponentError(
                f"search family {family.value} requires exactly one default engine, "
                f"found {len(defaults)}"
            )
        return defaults[0]

    def catalog(
        self,
        kind: ComponentKind | None = None,
    ) -> tuple[ComponentDescription, ...]:
        return tuple(registration.describe() for registration in self.describe(kind))
