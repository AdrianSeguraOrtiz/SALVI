"""Declarative composition contracts for SALVI search engines."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from salvi.components.protocols import Component, ComponentKind, CompositionAwareComponent
from salvi.domain.enums import SearchFamily
from salvi.exceptions import ComponentError


@dataclass(frozen=True, slots=True)
class RoleCardinality:
    """Allowed number of configured components for one composition role."""

    minimum: int = 0
    maximum: int | None = 1

    def __post_init__(self) -> None:
        if self.minimum < 0:
            raise ValueError("role minimum must be non-negative")
        if self.maximum is not None and self.maximum < self.minimum:
            raise ValueError("role maximum cannot be smaller than its minimum")

    def describe(self) -> str:
        if self.maximum is None:
            return f"at least {self.minimum}"
        if self.minimum == self.maximum:
            return f"exactly {self.minimum}"
        return f"between {self.minimum} and {self.maximum}"


@dataclass(frozen=True, slots=True)
class EngineCompositionContract:
    """Roles consumed by one search-engine implementation."""

    engine_name: str
    search_family: SearchFamily
    rules: tuple[tuple[ComponentKind, RoleCardinality], ...]

    def __post_init__(self) -> None:
        kinds = tuple(kind for kind, _ in self.rules)
        if not self.engine_name.strip():
            raise ValueError("engine contract name must not be blank")
        if len(set(kinds)) != len(kinds):
            raise ValueError("engine contract roles must be unique")

    @property
    def role_rules(self) -> Mapping[ComponentKind, RoleCardinality]:
        return dict(self.rules)

    def validate(self, components: Sequence[tuple[ComponentKind, str]]) -> None:
        counts = Counter(kind for kind, _ in components)
        rules = self.role_rules
        errors: list[str] = []
        for kind in ComponentKind:
            count = counts[kind]
            rule = rules.get(kind, RoleCardinality(0, 0))
            if count < rule.minimum or (rule.maximum is not None and count > rule.maximum):
                errors.append(f"{kind.value} expects {rule.describe()} component(s), found {count}")
        if errors:
            detail = "; ".join(errors)
            raise ComponentError(
                f"search engine {self.engine_name!r} has an invalid composition: {detail}"
            )


def _common_rules(
    *,
    minimum_objectives: int,
    minimum_descriptors: int,
) -> dict[ComponentKind, RoleCardinality]:
    return {
        ComponentKind.SOURCE_COLUMN_FILTER: RoleCardinality(0, None),
        ComponentKind.MISSING_VALUES_POLICY: RoleCardinality(1, 1),
        ComponentKind.COLUMN_AUGMENTATION: RoleCardinality(0, None),
        ComponentKind.NUMERIC_TRANSFORMATION: RoleCardinality(0, None),
        ComponentKind.CANDIDATE_VALIDITY_POLICY: RoleCardinality(1, 1),
        ComponentKind.EVALUATION_SUPPORT_POLICY: RoleCardinality(1, 1),
        ComponentKind.INITIALIZER: RoleCardinality(1, 1),
        ComponentKind.OBJECTIVE: RoleCardinality(minimum_objectives, None),
        ComponentKind.CONSTRAINT: RoleCardinality(0, None),
        ComponentKind.DESCRIPTOR: RoleCardinality(minimum_descriptors, None),
        ComponentKind.SEARCH_ENGINE: RoleCardinality(1, 1),
        ComponentKind.EVALUATION_EXECUTOR: RoleCardinality(1, 1),
        ComponentKind.OBSERVER: RoleCardinality(0, None),
        ComponentKind.TERMINATION: RoleCardinality(1, 1),
        ComponentKind.FINAL_SELECTOR: RoleCardinality(0, 1),
    }


def qd_engine_contract(engine_name: str) -> EngineCompositionContract:
    rules = _common_rules(minimum_objectives=1, minimum_descriptors=1)
    rules.update(
        {
            ComponentKind.ARCHIVE: RoleCardinality(1, 1),
            ComponentKind.PARENT_SELECTION_POLICY: RoleCardinality(0, 1),
            ComponentKind.MATE_SELECTION_POLICY: RoleCardinality(0, 1),
            ComponentKind.CROSSOVER_OPERATOR: RoleCardinality(0, 1),
            ComponentKind.MUTATION_OPERATOR: RoleCardinality(0, 1),
            ComponentKind.EMITTER: RoleCardinality(1, None),
            ComponentKind.SCHEDULER: RoleCardinality(1, 1),
        }
    )
    return EngineCompositionContract(
        engine_name=engine_name,
        search_family=SearchFamily.QUALITY_DIVERSITY,
        rules=tuple(rules.items()),
    )


OPTIONAL_STRATEGY_CAPABILITIES: Mapping[ComponentKind, str] = {
    ComponentKind.PARENT_SELECTION_POLICY: "parent-selection",
    ComponentKind.MATE_SELECTION_POLICY: "mate-selection",
    ComponentKind.CROSSOVER_OPERATOR: "crossover-operator",
    ComponentKind.MUTATION_OPERATOR: "mutation-operator",
}


def validate_component_capabilities(
    components: Sequence[tuple[ComponentKind, str, frozenset[str], frozenset[str]]],
) -> None:
    """Validate requirements without allowing self-provided capabilities."""

    provider_counts = Counter(
        capability for _, _, provides, _ in components for capability in provides
    )
    base_capabilities = frozenset({"dataset", "prepared-dataset"})
    for kind, name, provides, requires in components:
        available = base_capabilities.union(
            capability
            for capability, count in provider_counts.items()
            if count - int(capability in provides) > 0
        )
        unavailable = requires - available
        if unavailable:
            raise ComponentError(
                f"component {kind.value}:{name} requires unavailable capabilities: "
                f"{', '.join(sorted(unavailable))}"
            )


def validate_optional_strategy_consumers(
    components: Sequence[tuple[ComponentKind, str, frozenset[str]]],
) -> None:
    """Reject configured strategies that no active component consumes."""

    errors = optional_strategy_consumer_errors(components)
    if errors:
        raise ComponentError("; ".join(message for _, _, message in errors))


def component_composition_issues(
    components: Sequence[tuple[ComponentKind, Component]],
) -> tuple[tuple[ComponentKind, str, str], ...]:
    """Collect validation owned by configured component instances."""

    issues: list[tuple[ComponentKind, str, str]] = []
    for kind, component in components:
        if not isinstance(component, CompositionAwareComponent):
            continue
        issues.extend(
            (kind, component.component_name, message)
            for message in component.composition_issues(components)
        )
    return tuple(issues)


def validate_component_composition(
    components: Sequence[tuple[ComponentKind, Component]],
) -> None:
    """Run component-owned cross-role validation after construction."""

    issues = component_composition_issues(components)
    if issues:
        raise ComponentError(
            "; ".join(f"{kind.value}:{name}: {message}" for kind, name, message in issues)
        )


def optional_strategy_consumer_errors(
    components: Sequence[tuple[ComponentKind, str, frozenset[str]]],
) -> tuple[tuple[ComponentKind, str, str], ...]:
    """Return catalog-addressable errors for unconsumed optional strategies."""

    errors: list[tuple[ComponentKind, str, str]] = []
    for kind, capability in OPTIONAL_STRATEGY_CAPABILITIES.items():
        configured = tuple(name for role, name, _ in components if role is kind)
        consumed = any(
            role is not kind and capability in required for role, _, required in components
        )
        if configured and not consumed:
            name = configured[0]
            errors.append(
                (
                    kind,
                    name,
                    f"{kind.value}:{name} is configured but no active component "
                    f"consumes capability {capability!r}",
                )
            )
    return tuple(errors)


def runtime_capability_errors(
    engine_name: str,
    provides: frozenset[str],
    *,
    resume_requested: bool,
    periodic_checkpoints_requested: bool,
) -> tuple[str, ...]:
    """Report runtime features that the selected search engine cannot provide."""

    errors: list[str] = []
    if resume_requested and "checkpoint-resume" not in provides:
        errors.append(f"search engine {engine_name!r} does not support checkpoint resumption")
    if periodic_checkpoints_requested and "checkpoint-resume" not in provides:
        errors.append(
            f"search engine {engine_name!r} does not support resumable periodic checkpoints"
        )
    return tuple(errors)


def nsga2_engine_contract(engine_name: str) -> EngineCompositionContract:
    rules = _common_rules(minimum_objectives=2, minimum_descriptors=0)
    rules.update(
        {
            ComponentKind.DESCRIPTOR: RoleCardinality(0, 0),
            ComponentKind.CROSSOVER_OPERATOR: RoleCardinality(1, 1),
            ComponentKind.MUTATION_OPERATOR: RoleCardinality(1, 1),
        }
    )
    return EngineCompositionContract(
        engine_name=engine_name,
        search_family=SearchFamily.CONVENTIONAL_MULTI_OBJECTIVE,
        rules=tuple(rules.items()),
    )


__all__ = [
    "OPTIONAL_STRATEGY_CAPABILITIES",
    "EngineCompositionContract",
    "RoleCardinality",
    "component_composition_issues",
    "nsga2_engine_contract",
    "optional_strategy_consumer_errors",
    "qd_engine_contract",
    "runtime_capability_errors",
    "validate_component_capabilities",
    "validate_component_composition",
    "validate_optional_strategy_consumers",
]
