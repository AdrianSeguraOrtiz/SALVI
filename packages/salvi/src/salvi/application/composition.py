"""Partial composition resolution for catalog-driven user interfaces."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any, cast

from pydantic import Field

from salvi.components.catalog import (
    ComponentDescription,
    RolePresentation,
    WorkflowConnectionKind,
    WorkflowStage,
    role_catalog,
)
from salvi.components.contracts import (
    RoleCardinality,
    component_composition_issues,
    optional_strategy_consumer_errors,
    runtime_capability_errors,
)
from salvi.components.defaults import default_component_registry
from salvi.components.protocols import Component, ComponentKind, EvaluationExecutor
from salvi.components.registry import ComponentRegistration, ComponentRegistry
from salvi.domain.enums import PatternKind
from salvi.domain.models import FrozenModel


class RoleState(StrEnum):
    REQUIRED = "REQUIRED"
    AVAILABLE = "AVAILABLE"
    CONFIGURED = "CONFIGURED"
    UNAVAILABLE = "UNAVAILABLE"
    INVALID = "INVALID"


class InstanceResolution(FrozenModel):
    component: ComponentDescription
    available: bool
    reasons: tuple[str, ...] = ()


class RoleResolution(FrozenModel):
    role: RolePresentation
    state: RoleState
    minimum: int = Field(ge=0)
    maximum: int | None = Field(default=1, ge=0)
    configured: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    instances: tuple[InstanceResolution, ...] = ()


class WorkflowConnectionResolution(FrozenModel):
    """One effective connection between configured workflow roles."""

    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    kind: WorkflowConnectionKind


class CompositionResolution(FrozenModel):
    valid: bool
    complete: bool
    allowed_patterns: tuple[PatternKind, ...]
    roles: tuple[RoleResolution, ...]
    workflow_connections: tuple[WorkflowConnectionResolution, ...] = ()
    errors: tuple[str, ...] = ()


_ROLE_PATHS = {role.kind: role.configuration_path for role in role_catalog()}
_REPEATABLE = {role.kind: role.repeatable for role in role_catalog()}


def _at_path(mapping: Mapping[str, Any], path: Sequence[str]) -> object:
    value: object = mapping
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _specifications(mapping: Mapping[str, Any], kind: ComponentKind) -> tuple[dict[str, Any], ...]:
    value = _at_path(mapping, _ROLE_PATHS[kind])
    if _REPEATABLE[kind]:
        if not isinstance(value, list | tuple):
            return ()
        return tuple(item for item in value if isinstance(item, dict))
    return (value,) if isinstance(value, dict) else ()


def _patterns(mapping: Mapping[str, Any]) -> tuple[PatternKind, ...]:
    patterns = mapping.get("patterns")
    raw = patterns.get("allowed") if isinstance(patterns, Mapping) else None
    if not isinstance(raw, list | tuple):
        return (PatternKind.CONSTANT,)
    parsed: list[PatternKind] = []
    for value in raw:
        try:
            parsed.append(PatternKind(str(value).upper()))
        except ValueError:
            continue
    return tuple(dict.fromkeys(parsed)) or (PatternKind.CONSTANT,)


def _effective_workflow_connections(
    roles: Sequence[RoleResolution],
) -> tuple[WorkflowConnectionResolution, ...]:
    active = {role.role.kind: role for role in roles if role.configured}
    connections: list[WorkflowConnectionResolution] = []

    def add(
        source: str | ComponentKind,
        target: str | ComponentKind,
        kind: WorkflowConnectionKind,
    ) -> None:
        source_value = source.value if isinstance(source, ComponentKind) else source
        target_value = target.value if isinstance(target, ComponentKind) else target
        connection = WorkflowConnectionResolution(
            source=source_value,
            target=target_value,
            kind=kind,
        )
        if connection not in connections:
            connections.append(connection)

    preparation = sorted(
        (role for role in active.values() if role.role.stage is WorkflowStage.PREPARATION),
        key=lambda item: item.role.order,
    )
    data_source: str | ComponentKind = "__input__"
    for role in preparation:
        add(data_source, role.role.kind, WorkflowConnectionKind.PRIMARY)
        data_source = role.role.kind

    prepared_data_consumers = (
        ComponentKind.CANDIDATE_VALIDITY_POLICY,
        ComponentKind.EVALUATION_SUPPORT_POLICY,
        ComponentKind.INITIALIZER,
    )
    for kind in prepared_data_consumers:
        if kind in active:
            add(data_source, kind, WorkflowConnectionKind.PRIMARY)

    preparation_kinds = {
        role.role.kind for role in roles if role.role.stage is WorkflowStage.PREPARATION
    }
    for target in roles:
        if target.role.kind not in active:
            continue
        for incoming in target.role.incoming:
            if incoming.source not in active:
                continue
            if incoming.source in preparation_kinds and target.role.kind in prepared_data_consumers:
                continue
            if incoming.source in preparation_kinds and target.role.kind in preparation_kinds:
                continue
            add(incoming.source, target.role.kind, incoming.kind)

    output_roles = sorted(
        (role for role in active.values() if role.role.emits_pipeline_output),
        key=lambda item: item.role.order,
    )
    if output_roles:
        add(output_roles[-1].role.kind, "__analysis__", WorkflowConnectionKind.PRIMARY)
    elif ComponentKind.SEARCH_ENGINE in active:
        add(ComponentKind.SEARCH_ENGINE, "__analysis__", WorkflowConnectionKind.PRIMARY)

    return tuple(connections)


class CompositionResolutionService:
    """Resolve a draft without requiring it to be a complete pipeline."""

    def __init__(self, registry: ComponentRegistry | None = None) -> None:
        self._registry = registry or default_component_registry()

    def resolve(self, draft: Mapping[str, Any]) -> CompositionResolution:
        patterns = _patterns(draft)
        specs = {kind: _specifications(draft, kind) for kind in ComponentKind}
        names = {
            kind: tuple(str(item.get("name", "")) for item in values if item.get("name"))
            for kind, values in specs.items()
        }
        selected = frozenset((kind, name) for kind, values in names.items() for name in values)
        errors: list[str] = []
        registrations: dict[tuple[ComponentKind, str], ComponentRegistration] = {}
        for key in selected:
            try:
                registrations[key] = self._registry.get(*key)
            except Exception as error:
                errors.append(str(error))

        instance_issues: dict[tuple[ComponentKind, str], list[str]] = defaultdict(list)
        instances: dict[tuple[ComponentKind, str], Component] = {}
        for kind, specifications in specs.items():
            for specification in specifications:
                name = str(specification.get("name", ""))
                key = (kind, name)
                if not name or key not in registrations:
                    continue
                parameters = specification.get("parameters", {})
                if not isinstance(parameters, dict):
                    instance_issues[key].append("parameters must be a mapping")
                    continue
                try:
                    instances[key] = self._registry.create(kind, name, parameters)
                except Exception as error:
                    instance_issues[key].append(str(error))

        execution = draft.get("execution")
        workers = execution.get("workers") if isinstance(execution, Mapping) else None
        if isinstance(workers, int) and not isinstance(workers, bool):
            for key, component in instances.items():
                if key[0] is not ComponentKind.EVALUATION_EXECUTOR:
                    continue
                try:
                    cast(EvaluationExecutor, component).validate_worker_count(workers)
                except Exception as error:
                    instance_issues[key].append(str(error))

        for kind, name, message in component_composition_issues(
            tuple((kind, component) for (kind, _name), component in instances.items())
        ):
            instance_issues[(kind, name)].append(message)

        engine_names = names[ComponentKind.SEARCH_ENGINE]
        contract = None
        if len(engine_names) == 1:
            registration = registrations.get((ComponentKind.SEARCH_ENGINE, engine_names[0]))
            contract = None if registration is None else registration.composition_contract
        elif len(engine_names) > 1:
            errors.append("only one search engine can be configured")

        def role_consumed(registration: ComponentRegistration) -> bool:
            return (
                contract is None
                or contract.role_rules.get(
                    registration.kind,
                    RoleCardinality(0, 0),
                ).maximum
                != 0
            )

        provider_counts = Counter(
            capability
            for registration in registrations.values()
            if role_consumed(registration)
            for capability in registration.provides
        )
        optional_strategy_errors: dict[tuple[ComponentKind, str], tuple[str, ...]] = {}
        for kind, strategy_name, message in optional_strategy_consumer_errors(
            tuple(
                (registration.kind, registration.name, registration.requires)
                for registration in registrations.values()
            )
        ):
            optional_strategy_errors[(kind, strategy_name)] = (message,)
        possible_providers = {
            capability
            for registration in self._registry.describe()
            if role_consumed(registration)
            for capability in registration.provides
        }
        monitoring = draft.get("monitoring")
        checkpoint_interval = (
            monitoring.get("checkpoint_interval_evaluations")
            if isinstance(monitoring, Mapping)
            else None
        )
        engine_runtime_errors: tuple[str, ...] = ()
        if len(engine_names) == 1:
            engine_registration = registrations.get((ComponentKind.SEARCH_ENGINE, engine_names[0]))
            if engine_registration is not None:
                engine_runtime_errors = runtime_capability_errors(
                    engine_registration.name,
                    engine_registration.provides,
                    resume_requested=False,
                    periodic_checkpoints_requested=checkpoint_interval is not None,
                )
        role_results: list[RoleResolution] = []
        for role in role_catalog():
            rule = (
                RoleCardinality(1, 1)
                if contract is None and role.kind is ComponentKind.SEARCH_ENGINE
                else (
                    RoleCardinality(0, None if role.repeatable else 1)
                    if contract is None
                    else contract.role_rules.get(role.kind, RoleCardinality(0, 0))
                )
            )
            configured = names[role.kind]
            role_reasons: list[str] = []
            if role.kind is ComponentKind.SEARCH_ENGINE:
                role_reasons.extend(engine_runtime_errors)
            if len(configured) < rule.minimum:
                state = RoleState.REQUIRED
                role_reasons.append(f"requires {rule.describe()} component(s)")
            elif rule.maximum is not None and len(configured) > rule.maximum:
                state = RoleState.INVALID
                role_reasons.append(
                    f"allows {rule.describe()} component(s), found {len(configured)}"
                )
            elif rule.maximum == 0:
                state = RoleState.INVALID if configured else RoleState.UNAVAILABLE
                role_reasons.append(
                    "the selected search engine does not consume this role"
                    if contract is not None
                    else "this role is unavailable"
                )
            else:
                state = RoleState.CONFIGURED if configured else RoleState.AVAILABLE

            instance_results: list[InstanceResolution] = []
            for registration in self._registry.describe(role.kind):
                reasons: list[str] = []
                unsupported = set(patterns) - registration.supported_patterns
                if unsupported:
                    reasons.append(
                        "does not support "
                        + ", ".join(sorted(pattern.value for pattern in unsupported))
                    )
                conflicts = registration.conflicts & selected
                if conflicts:
                    reasons.append(
                        "conflicts with "
                        + ", ".join(
                            f"{kind.value}:{name}"
                            for kind, name in sorted(
                                conflicts,
                                key=lambda item: (item[0].value, item[1]),
                            )
                        )
                    )
                impossible = (
                    registration.requires
                    - possible_providers
                    - {
                        "dataset",
                        "prepared-dataset",
                    }
                )
                if impossible:
                    reasons.append(
                        "requires unavailable capabilities: " + ", ".join(sorted(impossible))
                    )
                if rule.maximum == 0:
                    reasons.append("forbidden by the selected search engine")
                instance_results.append(
                    InstanceResolution(
                        component=registration.describe(),
                        available=not reasons,
                        reasons=tuple(reasons),
                    )
                )

            for specification in specs[role.kind]:
                raw_name = specification.get("name")
                registration = registrations.get((role.kind, str(raw_name)))
                if registration is None:
                    continue
                role_reasons.extend(instance_issues.get((role.kind, registration.name), ()))
                unsupported = set(patterns) - registration.supported_patterns
                if unsupported:
                    role_reasons.append(f"{raw_name} does not support configured patterns")
                active_conflicts = registration.conflicts & selected
                if active_conflicts:
                    role_reasons.append(
                        f"{registration.name} conflicts with the current composition"
                    )
                role_reasons.extend(
                    optional_strategy_errors.get((role.kind, registration.name), ())
                )
                parameters = specification.get("parameters", {})
                try:
                    registration.configuration_model.model_validate(parameters)
                except Exception as error:
                    role_reasons.append(f"{raw_name}: {error}")
                missing_now = {
                    capability
                    for capability in registration.requires
                    if provider_counts[capability] - int(capability in registration.provides) <= 0
                    and capability not in {"dataset", "prepared-dataset"}
                }
                if missing_now:
                    role_reasons.append(
                        f"{registration.name} still requires: {', '.join(sorted(missing_now))}"
                    )

            if role_reasons and configured:
                state = RoleState.INVALID
            role_results.append(
                RoleResolution(
                    role=role,
                    state=state,
                    minimum=rule.minimum,
                    maximum=rule.maximum,
                    configured=configured,
                    reasons=tuple(dict.fromkeys(role_reasons)),
                    instances=tuple(instance_results),
                )
            )

        complete = all(
            role.state not in {RoleState.REQUIRED, RoleState.INVALID} for role in role_results
        )
        resolved_roles = tuple(role_results)
        return CompositionResolution(
            valid=not errors and all(role.state is not RoleState.INVALID for role in role_results),
            complete=complete and not errors,
            allowed_patterns=patterns,
            roles=resolved_roles,
            workflow_connections=_effective_workflow_connections(resolved_roles),
            errors=tuple(dict.fromkeys(errors)),
        )


__all__ = [
    "CompositionResolution",
    "CompositionResolutionService",
    "InstanceResolution",
    "RoleResolution",
    "RoleState",
    "WorkflowConnectionResolution",
]
