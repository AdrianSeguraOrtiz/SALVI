"""User-facing descriptions derived from registered component contracts."""

from __future__ import annotations

from typing import Any

from salvi.components.catalog_models import (
    ComponentDescription,
    ComponentMaturity,
    ComponentParameterDescription,
    ComponentReference,
    MetricPopulation,
    MetricTemporalScope,
    MetricValueKind,
    ObserverMetricGroupPresentation,
    ObserverMetricPresentation,
    ObserverPresentation,
    ObserverViewKind,
    ParameterWidget,
    RolePresentation,
    WorkflowConnection,
    WorkflowConnectionKind,
    WorkflowStage,
    WorkflowStagePresentation,
)
from salvi.components.catalog_presentations import (
    _ADVANCED_PARAMETERS,
    _COMPONENT_DESCRIPTIONS,
    _CONFIGURATION_PATHS,
    _OBSERVER_PRESENTATIONS,
    _PARAMETER_DESCRIPTIONS,
    _PARAMETER_UNITS,
    _ROLE_ICONS,
    _ROLE_PRESENTATIONS,
    _WORKFLOW_STAGE_ORDER,
    _WORKFLOW_STAGES,
)
from salvi.components.protocols import ComponentKind
from salvi.domain.enums import PatternKind


def role_catalog() -> tuple[RolePresentation, ...]:
    return tuple(
        item.model_copy(
            update={
                "configuration_path": _CONFIGURATION_PATHS[item.kind],
                "icon": _ROLE_ICONS[item.kind],
            }
        )
        for item in sorted(
            _ROLE_PRESENTATIONS.values(),
            key=lambda value: (_WORKFLOW_STAGE_ORDER[value.stage], value.order),
        )
    )


def workflow_stage_catalog() -> tuple[WorkflowStagePresentation, ...]:
    """Return stages in their visual and execution order."""

    return _WORKFLOW_STAGES


def _widget(schema: dict[str, Any]) -> ParameterWidget:
    if "enum" in schema:
        return ParameterWidget.SELECT
    alternatives = schema.get("anyOf")
    if isinstance(alternatives, list):
        non_null = tuple(
            item for item in alternatives if isinstance(item, dict) and item.get("type") != "null"
        )
        if len(non_null) == 1:
            return _widget(non_null[0])
    value_type = schema.get("type")
    if value_type == "boolean":
        return ParameterWidget.BOOLEAN
    if value_type in {"integer", "number"}:
        return ParameterWidget.NUMBER
    if value_type == "string":
        return ParameterWidget.TEXT
    return ParameterWidget.STRUCTURED


def humanize_identifier(value: str) -> str:
    return value.replace("_", " ").strip().title()


def default_component_description(kind: ComponentKind, name: str) -> str:
    title = humanize_identifier(name)
    return _COMPONENT_DESCRIPTIONS.get(
        (kind, name),
        f"Configures the {title.lower()} {humanize_identifier(kind.value).lower()}.",
    )


def describe_configuration(
    *,
    kind: ComponentKind,
    name: str,
    title: str,
    description: str,
    provides: frozenset[str],
    requires: frozenset[str],
    supported_patterns: frozenset[PatternKind],
    conflicts: frozenset[tuple[ComponentKind, str]],
    compatibility_notes: tuple[str, ...],
    maturity: ComponentMaturity,
    parameter_patterns: dict[str, frozenset[PatternKind]],
    schema: dict[str, Any],
) -> ComponentDescription:
    required = frozenset(str(item) for item in schema.get("required", ()))
    properties = schema.get("properties", {})
    definitions = schema.get("$defs", {})

    def resolve(value: Any, resolving: frozenset[str] = frozenset()) -> Any:
        if isinstance(value, list):
            return [resolve(item, resolving) for item in value]
        if not isinstance(value, dict):
            return value
        raw = dict(value)
        reference = raw.get("$ref")
        if isinstance(reference, str) and reference.startswith("#/$defs/"):
            definition_name = reference.rsplit("/", maxsplit=1)[-1]
            referenced = definitions.get(definition_name)
            if isinstance(referenced, dict) and definition_name not in resolving:
                overrides = {key: item for key, item in raw.items() if key != "$ref"}
                return resolve(
                    {**referenced, **overrides},
                    resolving | {definition_name},
                )
        return {key: resolve(item, resolving) for key, item in raw.items()}

    parameters: list[ComponentParameterDescription] = []
    if isinstance(properties, dict):
        for parameter_name, raw_schema in properties.items():
            parameter_schema = resolve(raw_schema) if isinstance(raw_schema, dict) else {}
            parameter_title = str(
                parameter_schema.get("title") or humanize_identifier(parameter_name)
            )
            parameter_description = str(
                parameter_schema.get("description")
                or _PARAMETER_DESCRIPTIONS.get(parameter_name)
                or f"Configuration value for {parameter_title.lower()}."
            )
            parameters.append(
                ComponentParameterDescription(
                    name=parameter_name,
                    title=parameter_title,
                    description=parameter_description,
                    required=parameter_name in required,
                    default=parameter_schema.get("default"),
                    value_schema=parameter_schema,
                    applicable_patterns=tuple(
                        sorted(
                            parameter_patterns.get(parameter_name, frozenset()),
                            key=lambda pattern: pattern.value,
                        )
                    ),
                    widget=_widget(parameter_schema),
                    unit=_PARAMETER_UNITS.get(parameter_name),
                    advanced=parameter_name in _ADVANCED_PARAMETERS,
                )
            )
    role = _ROLE_PRESENTATIONS[kind]
    return ComponentDescription(
        kind=kind,
        name=name,
        title=title,
        description=description,
        provides=tuple(sorted(provides)),
        requires=tuple(sorted(requires)),
        supported_patterns=tuple(sorted(supported_patterns, key=lambda pattern: pattern.value)),
        conflicts=tuple(
            ComponentReference(kind=kind_value, name=name_value)
            for kind_value, name_value in sorted(
                conflicts,
                key=lambda item: (item[0].value, item[1]),
            )
        ),
        compatibility_notes=compatibility_notes,
        maturity=maturity,
        parameters=tuple(parameters),
        stage=role.stage,
        order=role.order,
        observer_view=(
            _OBSERVER_PRESENTATIONS.get(name) if kind is ComponentKind.OBSERVER else None
        ),
    )


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
    "default_component_description",
    "describe_configuration",
    "humanize_identifier",
    "role_catalog",
    "workflow_stage_catalog",
]
