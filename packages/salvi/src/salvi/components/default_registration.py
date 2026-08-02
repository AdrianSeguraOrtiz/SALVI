"""Composition of the built-in component registry."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Literal, cast

from pydantic import BaseModel

from salvi.components.catalog import ComponentMaturity, default_component_description
from salvi.components.membership_emitters import (
    MembershipEmitterConfiguration,
    MembershipMoveEmitter,
)
from salvi.components.protocols import Component, ComponentKind, SearchEngine
from salvi.components.registry import ComponentRegistration
from salvi.domain.enums import PatternKind


def _registration(
    kind: ComponentKind,
    name: str,
    configuration_model: type[BaseModel],
    factory: Callable[[BaseModel], Component],
    *,
    supported_patterns: frozenset[PatternKind] | None = None,
    conflicts: frozenset[tuple[ComponentKind, str]] = frozenset(),
    compatibility_notes: tuple[str, ...] = (),
    maturity: ComponentMaturity = ComponentMaturity.STABLE,
    parameter_patterns: tuple[tuple[str, frozenset[PatternKind]], ...] = (),
    continuation_fingerprint_exclusions: frozenset[str] = frozenset(),
    prototype_component: Component | None = None,
) -> ComponentRegistration:
    prototype = (
        factory(configuration_model.model_validate({}))
        if prototype_component is None
        else prototype_component
    )
    declared_docstring = type(prototype).__dict__.get("__doc__")
    description = (
        inspect.cleandoc(declared_docstring)
        if (
            isinstance(declared_docstring, str)
            and not declared_docstring.lstrip().startswith(f"{type(prototype).__name__}(")
        )
        else default_component_description(kind, name)
    )
    return ComponentRegistration(
        kind=kind,
        name=name,
        configuration_model=configuration_model,
        factory=factory,
        provides=prototype.provides,
        requires=prototype.requires,
        description=description,
        supported_patterns=(
            frozenset(PatternKind) if supported_patterns is None else supported_patterns
        ),
        conflicts=conflicts,
        compatibility_notes=compatibility_notes,
        maturity=maturity,
        parameter_patterns=parameter_patterns,
        continuation_fingerprint_exclusions=continuation_fingerprint_exclusions,
        composition_contract=(
            cast(SearchEngine, prototype).composition_contract
            if kind is ComponentKind.SEARCH_ENGINE
            else None
        ),
    )


def _membership_registration(
    dimension: Literal["rows", "columns"],
    operation: Literal["add", "remove", "swap"],
) -> ComponentRegistration:
    name = f"{operation}_{dimension[:-1]}"

    def create(config: BaseModel) -> Component:
        typed = cast(MembershipEmitterConfiguration, config)
        return MembershipMoveEmitter(
            dimension=dimension,
            operation=operation,
            guided=typed.guided,
            parent_pool_size=typed.parent_pool_size,
            candidate_pool_size=typed.candidate_pool_size,
            component_name=name,
        )

    return _registration(
        ComponentKind.EMITTER,
        name,
        MembershipEmitterConfiguration,
        create,
    )


__all__ = ["_membership_registration", "_registration"]
