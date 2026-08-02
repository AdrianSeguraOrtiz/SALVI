"""Composition of the built-in component registry."""

from salvi.components.default_data_registrations import default_data_registrations
from salvi.components.default_runtime_registrations import default_runtime_registrations
from salvi.components.default_search_registrations import default_search_registrations
from salvi.components.registry import ComponentRegistry
from salvi.domain.enums import SearchFamily


def default_component_registry() -> ComponentRegistry:
    """Build an isolated registry containing every built-in component."""

    registry = ComponentRegistry()
    entries = (
        *default_data_registrations(),
        *default_search_registrations(),
        *default_runtime_registrations(),
    )
    for entry in entries:
        registry.register(entry)
    for family in SearchFamily:
        if registry.search_engines(family):
            registry.default_search_engine(family)
    return registry


__all__ = ["default_component_registry"]
