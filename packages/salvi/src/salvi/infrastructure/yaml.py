"""Strict YAML loading and deterministic serialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from yaml.events import AliasEvent
from yaml.nodes import MappingNode

from salvi.exceptions import ConfigurationError


class StrictSafeLoader(yaml.SafeLoader):
    """Safe loader that rejects aliases, merge keys, and duplicate keys."""

    def compose_node(self, parent: Any, index: Any) -> Any:
        if self.check_event(AliasEvent):  # type: ignore[no-untyped-call]
            event = self.peek_event()  # type: ignore[no-untyped-call]
            raise ConfigurationError(
                f"YAML aliases are not supported (line {event.start_mark.line + 1})"
            )
        return super().compose_node(parent, index)

    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict[Any, Any]:
        if not isinstance(node, MappingNode):
            raise ConfigurationError("expected a YAML mapping")
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key == "<<":
                raise ConfigurationError(
                    f"YAML merge keys are not supported (line {key_node.start_mark.line + 1})"
                )
            if key in mapping:
                raise ConfigurationError(
                    f"duplicate YAML key {key!r} (line {key_node.start_mark.line + 1})"
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def _reject_environment_interpolation(value: Any, location: str = "configuration") -> None:
    if isinstance(value, str) and "${" in value:
        raise ConfigurationError(f"environment interpolation is not supported at {location}")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_environment_interpolation(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_environment_interpolation(item, f"{location}[{index}]")


def load_strict_yaml_text(content: str, *, source: str = "configuration") -> dict[str, Any]:
    """Parse one strict YAML mapping from an in-memory document."""

    try:
        value = yaml.load(content, Loader=StrictSafeLoader)
    except ConfigurationError:
        raise
    except yaml.YAMLError as error:
        raise ConfigurationError(f"invalid YAML in {source}: {error}") from error
    if not isinstance(value, dict):
        raise ConfigurationError(f"YAML document {source} must contain a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ConfigurationError(f"all top-level keys in {source} must be strings")
    _reject_environment_interpolation(value)
    return value


def load_strict_yaml(path: Path) -> dict[str, Any]:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigurationError(f"cannot read YAML file {path}: {error}") from error
    return load_strict_yaml_text(content, source=str(path))


def dump_yaml_text(value: Any) -> str:
    """Serialize YAML deterministically without writing a file."""

    return yaml.safe_dump(value, sort_keys=False, allow_unicode=True)


def dump_yaml(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_yaml_text(value), encoding="utf-8")
