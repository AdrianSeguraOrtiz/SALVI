"""Backend-neutral descriptions of externally implemented variation operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class BackendOperatorSpec:
    """Lazy factory reference used by optional evolutionary backends."""

    factory_path: str
    keyword_arguments: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        module_name, separator, attribute = self.factory_path.partition(":")
        if not separator or not module_name or not attribute:
            raise ValueError("backend operator factory_path must use 'module:attribute'")
        names = tuple(name for name, _ in self.keyword_arguments)
        if len(set(names)) != len(names):
            raise ValueError("backend operator keyword arguments must be unique")


@runtime_checkable
class BackendOperatorProvider(Protocol):
    def backend_operator_spec(self, backend: str) -> BackendOperatorSpec: ...


__all__ = ["BackendOperatorProvider", "BackendOperatorSpec"]
