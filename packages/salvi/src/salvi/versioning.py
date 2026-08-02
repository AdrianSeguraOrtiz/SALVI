"""Public package and artifact version information."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from types import MappingProxyType
from typing import Final


def package_version() -> str:
    """Return the installed core version without making source-tree imports fail."""

    try:
        return version("salvi")
    except PackageNotFoundError:  # pragma: no cover - source tree without installation
        return "0.0.0+uninstalled"


@dataclass(frozen=True, slots=True)
class SchemaVersion:
    """Current writer version and reader range for one public contract."""

    current: int
    minimum_readable: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


SCHEMA_VERSIONS: Final = MappingProxyType(
    {
        "run_configuration": SchemaVersion(current=1, minimum_readable=1),
        "dataset_bundle": SchemaVersion(current=1, minimum_readable=1),
        "ground_truth": SchemaVersion(current=1, minimum_readable=1),
        "bicluster_set": SchemaVersion(current=7, minimum_readable=5),
        "search_checkpoint": SchemaVersion(current=4, minimum_readable=4),
        "run_metadata": SchemaVersion(current=2, minimum_readable=2),
        "profile_report": SchemaVersion(current=1, minimum_readable=1),
    }
)


def public_version_info() -> dict[str, object]:
    return {
        "salvi_version": package_version(),
        "schemas": {name: schema.as_dict() for name, schema in SCHEMA_VERSIONS.items()},
    }


__all__ = [
    "SCHEMA_VERSIONS",
    "SchemaVersion",
    "package_version",
    "public_version_info",
]
