"""Public schema versions owned by the scientific experiment package."""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

from salvi.versioning import SchemaVersion
from salvi_experiments import __version__

EXPERIMENT_SCHEMA_VERSIONS: Final = MappingProxyType(
    {
        "clinical_dataset_bundle": SchemaVersion(current=1, minimum_readable=1),
        "clinical_validation_configuration": SchemaVersion(
            current=1,
            minimum_readable=1,
        ),
        "clinical_validation_report": SchemaVersion(current=1, minimum_readable=1),
        "experiment_configuration": SchemaVersion(current=1, minimum_readable=1),
        "experiment_report": SchemaVersion(current=1, minimum_readable=1),
        "uci_import_recipe": SchemaVersion(current=1, minimum_readable=1),
    }
)


def public_version_info() -> dict[str, object]:
    return {
        "salvi_experiments_version": __version__,
        "schemas": {name: schema.as_dict() for name, schema in EXPERIMENT_SCHEMA_VERSIONS.items()},
    }


__all__ = ["EXPERIMENT_SCHEMA_VERSIONS", "public_version_info"]
