"""Stable, presentation-oriented exports from canonical BiclusterSets."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from salvi.exceptions import ArtifactError
from salvi.infrastructure.bicluster_set import BiclusterSetReader
from salvi.infrastructure.files import atomic_directory


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    fields = tuple(records[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


@dataclass(frozen=True, slots=True)
class CsvBiclusterSetExporter:
    """Export a canonical result into stable normalized CSV tables."""

    overwrite: bool = False

    def export(self, source: Path, destination: Path) -> Path:
        contents = BiclusterSetReader().read_contents(source)
        target = destination.resolve()
        with atomic_directory(target, overwrite=self.overwrite) as temporary:
            _write_csv(
                temporary / "columns.csv",
                [
                    {
                        "column_index": column.index,
                        "name": column.name,
                        "kind": column.kind.value,
                        "categories_json": _json(column.categories),
                        "source_column_index": column.source_column_index,
                        "derivation": column.derivation or "",
                    }
                    for column in contents.columns
                ],
            )
            _write_csv(
                temporary / "biclusters.csv",
                [
                    {
                        "bicluster_id": evaluation.candidate.identifier,
                        "row_indices_json": _json(evaluation.candidate.bicluster.row_indices),
                        "column_indices_json": _json(evaluation.candidate.bicluster.column_indices),
                        "archive_coordinate_json": _json(evaluation.archive_coordinate or ()),
                        "objectives_json": _json(
                            [
                                objective.model_dump(mode="json", exclude={"columns"})
                                for objective in evaluation.objectives
                            ]
                        ),
                        "constraints_json": _json(
                            [
                                constraint.model_dump(mode="json")
                                for constraint in evaluation.constraints
                            ]
                        ),
                        "descriptors_json": _json(
                            [
                                descriptor.model_dump(mode="json")
                                for descriptor in evaluation.descriptors
                            ]
                        ),
                        "valid": evaluation.valid,
                        "feasible": evaluation.feasible,
                        "constraint_violation": evaluation.constraint_violation,
                    }
                    for evaluation in contents.repertoire.evaluations
                ],
            )
            detail_tables = {
                "column-patterns.csv": contents.column_patterns,
                "column-objectives.csv": contents.column_objectives,
                "pattern-groups.csv": contents.pattern_groups,
                "pattern-row-parameters.csv": contents.row_parameters,
                "final-selection.csv": contents.final_selection,
            }
            for filename, models in detail_tables.items():
                records = []
                for model in models:
                    record = model.model_dump(mode="json")
                    records.append(
                        {
                            key: (_json(value) if isinstance(value, dict | list | tuple) else value)
                            for key, value in record.items()
                        }
                    )
                _write_csv(temporary / filename, records)
            manifest = {
                "schema_version": 1,
                "format": "salvi-bicluster-set-csv",
                "source_manifest": contents.manifest.model_dump(mode="json"),
                "files": sorted(path.name for path in temporary.glob("*.csv")),
            }
            (temporary / "export-manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if not (target / "export-manifest.json").is_file():
            raise ArtifactError("CSV export did not produce its manifest")
        return target


__all__ = ["CsvBiclusterSetExporter"]
