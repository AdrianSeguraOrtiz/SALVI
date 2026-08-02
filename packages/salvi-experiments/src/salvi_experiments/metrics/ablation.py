"""Search-cost and structural-diversity measurements for SALVI ablations."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean, median

from salvi import Repertoire, structural_distance


def _summary(values: tuple[float, ...], prefix: str) -> dict[str, float | None]:
    if not values:
        return {
            f"{prefix}_minimum": None,
            f"{prefix}_median": None,
            f"{prefix}_mean": None,
            f"{prefix}_maximum": None,
        }
    return {
        f"{prefix}_minimum": min(values),
        f"{prefix}_median": median(values),
        f"{prefix}_mean": fmean(values),
        f"{prefix}_maximum": max(values),
    }


def repertoire_diversity(
    repertoire: Repertoire,
    *,
    row_weight: float,
    sample_size: int,
) -> dict[str, int | float | None]:
    """Measure exact and nearest-neighbour diversity in a canonical repertoire."""

    evaluations = repertoire.evaluations
    signatures = tuple(item.candidate.bicluster.signature for item in evaluations)
    unique = len(set(signatures))
    coordinates = {
        item.archive_coordinate for item in evaluations if item.archive_coordinate is not None
    }
    sampled = tuple(
        sorted(evaluations, key=lambda item: item.candidate.bicluster.signature)[:sample_size]
    )
    nearest = [1.0] * len(sampled)
    for left_index, left in enumerate(sampled):
        left_bicluster = left.candidate.bicluster
        left_rows = frozenset(left_bicluster.row_indices)
        left_columns = frozenset(left_bicluster.column_indices)
        for right_index in range(left_index + 1, len(sampled)):
            right_bicluster = sampled[right_index].candidate.bicluster
            distance = structural_distance(
                left_rows,
                left_columns,
                frozenset(right_bicluster.row_indices),
                frozenset(right_bicluster.column_indices),
                row_weight=row_weight,
            )
            nearest[left_index] = min(nearest[left_index], distance)
            nearest[right_index] = min(nearest[right_index], distance)
    distances = tuple(nearest) if len(sampled) >= 2 else ()
    return {
        "repertoire_count": len(evaluations),
        "repertoire_unique_structures": unique,
        "repertoire_duplicate_ratio": (0.0 if not evaluations else 1.0 - unique / len(evaluations)),
        "repertoire_occupied_coordinates": len(coordinates),
        "repertoire_diversity_sample_size": len(sampled),
        **_summary(distances, "repertoire_nearest_distance"),
    }


def _resource_metrics(connection: sqlite3.Connection) -> dict[str, float | None]:
    result: dict[str, float | None] = {
        "process_cpu_seconds": None,
        "peak_resident_memory_bytes": None,
        "peak_active_threads": None,
    }
    queries = {
        "process_cpu_seconds": "resource.process_cpu_seconds",
        "peak_resident_memory_bytes": "resource.resident_memory_bytes",
        "peak_active_threads": "resource.active_threads",
    }
    for field, name in queries.items():
        row = connection.execute(
            "SELECT MAX(value) FROM metrics WHERE name = ?",
            (name,),
        ).fetchone()
        if row is not None and row[0] is not None:
            result[field] = float(row[0])
    return result


def _latest_observer_metrics(
    connection: sqlite3.Connection,
) -> dict[str, float | None]:
    names = {
        "candidate_window_size": "diversity.window_size",
        "candidate_window_duplicate_ratio": "diversity.window_duplicate_ratio",
        "candidate_window_nearest_distance_minimum": ("diversity.nearest_distance.minimum"),
        "candidate_window_nearest_distance_median": ("diversity.nearest_distance.median"),
        "candidate_window_nearest_distance_mean": "diversity.nearest_distance.mean",
        "candidate_window_nearest_distance_maximum": ("diversity.nearest_distance.maximum"),
    }
    result: dict[str, float | None] = {}
    for field, name in names.items():
        row = connection.execute(
            "SELECT value FROM metrics WHERE name = ? ORDER BY sequence DESC LIMIT 1",
            (name,),
        ).fetchone()
        result[field] = None if row is None else float(row[0])
    return result


def analyze_run_event_store(
    path: Path,
) -> tuple[dict[str, int | float | None], tuple[dict[str, object], ...]]:
    """Derive search diagnostics directly from the durable event stream.

    The calculation does not depend on optional observers. Observer-produced
    resource samples are included when they are present.
    """

    source = path.expanduser().resolve()
    connection = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    stage_seconds: defaultdict[str, float] = defaultdict(float)
    status_counts: Counter[str] = Counter()
    signatures: set[str] = set()
    evaluated = 0
    invalid = 0
    created_cells = 0
    cell_attempts: Counter[tuple[int, ...]] = Counter()
    cell_accepted: Counter[tuple[int, ...]] = Counter()
    emitter_reports: dict[str, dict[str, object]] = {}
    try:
        rows = connection.execute("SELECT event_type, payload_json FROM events ORDER BY sequence")
        for event_type, payload_json in rows:
            payload = json.loads(payload_json)
            runtime = payload.get("runtime")
            if isinstance(runtime, dict):
                duration = runtime.get("duration_seconds")
                if isinstance(duration, int | float):
                    stage = {
                        "candidates.asked": "candidate_generation_seconds",
                        "candidates.evaluated": "evaluation_seconds",
                        "archive.updated": "archive_update_seconds",
                    }.get(str(event_type))
                    if stage is not None:
                        stage_seconds[stage] += float(duration)
            if event_type == "candidates.evaluated":
                for item in payload.get("items", ()):
                    if not isinstance(item, dict):
                        continue
                    evaluated += 1
                    signatures.add(str(item.get("signature", "")))
                    invalid += not bool(item.get("valid", False))
            elif event_type == "archive.updated":
                for outcome in payload.get("outcomes", ()):
                    if not isinstance(outcome, dict):
                        continue
                    status = str(outcome.get("status", "UNKNOWN")).lower()
                    status_counts[status] += 1
                    created_cells += bool(outcome.get("created_cell", False))
                    coordinate_raw = outcome.get("coordinate")
                    if isinstance(coordinate_raw, dict):
                        coordinate_raw = coordinate_raw.get("indices")
                    if isinstance(coordinate_raw, list | tuple):
                        coordinate = tuple(int(value) for value in coordinate_raw)
                        cell_attempts[coordinate] += 1
                        if status in {"inserted", "inserted_with_evictions"}:
                            cell_accepted[coordinate] += 1
            elif event_type == "emitter.credit.updated":
                for report in payload.get("reports", ()):
                    if isinstance(report, dict) and isinstance(
                        report.get("emitter_name"),
                        str,
                    ):
                        emitter_reports[str(report["emitter_name"])] = dict(report)
        acceptance = tuple(
            cell_accepted[coordinate] / attempts for coordinate, attempts in cell_attempts.items()
        )
        accepted = sum(
            count
            for status, count in status_counts.items()
            if status in {"inserted", "inserted_with_evictions"}
        )
        diagnostics: dict[str, int | float | None] = {
            "evaluated_candidates": evaluated,
            "unique_evaluated_candidates": len(signatures),
            "candidate_duplicate_ratio": (
                0.0 if evaluated == 0 else 1.0 - len(signatures) / evaluated
            ),
            "invalid_candidates": invalid,
            "accepted_candidates": accepted,
            "acceptance_ratio": 0.0 if evaluated == 0 else accepted / evaluated,
            "created_cells": created_cells,
            "visited_cells": len(cell_attempts),
            **{key: float(value) for key, value in stage_seconds.items()},
            **_summary(acceptance, "cell_acceptance_ratio"),
            **_resource_metrics(connection),
            **_latest_observer_metrics(connection),
        }
        for stage in (
            "candidate_generation_seconds",
            "evaluation_seconds",
            "archive_update_seconds",
        ):
            diagnostics.setdefault(stage, 0.0)
        evaluation_seconds = float(diagnostics["evaluation_seconds"] or 0.0)
        diagnostics["evaluations_per_second"] = (
            None if evaluation_seconds <= 0.0 else evaluated / evaluation_seconds
        )
        diagnostics.update(
            {f"archive_status_{status}": count for status, count in sorted(status_counts.items())}
        )
        emitters = tuple(
            {
                "emitter_name": name,
                **{key: value for key, value in report.items() if key != "emitter_name"},
            }
            for name, report in sorted(emitter_reports.items())
        )
        return diagnostics, emitters
    finally:
        connection.close()


def flatten_configuration(value: object, prefix: str = "") -> dict[str, object]:
    """Flatten mappings while retaining lists as atomic configuration values."""

    if not isinstance(value, dict):
        return {prefix: value}
    flattened: dict[str, object] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            flattened.update(flatten_configuration(item, path))
        else:
            flattened[path] = item
    return flattened


__all__ = [
    "analyze_run_event_store",
    "flatten_configuration",
    "repertoire_diversity",
]
