"""Paged result and source-matrix projections for the web interface."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq

from salvi.domain.enums import PatternKind
from salvi.domain.models import Evaluation
from salvi.domain.prepared import PreparedColumnMetadata
from salvi.exceptions import ArtifactError
from salvi.infrastructure.bicluster_set import PagedBiclusterSetReader
from salvi.infrastructure.dataset_bundle import DatasetBundleReader
from salvi.web.providers import WebProviderRegistry
from salvi.web.storage import WebStateStore

ResultKind = Literal["raw", "selected"]


def _value_map(values: Sequence[Any]) -> dict[str, float]:
    return {str(value.name): float(value.value) for value in values}


def _evaluation_summary(evaluation: Evaluation) -> dict[str, Any]:
    bicluster = evaluation.candidate.bicluster
    selection = evaluation.final_selection
    return {
        "identifier": evaluation.candidate.identifier,
        "generation": evaluation.candidate.generation,
        "row_count": len(bicluster.row_indices),
        "column_count": len(bicluster.column_indices),
        "objectives": _value_map(evaluation.objectives),
        "constraints": _value_map(evaluation.constraints),
        "descriptors": _value_map(evaluation.descriptors),
        "feasible": evaluation.feasible,
        "valid": evaluation.valid,
        "archive_coordinate": evaluation.archive_coordinate,
        "selection_rank": None if selection is None else selection.selection_rank,
        "quality_score": None if selection is None else selection.quality_score,
        "novelty_score": None if selection is None else selection.novelty_score,
        "patterns": sorted(
            {
                column.pattern.value
                for column in (
                    () if evaluation.pattern_fit is None else evaluation.pattern_fit.columns
                )
                if column.pattern is not None
            }
        ),
        "provenance": (
            None
            if evaluation.candidate.provenance is None
            else evaluation.candidate.provenance.model_dump(mode="json")
        ),
    }


class WebResultService:
    def __init__(
        self,
        store: WebStateStore,
        providers: WebProviderRegistry,
    ) -> None:
        self._store = store
        self._providers = providers

    def result_path(self, run_identifier: str, kind: ResultKind) -> Path:
        run = self._store.get_run(run_identifier)
        if run is None:
            raise ArtifactError(f"unknown run: {run_identifier}")
        selected = run.output_directory / "artifacts" / "repertoire"
        raw = run.output_directory / "artifacts" / "search-repertoire"
        path = raw if kind == "raw" else selected
        if not path.is_dir():
            raise ArtifactError(f"run {run_identifier!r} has no {kind} result")
        return path

    def page(
        self,
        run_identifier: str,
        kind: ResultKind,
        *,
        offset: int,
        limit: int,
        query: str | None = None,
        feasible: bool | None = None,
        pattern: PatternKind | None = None,
        min_rows: int | None = None,
        max_rows: int | None = None,
        min_columns: int | None = None,
        max_columns: int | None = None,
    ) -> dict[str, Any]:
        source = PagedBiclusterSetReader(self.result_path(run_identifier, kind))
        filtered = any(
            value is not None
            for value in (
                query,
                feasible,
                pattern,
                min_rows,
                max_rows,
                min_columns,
                max_columns,
            )
        )
        if filtered:
            identifiers = self._filtered_identifiers(
                source,
                query=query,
                feasible=feasible,
                pattern=pattern,
                min_rows=min_rows,
                max_rows=max_rows,
                min_columns=min_columns,
                max_columns=max_columns,
            )
            selected = identifiers[offset : offset + limit]
            evaluations = source.read_identifiers(selected)
            total = len(identifiers)
        else:
            evaluations = source.read_page(offset, limit=limit)
            total = source.row_count
        return {
            "offset": offset,
            "limit": limit,
            "total": total,
            "items": [_evaluation_summary(item) for item in evaluations],
        }

    @staticmethod
    def _filtered_identifiers(
        source: PagedBiclusterSetReader,
        *,
        query: str | None,
        feasible: bool | None,
        pattern: PatternKind | None,
        min_rows: int | None,
        max_rows: int | None,
        min_columns: int | None,
        max_columns: int | None,
    ) -> tuple[str, ...]:
        normalized_query = None if not query else query.strip().lower()
        pattern_members = None if pattern is None else source.pattern_identifiers(pattern)
        matched: list[str] = []
        columns = (
            "bicluster_id",
            "row_indices",
            "column_indices",
            "objectives_json",
            "constraints_json",
            "descriptors_json",
            "provenance_json",
            "feasible",
        )
        for batch in source.core_batches(columns=columns):
            for record in batch.to_pylist():
                identifier = str(record["bicluster_id"])
                row_count = len(record["row_indices"])
                column_count = len(record["column_indices"])
                if feasible is not None and bool(record["feasible"]) is not feasible:
                    continue
                if pattern_members is not None and identifier not in pattern_members:
                    continue
                if min_rows is not None and row_count < min_rows:
                    continue
                if max_rows is not None and row_count > max_rows:
                    continue
                if min_columns is not None and column_count < min_columns:
                    continue
                if max_columns is not None and column_count > max_columns:
                    continue
                if normalized_query is not None:
                    haystack = " ".join(
                        str(record[name])
                        for name in (
                            "bicluster_id",
                            "objectives_json",
                            "constraints_json",
                            "descriptors_json",
                            "provenance_json",
                        )
                    ).lower()
                    if normalized_query not in haystack:
                        continue
                matched.append(identifier)
        return tuple(matched)

    def detail(
        self,
        run_identifier: str,
        kind: ResultKind,
        bicluster_identifier: str,
    ) -> dict[str, Any]:
        source = PagedBiclusterSetReader(self.result_path(run_identifier, kind))
        evaluation = source.read_by_identifier(bicluster_identifier)
        result = _evaluation_summary(evaluation)
        result.update(
            {
                "row_indices": evaluation.candidate.bicluster.row_indices,
                "column_indices": evaluation.candidate.bicluster.column_indices,
                "columns": [
                    self._column_detail(column, evaluation)
                    for column in source.columns
                    if column.index in evaluation.candidate.bicluster.column_indices
                ],
                "pattern_groups": (
                    []
                    if evaluation.pattern_fit is None
                    else [group.model_dump(mode="json") for group in evaluation.pattern_fit.groups]
                ),
                "issues": [issue.model_dump(mode="json") for issue in evaluation.issues],
                "final_selection": (
                    None
                    if evaluation.final_selection is None
                    else evaluation.final_selection.model_dump(mode="json")
                ),
            }
        )
        return result

    @staticmethod
    def _column_detail(
        column: PreparedColumnMetadata,
        evaluation: Evaluation,
    ) -> dict[str, Any]:
        pattern = (
            None
            if evaluation.pattern_fit is None
            else next(
                (
                    item
                    for item in evaluation.pattern_fit.columns
                    if item.column_index == column.index
                ),
                None,
            )
        )
        objectives = {
            objective.name: next(
                (
                    value.model_dump(mode="json")
                    for value in objective.columns
                    if value.column_index == column.index
                ),
                None,
            )
            for objective in evaluation.objectives
        }
        return {
            **asdict(column),
            "kind": column.kind.value,
            "pattern_fit": None if pattern is None else pattern.model_dump(mode="json"),
            "objectives": objectives,
        }

    def matrix(
        self,
        run_identifier: str,
        kind: ResultKind,
        bicluster_identifier: str,
        *,
        row_offset: int,
        row_limit: int,
        column_offset: int,
        column_limit: int,
    ) -> dict[str, Any]:
        run = self._store.get_run(run_identifier)
        if run is None:
            raise ArtifactError(f"unknown run: {run_identifier}")
        dataset = self._store.get_dataset(run.dataset_identifier)
        if dataset is None:
            raise ArtifactError(f"run dataset is no longer available: {run.dataset_identifier}")
        source = PagedBiclusterSetReader(self.result_path(run_identifier, kind))
        evaluation = source.read_by_identifier(bicluster_identifier)
        rows = evaluation.candidate.bicluster.row_indices[row_offset : row_offset + row_limit]
        prepared_indices = evaluation.candidate.bicluster.column_indices[
            column_offset : column_offset + column_limit
        ]
        columns = tuple(source.columns[index] for index in prepared_indices)
        values = _source_values(dataset.bundle_path, rows, columns)
        return {
            "row_offset": row_offset,
            "column_offset": column_offset,
            "total_rows": len(evaluation.candidate.bicluster.row_indices),
            "total_columns": len(evaluation.candidate.bicluster.column_indices),
            "row_indices": rows,
            "columns": [
                {
                    "index": column.index,
                    "name": column.name,
                    "kind": column.kind.value,
                    "source_column_index": column.source_column_index,
                    "derivation": column.derivation,
                }
                for column in columns
            ],
            "values": values,
        }

    def accuracy(
        self,
        run_identifier: str,
        kind: ResultKind,
        analysis_name: str,
    ) -> dict[str, Any]:
        run = self._store.get_run(run_identifier)
        if run is None:
            raise ArtifactError(f"unknown run: {run_identifier}")
        analysis = self._providers.analysis(analysis_name)
        if analysis_name not in run.analyses:
            raise ArtifactError(
                f"analysis {analysis_name!r} was not enabled for run {run_identifier!r}"
            )
        dataset = self._store.get_dataset(run.dataset_identifier)
        if dataset is None:
            raise ArtifactError(f"unknown dataset: {run.dataset_identifier}")
        result = analysis.calculate(
            dataset_bundle=dataset.bundle_path,
            bicluster_set=self.result_path(run_identifier, kind),
        )
        return result.model_dump(mode="json")


def _source_values(
    bundle: Path,
    row_indices: Sequence[int],
    columns: Sequence[PreparedColumnMetadata],
) -> list[list[Any]]:
    if not row_indices or not columns:
        return []
    manifest = DatasetBundleReader().read(bundle)
    data_path = bundle / "data.parquet"
    parquet = pq.ParquetFile(data_path)
    source_indices = tuple(dict.fromkeys(column.source_column_index for column in columns))
    source_names = [manifest.columns[index].name for index in source_indices]
    values_by_row: dict[int, dict[int, Any]] = {}
    row_group_start = 0
    requested = set(row_indices)
    try:
        for row_group in range(parquet.num_row_groups):
            row_count = parquet.metadata.row_group(row_group).num_rows
            local = sorted(
                row - row_group_start
                for row in requested
                if row_group_start <= row < row_group_start + row_count
            )
            if local:
                table = parquet.read_row_group(row_group, columns=source_names)
                selected = table.take(pa.array(local, type=pa.int64()))
                for position, local_index in enumerate(local):
                    absolute = row_group_start + local_index
                    values_by_row[absolute] = {
                        source_index: selected.column(index)[position].as_py()
                        for index, source_index in enumerate(source_indices)
                    }
            row_group_start += row_count
    except (OSError, pa.ArrowException) as error:
        raise ArtifactError(f"cannot read dataset matrix fragment: {error}") from error
    return [
        [
            (
                values_by_row[row][column.source_column_index] is None
                if column.derivation == "missingness_indicators"
                else values_by_row[row][column.source_column_index]
            )
            for column in columns
        ]
        for row in row_indices
    ]


__all__ = ["ResultKind", "WebResultService"]
