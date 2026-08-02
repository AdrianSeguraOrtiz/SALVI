"""Canonical dataset and bicluster coordinate helpers for experiments."""

from __future__ import annotations

from pathlib import Path

from salvi import (
    BiclusterSetReader,
    DatasetBundleReader,
    GroundTruth,
    GroundTruthBicluster,
    PatternKind,
)
from salvi_experiments.configuration import TaskScope
from salvi_experiments.exceptions import ExperimentArtifactError
from salvi_experiments.metrics import BiclusterMembership


def read_scoped_ground_truth(
    dataset_bundle: Path,
    scope: TaskScope,
) -> tuple[GroundTruth, tuple[GroundTruthBicluster, ...]]:
    reader = DatasetBundleReader()
    ground_truth = reader.read_ground_truth(dataset_bundle)
    if ground_truth is None:
        raise ExperimentArtifactError(
            f"DatasetBundle has no canonical ground truth: {dataset_bundle}"
        )
    allowed = {PatternKind(value) for value in scope.included_patterns}
    selected = tuple(
        bicluster
        for bicluster in ground_truth.biclusters
        if {column.pattern for column in bicluster.column_patterns}.issubset(allowed)
    )
    if not selected:
        raise ExperimentArtifactError("the declared task scope contains no ground-truth biclusters")
    return ground_truth, selected


def ground_truth_memberships(
    biclusters: tuple[GroundTruthBicluster, ...],
) -> tuple[BiclusterMembership, ...]:
    return tuple(
        BiclusterMembership(
            identifier=bicluster.identifier,
            row_indices=bicluster.row_indices,
            column_indices=bicluster.column_indices,
        )
        for bicluster in biclusters
    )


def detected_memberships(
    dataset_bundle: Path,
    bicluster_set: Path,
) -> tuple[BiclusterMembership, ...]:
    dataset = DatasetBundleReader().inspect(dataset_bundle)
    contents = BiclusterSetReader().read_contents(bicluster_set)
    manifest = contents.manifest
    if manifest.dataset_identifier != dataset.identifier:
        raise ExperimentArtifactError(
            "BiclusterSet dataset identifier does not match the DatasetBundle"
        )
    if (
        manifest.row_count != dataset.row_count
        or manifest.source_column_count != dataset.column_count
    ):
        raise ExperimentArtifactError(
            "BiclusterSet source dimensions do not match the DatasetBundle"
        )
    source_by_prepared = {column.index: column.source_column_index for column in contents.columns}
    memberships: list[BiclusterMembership] = []
    for evaluation in contents.repertoire.evaluations:
        if not evaluation.feasible:
            continue
        source_columns = tuple(
            sorted(
                {
                    source_by_prepared[column_index]
                    for column_index in evaluation.candidate.bicluster.column_indices
                }
            )
        )
        memberships.append(
            BiclusterMembership(
                identifier=evaluation.candidate.identifier,
                row_indices=evaluation.candidate.bicluster.row_indices,
                column_indices=source_columns,
            )
        )
    return tuple(memberships)


__all__ = [
    "detected_memberships",
    "ground_truth_memberships",
    "read_scoped_ground_truth",
]
