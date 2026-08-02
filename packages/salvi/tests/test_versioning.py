from __future__ import annotations

from salvi.domain.search import SearchCheckpoint
from salvi.infrastructure.bicluster_set import BiclusterSetManifest
from salvi.infrastructure.dataset_bundle import DatasetManifest
from salvi.infrastructure.ground_truth import GroundTruth
from salvi.versioning import SCHEMA_VERSIONS, public_version_info


def test_public_schema_registry_matches_runtime_models() -> None:
    assert (
        DatasetManifest.model_fields["schema_version"].default
        == SCHEMA_VERSIONS["dataset_bundle"].current
    )
    assert (
        GroundTruth.model_fields["schema_version"].default
        == SCHEMA_VERSIONS["ground_truth"].current
    )
    assert (
        BiclusterSetManifest.model_fields["schema_version"].default
        == SCHEMA_VERSIONS["bicluster_set"].current
    )
    assert (
        SearchCheckpoint.model_fields["schema_version"].default
        == SCHEMA_VERSIONS["search_checkpoint"].current
    )

    information = public_version_info()
    assert information["schemas"]["run_configuration"] == {
        "current": 1,
        "minimum_readable": 1,
    }
