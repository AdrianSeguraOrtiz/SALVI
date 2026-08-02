from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

import salvi_experiments.interop.uci as uci_module
from salvi import (
    Bicluster,
    BiclusterSetWriter,
    Candidate,
    DatasetBundleReader,
    Evaluation,
)
from salvi.domain.models import CandidateProvenance, NamedValue, Repertoire
from salvi.domain.prepared import PreparedColumnMetadata
from salvi.exceptions import ConversionError
from salvi_experiments.dataset.clinical import (
    ClinicalTestingConfiguration,
    ClinicalValidationConfiguration,
    RepertoireReference,
    calculate_clinical_associations,
    calculate_repertoire_stability,
    run_clinical_validation,
)
from salvi_experiments.interop.uci import (
    ClinicalAnnotation,
    ClinicalAnnotationKind,
    ClinicalColumnRole,
    ClinicalDatasetBundleReader,
    ClinicalDatasetManifest,
    DerivedAnnotation,
    DerivedOperation,
    UciColumnRule,
    UciConverter,
    UciImportRecipe,
    UciRepositoryClient,
)


class _UciFixtureClient:
    def __init__(self, metadata: dict[str, object], source: Path) -> None:
        self.metadata = metadata
        self.source = source

    def fetch(self, dataset_id: int, expected_sha256: str) -> tuple[dict[str, object], Path]:
        assert dataset_id == 999
        assert expected_sha256 == "0" * 64
        return self.metadata, self.source


@pytest.fixture
def clinical_bundle(tmp_path: Path) -> Path:
    source = tmp_path / "uci.csv"
    rows = [
        {
            "id": f"patient-{index}",
            "age": str(40 + index),
            "phenotype": "A" if index < 4 else "B",
            "outcome": "1" if index < 4 else "0",
            "time": str(10 + index),
            "death": "1" if index < 4 else "0",
            "score": str(index if index < 4 else index + 10),
            "category": "X" if index < 4 else "Y",
            "stage": "1" if index < 4 else "2",
        }
        for index in range(8)
    ]
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    variables = [
        {"name": "id", "role": "ID", "type": "Categorical", "units": None},
        {"name": "age", "role": "Feature", "type": "Integer", "units": "years"},
        {"name": "phenotype", "role": "Feature", "type": "Categorical", "units": None},
        {"name": "outcome", "role": "Target", "type": "Binary", "units": None},
        {"name": "time", "role": "Target", "type": "Continuous", "units": "days"},
        {"name": "death", "role": "Other", "type": "Binary", "units": None},
        {"name": "score", "role": "Other", "type": "Continuous", "units": None},
        {"name": "category", "role": "Other", "type": "Categorical", "units": None},
        {"name": "stage", "role": "Other", "type": "Categorical", "units": None},
    ]
    recipe = UciImportRecipe(
        identifier="clinical-fixture",
        dataset_id=999,
        expected_sha256="0" * 64,
        columns=(
            UciColumnRule(name="age", copy_to_annotations=True),
            UciColumnRule(
                name="time",
                role=ClinicalColumnRole.OUTCOME,
                annotation_kind=ClinicalAnnotationKind.SURVIVAL_TIME,
                survival_event_column="death",
            ),
            UciColumnRule(
                name="death",
                role=ClinicalColumnRole.SUPPLEMENTARY,
                annotation_kind=ClinicalAnnotationKind.BOOLEAN,
            ),
            UciColumnRule(
                name="score",
                role=ClinicalColumnRole.COVARIATE,
                annotation_kind=ClinicalAnnotationKind.NUMERIC,
            ),
            UciColumnRule(
                name="category",
                role=ClinicalColumnRole.COVARIATE,
                annotation_kind=ClinicalAnnotationKind.CATEGORICAL,
            ),
            UciColumnRule(
                name="stage",
                role=ClinicalColumnRole.COVARIATE,
                annotation_kind=ClinicalAnnotationKind.ORDINAL,
                categories=("1", "2"),
            ),
        ),
    )
    return UciConverter(
        client=_UciFixtureClient(
            {"name": "Clinical fixture", "variables": variables},
            source,
        )
    ).convert(recipe, tmp_path / "clinical")


def _write_result(clinical_bundle: Path, destination: Path) -> Path:
    clinical = ClinicalDatasetBundleReader().load(clinical_bundle)
    dataset = DatasetBundleReader().inspect(clinical.dataset_bundle)
    candidate = Candidate(
        identifier="first-half",
        bicluster=Bicluster(row_indices=(0, 1, 2, 3), column_indices=(0, 1)),
        provenance=CandidateProvenance(
            producer="test",
            operation="fixture",
            sequence=0,
        ),
    )
    BiclusterSetWriter().write(
        destination,
        identifier=destination.name,
        dataset_identifier=dataset.identifier,
        row_count=dataset.row_count,
        source_column_count=dataset.column_count,
        columns=tuple(
            PreparedColumnMetadata(
                index=column.index,
                name=column.name,
                kind=column.kind,
                categories=column.categories,
                source_column_index=column.index,
            )
            for column in dataset.columns
        ),
        repertoire=Repertoire(
            evaluations=(
                Evaluation(
                    candidate=candidate,
                    objectives=(),
                    descriptors=(
                        NamedValue(name="row_cardinality", value=4),
                        NamedValue(name="column_cardinality", value=2),
                    ),
                ),
            )
        ),
        source_run="test",
    )
    return destination


def test_uci_conversion_separates_search_data_from_clinical_annotations(
    clinical_bundle: Path,
) -> None:
    clinical = ClinicalDatasetBundleReader().load(clinical_bundle)
    dataset = DatasetBundleReader().load(clinical.dataset_bundle)

    assert dataset.dataset.identifier == "clinical-fixture"
    assert tuple(column.name for column in dataset.dataset.columns) == ("age", "phenotype")
    assert clinical.annotations.column_names == [
        "row_identifier",
        "age",
        "outcome",
        "time",
        "death",
        "score",
        "category",
        "stage",
    ]
    assert not {"outcome", "time", "death"} & {column.name for column in dataset.dataset.columns}


def test_uci_client_rejects_a_modified_resource_but_keeps_the_actual_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = b"id,feature\nrow-0,1\n"
    actual = hashlib.sha256(data).hexdigest()
    metadata = json.dumps(
        {
            "status": 200,
            "data": {
                "uci_id": 999,
                "name": "Changed fixture",
                "variables": [
                    {"name": "id", "role": "ID", "type": "Categorical"},
                    {"name": "feature", "role": "Feature", "type": "Continuous"},
                ],
            },
        }
    ).encode()
    client = UciRepositoryClient(cache_directory=tmp_path / "cache")
    monkeypatch.setattr(
        client,
        "_download",
        lambda url: metadata if "api/dataset" in url else data,
    )

    with pytest.raises(ConversionError, match="checksum changed"):
        client.fetch(999, "0" * 64)
    assert (tmp_path / "cache" / "999" / actual / "data.csv").read_bytes() == data
    fetched, path = client.fetch(999, actual)
    assert fetched["name"] == "Changed fixture"
    assert path.read_bytes() == data


def test_uci_recipe_and_annotation_contracts_reject_ambiguous_metadata() -> None:
    with pytest.raises(ValidationError, match="SURVIVAL_TIME"):
        UciColumnRule(
            name="time",
            annotation_kind=ClinicalAnnotationKind.NUMERIC,
            survival_event_column="event",
        )
    with pytest.raises(ValidationError, match="categories must be unique"):
        UciColumnRule(name="stage", categories=("I", "I"))
    with pytest.raises(ValidationError, match="mapping keys must not be blank"):
        UciColumnRule(name="binary", mapping={"": 1})

    with pytest.raises(ValidationError, match="derived annotations must be"):
        DerivedAnnotation(
            name="invalid",
            source="source",
            operation=DerivedOperation.EQUAL,
            value=1,
            role=ClinicalColumnRole.SEARCH,
            annotation_kind=ClinicalAnnotationKind.BOOLEAN,
        )
    with pytest.raises(ValidationError, match="IN derivations require"):
        DerivedAnnotation(
            name="invalid",
            source="source",
            operation=DerivedOperation.IN,
            value=1,
            role=ClinicalColumnRole.OUTCOME,
            annotation_kind=ClinicalAnnotationKind.BOOLEAN,
        )
    with pytest.raises(ValidationError, match="require one scalar"):
        DerivedAnnotation(
            name="invalid",
            source="source",
            operation=DerivedOperation.EQUAL,
            value=(1, 2),
            role=ClinicalColumnRole.OUTCOME,
            annotation_kind=ClinicalAnnotationKind.BOOLEAN,
        )

    with pytest.raises(ValidationError, match="unique source names"):
        UciImportRecipe(
            identifier="duplicate",
            dataset_id=1,
            expected_sha256="0" * 64,
            columns=(UciColumnRule(name="x"), UciColumnRule(name="x")),
        )
    derivation = DerivedAnnotation(
        name="derived",
        source="source",
        operation=DerivedOperation.EQUAL,
        value=1,
        role=ClinicalColumnRole.OUTCOME,
        annotation_kind=ClinicalAnnotationKind.BOOLEAN,
    )
    with pytest.raises(ValidationError, match="derived annotation names must be unique"):
        UciImportRecipe(
            identifier="duplicate",
            dataset_id=1,
            expected_sha256="0" * 64,
            derived_annotations=(derivation, derivation),
        )

    with pytest.raises(ValidationError, match="require categories"):
        ClinicalAnnotation(
            name="stage",
            source_name="stage",
            role=ClinicalColumnRole.OUTCOME,
            kind=ClinicalAnnotationKind.ORDINAL,
        )
    with pytest.raises(ValidationError, match="only categorical"):
        ClinicalAnnotation(
            name="age",
            source_name="age",
            role=ClinicalColumnRole.COVARIATE,
            kind=ClinicalAnnotationKind.NUMERIC,
            categories=("invalid",),
        )
    with pytest.raises(ValidationError, match="SEARCH annotations"):
        ClinicalAnnotation(
            name="age",
            source_name="age",
            role=ClinicalColumnRole.SEARCH,
            kind=ClinicalAnnotationKind.NUMERIC,
        )

    with pytest.raises(ValidationError, match="checksums do not cover"):
        ClinicalDatasetManifest(
            identifier="clinical",
            uci_dataset_id=1,
            uci_dataset_name="Clinical",
            row_count=1,
            source_data_sha256="0" * 64,
            dataset_manifest_sha256="0" * 64,
            annotations=(),
            checksums={},
        )


def test_uci_primitive_coercion_and_derivation_are_explicit() -> None:
    defaults = uci_module.UciRoleDefaults()
    assert defaults.for_uci_role("ID") is ClinicalColumnRole.IDENTIFIER
    assert defaults.for_uci_role("Feature") is ClinicalColumnRole.SEARCH
    assert defaults.for_uci_role("Target") is ClinicalColumnRole.OUTCOME
    assert defaults.for_uci_role("unknown") is ClinicalColumnRole.EXCLUDED

    missing = frozenset({"NA"})
    assert (
        uci_module._coerce(
            "NA", kind=ClinicalAnnotationKind.NUMERIC, mapping={}, missing_tokens=missing
        )
        is None
    )
    assert (
        uci_module._coerce(
            "yes", kind=ClinicalAnnotationKind.BOOLEAN, mapping={}, missing_tokens=missing
        )
        is True
    )
    assert (
        uci_module._coerce(
            "0", kind=ClinicalAnnotationKind.BOOLEAN, mapping={}, missing_tokens=missing
        )
        is False
    )
    assert (
        uci_module._coerce(
            "yes",
            kind=ClinicalAnnotationKind.BOOLEAN,
            mapping={"yes": True},
            missing_tokens=missing,
        )
        is True
    )
    with pytest.raises(ConversionError, match="as Boolean"):
        uci_module._coerce(
            "maybe", kind=ClinicalAnnotationKind.BOOLEAN, mapping={}, missing_tokens=missing
        )
    with pytest.raises(ConversionError, match="as numeric"):
        uci_module._coerce(
            "text", kind=ClinicalAnnotationKind.NUMERIC, mapping={}, missing_tokens=missing
        )
    with pytest.raises(ConversionError, match="non-finite"):
        uci_module._coerce(
            "inf", kind=ClinicalAnnotationKind.NUMERIC, mapping={}, missing_tokens=missing
        )
    assert (
        uci_module._coerce(
            "group", kind=ClinicalAnnotationKind.CATEGORICAL, mapping={}, missing_tokens=missing
        )
        == "group"
    )

    values = (None, 1, 2)

    def derive(operation: DerivedOperation, value: object) -> list[object | None]:
        rule = DerivedAnnotation(
            name="derived",
            source="source",
            operation=operation,
            value=value,
            role=ClinicalColumnRole.OUTCOME,
            annotation_kind=ClinicalAnnotationKind.BOOLEAN,
        )
        return uci_module._derive(values, rule)

    assert derive(DerivedOperation.EQUAL, 1) == [None, True, False]
    assert derive(DerivedOperation.NOT_EQUAL, 1) == [None, False, True]
    assert derive(DerivedOperation.LESS_EQUAL, 1) == [None, True, False]
    assert derive(DerivedOperation.IN, (2, 3)) == [None, False, True]


def test_uci_repository_reports_network_and_metadata_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        UciRepositoryClient(timeout_seconds=0)

    client = UciRepositoryClient(cache_directory=tmp_path / "cache")
    with pytest.raises(ConversionError, match="does not exist"):
        client._metadata_payload({}, 7)
    with pytest.raises(ConversionError, match="identity"):
        client._metadata_payload({"status": 200, "data": {"uci_id": 8}}, 7)

    invalid_cache = tmp_path / "metadata.json"
    invalid_cache.write_text("not-json", encoding="utf-8")
    with pytest.raises(ConversionError, match="invalid cached"):
        client._read_metadata(invalid_cache, 7)

    monkeypatch.setattr(client, "_download", lambda _url: b"not-json")
    with pytest.raises(ConversionError, match="invalid metadata"):
        client.fetch_current(7)

    def fail_download(_url: str) -> bytes:
        raise OSError("offline")

    monkeypatch.setattr(client, "_download", fail_download)
    with pytest.raises(ConversionError, match="cannot download"):
        client.fetch_current(7)


def test_clinical_associations_and_validation_artifacts(
    clinical_bundle: Path,
    tmp_path: Path,
) -> None:
    result = _write_result(clinical_bundle, tmp_path / "result")
    testing = ClinicalTestingConfiguration(
        minimum_members=2,
        minimum_nonmembers=2,
        minimum_events=2,
    )
    associations = calculate_clinical_associations(
        clinical_bundle,
        result,
        testing=testing,
    )
    outcome = next(item for item in associations if item["annotation"] == "outcome")
    assert outcome["evaluable"] is True
    assert outcome["test"] == "fisher_exact"
    assert float(outcome["p_value"]) < 0.05
    survival = next(item for item in associations if item["annotation"] == "time")
    assert survival["test"] == "log_rank"
    assert survival["effect_type"] == "hazard_ratio"
    assert {
        item["test"]
        for item in associations
        if item["annotation"] in {"score", "category", "stage"}
    } == {"mann_whitney_u", "chi_square"}

    output = run_clinical_validation(
        ClinicalValidationConfiguration(
            identifier="validation",
            clinical_dataset_bundle=clinical_bundle,
            bicluster_set=result,
            output_directory=tmp_path / "validation",
            testing=testing,
        )
    )
    assert pq.read_table(output / "bicluster-characterization.parquet").num_rows == 1
    assert pq.read_table(output / "outcome-associations.parquet").num_rows == 5


def test_repertoire_stability_matches_identical_structures(
    clinical_bundle: Path,
    tmp_path: Path,
) -> None:
    first = _write_result(clinical_bundle, tmp_path / "first")
    second = _write_result(clinical_bundle, tmp_path / "second")
    dataset = ClinicalDatasetBundleReader().load(clinical_bundle).dataset_bundle

    stability = calculate_repertoire_stability(
        (
            RepertoireReference(
                identifier="first",
                dataset_bundle=dataset,
                bicluster_set=first,
            ),
            RepertoireReference(
                identifier="second",
                dataset_bundle=dataset,
                bicluster_set=second,
            ),
        )
    )
    assert stability[0]["mean_matched_stability"] == pytest.approx(1.0)
    assert stability[0]["coverage_at_0_75"] == pytest.approx(1.0)
