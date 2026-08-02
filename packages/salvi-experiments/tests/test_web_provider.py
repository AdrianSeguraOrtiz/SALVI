from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from salvi.infrastructure.dataset_bundle import DatasetBundleReader
from salvi.web.app import create_app
from salvi_experiments.interop.uci import UciRepositoryClient
from salvi_experiments.web_provider import (
    AccuracyWebAnalysis,
    GbicWebAdapter,
    UciWebAdapter,
    create_provider,
)


class _UciWebClient:
    def __init__(self, source: Path) -> None:
        self.source = source
        self.metadata = {
            "name": "Web fixture",
            "variables": [
                {"name": "id", "role": "ID", "type": "Categorical", "units": None},
                {"name": "marker", "role": "Feature", "type": "Continuous", "units": None},
                {"name": "outcome", "role": "Target", "type": "Binary", "units": None},
            ],
        }

    def fetch_current(self, dataset_id: int) -> tuple[dict[str, object], Path, str]:
        assert dataset_id == 999
        return self.metadata, self.source, "0" * 64

    def fetch(self, dataset_id: int, expected_sha256: str) -> tuple[dict[str, object], Path]:
        assert dataset_id == 999
        assert expected_sha256 == "0" * 64
        return self.metadata, self.source


def test_gbic_web_adapter_imports_tables_without_ground_truth(tmp_path: Path) -> None:
    source = tmp_path / "dataset.tsv"
    source.write_text(
        "X\tnumeric\tgroup\nrow-0\t1,5\tcase\nrow-1\t2,5\tcontrol\n",
        encoding="utf-8",
    )
    adapter = GbicWebAdapter()
    preview = adapter.inspect(
        {"data": source},
        identifier="G-Bic table",
        workspace=tmp_path / "inspect",
    )
    assert preview.identifier == "G-Bic-table"
    assert not preview.ground_truth_attached
    assert preview.confirmation_required
    destination = tmp_path / "bundle"
    adapter.convert(
        {"data": source},
        identifier=preview.identifier,
        columns=preview.columns,
        destination=destination,
        workspace=tmp_path / "convert",
    )
    loaded = DatasetBundleReader().load(destination)
    assert loaded.table.column("numeric").to_pylist() == [1.5, 2.5]
    assert DatasetBundleReader().read_ground_truth(destination) is None


def test_gbic_web_adapter_imports_the_optional_ground_truth(tmp_path: Path) -> None:
    source = tmp_path / "dataset.tsv"
    source.write_text(
        "X\tn0\tn1\nrow-0\t1\t2\nrow-1\t1\t2\nrow-2\t8\t9\n",
        encoding="utf-8",
    )
    ground_truth = tmp_path / "dataset.json"
    ground_truth.write_text(
        json.dumps(
            {
                "#DatasetRows": 3,
                "#DatasetColumns": 2,
                "#DatasetMinValue": 0,
                "#DatasetMaxValue": 9,
                "biclusters": {
                    "0": {
                        "Type": "Numeric",
                        "X": [0, 1],
                        "Y": [0, 1],
                        "#rows": 2,
                        "#columns": 2,
                        "%Noise": "0",
                        "%Errors": "0",
                        "%Missings": "0",
                        "RowPattern": "Constant",
                        "ColumnPattern": "None",
                        "PlaidCoherency": "No Overlapping",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    adapter = GbicWebAdapter()
    files = {"data": source, "ground_truth": ground_truth}
    preview = adapter.inspect(
        files,
        identifier="with-truth",
        workspace=tmp_path / "inspect",
    )
    assert preview.ground_truth_attached
    assert not preview.confirmation_required
    destination = tmp_path / "bundle"
    adapter.convert(
        files,
        identifier=preview.identifier,
        columns=preview.columns,
        destination=destination,
        workspace=tmp_path / "convert",
    )
    truth = DatasetBundleReader().read_ground_truth(destination)
    assert truth is not None
    assert truth.biclusters[0].row_indices == (0, 1)


def test_accuracy_web_analysis_uses_the_canonical_experiment_contract(
    experiment_dataset: Path,
    perfect_bicluster_set: Path,
) -> None:
    analysis = AccuracyWebAnalysis()
    assert analysis.description.name == "prelic_accuracy"
    result = analysis.calculate(
        dataset_bundle=experiment_dataset,
        bicluster_set=perfect_bicluster_set,
    )
    assert result.relevance == pytest.approx(1.0)
    assert result.recovery == pytest.approx(1.0)
    assert result.biclustering_error == pytest.approx(1.0)
    provider = create_provider()
    assert provider.adapters[0].description.name == "gbic"
    assert isinstance(provider.adapters[1], UciWebAdapter)
    assert provider.adapters[1].description.parameters[0].name == "dataset_id"
    assert provider.analyses == (analysis,)


def test_accuracy_web_analysis_requires_ground_truth(
    tmp_path: Path,
    perfect_bicluster_set: Path,
) -> None:
    source = tmp_path / "dataset.tsv"
    source.write_text("X\ta\nr0\t1\nr1\t2\n", encoding="utf-8")
    adapter = GbicWebAdapter()
    preview = adapter.inspect(
        {"data": source},
        identifier="no-truth",
        workspace=tmp_path / "inspect",
    )
    destination = tmp_path / "bundle"
    adapter.convert(
        {"data": source},
        identifier=preview.identifier,
        columns=preview.columns,
        destination=destination,
        workspace=tmp_path / "convert",
    )
    with pytest.raises(ValueError, match="ground truth"):
        AccuracyWebAnalysis().calculate(
            dataset_bundle=destination,
            bicluster_set=perfect_bicluster_set,
        )


def test_uci_web_adapter_uses_typed_parameters_and_keeps_outcomes_external(
    tmp_path: Path,
) -> None:
    source = tmp_path / "uci.csv"
    source.write_text(
        "id,marker,outcome\npatient-0,1.0,1\npatient-1,2.0,0\n",
        encoding="utf-8",
    )
    adapter = UciWebAdapter(client=_UciWebClient(source))
    preview = adapter.inspect(
        {},
        parameters={"dataset_id": 999},
        identifier="UCI fixture",
        workspace=tmp_path / "inspect",
    )
    assert preview.confirmation_required
    assert preview.clinical_annotations_attached
    assert preview.column_count == 1
    assert [column.role for column in preview.columns] == [
        "IDENTIFIER",
        "SEARCH",
        "OUTCOME",
    ]

    nested_bundle = adapter.convert(
        {},
        identifier=preview.identifier,
        columns=preview.columns,
        parameters={"dataset_id": 999},
        adapter_configuration=preview.adapter_configuration,
        destination=tmp_path / "import",
        workspace=tmp_path / "convert",
    )
    dataset = DatasetBundleReader().inspect(nested_bundle)
    assert tuple(column.name for column in dataset.columns) == ("marker",)


def test_web_api_imports_uci_without_a_required_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "uci.csv"
    source.write_text(
        "id,marker,outcome\npatient-0,1.0,1\npatient-1,2.0,0\n",
        encoding="utf-8",
    )
    fixture = _UciWebClient(source)
    monkeypatch.setattr(
        UciRepositoryClient,
        "fetch_current",
        lambda _self, dataset_id: fixture.fetch_current(dataset_id),
    )
    monkeypatch.setattr(
        UciRepositoryClient,
        "fetch",
        lambda _self, dataset_id, checksum: fixture.fetch(dataset_id, checksum),
    )

    app = create_app(data_directory=tmp_path / "web")
    with TestClient(app) as client:
        inspected = client.post(
            "/api/v1/imports/uci",
            data={
                "identifier": "clinical",
                "slot_names": "[]",
                "parameters": json.dumps({"dataset_id": 999}),
            },
        )
        assert inspected.status_code == 200
        preview = inspected.json()
        confirmed = client.post(
            f"/api/v1/imports/{preview['identifier']}/confirm",
            json={
                "columns": preview["preview"]["columns"],
                "adapter_configuration": preview["preview"]["adapter_configuration"],
            },
        )
        assert confirmed.status_code == 200
        assert confirmed.json()["clinical_annotations_attached"] is True
