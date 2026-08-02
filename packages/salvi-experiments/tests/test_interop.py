from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from salvi.domain.enums import PatternKind
from salvi.exceptions import ArtifactError, ConversionError
from salvi.infrastructure.bicluster_set import BiclusterSetReader
from salvi_experiments.cli import main
from salvi_experiments.interop.bicpams import BicPamsConverter
from salvi_experiments.interop.export import CsvBiclusterSetExporter
from salvi_experiments.interop.hbic import HbicConverter


def _write_hbic_document(path: Path, dataset_identifier: str = "test-dataset") -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_identifier": dataset_identifier,
                "biclusters": [
                    {
                        "identifier": "hbic-result-0",
                        "row_indices": [0, 2],
                        "column_indices": [0, 1],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_hbic_document_and_in_memory_masks_convert_to_canonical_results(
    tmp_path: Path,
    dataset_bundle: Path,
) -> None:
    source = _write_hbic_document(tmp_path / "hbic.json")
    destination = HbicConverter(dataset_bundle=dataset_bundle).convert(
        source,
        tmp_path / "result",
    )
    contents = BiclusterSetReader().read_contents(destination)
    evaluation = contents.repertoire.evaluations[0]
    assert evaluation.candidate.identifier == "hbic-result-0"
    assert evaluation.candidate.bicluster.row_indices == (0, 2)
    assert evaluation.objectives == ()
    assert evaluation.pattern_fit is None
    assert evaluation.candidate.provenance is not None
    assert evaluation.candidate.provenance.producer == "hbic"
    assert evaluation.candidate.provenance.pattern_hint is PatternKind.CONSTANT

    masks = [
        (
            np.array([True, False, True, False]),
            np.array([False, True, True]),
        )
    ]
    direct = HbicConverter(dataset_bundle=dataset_bundle).convert_result(
        masks,
        tmp_path / "direct",
    )
    converted = BiclusterSetReader().read(direct).evaluations[0]
    assert converted.candidate.bicluster.row_indices == (0, 2)
    assert converted.candidate.bicluster.column_indices == (1, 2)


def test_hbic_conversion_validates_dataset_identity_and_bounds(
    tmp_path: Path,
    dataset_bundle: Path,
) -> None:
    mismatch = _write_hbic_document(tmp_path / "mismatch.json", "other-dataset")
    with pytest.raises(ConversionError, match="identifier"):
        HbicConverter(dataset_bundle=dataset_bundle).convert(
            mismatch,
            tmp_path / "mismatch-result",
        )

    outside = tmp_path / "outside.json"
    outside.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_identifier": "test-dataset",
                "biclusters": [
                    {
                        "identifier": "outside",
                        "row_indices": [0, 99],
                        "column_indices": [0],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConversionError, match="dimensions"):
        HbicConverter(dataset_bundle=dataset_bundle).convert(
            outside,
            tmp_path / "outside-result",
        )
    with pytest.raises(ConversionError, match="empty or outside"):
        HbicConverter(dataset_bundle=dataset_bundle).convert_result(
            [([], [0])],
            tmp_path / "empty",
        )


def test_bicpams_document_preserves_declared_pattern_hint(
    tmp_path: Path,
    dataset_bundle: Path,
) -> None:
    source = tmp_path / "bicpams.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_identifier": "test-dataset",
                "algorithm": "BicPAMS",
                "pattern": "ADDITIVE",
                "biclusters": [
                    {
                        "identifier": "bicpams-000000",
                        "row_indices": [0, 1],
                        "column_indices": [0, 2],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    destination = BicPamsConverter(dataset_bundle=dataset_bundle).convert(
        source,
        tmp_path / "bicpams-result",
    )
    evaluation = BiclusterSetReader().read(destination).evaluations[0]
    assert evaluation.candidate.provenance is not None
    assert evaluation.candidate.provenance.pattern_hint is PatternKind.ADDITIVE


def test_csv_export_and_cli_adapters_preserve_canonical_information(
    tmp_path: Path,
    dataset_bundle: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _write_hbic_document(tmp_path / "hbic.json")
    result = tmp_path / "result"
    assert (
        main(
            [
                "convert",
                "hbic",
                str(source),
                str(result),
                "--dataset-bundle",
                str(dataset_bundle),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["converter"] == "hbic"

    exported = tmp_path / "csv"
    assert main(["export", "csv", str(result), str(exported)]) == 0
    assert json.loads(capsys.readouterr().out)["exporter"] == "csv"
    manifest = json.loads((exported / "export-manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "salvi-bicluster-set-csv"
    with (exported / "biclusters.csv").open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    assert records[0]["bicluster_id"] == "hbic-result-0"
    assert json.loads(records[0]["row_indices_json"]) == [0, 2]

    with pytest.raises(ArtifactError, match="already exists"):
        CsvBiclusterSetExporter().export(result, exported)
