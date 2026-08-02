from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import salvi_experiments.cli as experiment_cli
from salvi_experiments.artifacts import atomic_experiment_directory, read_report
from salvi_experiments.exceptions import ExperimentArtifactError


def _write_yaml(path: Path, value: object) -> Path:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _algorithm() -> dict[str, object]:
    return {
        "algorithm": "fixture",
        "target_count_known": False,
        "postprocessing_policy": "none",
        "final_selection_policy": "none",
    }


def test_cli_dispatches_all_benchmark_and_alignment_protocols(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    objective_alignment = _write_yaml(
        tmp_path / "objective-alignment.yaml",
        {
            "schema_version": 1,
            "identifier": "alignment",
            "pipeline_configuration": "pipeline.yaml",
            "dataset_bundle": "dataset",
            "output_directory": "alignment-output",
        },
    )
    objective_benchmark = _write_yaml(
        tmp_path / "objective-benchmark.yaml",
        {
            "schema_version": 1,
            "identifier": "alignment-benchmark",
            "cases": [
                {
                    "identifier": "case",
                    "pipeline_configuration": "pipeline.yaml",
                    "dataset_bundle": "dataset",
                }
            ],
            "output_directory": "alignment-benchmark-output",
        },
    )
    accuracy_benchmark = _write_yaml(
        tmp_path / "accuracy-benchmark.yaml",
        {
            "schema_version": 1,
            "identifier": "accuracy-benchmark",
            "cases": [
                {
                    "identifier": "case",
                    "dataset_bundle": "dataset",
                    "bicluster_set": "results",
                    "algorithm": _algorithm(),
                }
            ],
            "output_directory": "accuracy-benchmark-output",
        },
    )
    comparison = _write_yaml(
        tmp_path / "comparison.yaml",
        {
            "schema_version": 1,
            "identifier": "comparison",
            "algorithms": [
                {"identifier": "first", "accuracy_results": ["first"]},
                {"identifier": "second", "accuracy_results": ["second"]},
            ],
            "output_directory": "comparison-output",
        },
    )
    monkeypatch.setattr(
        experiment_cli,
        "run_objective_alignment",
        lambda configuration, *, progress=None: configuration.output_directory,
    )
    monkeypatch.setattr(
        experiment_cli,
        "run_objective_alignment_benchmark",
        lambda configuration, *, progress=None: configuration.output_directory,
    )
    monkeypatch.setattr(
        experiment_cli,
        "run_accuracy_benchmark",
        lambda configuration, *, progress=None: configuration.output_directory,
    )
    monkeypatch.setattr(
        experiment_cli,
        "run_comparison",
        lambda configuration, *, progress=None: configuration.output_directory,
    )

    commands = (
        ("dataset", "objective-alignment", objective_alignment),
        ("benchmark", "objective-alignment", objective_benchmark),
        ("benchmark", "accuracy", accuracy_benchmark),
        ("benchmark", "compare", comparison),
    )
    for level, protocol, configuration in commands:
        assert experiment_cli.main([level, protocol, str(configuration)]) == 0
    captured = capsys.readouterr()
    assert "comparison-output" in captured.out
    assert "salvi-exp: starting benchmark compare" in captured.err


def test_cli_quiet_suppresses_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configuration = _write_yaml(
        tmp_path / "objective-alignment.yaml",
        {
            "schema_version": 1,
            "identifier": "alignment",
            "pipeline_configuration": "pipeline.yaml",
            "dataset_bundle": "dataset",
            "output_directory": "alignment-output",
        },
    )
    monkeypatch.setattr(
        experiment_cli,
        "run_objective_alignment",
        lambda configuration, *, progress=None: configuration.output_directory,
    )

    assert (
        experiment_cli.main(["--quiet", "dataset", "objective-alignment", str(configuration)]) == 0
    )
    captured = capsys.readouterr()
    assert "alignment-output" in captured.out
    assert captured.err == ""


def test_cli_reports_invalid_configuration(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    invalid = _write_yaml(tmp_path / "invalid.yaml", {"schema_version": 1})
    assert experiment_cli.main(["dataset", "accuracy", str(invalid)]) == 2
    assert "invalid experiment configuration" in capsys.readouterr().err


def test_cli_dispatches_uci_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    recipe = _write_yaml(
        tmp_path / "uci.yaml",
        {
            "schema_version": 1,
            "identifier": "uci-fixture",
            "dataset_id": 999,
            "expected_sha256": "0" * 64,
        },
    )
    output = tmp_path / "clinical"
    monkeypatch.setattr(
        experiment_cli.UciConverter,
        "convert",
        lambda _self, _recipe, destination: destination,
    )

    assert experiment_cli.main(["convert", "uci", str(recipe), str(output)]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "converter": "uci",
        "destination": str(output),
    }


def test_cli_reports_owned_schema_versions(capsys: pytest.CaptureFixture[str]) -> None:
    assert experiment_cli.main(["schemas"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["schemas"] == {
        "clinical_dataset_bundle": {"current": 1, "minimum_readable": 1},
        "clinical_validation_configuration": {"current": 1, "minimum_readable": 1},
        "clinical_validation_report": {"current": 1, "minimum_readable": 1},
        "experiment_configuration": {"current": 1, "minimum_readable": 1},
        "experiment_report": {"current": 1, "minimum_readable": 1},
        "uci_import_recipe": {"current": 1, "minimum_readable": 1},
    }


def test_atomic_output_preserves_existing_result_without_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "result"
    destination.mkdir()
    (destination / "original.txt").write_text("original", encoding="utf-8")
    with (
        pytest.raises(ExperimentArtifactError, match="already exists"),
        atomic_experiment_directory(destination, overwrite=False) as temporary,
    ):
        (temporary / "new.txt").write_text("new", encoding="utf-8")
    assert (destination / "original.txt").read_text(encoding="utf-8") == "original"


def test_report_reader_rejects_invalid_documents(tmp_path: Path) -> None:
    directory = tmp_path / "result"
    directory.mkdir()
    (directory / "report.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ExperimentArtifactError, match="must be an object"):
        read_report(directory)
    (directory / "report.json").write_text("{", encoding="utf-8")
    with pytest.raises(ExperimentArtifactError, match="invalid experiment report"):
        read_report(directory)
    (directory / "report.json").write_text(json.dumps({"valid": True}), encoding="utf-8")
    assert read_report(directory) == {"valid": True}
