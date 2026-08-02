from __future__ import annotations

import json
import sqlite3
from io import StringIO
from pathlib import Path

import pytest

from salvi.application.configuration import (
    RunBinding,
    load_bound_configuration,
    load_pipeline_configuration,
)
from salvi.application.factory import build_specification
from salvi.application.inspection import inspect_pipeline
from salvi.application.run_service import CancellationToken, RunService
from salvi.application.selection_service import FinalSelectionService
from salvi.cli.main import main
from salvi.cli.monitor import ConsoleRunMonitor
from salvi.domain import EventType, RunStatus
from salvi.exceptions import (
    ArtifactError,
    ComponentError,
    RunCancelledError,
    RunError,
)
from salvi.infrastructure.bicluster_set import BiclusterSetReader
from salvi.infrastructure.events import SQLiteEventStore

from .conftest import configuration_mapping, create_dataset_bundle, write_configuration


def test_validate_inspect_and_run_complete_the_expected_lifecycle(
    configuration_path: Path,
    dataset_bundle: Path,
    tmp_path: Path,
) -> None:
    service = RunService()
    loaded = service.validate(configuration_path)
    pipeline = _write_pipeline(
        tmp_path / "pipeline.yaml",
        configuration_mapping(dataset_bundle, tmp_path / "unused"),
    )
    inspection = inspect_pipeline(pipeline, dataset_bundle=dataset_bundle)
    result = service.run(configuration_path)

    assert inspection.dataset_identifier == "test-dataset"
    assert inspection.source_rows == 4
    assert inspection.source_columns == 3
    assert inspection.reachable_archive_cells == 6
    assert result.status is RunStatus.COMPLETED
    assert result.output_directory == loaded.configuration.output.directory
    assert result.repertoire.evaluations
    assert (result.output_directory / "effective-configuration.yaml").is_file()
    assert (result.output_directory / "logs").is_dir()
    assert (result.output_directory / "artifacts" / "repertoire" / "manifest.json").is_file()

    metadata = json.loads((result.output_directory / "run-metadata.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["finished_at"] is not None
    events = SQLiteEventStore(result.event_store).read_after()
    event_types = tuple(event.event_type for event in events)
    assert event_types[0] is EventType.RUN_STARTED
    assert EventType.DATASET_PREPARED in event_types
    assert EventType.CANDIDATES_EVALUATED in event_types
    assert EventType.ARCHIVE_UPDATED in event_types
    assert event_types[-1] is EventType.RUN_COMPLETED
    evaluated_event = next(
        event for event in events if event.event_type is EventType.CANDIDATES_EVALUATED
    )
    assert "items" not in evaluated_event.payload
    assert "candidate_duration_seconds" not in evaluated_event.payload["runtime"]


def test_run_honors_output_overwrite(configuration_path: Path) -> None:
    service = RunService()
    service.run(configuration_path)
    with pytest.raises(RunError, match="already exists"):
        service.run(configuration_path)


def test_run_persists_cancellation(configuration_path: Path) -> None:
    token = CancellationToken()
    token.cancel()
    with pytest.raises(RunCancelledError):
        RunService().run(configuration_path, cancellation=token)
    output = RunService().validate(configuration_path).configuration.output.directory
    metadata = json.loads((output / "run-metadata.json").read_text())
    assert metadata["status"] == "cancelled"
    events = SQLiteEventStore(output / "run.sqlite").read_after()
    assert events[-1].event_type is EventType.RUN_CANCELLED


def test_run_persists_component_failure(tmp_path: Path) -> None:
    dataset = create_dataset_bundle(tmp_path / "dataset", rows=1, columns=1)
    mapping = configuration_mapping(dataset, tmp_path / "output")
    path = write_configuration(tmp_path / "configuration.yaml", mapping)
    with pytest.raises(ComponentError, match="minimum row cardinality 2"):
        RunService().run(path)
    metadata = json.loads((tmp_path / "output" / "run-metadata.json").read_text())
    assert metadata["status"] == "failed"
    events = SQLiteEventStore(tmp_path / "output" / "run.sqlite").read_after()
    assert events[-1].event_type is EventType.RUN_FAILED


def test_cli_validate_inspect_and_run(
    tmp_path: Path,
    dataset_bundle: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pipeline = _write_pipeline(
        tmp_path / "pipeline.yaml",
        configuration_mapping(dataset_bundle, tmp_path / "ignored"),
    )
    output = tmp_path / "run-output"
    dataset_binding = ["--dataset", str(dataset_bundle)]
    assert main(["validate", str(pipeline), *dataset_binding]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True

    assert main(["inspect", str(pipeline), *dataset_binding]) == 0
    inspection = json.loads(capsys.readouterr().out)
    assert inspection["dataset_identifier"] == "test-dataset"
    assert inspection["reachable_archive_cells"] == 6

    assert (
        main(
            [
                "run",
                str(pipeline),
                *dataset_binding,
                "--output",
                str(output),
                "--progress",
                "never",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


def test_cli_run_reports_sqlite_progress_on_stderr(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configuration, dataset, output = _write_scientific_cli_configuration(tmp_path, evaluations=4)
    assert (
        main(
            [
                "run",
                str(configuration),
                "--dataset",
                str(dataset),
                "--output",
                str(output),
                "--progress",
                "always",
                "--monitor-interval",
                "0.001",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert json.loads(captured.out)["status"] == "completed"
    assert "4/4 evals" in captured.err
    assert "Completed" in captured.err


def test_console_monitor_retries_while_sqlite_schema_is_initializing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_store = tmp_path / "run.sqlite"
    event_store.touch()

    def locked_source(_: Path) -> object:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr("salvi.cli.monitor.SQLiteRunEventSource", locked_source)
    monitor = ConsoleRunMonitor(
        event_store,
        interval_seconds=0.001,
        stream=StringIO(),
    )

    monitor._poll_once()

    assert monitor._source is None


def test_cli_run_can_suppress_progress(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configuration, dataset, output = _write_scientific_cli_configuration(tmp_path, evaluations=4)
    assert (
        main(
            [
                "run",
                str(configuration),
                "--dataset",
                str(dataset),
                "--output",
                str(output),
                "--progress",
                "always",
                "--quiet",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert json.loads(captured.out)["status"] == "completed"
    assert captured.err == ""


def test_cli_can_reapply_final_selection_without_rerunning_search(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configuration, dataset, output = _write_scientific_cli_configuration(
        tmp_path,
        evaluations=4,
    )
    assert (
        main(
            [
                "run",
                str(configuration),
                "--dataset",
                str(dataset),
                "--output",
                str(output),
                "--progress",
                "never",
            ]
        )
        == 0
    )
    capsys.readouterr()
    selection_mapping = load_pipeline_configuration(configuration).pipeline.model_dump(mode="json")
    selection_mapping["final_selection"] = {
        "name": "containment_marginal_quality",
        "parameters": {},
    }
    selection_configuration = write_configuration(
        tmp_path / "selection.yaml",
        selection_mapping,
    )
    selected = tmp_path / "selected"
    assert (
        main(
            [
                "select",
                str(selection_configuration),
                "--dataset",
                str(dataset),
                "--repertoire",
                str(output / "artifacts" / "repertoire"),
                "--output",
                str(selected),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    contents = BiclusterSetReader().read_contents(selected)
    assert summary["selector"] == "containment_marginal_quality"
    assert summary["input_count"] >= summary["output_count"]
    assert summary["output_count"] == len(contents.repertoire.evaluations)
    assert contents.manifest.source_run == "scientific"


def test_selection_service_validates_source_and_dataset(
    tmp_path: Path,
) -> None:
    configuration, dataset, output = _write_scientific_cli_configuration(
        tmp_path,
        evaluations=4,
    )
    RunService().run_pipeline(
        configuration,
        RunBinding(
            identifier="selection-source",
            dataset_bundle=dataset,
            output_directory=output,
            overwrite=True,
        ),
    )
    source = output / "artifacts" / "repertoire"
    service = FinalSelectionService()
    with pytest.raises(ArtifactError, match="must differ"):
        service.select(
            configuration,
            dataset_bundle=dataset,
            repertoire=source,
            output=source,
        )
    with pytest.raises(ArtifactError, match="does not configure"):
        service.select(
            configuration,
            dataset_bundle=dataset,
            repertoire=source,
            output=tmp_path / "missing-selector",
        )

    mapping = load_pipeline_configuration(configuration).pipeline.model_dump(mode="json")
    mapping["final_selection"] = {
        "name": "containment_marginal_quality",
        "parameters": {},
    }
    selection_configuration = write_configuration(
        tmp_path / "selection-artifacts.yaml",
        mapping,
    )
    incompatible_dataset = create_dataset_bundle(
        tmp_path / "incompatible-dataset",
        rows=7,
        columns=3,
    )
    with pytest.raises(ArtifactError, match="preprocessing pipeline"):
        service.select(
            selection_configuration,
            dataset_bundle=incompatible_dataset,
            repertoire=source,
            output=tmp_path / "selected-incompatible",
        )


def test_search_fingerprint_excludes_final_selection_configuration(
    tmp_path: Path,
) -> None:
    first_path, dataset, _ = _write_scientific_cli_configuration(
        tmp_path,
        evaluations=4,
    )
    second_mapping = load_pipeline_configuration(first_path).pipeline.model_dump(mode="json")
    second_mapping["final_selection"] = {
        "name": "containment_marginal_quality",
        "parameters": {},
    }
    second_path = write_configuration(tmp_path / "second.yaml", second_mapping)
    binding = RunBinding(
        identifier="fingerprint",
        dataset_bundle=dataset,
        output_directory=tmp_path / "output",
    )

    first = build_specification(load_bound_configuration(first_path, binding).configuration)
    second = build_specification(load_bound_configuration(second_path, binding).configuration)

    assert first.final_selector is None
    assert second.final_selector is not None
    assert first.search_fingerprint == second.search_fingerprint


def test_search_fingerprint_uses_termination_registration_policy(
    tmp_path: Path,
) -> None:
    first_path, dataset, _ = _write_scientific_cli_configuration(
        tmp_path,
        evaluations=4,
    )
    second_mapping = load_pipeline_configuration(first_path).pipeline.model_dump(mode="json")
    second_mapping["search"]["termination"]["parameters"]["max_evaluations"] = 40
    second_path = write_configuration(tmp_path / "extended-budget.yaml", second_mapping)
    binding = RunBinding(
        identifier="fingerprint",
        dataset_bundle=dataset,
        output_directory=tmp_path / "output",
    )

    first = build_specification(load_bound_configuration(first_path, binding).configuration)
    second = build_specification(load_bound_configuration(second_path, binding).configuration)

    assert first.search_fingerprint == second.search_fingerprint


def test_search_fingerprint_normalizes_registered_component_defaults(
    tmp_path: Path,
) -> None:
    first_path, dataset, _ = _write_scientific_cli_configuration(
        tmp_path,
        evaluations=4,
    )
    second_mapping = load_pipeline_configuration(first_path).pipeline.model_dump(mode="json")
    second_mapping["search"]["objectives"][1]["parameters"] = {}
    second_path = write_configuration(tmp_path / "implicit-default.yaml", second_mapping)
    binding = RunBinding(
        identifier="fingerprint",
        dataset_bundle=dataset,
        output_directory=tmp_path / "output",
    )

    first = build_specification(load_bound_configuration(first_path, binding).configuration)
    second = build_specification(load_bound_configuration(second_path, binding).configuration)

    assert first.search_fingerprint == second.search_fingerprint


def test_cli_reports_versions_and_delegates_profiling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from salvi.cli import commands as cli_commands

    assert main(["schemas"]) == 0
    versions = json.loads(capsys.readouterr().out)
    assert versions["schemas"]["dataset_bundle"]["current"] == 1
    assert versions["schemas"]["bicluster_set"]["current"] == 7

    captured: dict[str, object] = {}

    def profile(
        configuration: Path,
        destination: Path,
        *,
        binding: object,
        repetitions: int,
        overwrite: bool,
        instrument: bool,
    ) -> Path:
        captured.update(
            configuration=configuration,
            destination=destination,
            repetitions=repetitions,
            overwrite=overwrite,
            instrument=instrument,
            binding=binding,
        )
        return destination

    monkeypatch.setattr(cli_commands, "profile_configuration", profile)
    configuration = tmp_path / "configuration.yaml"
    destination = tmp_path / "profile"
    assert (
        main(
            [
                "profile",
                str(configuration),
                str(destination),
                "--dataset",
                str(tmp_path / "dataset"),
                "--output",
                str(tmp_path / "run-output"),
                "--repetitions",
                "3",
                "--overwrite",
                "--run-overwrite",
                "--lightweight",
            ]
        )
        == 0
    )
    assert captured == {
        "configuration": configuration,
        "destination": destination,
        "repetitions": 3,
        "overwrite": True,
        "instrument": False,
        "binding": RunBinding(
            identifier="configuration",
            dataset_bundle=tmp_path / "dataset",
            output_directory=tmp_path / "run-output",
            seed=42,
            overwrite=True,
        ),
    }


def test_cli_lists_components_and_formats_compact_pipeline_yaml(
    tmp_path: Path,
    dataset_bundle: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["components", "--kind", "search_engine", "--format", "json"]) == 0
    engines = json.loads(capsys.readouterr().out)
    assert {item["name"] for item in engines} == {"pymoo_nsga2", "serial_mome"}

    pipeline = _write_pipeline(
        tmp_path / "pipeline.yaml",
        configuration_mapping(dataset_bundle, tmp_path / "unused"),
    )
    formatted = tmp_path / "formatted.yaml"
    assert main(["config", "format", str(pipeline), "--output", str(formatted)]) == 0
    assert (
        load_pipeline_configuration(formatted).pipeline
        == load_pipeline_configuration(pipeline).pipeline
    )
    assert "source_column_filters:" not in formatted.read_text(encoding="utf-8")


def test_cli_renders_human_and_markdown_component_catalogs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["components", "--kind", "search_engine"]) == 0
    human = capsys.readouterr().out
    assert "search_engine:" in human
    assert "serial_mome" in human

    assert main(["components", "--kind", "search_engine", "--format", "markdown"]) == 0
    markdown = capsys.readouterr().out
    assert "### `search_engine:serial_mome`" in markdown
    assert "| Parameter | Default | Description |" in markdown


def test_cli_writes_expanded_pipeline_yaml_to_stdout(
    tmp_path: Path,
    dataset_bundle: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pipeline = _write_pipeline(
        tmp_path / "pipeline.yaml",
        configuration_mapping(dataset_bundle, tmp_path / "unused"),
    )

    assert main(["config", "format", str(pipeline), "--expanded"]) == 0

    expanded = capsys.readouterr().out
    assert "source_column_filters: []" in expanded
    assert "column_augmentations: []" in expanded


def test_cli_reports_domain_errors_consistently(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import salvi.cli.main as cli_main

    def fail(_namespace: object) -> int:
        raise RunError("deliberate failure")

    monkeypatch.setattr(cli_main, "dispatch", fail)

    assert cli_main.main(["schemas"]) == 2
    assert capsys.readouterr().err == "salvi: deliberate failure\n"


def test_cli_launches_local_web_gui(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import salvi.web.main as web_main

    captured: dict[str, object] = {}

    def launch(**options: object) -> int:
        captured.update(options)
        return 0

    monkeypatch.setattr(web_main, "launch_web_gui", launch)
    assert (
        main(
            [
                "gui",
                "--host",
                "127.0.0.1",
                "--port",
                "8877",
                "--no-open",
                "--data-directory",
                str(tmp_path),
                "--max-upload-mib",
                "128",
            ]
        )
        == 0
    )
    assert captured == {
        "host": "127.0.0.1",
        "port": 8877,
        "open_browser": False,
        "data_directory": tmp_path,
        "max_upload_mib": 128,
    }


def test_cli_rejects_invalid_gui_port(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["gui", "--port", "70000"])
    assert error.value.code == 2
    assert "between 1 and 65535" in capsys.readouterr().err


def test_cli_rejects_non_positive_profile_repetitions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        main(
            [
                "profile",
                "configuration.yaml",
                "profile",
                "--dataset",
                "dataset",
                "--output",
                "output",
                "--repetitions",
                "0",
            ]
        )
    assert error.value.code == 2
    assert "must be >= 1" in capsys.readouterr().err


def test_cli_rejects_non_positive_monitor_interval(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        main(
            [
                "run",
                "configuration.yaml",
                "--dataset",
                "dataset",
                "--output",
                "output",
                "--monitor-interval",
                "0",
            ]
        )
    assert error.value.code == 2
    assert "must be positive" in capsys.readouterr().err


def _write_scientific_cli_configuration(
    tmp_path: Path,
    *,
    evaluations: int,
) -> tuple[Path, Path, Path]:
    dataset = create_dataset_bundle(tmp_path / "dataset", rows=8, columns=3)
    mapping = configuration_mapping(dataset, tmp_path / "output", overwrite=True)
    mapping["search"] = {
        "engine": {
            "name": "serial_mome",
            "parameters": {"initial_population_size": 2, "batch_size": 1},
        },
        "objectives": [
            {"name": "internal_coherence", "parameters": {}},
            {"name": "contrast", "parameters": {"min_background_ratio": 0.1}},
        ],
        "descriptors": [
            {"name": "row_cardinality", "parameters": {}},
            {"name": "column_cardinality", "parameters": {}},
        ],
        "archive": {
            "name": "deep_grid_mome",
            "parameters": {
                "axes": [
                    {"descriptor": "row_cardinality", "binning": "EXACT"},
                    {"descriptor": "column_cardinality", "binning": "EXACT"},
                ],
                "cell_capacity": 2,
            },
        },
        "parent_selection": {"name": "repertoire_uniform", "parameters": {}},
        "initialization": {"name": "uniform_random", "parameters": {}},
        "emitters": [{"name": "random_move", "parameters": {}}],
        "scheduler": {"name": "first", "parameters": {}},
        "termination": {
            "name": "evaluation_budget",
            "parameters": {"max_evaluations": evaluations},
        },
    }
    mapping["monitoring"]["observers"] = [
        {"name": "search_progress", "parameters": {}},
        {"name": "runtime_throughput", "parameters": {}},
    ]
    mapping["final_selection"] = None
    output = tmp_path / "output"
    pipeline = _write_pipeline(tmp_path / "scientific.yaml", mapping)
    return pipeline, dataset, output


def _write_pipeline(path: Path, mapping: dict[str, object]) -> Path:
    pipeline = dict(mapping)
    for key in ("run", "dataset", "output"):
        pipeline.pop(key)
    return write_configuration(path, pipeline)
