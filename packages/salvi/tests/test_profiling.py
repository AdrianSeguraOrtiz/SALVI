from __future__ import annotations

import cProfile
import json
from pathlib import Path

import pytest

from salvi.application import profiling
from salvi.exceptions import RunError

from .conftest import configuration_mapping, create_dataset_bundle, write_configuration


def test_profile_helpers_measure_files_and_rank_functions(tmp_path: Path) -> None:
    (tmp_path / "first").write_bytes(b"123")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "second").write_bytes(b"4567")
    assert profiling._directory_size(tmp_path) == 7

    profile = cProfile.Profile()
    profile.runcall(sum, range(10))
    ranked = profiling._top_functions(profile, limit=1)
    assert len(ranked) == 1
    assert ranked[0]["cumulative_seconds"] >= 0


def test_profiler_rejects_unsafe_destination_and_repetition_contract(tmp_path: Path) -> None:
    dataset = create_dataset_bundle(tmp_path / "dataset", rows=4, columns=2)
    mapping = configuration_mapping(dataset, tmp_path / "output")
    configuration = write_configuration(tmp_path / "configuration.yaml", mapping)

    with pytest.raises(RunError, match="outside"):
        profiling.profile_configuration(configuration, tmp_path / "output" / "profile")

    mapping["output"]["overwrite"] = False
    configuration = write_configuration(tmp_path / "configuration.yaml", mapping)
    with pytest.raises(RunError, match="run-output overwrite"):
        profiling.profile_configuration(
            configuration,
            tmp_path / "profile",
            repetitions=2,
        )


def test_profiler_writes_versioned_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = create_dataset_bundle(tmp_path / "dataset", rows=4, columns=2)
    mapping = configuration_mapping(dataset, tmp_path / "output")
    mapping["search"]["engine"] = {
        "name": "serial_mome",
        "parameters": {"initial_population_size": 2, "batch_size": 1},
    }
    mapping["search"]["objectives"] = [
        {"name": "internal_coherence", "parameters": {}},
        {"name": "contrast", "parameters": {"min_background_ratio": 0.1}},
    ]
    mapping["search"]["descriptors"] = [
        {"name": "row_cardinality", "parameters": {}},
        {"name": "column_cardinality", "parameters": {}},
    ]
    mapping["search"]["archive"] = {
        "name": "deep_grid_mome",
        "parameters": {
            "axes": [
                {"descriptor": "row_cardinality", "binning": "EXACT"},
                {"descriptor": "column_cardinality", "binning": "EXACT"},
            ],
            "cell_capacity": 2,
        },
    }
    mapping["search"]["initialization"] = {
        "name": "uniform_random",
        "parameters": {},
    }
    mapping["search"]["termination"] = {
        "name": "evaluation_budget",
        "parameters": {"max_evaluations": 2},
    }
    mapping["execution"] = {
        "executor": {"name": "serial", "parameters": {}},
        "workers": 1,
        "cancellation_grace_seconds": 1.0,
    }
    mapping["monitoring"]["checkpoint_interval_evaluations"] = None
    mapping["monitoring"]["observers"] = []
    mapping["final_selection"] = None
    configuration = write_configuration(tmp_path / "configuration.yaml", mapping)

    monkeypatch.setattr(profiling, "_rusage", lambda: (1.0, 1024, 0.0, 0))
    monkeypatch.setattr(profiling, "_proc_io", lambda: (0, 0))
    destination = profiling.profile_configuration(configuration, tmp_path / "profile")

    report = json.loads((destination / "profile-report.json").read_text())
    assert report["schema_version"] == 1
    assert report["instrumentation"]["cpu_profile"] is True
    assert report["scientific_scope"]["patterns"] == ["CONSTANT"]
    assert report["repetitions"][0]["evaluations"] == 2
    assert (destination / "profile-001.pstats").is_file()


def test_lightweight_profiler_omits_coordinator_instrumentation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = create_dataset_bundle(tmp_path / "dataset", rows=4, columns=2)
    mapping = configuration_mapping(dataset, tmp_path / "output")
    mapping["search"]["engine"] = {
        "name": "serial_mome",
        "parameters": {"initial_population_size": 2, "batch_size": 1},
    }
    mapping["search"]["objectives"] = [
        {"name": "internal_coherence", "parameters": {}},
        {"name": "contrast", "parameters": {"min_background_ratio": 0.1}},
    ]
    mapping["search"]["descriptors"] = [
        {"name": "row_cardinality", "parameters": {}},
        {"name": "column_cardinality", "parameters": {}},
    ]
    mapping["search"]["archive"] = {
        "name": "deep_grid_mome",
        "parameters": {
            "axes": [
                {"descriptor": "row_cardinality", "binning": "EXACT"},
                {"descriptor": "column_cardinality", "binning": "EXACT"},
            ],
            "cell_capacity": 2,
        },
    }
    mapping["search"]["initialization"] = {
        "name": "uniform_random",
        "parameters": {},
    }
    mapping["search"]["termination"] = {
        "name": "evaluation_budget",
        "parameters": {"max_evaluations": 2},
    }
    mapping["execution"] = {
        "executor": {"name": "serial", "parameters": {}},
        "workers": 1,
        "cancellation_grace_seconds": 1.0,
    }
    mapping["monitoring"]["checkpoint_interval_evaluations"] = None
    mapping["monitoring"]["observers"] = []
    mapping["final_selection"] = None
    configuration = write_configuration(tmp_path / "configuration.yaml", mapping)
    monkeypatch.setattr(profiling, "_rusage", lambda: (1.0, 1024, 0.0, 0))
    monkeypatch.setattr(profiling, "_proc_io", lambda: (0, 0))

    destination = profiling.profile_configuration(
        configuration,
        tmp_path / "profile",
        instrument=False,
    )
    report = json.loads((destination / "profile-report.json").read_text())
    repetition = report["repetitions"][0]
    assert report["instrumentation"] == {
        "cpu_profile": False,
        "python_allocation_tracing": False,
    }
    assert repetition["peak_traced_python_bytes"] is None
    assert repetition["profile_file"] is None
    assert repetition["top_cumulative_functions"] == []
    assert not tuple(destination.glob("*.pstats"))
