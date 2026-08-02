from __future__ import annotations

import copy
from pathlib import Path

import pytest

from salvi.application.configuration import (
    RunBinding,
    bind_pipeline,
    load_configuration,
    load_pipeline_configuration,
    parse_pipeline_configuration,
    serialize_pipeline_configuration,
    write_effective_configuration,
)
from salvi.application.defaults import default_scientific_configuration
from salvi.application.factory import build_specification
from salvi.application.run_service import RunService
from salvi.domain import PatternKind
from salvi.exceptions import ComponentError, ConfigurationError

from .conftest import configuration_mapping, write_configuration


@pytest.mark.parametrize(
    "patterns",
    [
        ["CONSTANT"],
        ["ADDITIVE"],
        ["MULTIPLICATIVE"],
        ["CONSTANT", "ADDITIVE"],
        ["CONSTANT", "MULTIPLICATIVE"],
        ["ADDITIVE", "MULTIPLICATIVE"],
        ["CONSTANT", "ADDITIVE", "MULTIPLICATIVE"],
    ],
)
def test_all_pattern_modes_and_relative_paths_are_supported(
    tmp_path: Path,
    dataset_bundle: Path,
    patterns: list[str],
) -> None:
    relative_dataset = dataset_bundle.relative_to(tmp_path)
    mapping = configuration_mapping(
        relative_dataset,
        Path("relative-output"),
        patterns=patterns,
    )
    path = write_configuration(tmp_path / "configuration.yaml", mapping)
    loaded = load_configuration(path)
    assert loaded.configuration.dataset.bundle == dataset_bundle.resolve()
    assert loaded.configuration.output.directory == (tmp_path / "relative-output").resolve()
    assert loaded.configuration.patterns.allowed == tuple(PatternKind(item) for item in patterns)
    assert build_specification(loaded.configuration).patterns == loaded.configuration.patterns


def test_effective_configuration_round_trips(configuration_path: Path, tmp_path: Path) -> None:
    loaded = load_configuration(configuration_path)
    destination = tmp_path / "effective.yaml"
    write_effective_configuration(loaded.configuration, destination)
    reloaded = load_configuration(destination)
    assert reloaded.configuration == loaded.configuration


def test_run_materializes_registered_component_defaults(
    tmp_path: Path,
    dataset_bundle: Path,
) -> None:
    mapping = configuration_mapping(dataset_bundle, tmp_path / "output")
    mapping["search"]["engine"]["parameters"] = {}
    mapping["search"]["objectives"][1]["parameters"] = {}
    path = write_configuration(tmp_path / "configuration.yaml", mapping)

    result = RunService().run(path)
    effective = load_configuration(result.output_directory / "effective-configuration.yaml")

    assert effective.configuration.search.engine.parameters == {
        "initial_population_size": 64,
        "batch_size": 16,
    }
    assert effective.configuration.search.objectives[1].parameters == {"min_background_ratio": 0.1}


def test_pipeline_is_reusable_and_run_binding_is_external(
    tmp_path: Path,
    dataset_bundle: Path,
) -> None:
    full = configuration_mapping(dataset_bundle, tmp_path / "ignored-output")
    for key in ("run", "dataset", "output"):
        full.pop(key)
    pipeline_path = write_configuration(tmp_path / "pipeline.yaml", full)
    loaded = load_pipeline_configuration(pipeline_path)
    binding = RunBinding(
        identifier="bound-run",
        dataset_bundle=dataset_bundle.relative_to(tmp_path),
        output_directory=Path("run-output"),
        seed=11,
        overwrite=True,
    )
    bound = bind_pipeline(loaded.pipeline, binding, base_directory=tmp_path)
    assert bound.dataset.bundle == dataset_bundle.resolve()
    assert bound.output.directory == (tmp_path / "run-output").resolve()
    assert bound.run.identifier == "bound-run"
    assert bound.run.seed == 11
    assert bound.output.overwrite is True

    full_pipeline = dict(full)
    full_pipeline["dataset"] = {"bundle": "dataset"}
    with pytest.raises(ConfigurationError, match="extra_forbidden"):
        load_pipeline_configuration(write_configuration(tmp_path / "invalid.yaml", full_pipeline))


def test_pipeline_serialization_is_compact_but_semantically_complete(
    tmp_path: Path,
    dataset_bundle: Path,
) -> None:
    mapping = configuration_mapping(dataset_bundle, tmp_path / "unused")
    for key in ("run", "dataset", "output"):
        mapping.pop(key)
    pipeline = load_pipeline_configuration(
        write_configuration(tmp_path / "pipeline.yaml", mapping)
    ).pipeline

    compact = serialize_pipeline_configuration(pipeline)
    expanded = serialize_pipeline_configuration(pipeline, compact=False)

    assert parse_pipeline_configuration(compact) == pipeline
    assert parse_pipeline_configuration(expanded) == pipeline
    assert "allowed:" in compact
    assert "source_column_filters:" not in compact
    assert "column_augmentations:" not in compact
    assert len(compact) < len(expanded)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("schema_version: 1\nschema_version: 1\n", "duplicate YAML key"),
        ("base: &base {value: 1}\ncopy: *base\n", "aliases are not supported"),
        ("value: ${SALVI_PATH}\n", "environment interpolation"),
        ("!include other.yaml\n", "invalid YAML"),
    ],
)
def test_forbidden_yaml_constructs_are_rejected(
    tmp_path: Path,
    content: str,
    message: str,
) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigurationError, match=message):
        load_configuration(path)


def test_merge_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("base: &base {value: 1}\ncopy: {<<: *base}\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match=r"aliases are not supported|merge keys"):
        load_configuration(path)


def test_unknown_fields_duplicates_and_empty_patterns_are_rejected(
    tmp_path: Path,
    dataset_bundle: Path,
) -> None:
    mapping = configuration_mapping(dataset_bundle, tmp_path / "output")
    mapping["unknown"] = True
    with pytest.raises(ConfigurationError, match="extra_forbidden"):
        load_configuration(write_configuration(tmp_path / "unknown.yaml", mapping))

    mapping = configuration_mapping(dataset_bundle, tmp_path / "output")
    mapping["patterns"]["allowed"] = []
    with pytest.raises(ConfigurationError, match="at least one pattern"):
        load_configuration(write_configuration(tmp_path / "empty.yaml", mapping))

    mapping = configuration_mapping(dataset_bundle, tmp_path / "output")
    mapping["patterns"]["allowed"] = ["CONSTANT", "CONSTANT"]
    with pytest.raises(ConfigurationError, match="duplicates"):
        load_configuration(write_configuration(tmp_path / "patterns.yaml", mapping))

    mapping = configuration_mapping(dataset_bundle, tmp_path / "output")
    mapping["search"]["descriptors"].append(copy.deepcopy(mapping["search"]["descriptors"][0]))
    with pytest.raises(ConfigurationError, match="duplicate component names"):
        load_configuration(write_configuration(tmp_path / "components.yaml", mapping))


def test_duplicate_emitter_instances_are_rejected_without_instance_identifiers(
    tmp_path: Path,
    dataset_bundle: Path,
) -> None:
    mapping = configuration_mapping(dataset_bundle, tmp_path / "output")
    mapping["search"]["emitters"].append(copy.deepcopy(mapping["search"]["emitters"][0]))
    with pytest.raises(ConfigurationError, match="emitters must not contain duplicate"):
        load_configuration(write_configuration(tmp_path / "emitters.yaml", mapping))


def test_serial_executor_rejects_multiple_execution_workers(
    tmp_path: Path,
    dataset_bundle: Path,
) -> None:
    mapping = configuration_mapping(dataset_bundle, tmp_path / "output")
    mapping["search"]["engine"] = {"name": "serial_mome", "parameters": {}}
    mapping["execution"]["workers"] = 2
    loaded = load_configuration(write_configuration(tmp_path / "parallel.yaml", mapping))
    with pytest.raises(ComponentError, match="serial evaluation requires exactly one worker"):
        build_specification(loaded.configuration)


def test_cell_coverage_requires_cardinality_descriptors_and_matching_archive_axes(
    tmp_path: Path,
    dataset_bundle: Path,
) -> None:
    mapping = default_scientific_configuration()
    mapping["run"] = {"identifier": "cell-coverage", "seed": 0}
    mapping["dataset"] = {"bundle": str(dataset_bundle)}
    mapping["output"] = {"directory": str(tmp_path / "output")}
    mapping["evaluation"]["candidate_validity"]["parameters"] = {
        "min_rows": 2,
        "min_columns": 2,
    }
    mapping["search"]["descriptors"] = [
        {"name": "row_cardinality", "parameters": {}},
    ]
    mapping["search"]["archive"]["parameters"]["axes"] = [
        {"descriptor": "row_cardinality", "binning": "EXACT"},
    ]
    loaded = load_configuration(write_configuration(tmp_path / "cell-coverage.yaml", mapping))

    with pytest.raises(ComponentError, match="descriptor:column-cardinality"):
        build_specification(loaded.configuration)


def test_preprocessing_families_and_evaluation_policies_are_explicit(
    tmp_path: Path,
    dataset_bundle: Path,
) -> None:
    mapping = configuration_mapping(dataset_bundle, tmp_path / "output")
    mapping["preprocessing"]["column_augmentations"] = [
        {
            "name": "missingness_indicators",
            "parameters": {"min_missing_ratio": 0.2},
        }
    ]
    mapping["preprocessing"]["source_column_filters"] = [
        {"name": "drop_all_missing_columns", "parameters": {}}
    ]
    loaded = load_configuration(write_configuration(tmp_path / "families.yaml", mapping))
    specification = build_specification(loaded.configuration)
    assert specification.column_augmentations[0].component_name == "missingness_indicators"
    assert specification.source_column_filters[0].component_name == "drop_all_missing_columns"


def test_non_mapping_and_missing_files_are_clear_configuration_errors(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- one\n- two\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="must contain a mapping"):
        load_configuration(path)
    with pytest.raises(ConfigurationError, match="cannot read"):
        load_configuration(tmp_path / "missing.yaml")
