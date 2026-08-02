from __future__ import annotations

import io
import json
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from salvi.application.composition import CompositionResolutionService, RoleState
from salvi.application.configuration import (
    PipelineConfiguration,
    load_pipeline_configuration,
    parse_pipeline_configuration,
    serialize_pipeline_configuration,
)
from salvi.application.defaults import default_scientific_configuration
from salvi.components.catalog import WorkflowConnectionKind, WorkflowStage, role_catalog
from salvi.components.defaults import default_component_registry
from salvi.domain.enums import ColumnKind, RunStatus, SearchFamily
from salvi.exceptions import ConfigurationError, ConversionError, RunCancelledError
from salvi.web.adapters import (
    DatasetBundleZipAdapter,
    TabularInputAdapter,
    built_in_adapters,
    normalized_identifier,
)
from salvi.web.app import create_app
from salvi.web.imports import DatasetImportService
from salvi.web.main import _is_loopback, _open_when_ready, launch_web_gui
from salvi.web.models import WebRunRecord
from salvi.web.providers import WebExtensionProvider, WebProviderRegistry
from salvi.web.run_manager import ProcessCancellationSignal, WebRunManager, _run_child
from salvi.web.storage import WebApplicationPaths, WebStateStore

from .conftest import create_dataset_bundle


def test_pipeline_text_api_round_trips_the_cli_model() -> None:
    configuration = PipelineConfiguration.model_validate(default_scientific_configuration())
    serialized = serialize_pipeline_configuration(configuration)
    assert parse_pipeline_configuration(serialized) == configuration


def test_public_default_matches_the_scientific_example() -> None:
    example = Path(__file__).resolve().parents[3] / "examples" / "scientific-configuration.yaml"
    expected = load_pipeline_configuration(example).pipeline
    actual = PipelineConfiguration.model_validate(default_scientific_configuration())
    assert actual == expected


def test_default_pipeline_uses_cell_compartmentalized_qd_components() -> None:
    configuration = PipelineConfiguration.model_validate(default_scientific_configuration())
    assert configuration.evaluation.candidate_validity.parameters == {
        "min_rows": 10,
        "min_columns": 10,
    }
    assert configuration.search.engine.name == "serial_mome"
    assert configuration.search.archive is not None
    assert configuration.search.archive.name == "deep_grid_mome"
    assert configuration.search.mate_selection is not None
    assert configuration.search.parent_selection is not None
    assert configuration.search.parent_selection.name == "cell_uniform_quality"
    assert configuration.search.mate_selection.name == "cell_first_evidence_compatible"
    assert configuration.search.crossover is not None
    assert configuration.search.crossover.name == "evidence_weighted_recombination"
    assert configuration.search.initialization.name == "cell_coverage_pattern_aware"
    assert configuration.search.scheduler is not None
    assert configuration.search.scheduler.name == "cell_balanced_adaptive_credit"
    assert configuration.search.termination.parameters["max_evaluations"] == 50_000
    assert configuration.final_selection is not None
    assert configuration.final_selection.name == "adaptive_residual_evidence_cover"
    assert configuration.final_selection.parameters["objective_names"] == [
        "internal_coherence",
        "contrast",
    ]
    assert configuration.final_selection.parameters["minimum_quality_floor"] == 0.5
    assert configuration.final_selection.parameters["maximum_quality_floor"] == 0.85
    assert tuple(observer.name for observer in configuration.monitoring.observers) == (
        "search_progress",
        "archive_coverage",
        "candidate_outcomes",
        "emitter_credit",
        "component_timing",
        "evaluation_issues",
        "qd_archive_diagnostics",
        "objective_distribution",
        "descriptor_distribution",
        "archive_descriptor_distribution",
        "candidate_diversity",
        "runtime_throughput",
        "resource_usage",
    )


def test_workflow_catalog_has_semantic_order_layout_and_connections() -> None:
    roles = role_catalog()
    stages = tuple(dict.fromkeys(role.stage for role in roles))
    assert stages == (
        WorkflowStage.PREPARATION,
        WorkflowStage.EVALUATION,
        WorkflowStage.SEARCH,
        WorkflowStage.OUTPUT,
    )
    source_filter = next(role for role in roles if role.kind.value == "source_column_filter")
    search_engine = next(role for role in roles if role.kind.value == "search_engine")
    final_selector = next(role for role in roles if role.kind.value == "final_selector")
    assert source_filter.accepts_pipeline_input
    assert any(
        connection.source.value == "initializer"
        and connection.kind is WorkflowConnectionKind.PRIMARY
        for connection in search_engine.incoming
    )
    assert final_selector.emits_pipeline_output


def test_component_parameter_schemas_are_self_contained() -> None:
    for component in default_component_registry().catalog():
        for parameter in component.parameters:
            assert '"$ref"' not in json.dumps(parameter.value_schema)
        if component.observer_view is not None:
            view = component.observer_view
            assert view.x_axis_label or view.view_kind.value == "TABLE"
            assert view.metrics
            assert view.groups
            assert {metric.display_group for metric in view.metrics} <= {
                group.name for group in view.groups
            }
            for metric in view.metrics:
                assert metric.description
                assert metric.unit
                assert metric.display_group


def test_partial_resolution_changes_roles_with_the_search_engine() -> None:
    service = CompositionResolutionService()
    draft = default_scientific_configuration()
    mome = service.resolve(draft)
    assert mome.complete
    assert (
        next(role for role in mome.roles if role.role.kind.value == "emitter").state
        is RoleState.CONFIGURED
    )

    search = draft["search"]
    assert isinstance(search, dict)
    search["engine"] = {
        "name": "pymoo_nsga2",
        "parameters": {"population_size": 16, "eliminate_duplicates": True},
    }
    search["initialization"] = {"name": "pattern_aware", "parameters": {}}
    search["crossover"] = None
    nsga = service.resolve(draft)
    emitter = next(role for role in nsga.roles if role.role.kind.value == "emitter")
    descriptors = next(role for role in nsga.roles if role.role.kind.value == "descriptor")
    crossover = next(role for role in nsga.roles if role.role.kind.value == "crossover_operator")
    observers = next(role for role in nsga.roles if role.role.kind.value == "observer")
    assert emitter.state is RoleState.INVALID
    assert descriptors.state is RoleState.INVALID
    assert descriptors.maximum == 0
    assert crossover.state is RoleState.REQUIRED
    assert observers.state is RoleState.INVALID
    assert "archive_coverage still requires: archive" in observers.reasons
    assert "emitter_credit still requires: scheduler" in observers.reasons


def test_search_family_transition_builds_complete_family_specific_topologies() -> None:
    service = CompositionResolutionService()
    original = default_scientific_configuration()

    conventional = service.switch_search_family(
        original,
        SearchFamily.CONVENTIONAL_MULTI_OBJECTIVE,
    )
    assert conventional.resolution.complete
    assert conventional.resolution.search_family is SearchFamily.CONVENTIONAL_MULTI_OBJECTIVE
    search = conventional.configuration["search"]
    assert search["engine"]["name"] == "pymoo_nsga2"
    assert search["descriptors"] == []
    assert search["archive"] is None
    assert search["emitters"] == []
    assert search["crossover"]["name"] == "half_uniform_membership"
    assert search["mutation"]["name"] == "bit_flip_membership"
    engine = next(
        role for role in conventional.resolution.roles if role.role.kind.value == "search_engine"
    )
    engines = {instance.component.name: instance for instance in engine.instances}
    assert engines["pymoo_nsga2"].available
    assert not engines["serial_mome"].available
    assert "QUALITY_DIVERSITY" in engines["serial_mome"].reasons[0]

    quality_diversity = service.switch_search_family(
        conventional.configuration,
        SearchFamily.QUALITY_DIVERSITY,
    )
    assert quality_diversity.resolution.complete
    assert quality_diversity.resolution.search_family is SearchFamily.QUALITY_DIVERSITY
    qd_search = quality_diversity.configuration["search"]
    assert qd_search["engine"]["name"] == "serial_mome"
    assert len(qd_search["descriptors"]) == 2
    assert qd_search["archive"]["name"] == "deep_grid_mome"


def test_partial_resolution_exposes_runtime_and_observer_incompatibilities() -> None:
    service = CompositionResolutionService()
    draft = default_scientific_configuration()
    search = draft["search"]
    monitoring = draft["monitoring"]
    assert isinstance(search, dict)
    assert isinstance(monitoring, dict)
    search.update(
        {
            "engine": {
                "name": "pymoo_nsga2",
                "parameters": {
                    "population_size": 16,
                    "eliminate_duplicates": True,
                },
            },
            "archive": None,
            "parent_selection": None,
            "mate_selection": None,
            "crossover": {
                "name": "membership_recombination",
                "parameters": {},
            },
            "mutation": {
                "name": "bit_flip_membership",
                "parameters": {},
            },
            "emitters": [],
            "scheduler": None,
            "descriptors": [],
            "initialization": {"name": "pattern_aware", "parameters": {}},
        }
    )
    monitoring["checkpoint_interval_evaluations"] = 100
    monitoring["observers"] = [
        {"name": "archive_coverage", "parameters": {}},
        {"name": "emitter_credit", "parameters": {}},
    ]

    resolution = service.resolve(draft)

    engine = next(role for role in resolution.roles if role.role.kind.value == "search_engine")
    observers = next(role for role in resolution.roles if role.role.kind.value == "observer")
    assert engine.state is RoleState.INVALID
    assert engine.reasons == (
        "search engine 'pymoo_nsga2' does not support resumable periodic checkpoints",
    )
    assert observers.state is RoleState.INVALID
    assert "archive_coverage still requires: archive" in observers.reasons
    assert "emitter_credit still requires: scheduler" in observers.reasons
    by_name = {instance.component.name: instance for instance in observers.instances}
    assert not by_name["archive_coverage"].available
    assert not by_name["emitter_credit"].available
    assert by_name["search_progress"].available


def test_partial_resolution_exposes_an_unconsumed_optional_operator() -> None:
    service = CompositionResolutionService()
    draft = default_scientific_configuration()
    search = draft["search"]
    assert isinstance(search, dict)
    search["crossover"] = {
        "name": "membership_recombination",
        "parameters": {},
    }
    search["emitters"] = []

    resolution = service.resolve(draft)

    crossover = next(
        role for role in resolution.roles if role.role.kind.value == "crossover_operator"
    )
    assert crossover.state is RoleState.INVALID
    assert crossover.reasons == (
        "crossover_operator:membership_recombination is configured but no active component "
        "consumes capability 'crossover-operator'",
    )


def test_partial_resolution_explains_invalid_cell_coverage_axes() -> None:
    service = CompositionResolutionService()
    draft = default_scientific_configuration()
    search = draft["search"]
    assert isinstance(search, dict)
    search["descriptors"] = [{"name": "row_cardinality", "parameters": {}}]
    archive = search["archive"]
    assert isinstance(archive, dict)
    parameters = archive["parameters"]
    assert isinstance(parameters, dict)
    parameters["axes"] = [{"descriptor": "row_cardinality", "binning": "EXACT"}]

    resolution = service.resolve(draft)

    initializer = next(role for role in resolution.roles if role.role.kind.value == "initializer")
    archive_role = next(role for role in resolution.roles if role.role.kind.value == "archive")
    assert initializer.state is RoleState.INVALID
    assert any("descriptor:column-cardinality" in reason for reason in initializer.reasons)
    assert archive_role.state is RoleState.INVALID
    assert any("cell-target consumers require exactly" in reason for reason in archive_role.reasons)


def test_partial_resolution_remains_actionable_for_malformed_drafts() -> None:
    service = CompositionResolutionService()

    empty = service.resolve({})
    assert empty.allowed_patterns == ("CONSTANT",)
    assert not empty.complete
    assert (
        next(role for role in empty.roles if role.role.kind.value == "search_engine").state
        is RoleState.REQUIRED
    )

    unknown = service.resolve(
        {
            "patterns": {"allowed": ["unknown", "ADDITIVE", "ADDITIVE"]},
            "search": {"engine": {"name": "missing", "parameters": {}}},
        }
    )
    assert unknown.allowed_patterns == ("ADDITIVE",)
    assert not unknown.valid
    assert any("unknown search_engine" in error for error in unknown.errors)

    malformed = default_scientific_configuration()
    execution = malformed["execution"]
    assert isinstance(execution, dict)
    execution["executor"] = {"name": "serial", "parameters": []}
    execution["workers"] = 2
    resolved = service.resolve(malformed)
    executor = next(
        role for role in resolved.roles if role.role.kind.value == "evaluation_executor"
    )
    assert executor.state is RoleState.INVALID
    assert "parameters must be a mapping" in executor.reasons


@pytest.mark.parametrize(
    "crossover_name",
    (
        "membership_recombination",
        "evidence_weighted_recombination",
        "half_uniform_membership",
    ),
)
def test_partial_resolution_accepts_every_salvi_crossover_with_nsga2(
    crossover_name: str,
) -> None:
    service = CompositionResolutionService()
    draft = default_scientific_configuration()
    search = draft["search"]
    assert isinstance(search, dict)
    search.update(
        {
            "engine": {
                "name": "pymoo_nsga2",
                "parameters": {
                    "population_size": 16,
                    "eliminate_duplicates": True,
                },
            },
            "archive": None,
            "parent_selection": None,
            "mate_selection": None,
            "crossover": {
                "name": crossover_name,
                "parameters": {
                    "application_probability": 0.9,
                    "row_exchange_probability": 0.5,
                    "column_exchange_probability": 0.5,
                },
            },
            "mutation": {
                "name": "bit_flip_membership",
                "parameters": {
                    "application_probability": 1.0,
                    "bit_probability": None,
                },
            },
            "emitters": [],
            "scheduler": None,
            "descriptors": [],
            "initialization": {"name": "pattern_aware", "parameters": {}},
        }
    )
    monitoring = draft["monitoring"]
    assert isinstance(monitoring, dict)
    monitoring["observers"] = [
        {"name": "search_progress", "parameters": {}},
        {"name": "objective_distribution", "parameters": {"every_evaluations": 100}},
        {"name": "runtime_throughput", "parameters": {}},
    ]
    monitoring["checkpoint_interval_evaluations"] = None
    draft["final_selection"] = None

    resolution = service.resolve(draft)

    engine = next(role for role in resolution.roles if role.role.kind.value == "search_engine")
    crossover = next(
        role for role in resolution.roles if role.role.kind.value == "crossover_operator"
    )
    assert engine.state is RoleState.CONFIGURED
    assert not engine.reasons
    assert crossover.state is RoleState.CONFIGURED
    assert resolution.complete


def test_partial_resolution_exposes_only_effective_workflow_connections() -> None:
    service = CompositionResolutionService()
    draft = default_scientific_configuration()
    resolved = service.resolve(draft)
    connections = {(item.source, item.target, item.kind) for item in resolved.workflow_connections}
    assert (
        "__input__",
        "missing_values_policy",
        WorkflowConnectionKind.PRIMARY,
    ) in connections
    assert (
        "missing_values_policy",
        "numeric_transformation",
        WorkflowConnectionKind.PRIMARY,
    ) in connections
    assert not any(
        "source_column_filter" in {item.source, item.target}
        for item in resolved.workflow_connections
    )
    assert any(
        "crossover_operator" in {item.source, item.target} for item in resolved.workflow_connections
    )

    preprocessing = draft["preprocessing"]
    assert isinstance(preprocessing, dict)
    preprocessing["missing_values"] = None
    preprocessing["numeric_transformations"] = []
    without_preparation = service.resolve(draft)
    bypassed = {
        (item.source, item.target, item.kind) for item in without_preparation.workflow_connections
    }
    for target in (
        "candidate_validity_policy",
        "evaluation_support_policy",
        "initializer",
    ):
        assert ("__input__", target, WorkflowConnectionKind.PRIMARY) in bypassed


def test_tabular_adapter_requires_confirmed_types(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("id,value,group\nr1,1.5,A\nr2,,B\n", encoding="utf-8")
    adapter = TabularInputAdapter(",", "csv", "CSV", ".csv")
    preview = adapter.inspect(
        {"data": source},
        identifier="dataset",
        workspace=tmp_path,
    )
    assert preview.confirmation_required
    assert preview.columns[0].is_row_identifier
    assert preview.columns[1].inferred_kind.value == "NUMERIC"
    assert preview.columns[1].missing_ratio == 0.5
    destination = tmp_path / "dataset"
    adapter.convert(
        {"data": source},
        identifier="dataset",
        columns=preview.columns,
        destination=destination,
        workspace=tmp_path,
    )
    assert (destination / "dataset.yaml").is_file()


def test_tabular_adapter_rejects_invalid_tables_and_confirmations(tmp_path: Path) -> None:
    assert normalized_identifier("  Clinical cohort / 1 ") == "Clinical-cohort-1"
    with pytest.raises(ConversionError, match="letters or digits"):
        normalized_identifier("---")

    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text("a,a\n1,2\n", encoding="utf-8")
    adapter = TabularInputAdapter(",", "csv", "CSV", ".csv")
    with pytest.raises(ConversionError, match="unique"):
        adapter.inspect({"data": duplicate}, identifier="duplicate", workspace=tmp_path)

    identifier_only = tmp_path / "identifier.csv"
    identifier_only.write_text("id\nrow-1\nrow-2\n", encoding="utf-8")
    with pytest.raises(ConversionError, match="non-identifier"):
        adapter.inspect({"data": identifier_only}, identifier="identifier", workspace=tmp_path)

    source = tmp_path / "source.csv"
    source.write_text("id,value\nr1,A\nr2,B\n", encoding="utf-8")
    preview = adapter.inspect({"data": source}, identifier="data", workspace=tmp_path)
    invalid = tuple(
        item.model_copy(update={"selected_kind": ColumnKind.BOOLEAN})
        if item.name == "value"
        else item
        for item in preview.columns
    )
    with pytest.raises(ConversionError, match="not boolean"):
        adapter.convert(
            {"data": source},
            identifier="data",
            columns=invalid,
            destination=tmp_path / "invalid",
            workspace=tmp_path,
        )


def test_dataset_zip_adapter_rejects_path_traversal(tmp_path: Path) -> None:
    source = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("../outside.txt", "unsafe")
    adapter = DatasetBundleZipAdapter(maximum_expanded_bytes=1024)
    try:
        adapter.inspect(
            {"bundle": source},
            identifier="dataset",
            workspace=tmp_path / "workspace",
        )
    except ConversionError as error:
        assert "unsafe path" in str(error)
    else:  # pragma: no cover - explicit security assertion
        raise AssertionError("unsafe ZIP was accepted")


def test_dataset_zip_adapter_round_trips_a_canonical_bundle(tmp_path: Path) -> None:
    source_bundle = create_dataset_bundle(tmp_path / "source-bundle")
    archive_path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for item in source_bundle.iterdir():
            archive.write(item, f"bundle/{item.name}")
    adapter = DatasetBundleZipAdapter(maximum_expanded_bytes=10 * 1024 * 1024)
    preview = adapter.inspect(
        {"bundle": archive_path},
        identifier="ignored",
        workspace=tmp_path / "inspect",
    )
    assert preview.identifier == "test-dataset"
    assert preview.ground_truth_attached
    destination = tmp_path / "canonical"
    assert (
        adapter.convert(
            {"bundle": archive_path},
            identifier="ignored",
            columns=preview.columns,
            destination=destination,
            workspace=tmp_path / "convert",
        )
        == destination
    )
    assert (destination / "ground-truth.json").is_file()

    with pytest.raises(ConversionError, match="expanded size"):
        DatasetBundleZipAdapter(maximum_expanded_bytes=1).inspect(
            {"bundle": archive_path},
            identifier="ignored",
            workspace=tmp_path / "too-small",
        )


def test_provider_registry_rejects_duplicates_and_unknown_names() -> None:
    csv_adapter = TabularInputAdapter(",", "csv", "CSV", ".csv")
    registry = WebProviderRegistry(adapters=(csv_adapter,))
    assert registry.adapter("csv") is csv_adapter
    assert registry.adapter_descriptions[0].name == "csv"
    assert registry.analysis_descriptions == ()
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(WebExtensionProvider(adapters=(csv_adapter,)))
    with pytest.raises(KeyError, match="unknown input adapter"):
        registry.adapter("missing")
    with pytest.raises(KeyError, match="unknown result analysis"):
        registry.analysis("missing")


def test_import_service_rejects_inconsistent_files_and_confirmation(
    tmp_path: Path,
) -> None:
    paths = WebApplicationPaths.create(tmp_path / "web")
    store = WebStateStore(paths)
    service = DatasetImportService(
        store,
        WebProviderRegistry(adapters=built_in_adapters(1024 * 1024)),
    )
    upload = paths.uploads / "upload"
    upload.mkdir()
    source = upload / "data.csv"
    source.write_text("id,value\nr1,A\nr2,B\n", encoding="utf-8")
    with pytest.raises(ConversionError, match="missing files"):
        service.inspect(
            adapter_name="csv",
            dataset_identifier="dataset",
            files={},
            upload_directory=upload,
        )
    with pytest.raises(ConversionError, match="unknown files"):
        service.inspect(
            adapter_name="csv",
            dataset_identifier="dataset",
            files={"data": source, "extra": source},
            upload_directory=upload,
        )

    record = service.inspect(
        adapter_name="csv",
        dataset_identifier="dataset",
        files={"data": source},
        upload_directory=upload,
    )
    with pytest.raises(ConversionError, match="must be confirmed"):
        service.confirm(record.identifier)
    with pytest.raises(ConversionError, match="do not match"):
        service.confirm(record.identifier, columns=record.preview.columns[:-1])
    duplicate = tuple(
        item.model_copy(update={"source_index": 0}) for item in record.preview.columns
    )
    with pytest.raises(ConversionError, match="unique"):
        service.confirm(record.identifier, columns=duplicate)
    invalid = tuple(
        item.model_copy(update={"selected_kind": ColumnKind.BOOLEAN})
        if item.name == "value"
        else item
        for item in record.preview.columns
    )
    with pytest.raises(ConversionError, match="not boolean"):
        service.confirm(record.identifier, columns=invalid)
    failed = store.get_import(record.identifier)
    assert failed is not None
    assert failed.status.value == "failed"
    assert not (paths.datasets / "dataset").exists()

    with pytest.raises(Exception, match="unknown import"):
        service.confirm("missing")


def test_web_api_imports_csv_and_serves_catalog_and_static_assets(tmp_path: Path) -> None:
    app = create_app(data_directory=tmp_path / "web", load_extensions=False)
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        catalog = client.get("/api/v1/catalog").json()
        assert {item["name"] for item in catalog["input_adapters"]} == {
            "csv",
            "dataset_bundle",
            "tsv",
        }
        assert [item["stage"] for item in catalog["workflow_stages"]] == [
            "INPUT",
            "PREPARATION",
            "EVALUATION",
            "SEARCH",
            "OUTPUT",
            "ANALYSIS",
        ]
        assert {item["family"]: item["default_engine"] for item in catalog["search_families"]} == {
            "QUALITY_DIVERSITY": "serial_mome",
            "CONVENTIONAL_MULTI_OBJECTIVE": "pymoo_nsga2",
        }
        assert catalog["analyses"] == []
        candidate_outcomes = next(
            item
            for item in catalog["components"]
            if item["kind"] == "observer" and item["name"] == "candidate_outcomes"
        )
        outcome_view = candidate_outcomes["observer_view"]
        assert outcome_view["view_kind"] == "STACKED_SERIES"
        assert {metric["temporal_scope"] for metric in outcome_view["metrics"]} == {"WINDOW"}
        assert {metric["unit"] for metric in outcome_view["metrics"]} == {"ratio"}
        qd_diagnostics = next(
            item
            for item in catalog["components"]
            if item["kind"] == "observer" and item["name"] == "qd_archive_diagnostics"
        )
        assert qd_diagnostics["observer_view"]["view_kind"] == "QD_DIAGNOSTICS"
        response = client.post(
            "/api/v1/imports/csv",
            data={"identifier": "browser-data", "slot_names": '["data"]'},
            files={"files": ("data.csv", b"id,a,b\nr1,1,X\nr2,2,Y\n", "text/csv")},
        )
        assert response.status_code == 200
        imported = response.json()
        confirmed = client.post(
            f"/api/v1/imports/{imported['identifier']}/confirm",
            json={"columns": imported["preview"]["columns"]},
        )
        assert confirmed.status_code == 200
        datasets = client.get("/api/v1/datasets").json()["items"]
        assert [item["identifier"] for item in datasets] == ["browser-data"]


def test_web_api_validates_pipeline_text_and_upload_boundaries(tmp_path: Path) -> None:
    app = create_app(
        data_directory=tmp_path / "web",
        max_upload_mib=1,
        load_extensions=False,
    )
    with TestClient(app) as client:
        default = client.get("/api/v1/pipelines/default").json()["yaml"]
        validated = client.post(
            "/api/v1/pipelines/validate",
            content=default,
            headers={"Content-Type": "text/yaml"},
        )
        assert validated.status_code == 200
        serialized = client.post(
            "/api/v1/pipelines/serialize",
            json=validated.json()["configuration"],
        )
        assert parse_pipeline_configuration(serialized.json()["yaml"])
        resolved = client.post(
            "/api/v1/pipelines/resolve",
            json={"configuration": validated.json()["configuration"]},
        )
        assert resolved.json()["complete"]
        assert resolved.json()["search_family"] == "QUALITY_DIVERSITY"
        transitioned = client.post(
            "/api/v1/pipelines/search-family",
            json={
                "configuration": validated.json()["configuration"],
                "search_family": "CONVENTIONAL_MULTI_OBJECTIVE",
            },
        )
        assert transitioned.status_code == 200
        transition = transitioned.json()
        assert transition["resolution"]["complete"]
        assert transition["resolution"]["search_family"] == "CONVENTIONAL_MULTI_OBJECTIVE"
        assert transition["configuration"]["search"]["descriptors"] == []
        assert (
            client.post(
                "/api/v1/pipelines/validate",
                content="schema_version: 1\nunknown: true\n",
                headers={"Content-Type": "text/yaml"},
            ).status_code
            == 422
        )

        malformed_slots = client.post(
            "/api/v1/imports/csv",
            data={"identifier": "bad", "slot_names": "{}"},
            files={"files": ("data.csv", b"a\n1\n", "text/csv")},
        )
        assert malformed_slots.status_code == 422
        unknown_slot = client.post(
            "/api/v1/imports/csv",
            data={"identifier": "bad", "slot_names": '["unknown"]'},
            files={"files": ("data.csv", b"a\n1\n", "text/csv")},
        )
        assert unknown_slot.status_code == 422
        wrong_extension = client.post(
            "/api/v1/imports/csv",
            data={"identifier": "bad", "slot_names": '["data"]'},
            files={"files": ("data.tsv", b"a\n1\n", "text/tab-separated-values")},
        )
        assert wrong_extension.status_code == 422
        too_large = client.post(
            "/api/v1/imports/csv",
            data={"identifier": "large", "slot_names": '["data"]'},
            files={"files": ("data.csv", b"a\n" + b"1" * (1024 * 1024), "text/csv")},
        )
        assert too_large.status_code == 422


def test_web_api_rejects_tampered_column_confirmation(tmp_path: Path) -> None:
    app = create_app(data_directory=tmp_path / "web", load_extensions=False)
    with TestClient(app) as client:
        imported = client.post(
            "/api/v1/imports/csv",
            data={"identifier": "browser-data", "slot_names": '["data"]'},
            files={"files": ("data.csv", b"id,a\nr1,1\nr2,2\n", "text/csv")},
        ).json()
        columns = imported["preview"]["columns"]
        columns[1]["name"] = "substituted"
        response = client.post(
            f"/api/v1/imports/{imported['identifier']}/confirm",
            json={"columns": columns},
        )
        assert response.status_code == 422
        assert "changed immutable source metadata" in response.json()["detail"]


def test_web_api_requires_confirmation_and_can_discard_an_import(tmp_path: Path) -> None:
    app = create_app(data_directory=tmp_path / "web", load_extensions=False)
    with TestClient(app) as client:
        imported = client.post(
            "/api/v1/imports/csv",
            data={"identifier": "discarded", "slot_names": '["data"]'},
            files={"files": ("data.csv", b"id,a\nr1,1\nr2,2\n", "text/csv")},
        ).json()
        missing_confirmation = client.post(
            f"/api/v1/imports/{imported['identifier']}/confirm",
            json={"columns": None},
        )
        assert missing_confirmation.status_code == 422
        assert client.delete(f"/api/v1/imports/{imported['identifier']}").json()["deleted"] is True
        unknown = client.post(
            "/api/v1/imports/unknown",
            data={"identifier": "unknown", "slot_names": '["data"]'},
            files={"files": ("data.csv", b"a\n1\n", "text/csv")},
        )
        assert unknown.status_code == 404


def test_installed_experiment_provider_contributes_optional_web_features(
    tmp_path: Path,
) -> None:
    app = create_app(data_directory=tmp_path / "web")
    with TestClient(app) as client:
        catalog = client.get("/api/v1/catalog").json()
    adapter_names = {item["name"] for item in catalog["input_adapters"]}
    assert {"gbic", "uci"} <= adapter_names
    uci = next(item for item in catalog["input_adapters"] if item["name"] == "uci")
    assert uci["parameters"][0]["name"] == "dataset_id"
    assert "prelic_accuracy" in {item["name"] for item in catalog["analyses"]}


def test_web_api_enables_analysis_only_for_an_imported_ground_truth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_start(
        _manager: WebRunManager,
        *,
        pipeline_text: str,
        dataset_identifier: str,
        run_identifier: str,
        seed: int,
        analyses: tuple[str, ...],
    ) -> WebRunRecord:
        captured.update(
            {
                "pipeline": pipeline_text,
                "dataset": dataset_identifier,
                "analyses": analyses,
            }
        )
        return WebRunRecord(
            identifier=run_identifier,
            dataset_identifier=dataset_identifier,
            pipeline_path=tmp_path / "pipeline.yaml",
            output_directory=tmp_path / "output",
            seed=seed,
            analyses=analyses,
            status=RunStatus.RUNNING,
        )

    monkeypatch.setattr(WebRunManager, "start", fake_start)
    app = create_app(data_directory=tmp_path / "web")
    ground_truth = json.dumps(
        {
            "#DatasetRows": 4,
            "#DatasetColumns": 3,
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
    ).encode()
    with TestClient(app) as client:
        imported = client.post(
            "/api/v1/imports/gbic",
            data={
                "identifier": "gbic-with-truth",
                "slot_names": '["data", "ground_truth"]',
            },
            files=[
                (
                    "files",
                    (
                        "dataset.tsv",
                        b"X\tn0\tn1\tn2\nr0\t1\t2\t3\nr1\t1\t2\t3\nr2\t8\t9\t7\nr3\t9\t8\t6\n",
                        "text/tab-separated-values",
                    ),
                ),
                ("files", ("dataset.json", ground_truth, "application/json")),
            ],
        )
        assert imported.status_code == 200
        confirmed = client.post(
            f"/api/v1/imports/{imported.json()['identifier']}/confirm",
            json={"columns": None},
        )
        assert confirmed.status_code == 200
        assert confirmed.json()["ground_truth_attached"] is True

        started = client.post(
            "/api/v1/runs",
            json={
                "pipeline": client.get("/api/v1/pipelines/default").json()["yaml"],
                "dataset_identifier": "gbic-with-truth",
                "run_identifier": "analysis-run",
                "seed": 2,
                "analyses": ["prelic_accuracy"],
            },
        )
        assert started.status_code == 201
        assert started.json()["analyses"] == ["prelic_accuracy"]
        assert captured["analyses"] == ("prelic_accuracy",)

        plain = client.post(
            "/api/v1/imports/csv",
            data={"identifier": "plain", "slot_names": '["data"]'},
            files={"files": ("plain.csv", b"id,a,b\nr0,1,2\nr1,2,3\n", "text/csv")},
        ).json()
        client.post(
            f"/api/v1/imports/{plain['identifier']}/confirm",
            json={"columns": plain["preview"]["columns"]},
        ).raise_for_status()
        rejected = client.post(
            "/api/v1/runs",
            json={
                "pipeline": client.get("/api/v1/pipelines/default").json()["yaml"],
                "dataset_identifier": "plain",
                "run_identifier": "invalid-analysis-run",
                "analyses": ["prelic_accuracy"],
            },
        )
        assert rejected.status_code == 422
        assert "requires an attached canonical ground truth" in rejected.json()["detail"]


def test_web_api_runs_one_small_parallel_search_in_a_spawned_process(tmp_path: Path) -> None:
    app = create_app(data_directory=tmp_path / "web", load_extensions=False)
    pipeline = default_scientific_configuration()
    evaluation = pipeline["evaluation"]
    assert isinstance(evaluation, dict)
    evaluation["candidate_validity"] = {
        "name": "minimum_cardinality",
        "parameters": {"min_rows": 2, "min_columns": 2},
    }
    search = pipeline["search"]
    assert isinstance(search, dict)
    search["engine"] = {
        "name": "serial_mome",
        "parameters": {"initial_population_size": 8, "batch_size": 2},
    }
    search["initialization"] = {"name": "uniform_random", "parameters": {}}
    search["termination"] = {
        "name": "evaluation_budget",
        "parameters": {"max_evaluations": 8},
    }
    pipeline["final_selection"] = {
        "name": "adaptive_residual_evidence_cover",
        "parameters": {
            "objective_names": ["internal_coherence", "contrast"],
            "complexity_penalty": 0.0,
            "minimum_marginal_evidence": 0.0,
            "minimum_quality_floor": 0.0,
            "maximum_quality_floor": 0.0,
        },
    }
    monitoring = pipeline["monitoring"]
    assert isinstance(monitoring, dict)
    monitoring["observers"] = [{"name": "search_progress", "parameters": {}}]
    monitoring["checkpoint_interval_evaluations"] = None
    execution = pipeline["execution"]
    assert isinstance(execution, dict)
    execution["executor"] = {
        "name": "process_pool",
        "parameters": {
            "integration_mode": "DETERMINISTIC",
            "max_in_flight": 2,
        },
    }
    execution["workers"] = 2
    yaml = serialize_pipeline_configuration(PipelineConfiguration.model_validate(pipeline))

    with TestClient(app) as client:
        imported = client.post(
            "/api/v1/imports/csv",
            data={"identifier": "run-data", "slot_names": '["data"]'},
            files={
                "files": (
                    "data.csv",
                    io.BytesIO(b"id,a,b,c\nr1,1,1,X\nr2,1,1,X\nr3,4,5,Y\nr4,4,5,Y\n"),
                    "text/csv",
                )
            },
        ).json()
        client.post(
            f"/api/v1/imports/{imported['identifier']}/confirm",
            json={"columns": imported["preview"]["columns"]},
        ).raise_for_status()
        started = client.post(
            "/api/v1/runs",
            json={
                "pipeline": yaml,
                "dataset_identifier": "run-data",
                "run_identifier": "web-run",
                "seed": 3,
            },
        )
        assert started.status_code == 201
        for _ in range(100):
            record = client.get("/api/v1/runs/web-run").json()
            if record["status"] != "running":
                break
            time.sleep(0.05)
        assert record["status"] == "completed"
        assert record["has_selected_results"]
        assert client.get("/api/v1/runs").json()["items"][0]["identifier"] == "web-run"
        events = client.get("/api/v1/runs/web-run/events?limit=2000").json()
        assert events["total"] > 0
        metrics = client.get("/api/v1/runs/web-run/metrics").json()
        assert "search.evaluations" in metrics["names"]
        metrics_without_names = client.get(
            "/api/v1/runs/web-run/metrics?limit=25000&include_names=false"
        ).json()
        assert metrics_without_names["names"] == []
        assert metrics_without_names["items"]
        assert client.get("/api/v1/runs/web-run/artifacts").status_code == 200
        stream = client.get("/api/v1/runs/web-run/stream?after=0")
        assert stream.status_code == 200
        assert "event:" in stream.text
        assert (
            client.get(
                "/api/v1/runs/web-run/stream",
                headers={"Last-Event-ID": "invalid"},
            ).status_code
            == 422
        )
        download = client.get("/api/v1/runs/web-run/download")
        assert download.status_code == 200
        assert zipfile.is_zipfile(io.BytesIO(download.content))
        page = client.get("/api/v1/runs/web-run/results/selected").json()
        assert page["total"] > 0
        identifier = page["items"][0]["identifier"]
        assert (
            client.get(f"/api/v1/runs/web-run/results/selected?query={identifier}").json()["total"]
            >= 1
        )
        assert (
            client.get("/api/v1/runs/web-run/results/selected?feasible=true").json()["total"]
            == page["total"]
        )
        assert (
            client.get("/api/v1/runs/web-run/results/selected?min_rows=99999").json()["total"] == 0
        )
        assert (
            client.get("/api/v1/runs/web-run/results/selected?feasible=false").json()["total"] == 0
        )
        assert (
            client.get("/api/v1/runs/web-run/results/selected?pattern=ADDITIVE").json()["total"]
            == 0
        )
        assert client.get("/api/v1/runs/web-run/results/selected?max_rows=1").json()["total"] == 0
        assert (
            client.get("/api/v1/runs/web-run/results/selected?min_columns=99999").json()["total"]
            == 0
        )
        assert (
            client.get("/api/v1/runs/web-run/results/selected?max_columns=1").json()["total"] == 0
        )
        assert (
            client.get("/api/v1/runs/web-run/results/selected?query=definitely-absent").json()[
                "total"
            ]
            == 0
        )
        assert (
            client.get("/api/v1/runs/web-run/results/selected?min_rows=3&max_rows=2").status_code
            == 422
        )
        detail = client.get(f"/api/v1/runs/web-run/results/selected/{identifier}").json()
        assert detail["row_indices"]
        assert detail["columns"]
        assert detail["patterns"]
        assert (
            client.get(
                f"/api/v1/runs/web-run/results/selected?pattern={detail['patterns'][0]}"
            ).json()["total"]
            >= 1
        )
        matrix = client.get(f"/api/v1/runs/web-run/results/selected/{identifier}/matrix").json()
        assert matrix["total_rows"] == detail["row_count"]
        assert client.get("/api/v1/runs/web-run/results/raw").status_code == 200
        assert client.post("/api/v1/runs/web-run/accuracy/selected/missing").status_code == 404
        assert client.delete("/api/v1/datasets/run-data").status_code == 422
        assert client.delete("/api/v1/runs/web-run").json()["deleted"] is True
        assert client.delete("/api/v1/datasets/run-data").json()["deleted"] is True
        assert client.delete("/api/v1/runs/unknown").status_code == 200
        assert client.delete("/api/v1/datasets/unknown").status_code == 200
        assert client.get("/api/v1/runs/unknown/results/selected").status_code == 404
        assert (
            client.get(f"/api/v1/runs/unknown/results/selected/{identifier}/matrix").status_code
            == 404
        )
        assert client.post("/api/v1/runs/unknown/accuracy/selected/missing").status_code == 404


def test_web_api_enforces_one_active_run_and_cancels_it(tmp_path: Path) -> None:
    app = create_app(data_directory=tmp_path / "web", load_extensions=False)
    pipeline = default_scientific_configuration()
    evaluation = pipeline["evaluation"]
    assert isinstance(evaluation, dict)
    evaluation["candidate_validity"] = {
        "name": "minimum_cardinality",
        "parameters": {"min_rows": 2, "min_columns": 2},
    }
    search = pipeline["search"]
    assert isinstance(search, dict)
    search["engine"] = {
        "name": "serial_mome",
        "parameters": {"initial_population_size": 64, "batch_size": 16},
    }
    search["initialization"] = {"name": "uniform_random", "parameters": {}}
    search["termination"] = {
        "name": "evaluation_budget",
        "parameters": {"max_evaluations": 1_000_000},
    }
    monitoring = pipeline["monitoring"]
    assert isinstance(monitoring, dict)
    monitoring["checkpoint_interval_evaluations"] = None
    yaml = serialize_pipeline_configuration(PipelineConfiguration.model_validate(pipeline))

    with TestClient(app) as client:
        imported = client.post(
            "/api/v1/imports/csv",
            data={"identifier": "cancel-data", "slot_names": '["data"]'},
            files={
                "files": (
                    "data.csv",
                    b"id,a,b,c\nr1,1,1,X\nr2,1,1,X\nr3,4,5,Y\nr4,4,5,Y\n",
                    "text/csv",
                )
            },
        ).json()
        client.post(
            f"/api/v1/imports/{imported['identifier']}/confirm",
            json={"columns": imported["preview"]["columns"]},
        ).raise_for_status()
        payload = {
            "pipeline": yaml,
            "dataset_identifier": "cancel-data",
            "run_identifier": "long-run",
            "seed": 9,
        }
        assert client.post("/api/v1/runs", json=payload).status_code == 201
        assert (
            client.post(
                "/api/v1/runs",
                json={**payload, "run_identifier": "second-run"},
            ).status_code
            == 409
        )
        assert client.delete("/api/v1/runs/long-run").status_code == 422
        cancelled = client.post("/api/v1/runs/long-run/cancel")
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] in {"cancelled", "failed"}
        assert client.delete("/api/v1/runs/long-run").status_code == 200


def test_web_launcher_is_loopback_only_and_delegates_to_uvicorn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assert _is_loopback("localhost")
    assert _is_loopback("::1")
    assert not _is_loopback("0.0.0.0")
    assert not _is_loopback("not-an-address")
    with pytest.raises(ConfigurationError, match="loopback"):
        launch_web_gui(host="0.0.0.0", open_browser=False)

    calls: list[tuple[object, str, int, str]] = []
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        SimpleNamespace(
            run=lambda app, host, port, log_level: calls.append((app, host, port, log_level))
        ),
    )
    assert (
        launch_web_gui(
            host="::1",
            port=9876,
            open_browser=False,
            data_directory=tmp_path,
            max_upload_mib=2,
        )
        == 0
    )
    assert calls[0][1:] == ("::1", 9876, "info")


def test_browser_waiter_opens_after_health_becomes_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[str] = []

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    results = iter((OSError("not ready"), Response()))

    def controlled_urlopen(_url: str, *, timeout: float) -> object:
        assert timeout == 0.25
        result = next(results)
        if isinstance(result, OSError):
            raise result
        return result

    monkeypatch.setattr("salvi.web.main.urllib.request.urlopen", controlled_urlopen)
    monkeypatch.setattr("salvi.web.main.webbrowser.open", lambda url: opened.append(url))
    monkeypatch.setattr("salvi.web.main.time.sleep", lambda _seconds: None)
    _open_when_ready("http://localhost/health", "http://localhost")
    assert opened == ["http://localhost"]


def test_process_signal_and_worker_outcome_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class Event:
        def __init__(self) -> None:
            self.value = False

        def is_set(self) -> bool:
            return self.value

    event = Event()
    signal = ProcessCancellationSignal(event)  # type: ignore[arg-type]
    assert not signal.cancelled
    signal.raise_if_cancelled()
    event.value = True
    with pytest.raises(RunCancelledError):
        signal.raise_if_cancelled()

    binding = {
        "identifier": "child",
        "dataset_bundle": str(tmp_path / "dataset"),
        "output_directory": str(tmp_path / "output"),
        "seed": 1,
    }
    called: list[str] = []
    monkeypatch.setattr(
        "salvi.web.run_manager.RunService.run_pipeline",
        lambda _self, _path, parsed, cancellation: called.append(parsed.identifier),
    )
    worker_error_path = tmp_path / "worker-error.json"
    _run_child(  # type: ignore[arg-type]
        str(tmp_path / "pipeline.yaml"), binding, event, str(worker_error_path)
    )
    assert called == ["child"]
    assert not worker_error_path.exists()
    monkeypatch.setattr(
        "salvi.web.run_manager.RunService.run_pipeline",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("failed")),
    )
    with pytest.raises(SystemExit):
        _run_child(  # type: ignore[arg-type]
            str(tmp_path / "pipeline.yaml"), binding, event, str(worker_error_path)
        )
    worker_error = json.loads(worker_error_path.read_text(encoding="utf-8"))
    assert worker_error["error_type"] == "RuntimeError"
    assert worker_error["message"] == "failed"
    assert "RuntimeError: failed" in worker_error["traceback"]

    record = WebRunRecord(
        identifier="run",
        dataset_identifier="dataset",
        pipeline_path=tmp_path / "run" / "pipeline.yaml",
        output_directory=tmp_path / "run" / "output",
        status=RunStatus.RUNNING,
        started_at=datetime.now(UTC),
    )
    record.pipeline_path.parent.mkdir(parents=True)
    worker_error_path.replace(record.pipeline_path.parent / "worker-error.json")
    status, error, started_at, finished_at = WebRunManager._read_outcome(record, 1)
    assert status is RunStatus.FAILED
    assert error == "RuntimeError: failed"
    assert started_at is None
    assert finished_at is not None

    (record.pipeline_path.parent / "worker-error.json").unlink()
    status, error, _, _ = WebRunManager._read_outcome(record, 1)
    assert status is RunStatus.FAILED
    assert "code 1" in str(error)

    record.output_directory.mkdir(parents=True)
    now = datetime.now(UTC)
    (record.output_directory / "run-metadata.json").write_text(
        (
            '{"status":"completed","error":null,'
            f'"started_at":"{now.isoformat()}","finished_at":"{now.isoformat()}"'
            "}"
        ),
        encoding="utf-8",
    )
    status, error, started_at, finished_at = WebRunManager._read_outcome(record, 0)
    assert (status, error, started_at, finished_at) == (
        RunStatus.COMPLETED,
        None,
        now,
        now,
    )
