"""FastAPI application factory for the local SALVI web interface."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.resources import files
from pathlib import Path
from typing import Annotated, Any

from fastapi import (
    Body,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from starlette.background import BackgroundTask

from salvi.application.composition import CompositionResolutionService
from salvi.application.configuration import (
    PipelineConfiguration,
    parse_pipeline_configuration,
    serialize_pipeline_configuration,
)
from salvi.application.defaults import default_scientific_configuration
from salvi.components.catalog import role_catalog, workflow_stage_catalog
from salvi.components.defaults import default_component_registry
from salvi.domain.enums import PatternKind, RunStatus
from salvi.exceptions import ArtifactError, ConfigurationError, SalviError
from salvi.infrastructure.events import SQLiteRunEventSource
from salvi.patterns import default_pattern_catalog
from salvi.web.adapters import built_in_adapters
from salvi.web.imports import DatasetImportService
from salvi.web.models import WebColumnProposal
from salvi.web.providers import WebProviderRegistry
from salvi.web.results import ResultKind, WebResultService
from salvi.web.run_manager import WebRunManager
from salvi.web.storage import WebApplicationPaths, WebStateStore

API_PREFIX = "/api/v1"


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CompositionRequest(StrictRequest):
    configuration: dict[str, Any]


class ConfirmImportRequest(StrictRequest):
    columns: tuple[WebColumnProposal, ...] | None = None
    adapter_configuration: dict[str, Any] | None = None


class StartRunRequest(StrictRequest):
    pipeline: str = Field(min_length=1)
    dataset_identifier: str = Field(min_length=1)
    run_identifier: str = Field(min_length=1)
    seed: int = Field(default=0, ge=0)
    analyses: tuple[str, ...] = ()

    @field_validator("analyses")
    @classmethod
    def validate_analyses(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not name.strip() for name in value):
            raise ValueError("analysis names must not be blank")
        if len(set(value)) != len(value):
            raise ValueError("analysis names must be unique")
        return value


def _dataset_payload(record: Any) -> dict[str, Any]:
    return {
        "identifier": record.identifier,
        "adapter": record.adapter,
        "created_at": record.created_at.isoformat(),
        "ground_truth_attached": record.ground_truth_attached,
        "clinical_annotations_attached": record.clinical_annotations_attached,
    }


def _run_payload(record: Any) -> dict[str, Any]:
    monitoring: dict[str, Any] = {
        "observers": [],
        "archive_axes": [],
        "termination": None,
    }
    metadata_path = record.output_directory / "run-metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            persisted = metadata.get("monitoring")
            if isinstance(persisted, dict):
                monitoring = {**monitoring, **persisted}
        except (OSError, ValueError):
            pass
    return {
        "identifier": record.identifier,
        "dataset_identifier": record.dataset_identifier,
        "seed": record.seed,
        "analyses": record.analyses,
        "status": record.status.value,
        "created_at": record.created_at.isoformat(),
        "started_at": None if record.started_at is None else record.started_at.isoformat(),
        "finished_at": None if record.finished_at is None else record.finished_at.isoformat(),
        "error": record.error,
        "has_events": (record.output_directory / "run.sqlite").is_file(),
        "has_raw_results": (record.output_directory / "artifacts" / "search-repertoire").is_dir(),
        "has_selected_results": (record.output_directory / "artifacts" / "repertoire").is_dir(),
        "monitoring": monitoring,
    }


def create_app(
    *,
    data_directory: Path | None = None,
    max_upload_mib: int = 2048,
    load_extensions: bool = True,
) -> FastAPI:
    if max_upload_mib < 1:
        raise ValueError("max_upload_mib must be positive")
    maximum_bytes = max_upload_mib * 1024 * 1024
    paths = WebApplicationPaths.create(data_directory)
    store = WebStateStore(paths)
    providers = WebProviderRegistry(adapters=built_in_adapters(maximum_bytes))
    if load_extensions:
        providers.load_entry_points()
    imports = DatasetImportService(store, providers)
    manager = WebRunManager(store)
    results = WebResultService(store, providers)
    composition = CompositionResolutionService()
    component_registry = default_component_registry()

    @asynccontextmanager
    async def lifespan(_application: FastAPI) -> AsyncIterator[None]:
        yield
        await asyncio.to_thread(manager.shutdown)

    app = FastAPI(
        title="SALVI",
        version="1",
        docs_url=f"{API_PREFIX}/docs",
        openapi_url=f"{API_PREFIX}/openapi.json",
        lifespan=lifespan,
    )
    app.state.paths = paths
    app.state.store = store
    app.state.providers = providers
    app.state.manager = manager

    @app.exception_handler(SalviError)
    async def salvi_error_handler(_request: Request, error: SalviError) -> JSONResponse:
        status = 404 if isinstance(error, ArtifactError) and "unknown" in str(error) else 422
        return JSONResponse(status_code=status, content={"detail": str(error)})

    @app.exception_handler(KeyError)
    async def key_error_handler(_request: Request, error: KeyError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(error)})

    @app.get(f"{API_PREFIX}/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get(f"{API_PREFIX}/catalog")
    async def catalog() -> dict[str, Any]:
        return {
            "workflow_stages": [item.model_dump(mode="json") for item in workflow_stage_catalog()],
            "roles": [item.model_dump(mode="json") for item in role_catalog()],
            "components": [item.model_dump(mode="json") for item in component_registry.catalog()],
            "patterns": [
                {
                    "kind": item.kind.value,
                    "scope": item.scope.value,
                    "supported_column_kinds": sorted(
                        value.value for value in item.supported_column_kinds
                    ),
                    "minimum_columns": item.minimum_columns,
                    "maximum_groups": item.maximum_groups,
                    "reference_model": item.reference_model,
                }
                for item in default_pattern_catalog().definitions()
            ],
            "input_adapters": [
                item.model_dump(mode="json") for item in providers.adapter_descriptions
            ],
            "analyses": [item.model_dump(mode="json") for item in providers.analysis_descriptions],
        }

    @app.get(f"{API_PREFIX}/pipelines/default")
    async def default_pipeline() -> dict[str, str]:
        pipeline = PipelineConfiguration.model_validate(default_scientific_configuration())
        return {"yaml": serialize_pipeline_configuration(pipeline)}

    @app.post(f"{API_PREFIX}/pipelines/resolve")
    async def resolve_composition(request: CompositionRequest) -> dict[str, Any]:
        return composition.resolve(request.configuration).model_dump(mode="json")

    @app.post(f"{API_PREFIX}/pipelines/validate")
    async def validate_pipeline(
        content: Annotated[str, Body(media_type="text/yaml")],
    ) -> dict[str, Any]:
        pipeline = parse_pipeline_configuration(content, source="web editor")
        resolution = composition.resolve(pipeline.model_dump(mode="json"))
        if not resolution.complete:
            raise ConfigurationError("pipeline composition is incomplete")
        return {
            "valid": True,
            "yaml": serialize_pipeline_configuration(pipeline),
            "configuration": pipeline.model_dump(mode="json"),
        }

    @app.post(f"{API_PREFIX}/pipelines/serialize")
    async def serialize_pipeline(configuration: dict[str, Any]) -> dict[str, str]:
        try:
            pipeline = PipelineConfiguration.model_validate(configuration)
        except ValidationError as error:
            raise ConfigurationError(f"invalid SALVI pipeline: {error}") from error
        return {"yaml": serialize_pipeline_configuration(pipeline)}

    @app.get(f"{API_PREFIX}/datasets")
    async def list_datasets() -> dict[str, Any]:
        return {"items": [_dataset_payload(item) for item in store.datasets()]}

    @app.delete(f"{API_PREFIX}/datasets/{{identifier}}")
    async def delete_dataset(identifier: str) -> dict[str, bool]:
        await asyncio.to_thread(store.delete_dataset, identifier)
        return {"deleted": True}

    @app.post(f"{API_PREFIX}/imports/{{adapter_name}}")
    async def inspect_import(
        adapter_name: str,
        identifier: Annotated[str, Form()],
        slot_names: Annotated[str, Form()],
        parameters: Annotated[str, Form()] = "{}",
        uploaded_files: Annotated[list[UploadFile] | None, File(alias="files")] = None,
    ) -> dict[str, Any]:
        uploads = uploaded_files or []
        try:
            slots = json.loads(slot_names)
            if (
                not isinstance(slots, list)
                or any(not isinstance(item, str) or not item for item in slots)
                or len(slots) != len(uploads)
                or len(set(slots)) != len(slots)
            ):
                raise ValueError
        except (json.JSONDecodeError, ValueError) as error:
            raise HTTPException(
                status_code=422,
                detail="slot_names must be a unique JSON string list aligned with files",
            ) from error
        try:
            raw_parameters = json.loads(parameters)
            if not isinstance(raw_parameters, dict) or any(
                not isinstance(key, str) or not isinstance(value, str | int | float | bool)
                for key, value in raw_parameters.items()
            ):
                raise ValueError
        except (json.JSONDecodeError, ValueError) as error:
            raise HTTPException(
                status_code=422,
                detail="parameters must be a JSON object of scalar values",
            ) from error
        adapter = providers.adapter(adapter_name)
        descriptions = {slot.name: slot for slot in adapter.description.files}
        upload_directory = paths.uploads / store.new_identifier("upload")
        upload_directory.mkdir()
        saved: dict[str, Path] = {}
        total = 0
        try:
            for slot_name, uploaded in zip(slots, uploads, strict=True):
                description = descriptions.get(slot_name)
                if description is None:
                    raise ConfigurationError(f"unknown adapter file slot: {slot_name}")
                filename = Path(uploaded.filename or "").name
                if not filename or filename != uploaded.filename:
                    raise ConfigurationError("uploaded filenames must be safe basenames")
                if description.accepted_extensions and Path(filename).suffix.lower() not in {
                    item.lower() for item in description.accepted_extensions
                }:
                    raise ConfigurationError(
                        f"{slot_name} requires one of: "
                        + ", ".join(description.accepted_extensions)
                    )
                destination = upload_directory / f"{slot_name}{Path(filename).suffix.lower()}"
                with destination.open("wb") as output:
                    while chunk := await uploaded.read(1024 * 1024):
                        total += len(chunk)
                        if total > maximum_bytes:
                            raise ConfigurationError(
                                f"upload exceeds the {max_upload_mib} MiB limit"
                            )
                        output.write(chunk)
                saved[slot_name] = destination
            record = await asyncio.to_thread(
                imports.inspect,
                adapter_name=adapter_name,
                dataset_identifier=identifier,
                files=saved,
                parameters=raw_parameters,
                upload_directory=upload_directory,
            )
            return {
                "identifier": record.identifier,
                "status": record.status.value,
                "preview": record.preview.model_dump(mode="json"),
            }
        except Exception:
            shutil.rmtree(upload_directory, ignore_errors=True)
            raise
        finally:
            for uploaded in uploads:
                await uploaded.close()

    @app.post(f"{API_PREFIX}/imports/{{identifier}}/confirm")
    async def confirm_import(
        identifier: str,
        request: ConfirmImportRequest,
    ) -> dict[str, Any]:
        dataset = await asyncio.to_thread(
            imports.confirm,
            identifier,
            columns=request.columns,
            adapter_configuration=request.adapter_configuration,
        )
        return _dataset_payload(dataset)

    @app.delete(f"{API_PREFIX}/imports/{{identifier}}")
    async def delete_import(identifier: str) -> dict[str, bool]:
        await asyncio.to_thread(store.delete_import, identifier)
        return {"deleted": True}

    @app.get(f"{API_PREFIX}/runs")
    async def list_runs() -> dict[str, Any]:
        return {"items": [_run_payload(item) for item in store.runs()]}

    @app.get(f"{API_PREFIX}/runs/{{identifier}}")
    async def get_run(identifier: str) -> dict[str, Any]:
        record = store.get_run(identifier)
        if record is None:
            raise ArtifactError(f"unknown run: {identifier}")
        return _run_payload(record)

    @app.post(f"{API_PREFIX}/runs", status_code=201)
    async def start_run(request: StartRunRequest) -> dict[str, Any]:
        if manager.active_identifier is not None:
            raise HTTPException(status_code=409, detail="another SALVI run is active")
        dataset = store.get_dataset(request.dataset_identifier)
        if dataset is None:
            raise ArtifactError(f"unknown dataset: {request.dataset_identifier}")
        for name in request.analyses:
            try:
                analysis = providers.analysis(name)
            except KeyError as error:
                raise ConfigurationError(str(error)) from error
            if analysis.description.requires_ground_truth and not dataset.ground_truth_attached:
                raise ConfigurationError(
                    f"analysis {name!r} requires an attached canonical ground truth"
                )
        record = await asyncio.to_thread(
            manager.start,
            pipeline_text=request.pipeline,
            dataset_identifier=request.dataset_identifier,
            run_identifier=request.run_identifier,
            seed=request.seed,
            analyses=request.analyses,
        )
        return _run_payload(record)

    @app.post(f"{API_PREFIX}/runs/{{identifier}}/cancel")
    async def cancel_run(identifier: str) -> dict[str, Any]:
        record = await asyncio.to_thread(manager.cancel, identifier)
        return _run_payload(record)

    @app.delete(f"{API_PREFIX}/runs/{{identifier}}")
    async def delete_run(identifier: str) -> dict[str, bool]:
        await asyncio.to_thread(store.delete_run, identifier)
        return {"deleted": True}

    @app.get(f"{API_PREFIX}/runs/{{identifier}}/events")
    async def event_page(
        identifier: str,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=256, ge=1, le=2000),
    ) -> dict[str, Any]:
        record = store.get_run(identifier)
        if record is None:
            raise ArtifactError(f"unknown run: {identifier}")
        source = SQLiteRunEventSource(record.output_directory / "run.sqlite")
        events = await asyncio.to_thread(source.event_page, offset, limit=limit)
        return {
            "offset": offset,
            "total": await asyncio.to_thread(source.event_count),
            "items": [event.model_dump(mode="json") for event in events],
        }

    @app.get(f"{API_PREFIX}/runs/{{identifier}}/metrics")
    async def metrics(
        identifier: str,
        name: str | None = None,
        after_sequence: int = Query(default=0, ge=0),
        limit: int = Query(default=1000, ge=1, le=25_000),
        include_names: bool = True,
    ) -> dict[str, Any]:
        record = store.get_run(identifier)
        if record is None:
            raise ArtifactError(f"unknown run: {identifier}")
        source = SQLiteRunEventSource(record.output_directory / "run.sqlite")
        values = await asyncio.to_thread(
            source.poll_metrics,
            after_sequence,
            name=name,
            limit=limit,
        )
        return {
            "names": await asyncio.to_thread(source.metric_names) if include_names else [],
            "items": [value.model_dump(mode="json") for value in values],
        }

    @app.get(f"{API_PREFIX}/runs/{{identifier}}/stream")
    async def event_stream(
        identifier: str,
        request: Request,
        after: int = Query(default=0, ge=0),
        last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
    ) -> StreamingResponse:
        record = store.get_run(identifier)
        if record is None:
            raise ArtifactError(f"unknown run: {identifier}")
        sequence = after
        if last_event_id is not None:
            try:
                sequence = max(sequence, int(last_event_id))
            except ValueError as error:
                raise HTTPException(
                    status_code=422,
                    detail="Last-Event-ID must be an integer",
                ) from error

        async def generate() -> AsyncIterator[str]:
            current = sequence
            idle_cycles = 0
            while not await request.is_disconnected():
                source = SQLiteRunEventSource(record.output_directory / "run.sqlite")
                events = await asyncio.to_thread(source.poll, current, limit=500)
                if events:
                    idle_cycles = 0
                    for event in events:
                        current = event.sequence or current
                        payload = event.model_dump_json()
                        yield f"id: {current}\nevent: run-event\ndata: {payload}\n\n"
                else:
                    idle_cycles += 1
                    latest = store.get_run(identifier)
                    if (
                        latest is not None
                        and latest.status
                        in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}
                        and idle_cycles > 1
                    ):
                        break
                    if idle_cycles % 40 == 0:
                        yield ": keep-alive\n\n"
                    await asyncio.sleep(0.25)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get(f"{API_PREFIX}/runs/{{identifier}}/artifacts")
    async def artifacts(identifier: str) -> dict[str, Any]:
        record = store.get_run(identifier)
        if record is None:
            raise ArtifactError(f"unknown run: {identifier}")
        source = SQLiteRunEventSource(record.output_directory / "run.sqlite")
        items = await asyncio.to_thread(source.artifacts)
        return {
            "items": [
                {
                    "identifier": item.identifier,
                    "media_type": item.media_type,
                    "checksum": item.checksum,
                    "event_sequence": item.event_sequence,
                }
                for item in items
            ]
        }

    @app.get(f"{API_PREFIX}/runs/{{identifier}}/download")
    async def download_run(identifier: str) -> FileResponse:
        record = store.get_run(identifier)
        if record is None:
            raise ArtifactError(f"unknown run: {identifier}")
        descriptor, name = tempfile.mkstemp(prefix=f"salvi-{identifier}-", suffix=".zip")
        os.close(descriptor)
        temporary = Path(name)
        temporary.unlink()
        archive = Path(
            shutil.make_archive(
                str(temporary.with_suffix("")),
                "zip",
                root_dir=record.pipeline_path.parent,
            )
        )
        return FileResponse(
            archive,
            media_type="application/zip",
            filename=f"{identifier}.zip",
            background=BackgroundTask(archive.unlink, missing_ok=True),
        )

    @app.get(f"{API_PREFIX}/runs/{{identifier}}/results/{{kind}}")
    async def result_page(
        identifier: str,
        kind: ResultKind,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=500),
        query: str | None = Query(default=None, max_length=200),
        feasible: bool | None = None,
        pattern: PatternKind | None = None,
        min_rows: int | None = Query(default=None, ge=1),
        max_rows: int | None = Query(default=None, ge=1),
        min_columns: int | None = Query(default=None, ge=1),
        max_columns: int | None = Query(default=None, ge=1),
    ) -> dict[str, Any]:
        if min_rows is not None and max_rows is not None and min_rows > max_rows:
            raise HTTPException(status_code=422, detail="min_rows cannot exceed max_rows")
        if min_columns is not None and max_columns is not None and min_columns > max_columns:
            raise HTTPException(
                status_code=422,
                detail="min_columns cannot exceed max_columns",
            )
        return await asyncio.to_thread(
            results.page,
            identifier,
            kind,
            offset=offset,
            limit=limit,
            query=query,
            feasible=feasible,
            pattern=pattern,
            min_rows=min_rows,
            max_rows=max_rows,
            min_columns=min_columns,
            max_columns=max_columns,
        )

    @app.get(f"{API_PREFIX}/runs/{{identifier}}/results/{{kind}}/{{bicluster_identifier}}")
    async def result_detail(
        identifier: str,
        kind: ResultKind,
        bicluster_identifier: str,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            results.detail,
            identifier,
            kind,
            bicluster_identifier,
        )

    @app.get(f"{API_PREFIX}/runs/{{identifier}}/results/{{kind}}/{{bicluster_identifier}}/matrix")
    async def result_matrix(
        identifier: str,
        kind: ResultKind,
        bicluster_identifier: str,
        row_offset: int = Query(default=0, ge=0),
        row_limit: int = Query(default=50, ge=1, le=200),
        column_offset: int = Query(default=0, ge=0),
        column_limit: int = Query(default=30, ge=1, le=100),
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            results.matrix,
            identifier,
            kind,
            bicluster_identifier,
            row_offset=row_offset,
            row_limit=row_limit,
            column_offset=column_offset,
            column_limit=column_limit,
        )

    @app.post(f"{API_PREFIX}/runs/{{identifier}}/accuracy/{{kind}}/{{analysis_name}}")
    async def accuracy(
        identifier: str,
        kind: ResultKind,
        analysis_name: str,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            results.accuracy,
            identifier,
            kind,
            analysis_name,
        )

    static_path = Path(str(files("salvi.web").joinpath("static")))
    if (static_path / "index.html").is_file():
        app.mount("/", StaticFiles(directory=static_path, html=True), name="web")

    return app


__all__ = ["API_PREFIX", "create_app"]
