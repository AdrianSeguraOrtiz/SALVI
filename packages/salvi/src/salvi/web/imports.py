"""Dataset upload inspection and canonicalization services."""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from salvi.exceptions import ArtifactError, ConversionError
from salvi.infrastructure.dataset_bundle import DatasetBundleReader
from salvi.web.models import (
    AdapterParameterDescription,
    AdapterParameterKind,
    ImportStatus,
    WebColumnProposal,
    WebDatasetRecord,
    WebImportRecord,
)
from salvi.web.providers import WebProviderRegistry
from salvi.web.storage import WebStateStore


class DatasetImportService:
    def __init__(self, store: WebStateStore, providers: WebProviderRegistry) -> None:
        self._store = store
        self._providers = providers

    def inspect(
        self,
        *,
        adapter_name: str,
        dataset_identifier: str,
        files: Mapping[str, Path],
        upload_directory: Path,
        parameters: Mapping[str, str | int | float | bool] | None = None,
    ) -> WebImportRecord:
        adapter = self._providers.adapter(adapter_name)
        expected = {slot.name for slot in adapter.description.files if slot.required}
        missing = expected - set(files)
        unknown = set(files) - {slot.name for slot in adapter.description.files}
        if missing:
            raise ConversionError(
                f"missing files for adapter {adapter_name}: {', '.join(sorted(missing))}"
            )
        if unknown:
            raise ConversionError(
                f"unknown files for adapter {adapter_name}: {', '.join(sorted(unknown))}"
            )
        validated_parameters = self._validate_parameters(
            adapter.description.parameters,
            parameters or {},
        )
        import_identifier = self._store.new_identifier("import")
        workspace = upload_directory / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        preview = adapter.inspect(
            files,
            parameters=validated_parameters,
            identifier=dataset_identifier,
            workspace=workspace,
        )
        record = WebImportRecord(
            identifier=import_identifier,
            adapter=adapter_name,
            upload_directory=upload_directory,
            files=dict(files),
            parameters=validated_parameters,
            preview=preview,
            status=(ImportStatus.READY if preview.confirmation_required else ImportStatus.UPLOADED),
        )
        self._store.put_import(record)
        return record

    def confirm(
        self,
        identifier: str,
        *,
        columns: Sequence[WebColumnProposal] | None = None,
        adapter_configuration: Mapping[str, object] | None = None,
    ) -> WebDatasetRecord:
        record = self._store.get_import(identifier)
        if record is None:
            raise ArtifactError(f"unknown import: {identifier}")
        adapter = self._providers.adapter(record.adapter)
        selected_columns = tuple(columns or record.preview.columns)
        if record.preview.confirmation_required and columns is None:
            raise ConversionError("the inferred tabular column types must be confirmed")
        self._validate_confirmed_columns(record.preview.columns, selected_columns)
        destination = self._store.paths.datasets / record.preview.identifier
        if destination.exists():
            raise ConversionError(f"a dataset named {record.preview.identifier!r} already exists")
        workspace = record.upload_directory / "workspace"
        try:
            bundle_path = adapter.convert(
                record.files,
                identifier=record.preview.identifier,
                columns=selected_columns,
                parameters=record.parameters,
                adapter_configuration=(
                    record.preview.adapter_configuration
                    if adapter_configuration is None
                    else adapter_configuration
                ),
                destination=destination,
                workspace=workspace,
            )
            manifest = DatasetBundleReader().read_manifest(bundle_path)
            dataset = WebDatasetRecord(
                identifier=manifest.identifier,
                adapter=record.adapter,
                bundle_path=bundle_path,
                storage_path=destination,
                created_at=datetime.now(UTC),
                ground_truth_attached=manifest.ground_truth_file is not None,
                clinical_annotations_attached=record.preview.clinical_annotations_attached,
            )
            self._store.put_dataset(dataset)
            self._store.put_import(record.model_copy(update={"status": ImportStatus.COMPLETED}))
            self._store.delete_import(record.identifier)
            return dataset
        except Exception as error:
            self._store.put_import(
                record.model_copy(update={"status": ImportStatus.FAILED, "error": str(error)})
            )
            if destination.exists():
                shutil.rmtree(destination)
            raise

    @staticmethod
    def _validate_parameters(
        descriptions: Sequence[AdapterParameterDescription],
        supplied: Mapping[str, str | int | float | bool],
    ) -> dict[str, str | int | float | bool]:
        by_name = {description.name: description for description in descriptions}
        unknown = set(supplied) - set(by_name)
        if unknown:
            raise ConversionError("unknown adapter parameters: " + ", ".join(sorted(unknown)))
        result: dict[str, str | int | float | bool] = {}
        for name, description in by_name.items():
            if name in supplied:
                value = supplied[name]
            elif description.default is not None:
                value = description.default
            elif description.required:
                raise ConversionError(f"missing adapter parameter {name!r}")
            else:
                continue
            valid_type = (
                (description.kind is AdapterParameterKind.STRING and isinstance(value, str))
                or (
                    description.kind is AdapterParameterKind.INTEGER
                    and isinstance(value, int)
                    and not isinstance(value, bool)
                )
                or (
                    description.kind is AdapterParameterKind.NUMBER
                    and isinstance(value, int | float)
                    and not isinstance(value, bool)
                )
                or (description.kind is AdapterParameterKind.BOOLEAN and isinstance(value, bool))
            )
            if not valid_type:
                raise ConversionError(
                    f"adapter parameter {name!r} must be {description.kind.value.lower()}"
                )
            if isinstance(value, int | float) and not isinstance(value, bool):
                if description.minimum is not None and value < description.minimum:
                    raise ConversionError(
                        f"adapter parameter {name!r} must be at least {description.minimum}"
                    )
                if description.maximum is not None and value > description.maximum:
                    raise ConversionError(
                        f"adapter parameter {name!r} must not exceed {description.maximum}"
                    )
            result[name] = value
        return result

    @staticmethod
    def _validate_confirmed_columns(
        proposed: Sequence[WebColumnProposal],
        confirmed: Sequence[WebColumnProposal],
    ) -> None:
        if len(proposed) != len(confirmed):
            raise ConversionError("confirmed columns do not match the inspected upload")
        by_index = {column.source_index: column for column in confirmed}
        if len(by_index) != len(confirmed):
            raise ConversionError("confirmed source column indices must be unique")
        for original in proposed:
            replacement = by_index.get(original.source_index)
            if replacement is None:
                raise ConversionError("confirmed columns do not match the inspected upload")
            immutable_fields = (
                "name",
                "inferred_kind",
                "missing_ratio",
                "sample_values",
                "units",
                "description",
            )
            if any(
                getattr(original, field) != getattr(replacement, field)
                for field in immutable_fields
            ):
                raise ConversionError("confirmed columns changed immutable source metadata")


__all__ = ["DatasetImportService"]
