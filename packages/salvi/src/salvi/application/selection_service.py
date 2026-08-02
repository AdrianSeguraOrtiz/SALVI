"""Reusable final selection over a previously persisted search repertoire."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from salvi.application.configuration import RunBinding, load_bound_configuration
from salvi.application.factory import build_specification, prepare_run
from salvi.components.registry import ComponentRegistry
from salvi.domain.models import Repertoire
from salvi.exceptions import ArtifactError
from salvi.infrastructure.bicluster_set import (
    BiclusterSetManifest,
    BiclusterSetReader,
    BiclusterSetWriter,
)
from salvi.infrastructure.files import sha256_file


@dataclass(frozen=True, slots=True)
class FinalSelectionResult:
    """Summary of one selector application to an existing archive."""

    output_directory: Path
    selector: str
    input_count: int
    output_count: int
    manifest: BiclusterSetManifest


class FinalSelectionService:
    """Apply the configured selector without rerunning candidate search."""

    def __init__(self, registry: ComponentRegistry | None = None) -> None:
        self._registry = registry

    def select(
        self,
        configuration: Path,
        *,
        dataset_bundle: Path,
        repertoire: Path,
        output: Path,
        identifier: str | None = None,
        overwrite: bool = False,
    ) -> FinalSelectionResult:
        source_repertoire = repertoire.expanduser().resolve()
        destination = output.expanduser().resolve()
        if source_repertoire == destination:
            raise ArtifactError("selection output must differ from its source repertoire")
        loaded = load_bound_configuration(
            configuration,
            RunBinding(
                identifier=identifier or f"{source_repertoire.name}-selection",
                dataset_bundle=dataset_bundle,
                output_directory=destination.parent,
            ),
        )
        specification = build_specification(loaded.configuration, self._registry)
        selector = specification.final_selector
        if selector is None:
            raise ArtifactError(
                "the selected pipeline does not configure a final-selection component"
            )
        prepared = prepare_run(specification)
        contents = BiclusterSetReader().read_contents(source_repertoire)
        dataset = prepared.context.dataset
        if contents.manifest.dataset_identifier != dataset.metadata.identifier:
            raise ArtifactError(
                "source archive and configured DatasetBundle have different identifiers"
            )
        if (
            contents.manifest.row_count != dataset.row_count
            or contents.manifest.source_column_count != dataset.source_column_count
            or contents.columns != dataset.columns
        ):
            raise ArtifactError(
                "source archive was not produced from the configured preprocessing pipeline"
            )
        try:
            selected = selector.select(prepared.context, contents.repertoire)
            source_checkpoint = contents.manifest.source_checkpoint
            if source_checkpoint is not None and not Path(source_checkpoint).is_absolute():
                source_checkpoint = str(
                    (source_repertoire.parents[1] / source_checkpoint).resolve()
                )
            writer = BiclusterSetWriter()

            def write_repertoire(
                target: Path,
                artifact_identifier: str,
                repertoire_to_write: Repertoire,
            ) -> BiclusterSetManifest:
                return writer.write(
                    target,
                    identifier=artifact_identifier,
                    dataset_identifier=dataset.metadata.identifier,
                    row_count=dataset.row_count,
                    source_column_count=dataset.source_column_count,
                    columns=dataset.columns,
                    repertoire=repertoire_to_write,
                    source_run=contents.manifest.source_run,
                    source_checkpoint=source_checkpoint,
                    source_checkpoint_sha256=(contents.manifest.source_checkpoint_sha256),
                    source_checkpoint_evaluations=(contents.manifest.source_checkpoint_evaluations),
                    overwrite=overwrite,
                )

            manifest = write_repertoire(
                destination,
                identifier or f"{contents.manifest.identifier}-selected",
                selected,
            )
        finally:
            specification.executor.close()
        return FinalSelectionResult(
            output_directory=destination,
            selector=selector.component_name,
            input_count=len(contents.repertoire.evaluations),
            output_count=len(selected.evaluations),
            manifest=manifest,
        )


def selection_manifest_sha256(result: FinalSelectionResult) -> str:
    """Return the canonical checksum used by orchestration layers."""

    return sha256_file(result.output_directory / "manifest.json")


__all__ = [
    "FinalSelectionResult",
    "FinalSelectionService",
    "selection_manifest_sha256",
]
