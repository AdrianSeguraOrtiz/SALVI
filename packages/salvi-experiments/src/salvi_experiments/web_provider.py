"""Optional browser integrations contributed by ``salvi-experiments``."""

from __future__ import annotations

import csv
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from salvi.domain.enums import ColumnKind
from salvi.infrastructure.dataset_bundle import DatasetBundleReader
from salvi.web.adapters import (
    TabularInputAdapter,
    canonical_bundle_preview,
    normalized_identifier,
)
from salvi.web.models import (
    AccuracySummary,
    AdapterFileSlot,
    AdapterParameterDescription,
    AdapterParameterKind,
    AnalysisDescription,
    DatasetImportPreview,
    InputAdapterDescription,
    WebColumnProposal,
)
from salvi.web.providers import WebExtensionProvider
from salvi_experiments.configuration import UncertaintyConfiguration
from salvi_experiments.dataset.common import detected_memberships, ground_truth_memberships
from salvi_experiments.interop.gbic import GbicConverter
from salvi_experiments.interop.uci import (
    ClinicalColumnRole,
    UciColumnRule,
    UciConverter,
    UciImportRecipe,
    UciRepositoryClient,
    load_uci_import_recipe,
)
from salvi_experiments.metrics import calculate_accuracy

_DECIMAL_COMMA = re.compile(r"^[+-]?(?:\d+,\d*|\d*,\d+)(?:[eE][+-]?\d+)?$")


def _staged_gbic(
    files: Mapping[str, Path],
    identifier: str,
    workspace: Path,
) -> tuple[Path, Path]:
    stage = workspace / "gbic-source"
    converted = workspace / "gbic-converted"
    shutil.rmtree(stage, ignore_errors=True)
    shutil.rmtree(converted, ignore_errors=True)
    stage.mkdir(parents=True)
    suffix = files["data"].suffix.lower()
    shutil.copy2(files["data"], stage / f"{identifier}_data{suffix}")
    shutil.copy2(files["ground_truth"], stage / f"{identifier}_bics.json")
    GbicConverter().convert(stage, converted)
    return converted / identifier, converted


def _normalize_gbic_table(source: Path, destination: Path) -> None:
    delimiter = "\t" if source.suffix.lower() == ".tsv" else ","
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        source.open("r", encoding="utf-8-sig", newline="") as input_stream,
        destination.open("w", encoding="utf-8", newline="") as output_stream,
    ):
        reader = csv.reader(input_stream, delimiter=delimiter)
        writer = csv.writer(output_stream, delimiter=delimiter, lineterminator="\n")
        for row in reader:
            writer.writerow(
                [
                    value.replace(",", ".") if _DECIMAL_COMMA.fullmatch(value.strip()) else value
                    for value in row
                ]
            )


@dataclass(frozen=True, slots=True)
class GbicWebAdapter:
    @property
    def description(self) -> InputAdapterDescription:
        return InputAdapterDescription(
            name="gbic",
            title="G-Bic dataset",
            description=(
                "Imports one G-Bic matrix. An optional ground-truth JSON enables exact "
                "semantic typing and post-run accuracy without entering the search."
            ),
            files=(
                AdapterFileSlot(
                    name="data",
                    title="G-Bic matrix",
                    description="The generated _data.tsv or _data.csv matrix.",
                    accepted_extensions=(".tsv", ".csv"),
                ),
                AdapterFileSlot(
                    name="ground_truth",
                    title="Ground truth",
                    description="The optional paired _bics.json file.",
                    required=False,
                    accepted_extensions=(".json",),
                ),
            ),
            supports_ground_truth=True,
            requires_confirmation=True,
        )

    def inspect(
        self,
        files: Mapping[str, Path],
        *,
        parameters: Mapping[str, str | int | float | bool] | None = None,
        identifier: str,
        workspace: Path,
    ) -> DatasetImportPreview:
        del parameters
        normalized = normalized_identifier(identifier)
        if "ground_truth" in files:
            bundle, _ = _staged_gbic(files, normalized, workspace)
            return canonical_bundle_preview(bundle, adapter="gbic")
        normalized_table = workspace / f"{normalized}{files['data'].suffix.lower()}"
        _normalize_gbic_table(files["data"], normalized_table)
        tabular = TabularInputAdapter(
            delimiter="\t" if normalized_table.suffix == ".tsv" else ",",
            name="gbic",
            title="G-Bic",
            extension=normalized_table.suffix,
        )
        return tabular.inspect(
            {"data": normalized_table},
            parameters={},
            identifier=normalized,
            workspace=workspace,
        ).model_copy(update={"adapter": "gbic", "ground_truth_attached": False})

    def convert(
        self,
        files: Mapping[str, Path],
        *,
        identifier: str,
        columns: Sequence[WebColumnProposal],
        parameters: Mapping[str, str | int | float | bool] | None = None,
        adapter_configuration: Mapping[str, object] | None = None,
        destination: Path,
        workspace: Path,
    ) -> Path:
        del parameters, adapter_configuration
        normalized = normalized_identifier(identifier)
        if "ground_truth" in files:
            bundle, _ = _staged_gbic(files, normalized, workspace)
            shutil.copytree(bundle, destination)
            return destination
        normalized_table = workspace / f"{normalized}{files['data'].suffix.lower()}"
        _normalize_gbic_table(files["data"], normalized_table)
        return TabularInputAdapter(
            delimiter="\t" if normalized_table.suffix == ".tsv" else ",",
            name="gbic",
            title="G-Bic",
            extension=normalized_table.suffix,
        ).convert(
            {"data": normalized_table},
            identifier=normalized,
            columns=columns,
            parameters={},
            adapter_configuration={},
            destination=destination,
            workspace=workspace,
        )


@dataclass(frozen=True, slots=True)
class UciWebAdapter:
    client: UciRepositoryClient = field(default_factory=UciRepositoryClient)

    @property
    def description(self) -> InputAdapterDescription:
        return InputAdapterDescription(
            name="uci",
            title="UCI Machine Learning Repository",
            description=(
                "Fetch an official UCI dataset by identifier, inspect its clinical roles, "
                "and convert an explicit import recipe without exposing outcomes to SALVI."
            ),
            files=(
                AdapterFileSlot(
                    name="recipe",
                    title="UCI import recipe",
                    description="Optional uci-import.yaml defining clinical curation.",
                    required=False,
                    accepted_extensions=(".yaml", ".yml"),
                ),
            ),
            parameters=(
                AdapterParameterDescription(
                    name="dataset_id",
                    title="UCI dataset ID",
                    description="Numeric identifier shown on the official UCI dataset page.",
                    kind=AdapterParameterKind.INTEGER,
                    minimum=1,
                ),
            ),
            requires_confirmation=True,
        )

    def inspect(
        self,
        files: Mapping[str, Path],
        *,
        parameters: Mapping[str, str | int | float | bool] | None = None,
        identifier: str,
        workspace: Path,
    ) -> DatasetImportPreview:
        del workspace
        dataset_id = int((parameters or {})["dataset_id"])
        if "recipe" in files:
            recipe = load_uci_import_recipe(files["recipe"])
            if recipe.dataset_id != dataset_id:
                raise ValueError(
                    f"recipe dataset_id {recipe.dataset_id} does not match {dataset_id}"
                )
            metadata, data_path = self.client.fetch(dataset_id, recipe.expected_sha256)
        else:
            metadata, data_path, checksum = self.client.fetch_current(dataset_id)
            recipe = UciImportRecipe(
                identifier=normalized_identifier(identifier),
                dataset_id=dataset_id,
                expected_sha256=checksum,
            )
        variables = metadata.get("variables")
        if not isinstance(variables, list):
            raise ValueError("UCI metadata has no variable descriptions")
        rules = {item.name: item for item in recipe.columns}
        with data_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
        proposals: list[WebColumnProposal] = []
        search_columns = 0
        for index, variable in enumerate(variables):
            if not isinstance(variable, dict):
                raise ValueError("UCI variable metadata is malformed")
            source_name = str(variable["name"])
            rule = rules.get(source_name)
            role = (
                recipe.role_defaults.for_uci_role(str(variable.get("role", "")))
                if rule is None or rule.role is None
                else rule.role
            )
            annotation_kind = None if rule is None else rule.annotation_kind
            inferred_kind = (
                ColumnKind.BOOLEAN
                if str(variable.get("type", "")).lower() == "binary"
                else (
                    ColumnKind.CATEGORICAL
                    if str(variable.get("type", "")).lower() == "categorical"
                    else ColumnKind.NUMERIC
                )
            )
            selected_kind = (
                inferred_kind if rule is None or rule.search_kind is None else rule.search_kind
            )
            raw_values = [row[source_name].strip() for row in rows]
            missing_tokens = set(
                recipe.missing_tokens
                if rule is None or rule.missing_tokens is None
                else rule.missing_tokens
            )
            observed = [value for value in raw_values if value not in missing_tokens]
            if role is ClinicalColumnRole.SEARCH:
                search_columns += 1
            proposals.append(
                WebColumnProposal(
                    source_index=index,
                    name=source_name,
                    inferred_kind=inferred_kind,
                    selected_kind=selected_kind,
                    missing_ratio=1.0 - len(observed) / max(1, len(rows)),
                    sample_values=tuple(dict.fromkeys(observed[:32]))[:5],
                    is_row_identifier=role is ClinicalColumnRole.IDENTIFIER,
                    role=role.value,
                    annotation_kind=(None if annotation_kind is None else annotation_kind.value),
                    units=(
                        rule.units
                        if rule is not None and rule.units is not None
                        else (None if variable.get("units") is None else str(variable.get("units")))
                    ),
                    description=(
                        None
                        if variable.get("description") is None
                        else str(variable.get("description"))
                    ),
                )
            )
        return DatasetImportPreview(
            identifier=normalized_identifier(identifier),
            adapter="uci",
            row_count=len(rows),
            column_count=search_columns,
            columns=tuple(proposals),
            confirmation_required=True,
            clinical_annotations_attached=True,
            warnings=("Only SEARCH columns enter SALVI; outcomes and covariates remain external.",),
            adapter_configuration=recipe.model_dump(mode="json"),
        )

    def convert(
        self,
        files: Mapping[str, Path],
        *,
        identifier: str,
        columns: Sequence[WebColumnProposal],
        parameters: Mapping[str, str | int | float | bool] | None = None,
        adapter_configuration: Mapping[str, object] | None = None,
        destination: Path,
        workspace: Path,
    ) -> Path:
        del files, parameters, workspace
        if adapter_configuration is None:
            raise ValueError("UCI conversion requires its inspected import recipe")
        recipe = UciImportRecipe.model_validate(adapter_configuration)
        existing = {rule.name: rule for rule in recipe.columns}
        updated: list[UciColumnRule] = []
        for proposal in columns:
            rule = existing.get(proposal.name, UciColumnRule(name=proposal.name))
            role = (
                ClinicalColumnRole.IDENTIFIER
                if proposal.is_row_identifier
                else (None if proposal.role is None else ClinicalColumnRole(proposal.role))
            )
            updated.append(
                rule.model_copy(
                    update={
                        "role": role,
                        "search_kind": (
                            proposal.selected_kind
                            if role is ClinicalColumnRole.SEARCH
                            else rule.search_kind
                        ),
                    }
                )
            )
        effective = recipe.model_copy(
            update={
                "identifier": normalized_identifier(identifier),
                "columns": tuple(updated),
            }
        )
        clinical = UciConverter(client=self.client).convert(
            effective,
            destination / "clinical",
        )
        return clinical / "dataset"


@dataclass(frozen=True, slots=True)
class AccuracyWebAnalysis:
    @property
    def description(self) -> AnalysisDescription:
        return AnalysisDescription(
            name="prelic_accuracy",
            title="REC / REL / BE",
            description=(
                "Calculates Prelić recovery, relevance, and biclustering error against "
                "the attached canonical ground truth."
            ),
        )

    def calculate(
        self,
        *,
        dataset_bundle: Path,
        bicluster_set: Path,
    ) -> AccuracySummary:
        ground_truth = DatasetBundleReader().read_ground_truth(dataset_bundle)
        if ground_truth is None:
            raise ValueError("accuracy requires an attached canonical ground truth")
        result = calculate_accuracy(
            detected_memberships(dataset_bundle, bicluster_set),
            ground_truth_memberships(ground_truth.biclusters),
            uncertainty=UncertaintyConfiguration(bootstrap_samples=0),
            coverage_thresholds=(0.25, 0.5, 0.75),
        )
        return AccuracySummary(
            relevance=result.relevance,
            recovery=result.recovery,
            biclustering_error=result.biclustering_error,
            detected_count=result.detected_count,
            ground_truth_count=result.ground_truth_count,
            coverage=result.coverage,
        )


def create_provider() -> WebExtensionProvider:
    return WebExtensionProvider(
        adapters=(GbicWebAdapter(), UciWebAdapter()),
        analyses=(AccuracyWebAnalysis(),),
    )


__all__ = ["AccuracyWebAnalysis", "GbicWebAdapter", "UciWebAdapter", "create_provider"]
