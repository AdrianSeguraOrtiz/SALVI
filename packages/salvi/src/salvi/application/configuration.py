"""Versioned pipeline specifications and concrete SALVI run bindings."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from salvi.exceptions import ConfigurationError
from salvi.infrastructure.yaml import (
    dump_yaml,
    dump_yaml_text,
    load_strict_yaml,
    load_strict_yaml_text,
)
from salvi.patterns.configuration import PatternConfiguration


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ComponentSpec(StrictModel):
    name: str = Field(min_length=1)
    parameters: dict[str, object] = Field(default_factory=dict)


class RunConfiguration(StrictModel):
    identifier: str = Field(min_length=1)
    seed: Annotated[int, Field(ge=0)] = 0
    resume_from_checkpoint: Path | None = None


class DatasetConfiguration(StrictModel):
    bundle: Path


class PreprocessingConfiguration(StrictModel):
    source_column_filters: tuple[ComponentSpec, ...] = ()
    missing_values: ComponentSpec
    column_augmentations: tuple[ComponentSpec, ...] = ()
    numeric_transformations: tuple[ComponentSpec, ...] = ()


class EvaluationConfiguration(StrictModel):
    candidate_validity: ComponentSpec
    observed_support: ComponentSpec


class SearchConfiguration(StrictModel):
    engine: ComponentSpec
    objectives: Annotated[tuple[ComponentSpec, ...], Field(min_length=1)]
    constraints: tuple[ComponentSpec, ...] = ()
    descriptors: tuple[ComponentSpec, ...] = ()
    archive: ComponentSpec | None = None
    parent_selection: ComponentSpec | None = None
    mate_selection: ComponentSpec | None = None
    crossover: ComponentSpec | None = None
    mutation: ComponentSpec | None = None
    initialization: ComponentSpec
    emitters: tuple[ComponentSpec, ...] = ()
    scheduler: ComponentSpec | None = None
    termination: ComponentSpec


class ExecutionConfiguration(StrictModel):
    executor: ComponentSpec
    workers: Annotated[int, Field(ge=1)] = 1
    cancellation_grace_seconds: Annotated[float, Field(gt=0.0)] = 5.0


class MonitoringConfiguration(StrictModel):
    queue_capacity: Annotated[int, Field(ge=8)] = 1024
    checkpoint_interval_evaluations: Annotated[int, Field(ge=1)] | None = None
    observers: tuple[ComponentSpec, ...] = ()


class OutputConfiguration(StrictModel):
    directory: Path
    overwrite: bool = False


class PipelineConfiguration(StrictModel):
    """Reusable component pipeline, deliberately independent of a dataset or run."""

    schema_version: Literal[1]
    patterns: PatternConfiguration = Field(default_factory=PatternConfiguration)
    preprocessing: PreprocessingConfiguration
    evaluation: EvaluationConfiguration
    search: SearchConfiguration
    execution: ExecutionConfiguration
    monitoring: MonitoringConfiguration = Field(default_factory=MonitoringConfiguration)
    final_selection: ComponentSpec | None = None

    @model_validator(mode="after")
    def validate_component_names(self) -> Self:
        named_collections = {
            "objectives": self.search.objectives,
            "constraints": self.search.constraints,
            "descriptors": self.search.descriptors,
            "emitters": self.search.emitters,
        }
        for label, specifications in named_collections.items():
            names = tuple(specification.name for specification in specifications)
            if len(set(names)) != len(names):
                raise ValueError(f"{label} must not contain duplicate component names")
        return self


class RunBinding(StrictModel):
    """Concrete data, identity, and output selected when launching a pipeline."""

    identifier: str = Field(min_length=1)
    dataset_bundle: Path
    output_directory: Path
    seed: Annotated[int, Field(ge=0)] = 0
    resume_from_checkpoint: Path | None = None
    overwrite: bool = False

    def resolved(self, base_directory: Path) -> Self:
        base = base_directory.resolve()

        def resolve(path: Path) -> Path:
            return path.resolve() if path.is_absolute() else (base / path).resolve()

        return self.model_copy(
            update={
                "dataset_bundle": resolve(self.dataset_bundle),
                "output_directory": resolve(self.output_directory),
                "resume_from_checkpoint": (
                    None
                    if self.resume_from_checkpoint is None
                    else resolve(self.resume_from_checkpoint)
                ),
            }
        )


class SalviConfiguration(PipelineConfiguration):
    """Fully bound configuration persisted with an executed SALVI run.

    Source YAML files supplied by users are :class:`PipelineConfiguration` documents.
    This model is the reproducibility artifact obtained after binding one pipeline to a
    dataset, run identity, seed, and output directory.
    """

    run: RunConfiguration
    dataset: DatasetConfiguration
    output: OutputConfiguration

    def resolved(self, base_directory: Path) -> Self:
        base = base_directory.resolve()

        def resolve(path: Path) -> Path:
            return path.resolve() if path.is_absolute() else (base / path).resolve()

        return self.model_copy(
            update={
                "run": self.run.model_copy(
                    update={
                        "resume_from_checkpoint": (
                            None
                            if self.run.resume_from_checkpoint is None
                            else resolve(self.run.resume_from_checkpoint)
                        )
                    }
                ),
                "dataset": self.dataset.model_copy(update={"bundle": resolve(self.dataset.bundle)}),
                "output": self.output.model_copy(
                    update={"directory": resolve(self.output.directory)}
                ),
            }
        )


class LoadedPipelineConfiguration(StrictModel):
    source: Path
    pipeline: PipelineConfiguration


class LoadedConfiguration(StrictModel):
    """A fully bound effective configuration, typically read from a run artifact."""

    source: Path
    configuration: SalviConfiguration


class LoadedRunConfiguration(StrictModel):
    """A pipeline together with the resolved binding used to execute it."""

    source: Path
    pipeline: PipelineConfiguration
    binding: RunBinding
    configuration: SalviConfiguration


def load_pipeline_configuration(path: str | Path) -> LoadedPipelineConfiguration:
    source = Path(path).expanduser().resolve()
    raw = load_strict_yaml(source)
    try:
        pipeline = PipelineConfiguration.model_validate(raw)
    except ValidationError as error:
        raise ConfigurationError(f"invalid SALVI pipeline {source}: {error}") from error
    return LoadedPipelineConfiguration(source=source, pipeline=pipeline)


def parse_pipeline_configuration(
    content: str,
    *,
    source: str = "uploaded pipeline",
) -> PipelineConfiguration:
    """Validate a reusable pipeline supplied by an API or GUI."""

    raw = load_strict_yaml_text(content, source=source)
    try:
        return PipelineConfiguration.model_validate(raw)
    except ValidationError as error:
        raise ConfigurationError(f"invalid SALVI pipeline {source}: {error}") from error


def serialize_pipeline_configuration(
    configuration: PipelineConfiguration,
    *,
    compact: bool = True,
) -> str:
    """Serialize a reusable document accepted by :func:`load_pipeline_configuration`.

    Compact source YAML omits defaults and empty optional roles. Effective run artifacts
    remain fully expanded so their meaning never depends on defaults from a later release.
    """

    if not compact:
        return dump_yaml_text(configuration.model_dump(mode="json"))
    raw = configuration.model_dump(
        mode="json",
        exclude_defaults=True,
        exclude_none=True,
    )
    pattern_options = configuration.patterns.model_dump(
        mode="json",
        exclude_defaults=True,
        exclude_none=True,
        exclude={"allowed"},
    )
    patterns = {
        "allowed": [pattern.value for pattern in configuration.patterns.allowed],
        **pattern_options,
    }
    remainder = {
        key: value for key, value in raw.items() if key not in {"schema_version", "patterns"}
    }
    return dump_yaml_text(
        {
            "schema_version": configuration.schema_version,
            "patterns": patterns,
            **remainder,
        }
    )


def bind_pipeline(
    pipeline: PipelineConfiguration,
    binding: RunBinding,
    *,
    base_directory: Path | None = None,
) -> SalviConfiguration:
    """Create one immutable effective run configuration from reusable inputs."""

    resolved_binding = binding.resolved(base_directory or Path.cwd())
    return SalviConfiguration.model_validate(
        {
            **pipeline.model_dump(mode="python"),
            "run": {
                "identifier": resolved_binding.identifier,
                "seed": resolved_binding.seed,
                "resume_from_checkpoint": resolved_binding.resume_from_checkpoint,
            },
            "dataset": {"bundle": resolved_binding.dataset_bundle},
            "output": {
                "directory": resolved_binding.output_directory,
                "overwrite": resolved_binding.overwrite,
            },
        }
    )


def load_bound_configuration(
    pipeline_path: str | Path,
    binding: RunBinding,
    *,
    binding_base_directory: Path | None = None,
) -> LoadedRunConfiguration:
    loaded = load_pipeline_configuration(pipeline_path)
    resolved_binding = binding.resolved(binding_base_directory or Path.cwd())
    return LoadedRunConfiguration(
        source=loaded.source,
        pipeline=loaded.pipeline,
        binding=resolved_binding,
        configuration=bind_pipeline(loaded.pipeline, resolved_binding),
    )


def load_configuration(path: str | Path) -> LoadedConfiguration:
    """Load a fully bound effective configuration from a run artifact.

    New source configurations must be loaded with :func:`load_pipeline_configuration`
    and combined with :class:`RunBinding` at launch time.
    """

    source = Path(path).expanduser().resolve()
    raw = load_strict_yaml(source)
    try:
        configuration = SalviConfiguration.model_validate(raw).resolved(source.parent)
    except ValidationError as error:
        raise ConfigurationError(f"invalid SALVI configuration {source}: {error}") from error
    return LoadedConfiguration(source=source, configuration=configuration)


def write_effective_configuration(configuration: SalviConfiguration, destination: Path) -> None:
    dump_yaml(configuration.model_dump(mode="json"), destination)
