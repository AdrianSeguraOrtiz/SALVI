"""Strict, self-contained configuration contracts for experiment protocols."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal, Self, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from salvi_experiments.exceptions import ExperimentConfigurationError

ExperimentConfiguration = TypeVar("ExperimentConfiguration", bound="FrozenExperimentModel")


class FrozenExperimentModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def resolved(self, base: Path) -> Self:
        return self


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ExperimentConfigurationError(
                f"duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}"
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def _load_yaml(path: Path) -> object:
    source = path.expanduser().resolve()
    try:
        return yaml.load(source.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except ExperimentConfigurationError:
        raise
    except (OSError, yaml.YAMLError) as error:
        raise ExperimentConfigurationError(
            f"cannot read experiment configuration {source}: {error}"
        ) from error


def load_experiment_configuration(
    path: str | Path,
    model: type[ExperimentConfiguration],
) -> ExperimentConfiguration:
    source = Path(path).expanduser().resolve()
    try:
        configuration = model.model_validate(_load_yaml(source))
    except ValidationError as error:
        raise ExperimentConfigurationError(
            f"invalid experiment configuration {source}: {error}"
        ) from error
    return configuration.resolved(source.parent)


class TaskScope(FrozenExperimentModel):
    """Ground-truth patterns included in one declared scientific task."""

    included_patterns: tuple[Literal["CONSTANT", "ADDITIVE", "MULTIPLICATIVE"], ...] = (
        "CONSTANT",
        "ADDITIVE",
        "MULTIPLICATIVE",
    )

    @model_validator(mode="after")
    def validate_patterns(self) -> Self:
        if not self.included_patterns:
            raise ValueError("task scope must include at least one pattern")
        if len(set(self.included_patterns)) != len(self.included_patterns):
            raise ValueError("task patterns must not contain duplicates")
        return self


class ObjectiveAlignmentSampling(FrozenExperimentModel):
    random_controls: Annotated[int, Field(ge=0)] = 100
    perturbations: Annotated[int, Field(ge=0)] = 20
    perturbation_ratio: Annotated[float, Field(gt=0.0, le=1.0)] = 0.10


class UncertaintyConfiguration(FrozenExperimentModel):
    bootstrap_samples: Annotated[int, Field(ge=0)] = 2000
    confidence_level: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.95
    seed: Annotated[int, Field(ge=0)] = 0


class BenchmarkExecutionConfiguration(FrozenExperimentModel):
    """How benchmark cases are orchestrated by salvi-experiments."""

    workers: Annotated[int, Field(ge=1)] = 1
    allow_nested_parallelism: bool = False
    allow_cpu_oversubscription: bool = False


class AblationExecutionConfiguration(BenchmarkExecutionConfiguration):
    """Execution and cache policy for a SALVI ablation."""

    resume: bool = True
    retry_failed: bool = False
    fail_fast: bool = False


class AblationDatasetSelection(FrozenExperimentModel):
    """Select DatasetBundles without changing their scientific contents."""

    replicates: Literal["ALL"] | tuple[Annotated[int, Field(ge=0)], ...] = "ALL"
    identifiers: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        if self.replicates != "ALL" and (
            not self.replicates or len(set(self.replicates)) != len(self.replicates)
        ):
            raise ValueError("dataset replicates must be a non-empty list without duplicates")
        if any(not identifier.strip() for identifier in self.identifiers):
            raise ValueError("dataset identifiers must not be blank")
        if len(set(self.identifiers)) != len(self.identifiers):
            raise ValueError("dataset identifiers must not contain duplicates")
        return self


class AblationPipeline(FrozenExperimentModel):
    identifier: str = Field(
        min_length=1,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    pipeline_configuration: Path


class AblationPairwiseComparison(FrozenExperimentModel):
    """One explicit paired contrast between two configured pipelines."""

    baseline_pipeline: str = Field(min_length=1)
    compared_pipeline: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_distinct_pipelines(self) -> Self:
        if self.baseline_pipeline == self.compared_pipeline:
            raise ValueError("paired comparison pipelines must be distinct")
        return self


class AblationSelector(FrozenExperimentModel):
    """Optional final selector applied offline to every completed search result."""

    identifier: str = Field(
        min_length=1,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    name: str | None = Field(default=None, min_length=1)
    parameters: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_raw_variant(self) -> Self:
        if self.name is None and self.parameters:
            raise ValueError("a raw selector variant cannot declare parameters")
        return self


class AblationMetricsConfiguration(FrozenExperimentModel):
    """Metrics calculated independently of the pipeline's observer selection."""

    artifacts: tuple[Literal["SEARCH", "FINAL"], ...] = ("SEARCH", "FINAL")
    coverage_thresholds: tuple[Annotated[float, Field(gt=0.0, le=1.0)], ...] = (
        0.25,
        0.50,
        0.75,
    )
    case_uncertainty: UncertaintyConfiguration = Field(
        default_factory=lambda: UncertaintyConfiguration(bootstrap_samples=0)
    )
    aggregate_uncertainty: UncertaintyConfiguration = Field(
        default_factory=UncertaintyConfiguration
    )
    structural_row_weight: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    diversity_sample_size: Annotated[int, Field(ge=2)] = 1000
    paired_analysis_unit: Literal["RUN", "DATASET"] = "RUN"
    paired_seed_aggregation: Literal["MEAN", "MEDIAN"] = "MEAN"

    @model_validator(mode="after")
    def validate_metrics(self) -> Self:
        if not self.artifacts or len(set(self.artifacts)) != len(self.artifacts):
            raise ValueError("ablation artifacts must be a non-empty list without duplicates")
        if not self.coverage_thresholds:
            raise ValueError("ablation accuracy requires at least one coverage threshold")
        if tuple(sorted(set(self.coverage_thresholds))) != self.coverage_thresholds:
            raise ValueError("coverage thresholds must be sorted and unique")
        return self


class SalviAblationConfiguration(FrozenExperimentModel):
    """Run multiple complete SALVI pipelines over the same benchmark."""

    schema_version: Literal[1] = 1
    identifier: str = Field(min_length=1)
    benchmark_root: Path
    datasets: AblationDatasetSelection = Field(default_factory=AblationDatasetSelection)
    pattern_binding: Literal["PIPELINE", "GROUND_TRUTH"]
    pipelines: Annotated[tuple[AblationPipeline, ...], Field(min_length=1)]
    paired_comparisons: tuple[AblationPairwiseComparison, ...] = ()
    selectors: tuple[AblationSelector, ...] = ()
    run_seeds: Annotated[tuple[Annotated[int, Field(ge=0)], ...], Field(min_length=1)] = (0,)
    task: TaskScope = Field(default_factory=TaskScope)
    metrics: AblationMetricsConfiguration = Field(default_factory=AblationMetricsConfiguration)
    execution: AblationExecutionConfiguration = Field(
        default_factory=AblationExecutionConfiguration
    )
    output_directory: Path

    @model_validator(mode="after")
    def validate_ablation(self) -> Self:
        identifiers = tuple(pipeline.identifier for pipeline in self.pipelines)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("ablation pipeline identifiers must be unique")
        configured = set(identifiers)
        comparison_pairs = tuple(
            (item.baseline_pipeline, item.compared_pipeline) for item in self.paired_comparisons
        )
        if len(set(comparison_pairs)) != len(comparison_pairs):
            raise ValueError("ablation paired comparisons must be unique")
        unknown = sorted(
            {
                identifier
                for pair in comparison_pairs
                for identifier in pair
                if identifier not in configured
            }
        )
        if unknown:
            raise ValueError(
                "ablation paired comparisons reference unknown pipelines: " + ", ".join(unknown)
            )
        selector_identifiers = tuple(selector.identifier for selector in self.selectors)
        if len(set(selector_identifiers)) != len(selector_identifiers):
            raise ValueError("ablation selector identifiers must be unique")
        if len(set(self.run_seeds)) != len(self.run_seeds):
            raise ValueError("ablation run seeds must not contain duplicates")
        return self

    def resolved(self, base: Path) -> Self:
        return self.model_copy(
            update={
                "benchmark_root": _resolve(self.benchmark_root, base),
                "pipelines": tuple(
                    pipeline.model_copy(
                        update={
                            "pipeline_configuration": _resolve(
                                pipeline.pipeline_configuration,
                                base,
                            )
                        }
                    )
                    for pipeline in self.pipelines
                ),
                "output_directory": _resolve(self.output_directory, base),
            }
        )


class ObjectiveAlignmentConfiguration(FrozenExperimentModel):
    schema_version: Literal[1] = 1
    identifier: str = Field(min_length=1)
    pipeline_configuration: Path
    dataset_bundle: Path
    output_directory: Path
    analysis_seed: Annotated[int, Field(ge=0)] = 0
    task: TaskScope = Field(default_factory=TaskScope)
    sampling: ObjectiveAlignmentSampling = Field(default_factory=ObjectiveAlignmentSampling)
    overwrite: bool = False

    def resolved(self, base: Path) -> Self:
        return self.model_copy(
            update={
                "pipeline_configuration": _resolve(self.pipeline_configuration, base),
                "dataset_bundle": _resolve(self.dataset_bundle, base),
                "output_directory": _resolve(self.output_directory, base),
            }
        )


class AlgorithmRunInformation(FrozenExperimentModel):
    """Information available to and resources consumed by one algorithm run."""

    algorithm: str = Field(min_length=1)
    version: str | None = None
    target_count_known: bool = False
    target_count_value: Annotated[int, Field(ge=1)] | None = None
    evaluation_budget: Annotated[int, Field(ge=1)] | None = None
    wall_clock_budget_seconds: Annotated[float, Field(gt=0.0)] | None = None
    wall_time_seconds: Annotated[float, Field(ge=0.0)] | None = None
    cpu_time_seconds: Annotated[float, Field(ge=0.0)] | None = None
    peak_memory_bytes: Annotated[int, Field(ge=0)] | None = None
    postprocessing_policy: str = Field(min_length=1)
    final_selection_policy: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_target_count(self) -> Self:
        if self.target_count_known != (self.target_count_value is not None):
            raise ValueError(
                "target_count_value must be present exactly when target_count_known is true"
            )
        return self


class AccuracyConfiguration(FrozenExperimentModel):
    schema_version: Literal[1] = 1
    identifier: str = Field(min_length=1)
    dataset_bundle: Path
    bicluster_set: Path
    output_directory: Path
    run_configuration: Path | None = None
    task: TaskScope = Field(default_factory=TaskScope)
    algorithm: AlgorithmRunInformation
    uncertainty: UncertaintyConfiguration = Field(default_factory=UncertaintyConfiguration)
    coverage_thresholds: tuple[Annotated[float, Field(gt=0.0, le=1.0)], ...] = (
        0.25,
        0.50,
        0.75,
    )
    overwrite: bool = False

    @model_validator(mode="after")
    def validate_thresholds(self) -> Self:
        if not self.coverage_thresholds:
            raise ValueError("accuracy requires at least one coverage threshold")
        if tuple(sorted(set(self.coverage_thresholds))) != self.coverage_thresholds:
            raise ValueError("coverage thresholds must be sorted and unique")
        return self

    def resolved(self, base: Path) -> Self:
        return self.model_copy(
            update={
                "dataset_bundle": _resolve(self.dataset_bundle, base),
                "bicluster_set": _resolve(self.bicluster_set, base),
                "output_directory": _resolve(self.output_directory, base),
                "run_configuration": (
                    None
                    if self.run_configuration is None
                    else _resolve(self.run_configuration, base)
                ),
            }
        )


class ObjectiveAlignmentBenchmarkCase(FrozenExperimentModel):
    identifier: str = Field(min_length=1)
    pipeline_configuration: Path
    dataset_bundle: Path


class ObjectiveAlignmentBenchmarkConfiguration(FrozenExperimentModel):
    schema_version: Literal[1] = 1
    identifier: str = Field(min_length=1)
    cases: Annotated[tuple[ObjectiveAlignmentBenchmarkCase, ...], Field(min_length=1)]
    output_directory: Path
    analysis_seed: Annotated[int, Field(ge=0)] = 0
    task: TaskScope = Field(default_factory=TaskScope)
    sampling: ObjectiveAlignmentSampling = Field(default_factory=ObjectiveAlignmentSampling)
    uncertainty: UncertaintyConfiguration = Field(default_factory=UncertaintyConfiguration)
    execution: BenchmarkExecutionConfiguration = Field(
        default_factory=BenchmarkExecutionConfiguration
    )
    overwrite: bool = False

    @model_validator(mode="after")
    def validate_cases(self) -> Self:
        identifiers = tuple(case.identifier for case in self.cases)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("benchmark case identifiers must be unique")
        return self

    def resolved(self, base: Path) -> Self:
        return self.model_copy(
            update={
                "cases": tuple(
                    case.model_copy(
                        update={
                            "pipeline_configuration": _resolve(case.pipeline_configuration, base),
                            "dataset_bundle": _resolve(case.dataset_bundle, base),
                        }
                    )
                    for case in self.cases
                ),
                "output_directory": _resolve(self.output_directory, base),
            }
        )


class AccuracyBenchmarkCase(FrozenExperimentModel):
    identifier: str = Field(min_length=1)
    dataset_bundle: Path
    bicluster_set: Path
    run_configuration: Path | None = None
    algorithm: AlgorithmRunInformation


class AccuracyBenchmarkConfiguration(FrozenExperimentModel):
    schema_version: Literal[1] = 1
    identifier: str = Field(min_length=1)
    cases: Annotated[tuple[AccuracyBenchmarkCase, ...], Field(min_length=1)]
    output_directory: Path
    task: TaskScope = Field(default_factory=TaskScope)
    uncertainty: UncertaintyConfiguration = Field(default_factory=UncertaintyConfiguration)
    coverage_thresholds: tuple[Annotated[float, Field(gt=0.0, le=1.0)], ...] = (
        0.25,
        0.50,
        0.75,
    )
    execution: BenchmarkExecutionConfiguration = Field(
        default_factory=BenchmarkExecutionConfiguration
    )
    overwrite: bool = False

    @model_validator(mode="after")
    def validate_cases_and_thresholds(self) -> Self:
        identifiers = tuple(case.identifier for case in self.cases)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("benchmark case identifiers must be unique")
        algorithms = {case.algorithm.algorithm for case in self.cases}
        if len(algorithms) != 1:
            raise ValueError("one accuracy benchmark must contain cases from exactly one algorithm")
        if not self.coverage_thresholds:
            raise ValueError("accuracy requires at least one coverage threshold")
        if tuple(sorted(set(self.coverage_thresholds))) != self.coverage_thresholds:
            raise ValueError("coverage thresholds must be sorted and unique")
        return self

    def resolved(self, base: Path) -> Self:
        return self.model_copy(
            update={
                "cases": tuple(
                    case.model_copy(
                        update={
                            "dataset_bundle": _resolve(case.dataset_bundle, base),
                            "bicluster_set": _resolve(case.bicluster_set, base),
                            "run_configuration": (
                                None
                                if case.run_configuration is None
                                else _resolve(case.run_configuration, base)
                            ),
                        }
                    )
                    for case in self.cases
                ),
                "output_directory": _resolve(self.output_directory, base),
            }
        )


class ComparisonAlgorithm(FrozenExperimentModel):
    identifier: str = Field(min_length=1)
    accuracy_results: Annotated[tuple[Path, ...], Field(min_length=1)]
    replicate_aggregation: Literal["ERROR", "MEAN", "MEDIAN"] = "ERROR"


class ComparisonConfiguration(FrozenExperimentModel):
    schema_version: Literal[1] = 1
    identifier: str = Field(min_length=1)
    algorithms: Annotated[tuple[ComparisonAlgorithm, ...], Field(min_length=2)]
    output_directory: Path
    uncertainty: UncertaintyConfiguration = Field(default_factory=UncertaintyConfiguration)
    overwrite: bool = False

    @model_validator(mode="after")
    def validate_algorithms(self) -> Self:
        identifiers = tuple(algorithm.identifier for algorithm in self.algorithms)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("comparison algorithm identifiers must be unique")
        return self

    def resolved(self, base: Path) -> Self:
        return self.model_copy(
            update={
                "algorithms": tuple(
                    algorithm.model_copy(
                        update={
                            "accuracy_results": tuple(
                                _resolve(path, base) for path in algorithm.accuracy_results
                            )
                        }
                    )
                    for algorithm in self.algorithms
                ),
                "output_directory": _resolve(self.output_directory, base),
            }
        )


def _resolve(path: Path, base: Path) -> Path:
    expanded = path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (base / expanded).resolve()


__all__ = [
    "AblationDatasetSelection",
    "AblationExecutionConfiguration",
    "AblationMetricsConfiguration",
    "AblationPipeline",
    "AblationSelector",
    "AccuracyBenchmarkCase",
    "AccuracyBenchmarkConfiguration",
    "AccuracyConfiguration",
    "AlgorithmRunInformation",
    "BenchmarkExecutionConfiguration",
    "ComparisonAlgorithm",
    "ComparisonConfiguration",
    "FrozenExperimentModel",
    "ObjectiveAlignmentBenchmarkCase",
    "ObjectiveAlignmentBenchmarkConfiguration",
    "ObjectiveAlignmentConfiguration",
    "ObjectiveAlignmentSampling",
    "SalviAblationConfiguration",
    "TaskScope",
    "UncertaintyConfiguration",
    "load_experiment_configuration",
]
