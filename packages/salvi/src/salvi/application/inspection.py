"""Read-only inspection of a concrete scientific pipeline."""

from __future__ import annotations

from pathlib import Path

from salvi.api.run import RunSpecification
from salvi.application.configuration import RunBinding, load_bound_configuration
from salvi.application.factory import build_specification, prepare_run
from salvi.components.protocols import CellTargetArchive, Component, ComponentKind
from salvi.components.registry import ComponentRegistry
from salvi.domain.models import FrozenModel
from salvi.domain.search import TerminationProgress


class InspectedComponent(FrozenModel):
    role: ComponentKind
    names: tuple[str, ...]


class InspectedDescriptor(FrozenModel):
    name: str
    minimum: float
    maximum: float
    value_kind: str
    recommended_binning: str


class PipelineInspection(FrozenModel):
    """Dataset-dependent shape and composition without starting a search."""

    dataset_identifier: str
    source_rows: int
    source_columns: int
    prepared_columns: int
    source_missing_values: int
    unavailable_values: int
    imputed_values: int
    memory_bytes: int
    preprocessing_steps: tuple[str, ...]
    allowed_patterns: tuple[str, ...]
    components: tuple[InspectedComponent, ...]
    descriptors: tuple[InspectedDescriptor, ...]
    reachable_archive_cells: int | None
    termination: TerminationProgress


def inspect_pipeline(
    path: str | Path,
    *,
    dataset_bundle: Path,
    seed: int = 0,
    registry: ComponentRegistry | None = None,
) -> PipelineInspection:
    """Inspect one pipeline using the real loader, preprocessors, and archive."""

    source = Path(path).expanduser().resolve()
    loaded = load_bound_configuration(
        source,
        RunBinding(
            identifier=f"inspect-{source.stem}",
            dataset_bundle=dataset_bundle,
            output_directory=Path.cwd() / ".salvi-inspect-unused",
            seed=seed,
        ),
    )
    specification = build_specification(loaded.configuration, registry)
    try:
        prepared = prepare_run(specification)
        archive = specification.archive
        reachable_cells: int | None = None
        if archive is not None:
            archive.initialize(
                prepared.context,
                specification.objectives,
                specification.descriptors,
                specification.constraints,
            )
            if isinstance(archive, CellTargetArchive):
                reachable_cells = len(archive.cell_targets())

        descriptor_details = tuple(
            InspectedDescriptor(
                name=descriptor.component_name,
                minimum=(domain := descriptor.domain(prepared.context)).minimum,
                maximum=domain.maximum,
                value_kind=domain.value_kind.value,
                recommended_binning=domain.recommended_binning.value,
            )
            for descriptor in specification.descriptors
        )
        components = _component_summary(specification)
        progress = specification.termination.progress(0)
        return PipelineInspection(
            dataset_identifier=prepared.context.dataset.metadata.identifier,
            source_rows=prepared.context.dataset.row_count,
            source_columns=prepared.context.dataset.source_column_count,
            prepared_columns=prepared.context.dataset.column_count,
            source_missing_values=prepared.context.dataset.missing_count,
            unavailable_values=prepared.context.dataset.unavailable_count,
            imputed_values=prepared.context.dataset.imputed_count,
            memory_bytes=prepared.preprocessing.final_memory_bytes,
            preprocessing_steps=tuple(step.component_name for step in prepared.preprocessing.steps),
            allowed_patterns=tuple(pattern.value for pattern in specification.patterns.allowed),
            components=components,
            descriptors=descriptor_details,
            reachable_archive_cells=reachable_cells,
            termination=progress,
        )
    finally:
        specification.executor.close()


def _component_summary(specification: RunSpecification) -> tuple[InspectedComponent, ...]:
    roles: tuple[tuple[ComponentKind, tuple[Component, ...]], ...] = (
        (ComponentKind.SOURCE_COLUMN_FILTER, specification.source_column_filters),
        (ComponentKind.MISSING_VALUES_POLICY, (specification.missing_values_policy,)),
        (ComponentKind.COLUMN_AUGMENTATION, specification.column_augmentations),
        (ComponentKind.NUMERIC_TRANSFORMATION, specification.numeric_transformations),
        (ComponentKind.CANDIDATE_VALIDITY_POLICY, (specification.candidate_validity_policy,)),
        (ComponentKind.EVALUATION_SUPPORT_POLICY, (specification.evaluation_support_policy,)),
        (ComponentKind.OBJECTIVE, specification.objectives),
        (ComponentKind.CONSTRAINT, specification.constraints),
        (ComponentKind.DESCRIPTOR, specification.descriptors),
        (ComponentKind.INITIALIZER, (specification.initializer,)),
        (ComponentKind.ARCHIVE, _optional(specification.archive)),
        (ComponentKind.PARENT_SELECTION_POLICY, _optional(specification.parent_selection_policy)),
        (ComponentKind.MATE_SELECTION_POLICY, _optional(specification.mate_selection_policy)),
        (ComponentKind.CROSSOVER_OPERATOR, _optional(specification.crossover_operator)),
        (ComponentKind.MUTATION_OPERATOR, _optional(specification.mutation_operator)),
        (ComponentKind.EMITTER, specification.emitters),
        (ComponentKind.SCHEDULER, _optional(specification.scheduler)),
        (ComponentKind.SEARCH_ENGINE, (specification.search_engine,)),
        (ComponentKind.EVALUATION_EXECUTOR, (specification.executor,)),
        (ComponentKind.OBSERVER, specification.observers),
        (ComponentKind.TERMINATION, (specification.termination,)),
        (ComponentKind.FINAL_SELECTOR, _optional(specification.final_selector)),
    )
    return tuple(
        InspectedComponent(
            role=kind,
            names=tuple(component.component_name for component in components),
        )
        for kind, components in roles
        if components
    )


def _optional(component: Component | None) -> tuple[Component, ...]:
    return () if component is None else (component,)


__all__ = ["InspectedComponent", "InspectedDescriptor", "PipelineInspection", "inspect_pipeline"]
