from __future__ import annotations

from pathlib import Path

import pytest

from salvi.application.context import RunContext
from salvi.components.final_selection import (
    ContainmentMarginalQualityConfiguration,
    ContainmentMarginalQualitySelector,
)
from salvi.components.residual_selection import (
    AdaptiveResidualEvidenceCoverConfiguration,
    AdaptiveResidualEvidenceCoverSelector,
)
from salvi.domain import (
    Bicluster,
    Candidate,
    ColumnObjectiveValue,
    Evaluation,
    EvaluationIssue,
    EvaluationIssueCode,
    NamedValue,
    ObjectiveDirection,
    ObjectiveValue,
    Repertoire,
)
from salvi.exceptions import ComponentError
from salvi.infrastructure.bicluster_set import BiclusterSetReader, BiclusterSetWriter


def _evaluation(
    identifier: str,
    rows: tuple[int, ...],
    columns: tuple[int, ...],
    coherence: float,
    contrast: float,
    *,
    coordinate: tuple[int, ...] | None = None,
    invalid: bool = False,
) -> Evaluation:
    return Evaluation(
        candidate=Candidate(
            identifier=identifier,
            bicluster=Bicluster(row_indices=rows, column_indices=columns),
        ),
        objectives=(
            ObjectiveValue(
                name="internal_coherence",
                value=coherence,
                direction=ObjectiveDirection.MINIMIZE,
            ),
            ObjectiveValue(
                name="contrast",
                value=contrast,
                direction=ObjectiveDirection.MAXIMIZE,
            ),
        ),
        descriptors=(
            NamedValue(name="row_cardinality", value=float(len(rows))),
            NamedValue(name="column_cardinality", value=float(len(columns))),
        ),
        issues=(
            (
                EvaluationIssue(
                    code=EvaluationIssueCode.INVALID_STRUCTURE,
                    message="invalid test candidate",
                ),
            )
            if invalid
            else ()
        ),
        archive_coordinate=coordinate,
    )


def _explained_evaluation(
    identifier: str,
    rows: tuple[int, ...],
    columns: tuple[int, ...],
    quality: float,
) -> Evaluation:
    return Evaluation(
        candidate=Candidate(
            identifier=identifier,
            bicluster=Bicluster(row_indices=rows, column_indices=columns),
        ),
        objectives=(
            ObjectiveValue(
                name="internal_coherence",
                value=1.0 - quality,
                direction=ObjectiveDirection.MINIMIZE,
                columns=tuple(
                    ColumnObjectiveValue(column_index=column, value=1.0 - quality)
                    for column in columns
                ),
            ),
            ObjectiveValue(
                name="contrast",
                value=quality,
                direction=ObjectiveDirection.MAXIMIZE,
                columns=tuple(
                    ColumnObjectiveValue(column_index=column, value=quality) for column in columns
                ),
            ),
        ),
        descriptors=(
            NamedValue(name="row_cardinality", value=float(len(rows))),
            NamedValue(name="column_cardinality", value=float(len(columns))),
        ),
    )


def test_adaptive_residual_selector_keeps_complementary_evidence_and_deduplicates(
    run_context: RunContext,
) -> None:
    repertoire = Repertoire(
        evaluations=(
            _explained_evaluation("left", (0, 1), (0, 1), 0.95),
            _explained_evaluation("left-duplicate", (0, 1), (0, 1), 0.90),
            _explained_evaluation("right", (2, 3), (1, 2), 0.92),
        )
    )
    dense = AdaptiveResidualEvidenceCoverSelector(
        objective_names=("internal_coherence", "contrast"),
        complexity_penalty=0.0,
        minimum_marginal_evidence=0.0,
        maximum_dense_cells=1_000,
        minimum_quality_floor=0.5,
        maximum_quality_floor=0.5,
    ).select(run_context, repertoire)
    sparse = AdaptiveResidualEvidenceCoverSelector(
        objective_names=("internal_coherence", "contrast"),
        complexity_penalty=0.0,
        minimum_marginal_evidence=0.0,
        maximum_dense_cells=1,
        minimum_quality_floor=0.5,
        maximum_quality_floor=0.5,
    ).select(run_context, repertoire)

    expected = ("left", "right")
    assert tuple(item.candidate.identifier for item in dense.evaluations) == expected
    assert tuple(item.candidate.identifier for item in sparse.evaluations) == expected
    assert tuple(
        item.final_selection.selection_rank
        for item in dense.evaluations
        if item.final_selection is not None
    ) == (0, 1)
    assert dense.evaluations[0].final_selection is not None
    assert dense.evaluations[0].final_selection.source_candidate_identifiers == (
        "left",
        "left-duplicate",
    )


def test_adaptive_residual_selector_configuration_is_strict() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        AdaptiveResidualEvidenceCoverConfiguration(
            minimum_quality_floor=0.8,
            maximum_quality_floor=0.7,
        )
    with pytest.raises(ValueError, match="unique"):
        AdaptiveResidualEvidenceCoverConfiguration(objective_names=("contrast", "contrast"))
    with pytest.raises(ValueError, match="Extra inputs"):
        AdaptiveResidualEvidenceCoverConfiguration.model_validate({"unknown": True})


def test_adaptive_residual_selector_requires_scaled_objectives_in_unit_interval(
    run_context: RunContext,
) -> None:
    unscaled = _evaluation("unscaled", (0, 1), (0, 1), 2.0, 3.0)
    with pytest.raises(ComponentError, match="unit-interval objective values"):
        AdaptiveResidualEvidenceCoverSelector().select(
            run_context,
            Repertoire(evaluations=(unscaled,)),
        )

    selected = AdaptiveResidualEvidenceCoverSelector(
        quality_scale="empirical",
        complexity_penalty=0.0,
        minimum_marginal_evidence=0.0,
    ).select(run_context, Repertoire(evaluations=(unscaled,)))
    assert tuple(item.candidate.identifier for item in selected.evaluations) == ("unscaled",)


def test_containment_selector_stops_at_the_first_material_quality_loss(
    run_context: RunContext,
) -> None:
    repertoire = Repertoire(
        evaluations=(
            _evaluation("small", (0, 1), (0, 1), 0.0, 1.0, coordinate=(0, 0)),
            _evaluation(
                "elbow",
                (0, 1, 2),
                (0, 1, 2),
                0.05,
                0.95,
                coordinate=(1, 1),
            ),
            _evaluation(
                "too-large",
                (0, 1, 2, 3),
                (0, 1, 2, 3),
                0.8,
                0.2,
                coordinate=(2, 2),
            ),
        )
    )

    selected = ContainmentMarginalQualitySelector(
        max_objective_degradation=0.2,
        max_degradation_per_log_area_gain=0.2,
    ).select(run_context, repertoire)

    assert tuple(item.candidate.identifier for item in selected.evaluations) == ("elbow",)
    provenance = selected.evaluations[0].final_selection
    assert provenance is not None
    assert provenance.selector == "containment_marginal_quality"
    assert provenance.source_candidate_identifiers == ("elbow", "small")
    assert provenance.source_archive_coordinates == ((0, 0), (1, 1))


def test_containment_selector_preserves_unrelated_structural_branches(
    run_context: RunContext,
) -> None:
    selected = ContainmentMarginalQualitySelector().select(
        run_context,
        Repertoire(
            evaluations=(
                _evaluation("left", (0, 1), (0, 1), 0.1, 0.9, coordinate=(0, 0)),
                _evaluation("right", (4, 5), (4, 5), 0.1, 0.9, coordinate=(0, 0)),
            )
        ),
    )

    assert {item.candidate.identifier for item in selected.evaluations} == {
        "left",
        "right",
    }


def test_containment_selector_filters_invalid_candidates_and_handles_empty_input(
    run_context: RunContext,
) -> None:
    selector = ContainmentMarginalQualitySelector()
    assert selector.select(run_context, Repertoire()) == Repertoire()
    invalid = _evaluation("invalid", (0, 1), (0, 1), 0.0, 1.0, invalid=True)
    assert (
        selector.select(
            run_context,
            Repertoire(evaluations=(invalid,)),
        )
        == Repertoire()
    )


def test_containment_selector_rejects_inconsistent_objective_schemas(
    run_context: RunContext,
) -> None:
    first = _evaluation("first", (0, 1), (0, 1), 0.1, 0.9)
    second = _evaluation("second", (2, 3), (2, 3), 0.2, 0.8).model_copy(
        update={
            "objectives": (
                ObjectiveValue(
                    name="different",
                    value=0.2,
                    direction=ObjectiveDirection.MINIMIZE,
                ),
            )
        }
    )
    with pytest.raises(ComponentError, match="consistent objective schema"):
        ContainmentMarginalQualitySelector().select(
            run_context,
            Repertoire(evaluations=(first, second)),
        )


def test_containment_selector_configuration_is_strict() -> None:
    with pytest.raises(ValueError, match="less than or equal to 1"):
        ContainmentMarginalQualityConfiguration(max_objective_degradation=1.1)
    with pytest.raises(ValueError, match="Extra inputs"):
        ContainmentMarginalQualityConfiguration.model_validate({"unknown": True})
    with pytest.raises(ValueError, match="non-empty"):
        ContainmentMarginalQualityConfiguration(objective_names=())
    with pytest.raises(ValueError, match="unique"):
        ContainmentMarginalQualityConfiguration(
            objective_names=("internal_coherence", "internal_coherence")
        )


def test_containment_selector_can_use_an_explicit_objective_subset(
    run_context: RunContext,
) -> None:
    preferred = _evaluation("preferred", (0, 1), (0, 1), 0.1, 0.9)
    size_biased = _evaluation("size-biased", (0, 1), (0, 1), 0.2, 0.8)

    def with_size(evaluation: Evaluation, value: float) -> Evaluation:
        return evaluation.model_copy(
            update={
                "objectives": (
                    *evaluation.objectives,
                    ObjectiveValue(
                        name="balanced_bicluster_size",
                        value=value,
                        direction=ObjectiveDirection.MAXIMIZE,
                    ),
                )
            }
        )

    selected = ContainmentMarginalQualitySelector(
        objective_names=("internal_coherence", "contrast")
    ).select(
        run_context,
        Repertoire(
            evaluations=(
                with_size(preferred, 0.1),
                with_size(size_biased, 1.0),
            )
        ),
    )

    assert tuple(item.candidate.identifier for item in selected.evaluations) == ("preferred",)
    provenance = selected.evaluations[0].final_selection
    assert provenance is not None
    assert provenance.source_candidate_identifiers == ("preferred", "size-biased")


def test_containment_selector_rejects_a_missing_selected_objective(
    run_context: RunContext,
) -> None:
    with pytest.raises(ComponentError, match="cannot find configured objectives"):
        ContainmentMarginalQualitySelector(
            objective_names=("internal_coherence", "unknown")
        ).select(
            run_context,
            Repertoire(evaluations=(_evaluation("candidate", (0, 1), (0, 1), 0.1, 0.9),)),
        )


def test_final_selection_provenance_round_trips_through_bicluster_set(
    tmp_path: Path,
    run_context: RunContext,
) -> None:
    repertoire = ContainmentMarginalQualitySelector().select(
        run_context,
        Repertoire(
            evaluations=(
                _evaluation("best", (0, 1), (0, 1), 0.10, 0.90, coordinate=(0, 0)),
                _evaluation(
                    "duplicate",
                    (0, 1),
                    (0, 1),
                    0.12,
                    0.88,
                    coordinate=(0, 0),
                ),
            )
        ),
    )
    destination = tmp_path / "results"
    manifest = BiclusterSetWriter().write(
        destination,
        identifier="selected-results",
        dataset_identifier=run_context.dataset.metadata.identifier,
        row_count=run_context.dataset.row_count,
        source_column_count=run_context.dataset.source_column_count,
        columns=run_context.dataset.columns,
        repertoire=repertoire,
        source_run="run",
        source_checkpoint="checkpoints/checkpoint-final.json",
        source_checkpoint_sha256="a" * 64,
        source_checkpoint_evaluations=100,
    )
    restored = BiclusterSetReader().read_contents(destination)

    assert manifest.schema_version == 7
    assert manifest.final_selection_file == "final-selection.parquet"
    assert manifest.source_checkpoint_evaluations == 100
    assert restored.repertoire == repertoire
    assert len(restored.final_selection) == len(repertoire.evaluations)
