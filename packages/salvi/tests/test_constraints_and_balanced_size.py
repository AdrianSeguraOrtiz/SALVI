from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from salvi.application.run_service import RunService
from salvi.components.constraints import (
    BalancedBiclusterSizeRange,
    BalancedBiclusterSizeRangeConfiguration,
    MaximumInternalCoherence,
    MaximumInternalCoherenceConfiguration,
)
from salvi.components.descriptors import ColumnCardinality, RowCardinality
from salvi.components.objectives import BalancedBiclusterSize
from salvi.components.protocols import Constraint, Objective
from salvi.domain import (
    Bicluster,
    BinningStrategy,
    Candidate,
    ConstraintResult,
    ConstraintValue,
    Evaluation,
    NamedValue,
    ObjectiveDirection,
    ObjectiveResult,
    ObjectiveValue,
)
from salvi.engine.archive import DeepGridMomeArchive
from salvi.engine.grid import ArchiveAxisConfiguration
from salvi.evaluation.workspace import EvaluationWorkspace

from .conftest import configuration_mapping, create_dataset_bundle, write_configuration


@dataclass(frozen=True, slots=True)
class _StaticObjective:
    component_name: str
    direction: ObjectiveDirection
    provides: frozenset[str] = frozenset({"objective"})
    requires: frozenset[str] = frozenset()

    def evaluate(self, candidate: Candidate, workspace: EvaluationWorkspace) -> ObjectiveResult:
        del candidate, workspace
        return ObjectiveResult(value=0.0, columns=())


@dataclass(frozen=True, slots=True)
class _StaticConstraint:
    component_name: str = "size_limit"
    provides: frozenset[str] = frozenset({"constraint"})
    requires: frozenset[str] = frozenset()

    def evaluate(self, candidate: Candidate, workspace: EvaluationWorkspace) -> ConstraintResult:
        del candidate, workspace
        return ConstraintResult(value=0.0)


def _candidate() -> Candidate:
    return Candidate(
        identifier="candidate",
        bicluster=Bicluster(
            row_indices=(0, 1),
            column_indices=(0, 1),
        ),
    )


def _evaluation(
    identifier: str,
    *,
    coherence: float,
    contrast: float,
    constraint: float,
    rows: tuple[int, ...] = (0, 1),
) -> Evaluation:
    return Evaluation(
        candidate=Candidate(
            identifier=identifier,
            bicluster=Bicluster(row_indices=rows, column_indices=(0, 1)),
        ),
        objectives=(
            ObjectiveValue(
                name="coherence",
                value=coherence,
                direction=ObjectiveDirection.MINIMIZE,
            ),
            ObjectiveValue(
                name="contrast",
                value=contrast,
                direction=ObjectiveDirection.MAXIMIZE,
            ),
        ),
        constraints=(ConstraintValue(name="size_limit", value=constraint),),
        descriptors=(
            NamedValue(name="row_cardinality", value=2.0),
            NamedValue(name="column_cardinality", value=2.0),
        ),
    )


def test_balanced_bicluster_size_is_harmonic_and_explains_each_column(
    run_context: object,
) -> None:
    from salvi.application.context import RunContext

    assert isinstance(run_context, RunContext)
    candidate = _candidate()
    result = BalancedBiclusterSize().evaluate(
        candidate,
        EvaluationWorkspace(run_context),
    )

    row_coverage = 2 / run_context.dataset.row_count
    column_coverage = 2 / run_context.dataset.column_count
    expected = 2 * row_coverage * column_coverage / (row_coverage + column_coverage)
    reduced = 2 * row_coverage * (1 / 3) / (row_coverage + (1 / 3))
    assert result.value == pytest.approx(expected)
    assert tuple(column.column_index for column in result.columns) == (0, 1)
    assert tuple(column.value for column in result.columns) == pytest.approx(
        (expected - reduced, expected - reduced)
    )


def test_balanced_size_range_uses_signed_constraint_convention(
    run_context: object,
) -> None:
    from salvi.application.context import RunContext

    assert isinstance(run_context, RunContext)
    workspace = EvaluationWorkspace(run_context)
    candidate = _candidate()
    feasible = BalancedBiclusterSizeRange(minimum=0.5, maximum=0.8).evaluate(
        candidate,
        workspace,
    )
    infeasible = BalancedBiclusterSizeRange(minimum=0.8, maximum=1.0).evaluate(
        candidate,
        workspace,
    )

    assert feasible.value <= 0.0
    assert feasible.violation == 0.0
    assert infeasible.value > 0.0
    assert infeasible.violation == pytest.approx(infeasible.value)


def test_constraint_configurations_reject_invalid_ranges() -> None:
    with pytest.raises(ValueError, match="minimum"):
        BalancedBiclusterSizeRangeConfiguration(minimum=0.8, maximum=0.2)
    with pytest.raises(ValueError):
        MaximumInternalCoherenceConfiguration(maximum_error=1.0)


def test_maximum_internal_coherence_reuses_the_workspace_fit(
    run_context: object,
) -> None:
    from salvi.application.context import RunContext

    assert isinstance(run_context, RunContext)
    workspace = EvaluationWorkspace(run_context)
    candidate = _candidate()
    result = MaximumInternalCoherence(maximum_error=0.9).evaluate(
        candidate,
        workspace,
    )

    assert len(workspace) == 1
    assert result.value <= 0.1
    assert dict(result.diagnostics)["maximum_error"] == pytest.approx(0.9)


def test_evaluation_distinguishes_validity_from_constraint_feasibility() -> None:
    evaluation = _evaluation(
        "candidate",
        coherence=0.1,
        contrast=0.9,
        constraint=0.25,
    )

    assert evaluation.valid
    assert not evaluation.feasible
    assert evaluation.constraint_violation == pytest.approx(0.25)


def test_mome_archive_applies_feasibility_before_pareto(run_context: object) -> None:
    from salvi.application.context import RunContext

    assert isinstance(run_context, RunContext)
    archive = DeepGridMomeArchive(
        axes=(
            ArchiveAxisConfiguration(
                descriptor="row_cardinality",
                binning=BinningStrategy.EXACT,
            ),
            ArchiveAxisConfiguration(
                descriptor="column_cardinality",
                binning=BinningStrategy.EXACT,
            ),
        ),
        cell_capacity=4,
    )
    objectives: tuple[Objective, ...] = (
        _StaticObjective("coherence", ObjectiveDirection.MINIMIZE),
        _StaticObjective("contrast", ObjectiveDirection.MAXIMIZE),
    )
    constraints: tuple[Constraint, ...] = (_StaticConstraint(),)
    archive.initialize(
        run_context,
        objectives,
        (RowCardinality(), ColumnCardinality()),
        constraints,
    )
    infeasible = _evaluation(
        "infeasible",
        coherence=0.0,
        contrast=1.0,
        constraint=0.1,
    )
    feasible = _evaluation(
        "feasible",
        coherence=0.8,
        contrast=0.2,
        constraint=-0.1,
        rows=(0, 2),
    )

    archive.add((infeasible,))
    outcome = archive.add((feasible,))[0]

    assert outcome.accepted
    assert outcome.evicted_candidate_identifiers == ("infeasible",)
    assert tuple(item.candidate.identifier for item in archive.repertoire().evaluations) == (
        "feasible",
    )


def test_configured_constraint_reaches_run_results_and_artifacts(tmp_path: Path) -> None:
    dataset = create_dataset_bundle(tmp_path / "dataset")
    mapping = configuration_mapping(
        dataset,
        tmp_path / "output",
        overwrite=True,
    )
    search = mapping["search"]
    assert isinstance(search, dict)
    search["constraints"] = [
        {
            "name": "balanced_bicluster_size_range",
            "parameters": {"minimum": 0.0, "maximum": 1.0},
        }
    ]
    mapping["final_selection"] = None

    result = RunService().run(write_configuration(tmp_path / "configuration.yaml", mapping))

    assert result.repertoire.evaluations
    assert all(evaluation.feasible for evaluation in result.repertoire.evaluations)
    assert all(
        tuple(item.name for item in evaluation.constraints) == ("balanced_bicluster_size_range",)
        for evaluation in result.repertoire.evaluations
    )
