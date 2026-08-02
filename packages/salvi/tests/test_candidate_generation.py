from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import pyarrow as pa
import pytest

from salvi.application.context import NamedRandomStreams, QdRunContext, RunContext
from salvi.components.advanced_emitters import AlternatingPatternLocalSearchEmitter
from salvi.components.candidate_initialization import (
    CellCoveragePatternAwareInitializer,
    CellCoveragePatternAwareInitializerConfiguration,
    Dimension,
    MembershipOperation,
    PatternAwareInitializer,
    StratifiedInitializer,
)
from salvi.components.descriptors import ColumnCardinality, RowCardinality
from salvi.components.evaluation_policies import MinimumCardinality
from salvi.components.execution import SerialEvaluationExecutor
from salvi.components.mate_selection import (
    CellFirstEvidenceCompatibleMateSelection,
    RepertoireRandomMateSelection,
)
from salvi.components.membership_emitters import (
    CellCoverageRestartEmitter,
    MembershipMoveEmitter,
    RestartEmitter,
    ShapeMoveEmitter,
)
from salvi.components.objectives import Contrast, InternalCoherence
from salvi.components.operators import (
    BitFlipMembershipMutation,
    EvidenceWeightedRecombinationCrossover,
    HalfUniformMembershipCrossover,
    MembershipRecombinationCrossover,
    _joint_groups,
    _recombine_membership,
    _weighted_without_replacement,
)
from salvi.components.preprocessing import RobustNumericScaling
from salvi.components.schedulers import (
    AdaptiveCreditScheduler,
    CellBalancedAdaptiveCreditScheduler,
    FixedProportionScheduler,
)
from salvi.components.variation_emitters import CrossoverEmitter, MutationEmitter
from salvi.domain import (
    ArchiveCellCoordinate,
    ArchiveCellTarget,
    ArchiveInsertionStatus,
    Bicluster,
    Candidate,
    CandidateProvenance,
    ColumnKind,
    ColumnMetadata,
    Dataset,
    EmitterCellFeedback,
    EmitterFeedback,
    Evaluation,
    PatternKind,
    Repertoire,
)
from salvi.domain.prepared import PreparedDataset
from salvi.evaluation.workspace import EvaluationWorkspace
from salvi.exceptions import ComponentError
from salvi.patterns import PatternConfiguration


def _numeric_context(
    tmp_path: Path,
    patterns: tuple[PatternKind, ...],
) -> QdRunContext:
    table = pa.table(
        {
            "a": pa.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            "b": pa.array([3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            "c": pa.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0]),
            "d": pa.array([6.0, 3.0, 8.0, 1.0, 7.0, 4.0]),
        }
    )
    metadata = Dataset(
        identifier="numeric",
        bundle_path=tmp_path,
        row_count=table.num_rows,
        column_count=table.num_columns,
        columns=tuple(
            ColumnMetadata(index=index, name=name, kind=ColumnKind.NUMERIC)
            for index, name in enumerate(table.column_names)
        ),
    )
    prepared = RobustNumericScaling().transform(
        PreparedDataset.from_arrow(
            metadata,
            table,
            pa.array([f"r{index}" for index in range(table.num_rows)]),
        )
    )
    from salvi.components.evaluation_policies import MinimumObservedSupport
    from salvi.components.parent_selection import RepertoireUniformParentSelection

    return QdRunContext(
        dataset=prepared,
        patterns=PatternConfiguration(allowed=patterns),
        random_streams=NamedRandomStreams(31),
        parent_selection_policy=RepertoireUniformParentSelection(),
        candidate_validity_policy=MinimumCardinality(),
        evaluation_support_policy=MinimumObservedSupport(),
    )


def test_stratified_initializer_covers_bounds_and_records_exact_provenance(
    run_context: RunContext,
) -> None:
    candidates = StratifiedInitializer(cardinality_levels=8).initialize(
        run_context,
        24,
        start_sequence=100,
    )
    row_counts = {len(candidate.bicluster.row_indices) for candidate in candidates}
    column_counts = {len(candidate.bicluster.column_indices) for candidate in candidates}

    assert {2, run_context.dataset.row_count} <= row_counts
    assert {2, run_context.dataset.column_count} <= column_counts
    assert tuple(
        candidate.provenance.sequence for candidate in candidates if candidate.provenance
    ) == tuple(range(100, 124))
    assert all(
        candidate.provenance is not None
        and candidate.provenance.producer == "stratified"
        and not candidate.provenance.parent_identifiers
        for candidate in candidates
    )


def test_stratified_initializer_covers_the_cartesian_shape_grid(
    run_context: RunContext,
) -> None:
    candidates = StratifiedInitializer(cardinality_levels=8).initialize(
        run_context,
        6,
    )
    shapes = {
        (
            len(candidate.bicluster.row_indices),
            len(candidate.bicluster.column_indices),
        )
        for candidate in candidates
    }

    assert shapes == {
        (2, 2),
        (2, 3),
        (3, 2),
        (3, 3),
        (4, 2),
        (4, 3),
    }


@pytest.mark.parametrize(
    "patterns",
    [
        (PatternKind.CONSTANT,),
        (PatternKind.ADDITIVE,),
        (PatternKind.MULTIPLICATIVE,),
        (PatternKind.CONSTANT, PatternKind.ADDITIVE, PatternKind.MULTIPLICATIVE),
    ],
)
def test_pattern_aware_initializer_allocates_attempts_to_allowed_families(
    tmp_path: Path,
    patterns: tuple[PatternKind, ...],
) -> None:
    context = _numeric_context(tmp_path, patterns)
    candidates = PatternAwareInitializer(
        cardinality_levels=4,
        joint_column_candidate_pool_size=4,
    ).initialize(context, len(patterns) * 3)
    hints = tuple(
        candidate.provenance.pattern_hint
        for candidate in candidates
        if candidate.provenance is not None
    )

    assert set(hints) == set(patterns)
    for candidate in candidates:
        context.candidate_validity_policy.validate(candidate, context.dataset)
        if candidate.provenance and candidate.provenance.pattern_hint is not PatternKind.CONSTANT:
            assert set(candidate.bicluster.column_indices) <= set(
                context.dataset.numeric_column_indices
            )


def test_constant_pattern_aware_initializer_uses_mixed_column_kinds(
    run_context: RunContext,
) -> None:
    candidates = PatternAwareInitializer(cardinality_levels=4).initialize(
        run_context,
        20,
    )
    assert any(
        any(
            run_context.dataset.column_metadata(column).kind is not ColumnKind.NUMERIC
            for column in candidate.bicluster.column_indices
        )
        for candidate in candidates
    )


def test_cell_coverage_initializer_targets_exact_shapes_and_balances_patterns(
    tmp_path: Path,
) -> None:
    context = _numeric_context(
        tmp_path,
        (PatternKind.CONSTANT, PatternKind.ADDITIVE),
    )
    initializer = CellCoveragePatternAwareInitializer(
        seeds_per_cell=2,
        max_attempts_per_cell=4,
        joint_column_candidate_pool_size=4,
    )
    target = ArchiveCellTarget(
        coordinate=ArchiveCellCoordinate(indices=(1, 2)),
        row_count=4,
        column_count=3,
    )

    plan = initializer.bootstrap_plan(context, (target,))
    candidates = initializer.initialize_bootstrap(context, plan, start_sequence=7)

    assert plan[0].required_patterns == (
        PatternKind.CONSTANT,
        PatternKind.ADDITIVE,
    )
    assert len(candidates) == 1
    candidate = candidates[0]
    assert len(candidate.bicluster.row_indices) == 4
    assert len(candidate.bicluster.column_indices) == 3
    assert candidate.provenance is not None
    assert candidate.provenance.sequence == 7
    assert candidate.provenance.pattern_hint is PatternKind.CONSTANT
    assert candidate.provenance.target_archive_coordinate == (1, 2)

    second_state = plan[0].model_copy(
        update={"accepted_patterns": (PatternKind.CONSTANT,), "attempts": 1}
    )
    additive = initializer.initialize_bootstrap(context, (second_state,), start_sequence=8)[0]
    assert additive.provenance is not None
    assert additive.provenance.pattern_hint is PatternKind.ADDITIVE


def test_cell_coverage_initializer_rejects_an_attempt_budget_below_seed_depth() -> None:
    with pytest.raises(ValueError, match="cannot be smaller"):
        CellCoveragePatternAwareInitializerConfiguration(
            seeds_per_cell=4,
            max_attempts_per_cell=3,
        )


def _repertoire_candidate(
    rows: tuple[int, ...],
    columns: tuple[int, ...],
    identifier: str = "parent",
) -> Repertoire:
    candidate = Candidate(
        identifier=identifier,
        bicluster=Bicluster(row_indices=rows, column_indices=columns),
    )
    return Repertoire(evaluations=(Evaluation(candidate=candidate, objectives=(), descriptors=()),))


def _scientific_evaluation(
    context: RunContext,
    rows: tuple[int, ...],
    columns: tuple[int, ...],
    identifier: str,
) -> Evaluation:
    candidate = Candidate(
        identifier=identifier,
        bicluster=Bicluster(row_indices=rows, column_indices=columns),
    )
    return SerialEvaluationExecutor().evaluate(
        (candidate,),
        (InternalCoherence(), Contrast(min_background_ratio=0.1)),
        (RowCardinality(), ColumnCardinality()),
        EvaluationWorkspace(context),
    )[0]


@pytest.mark.parametrize(
    ("dimension", "operation", "rows", "columns", "expected_delta"),
    [
        ("rows", "add", (0, 1), (0, 1), 1),
        ("rows", "remove", (0, 1, 2), (0, 1), -1),
        ("rows", "swap", (0, 1, 2), (0, 1), 0),
        ("columns", "add", (0, 1), (0, 1), 1),
        ("columns", "remove", (0, 1), (0, 1, 2), -1),
        ("columns", "swap", (0, 1), (0, 1), 0),
    ],
)
def test_membership_emitters_apply_one_declared_move(
    run_context: RunContext,
    dimension: str,
    operation: str,
    rows: tuple[int, ...],
    columns: tuple[int, ...],
    expected_delta: int,
) -> None:
    emitter = MembershipMoveEmitter(
        dimension=cast(Dimension, dimension),
        operation=cast(MembershipOperation, operation),
        component_name=f"{operation}_{dimension[:-1]}",
    )
    parent = _repertoire_candidate(rows, columns)
    child = emitter.emit(run_context, parent, 1, start_sequence=7)[0]
    before = rows if dimension == "rows" else columns
    after = child.bicluster.row_indices if dimension == "rows" else child.bicluster.column_indices

    assert len(after) - len(before) == expected_delta
    assert child.provenance is not None
    assert child.provenance.parent_identifiers == ("parent",)
    assert child.provenance.sequence == 7
    assert child.provenance.operation == f"{operation}_{dimension[:-1]}"


def test_shape_crossover_and_restart_emit_valid_non_frozen_shapes(
    run_context: RunContext,
) -> None:
    first = _repertoire_candidate((0, 1, 2), (0, 1), "first")
    second = _repertoire_candidate((1, 2, 3), (1, 2), "second")
    repertoire = Repertoire(evaluations=first.evaluations + second.evaluations)

    shape = ShapeMoveEmitter().emit(run_context, repertoire, 12)
    assert any(
        len(child.bicluster.row_indices) != 3 and len(child.bicluster.column_indices) != 2
        for child in shape
    )

    crossover_context = replace(
        run_context,
        mate_selection_policy=RepertoireRandomMateSelection(),
        crossover_operator=MembershipRecombinationCrossover(
            application_probability=1.0,
            row_exchange_probability=1.0,
            column_exchange_probability=1.0,
        ),
    )
    first_evaluation, second_evaluation = repertoire.evaluations
    same_cell_repertoire = Repertoire(
        evaluations=(
            first_evaluation.model_copy(update={"archive_coordinate": (0, 0)}),
            second_evaluation.model_copy(update={"archive_coordinate": (0, 0)}),
        )
    )
    recombined = CrossoverEmitter(max_attempts=16).emit(
        crossover_context,
        same_cell_repertoire,
        8,
    )
    assert all(
        child.provenance is not None and len(child.provenance.parent_identifiers) == 2
        for child in recombined
    )

    restarted = RestartEmitter(strategy="stratified").emit(
        run_context,
        repertoire,
        8,
        start_sequence=50,
    )
    assert {candidate.provenance.operation for candidate in restarted if candidate.provenance} == {
        "stratified_restart"
    }
    for candidate in (*shape, *recombined, *restarted):
        run_context.candidate_validity_policy.validate(candidate, run_context.dataset)


def test_reusable_variation_operators_preserve_valid_membership(
    run_context: RunContext,
) -> None:
    first = _repertoire_candidate((0, 1, 2), (0, 1), "first").evaluations[0]
    second = _repertoire_candidate((1, 2, 3), (1, 2), "second").evaluations[0]

    unchanged_crossover = HalfUniformMembershipCrossover(application_probability=0.0).cross(
        run_context,
        first,
        second,
        run_context.random_streams.generator("test.hux.unchanged"),
    )
    assert unchanged_crossover == first.candidate.bicluster

    exchanged = HalfUniformMembershipCrossover(
        application_probability=1.0,
        row_exchange_probability=1.0,
        column_exchange_probability=1.0,
    ).cross(
        run_context,
        first,
        second,
        run_context.random_streams.generator("test.hux.exchanged"),
    )
    assert exchanged == second.candidate.bicluster

    unchanged_mutation = BitFlipMembershipMutation(application_probability=0.0).mutate(
        run_context,
        first,
        run_context.random_streams.generator("test.mutation.unchanged"),
    )
    assert unchanged_mutation == first.candidate.bicluster

    mutated = BitFlipMembershipMutation(
        application_probability=1.0,
        bit_probability=1.0,
    ).mutate(
        run_context,
        first,
        run_context.random_streams.generator("test.mutation.changed"),
    )
    assert mutated != first.candidate.bicluster
    assert len(mutated.row_indices) >= 2
    assert len(mutated.column_indices) >= 2


def test_recombination_operators_can_be_disabled_without_changing_the_parent(
    run_context: RunContext,
) -> None:
    first = _repertoire_candidate((0, 1, 2), (0, 1), "first").evaluations[0]
    second = _repertoire_candidate((1, 2, 3), (1, 2), "second").evaluations[0]
    generator = run_context.random_streams.generator("test.disabled-recombination")

    assert (
        MembershipRecombinationCrossover(application_probability=0.0).cross(
            run_context,
            first,
            second,
            generator,
        )
        == first.candidate.bicluster
    )
    assert (
        EvidenceWeightedRecombinationCrossover(application_probability=0.0).cross(
            run_context,
            first,
            second,
            generator,
        )
        == first.candidate.bicluster
    )

    assert (
        EvidenceWeightedRecombinationCrossover(
            application_probability=1.0,
            row_exchange_probability=0.0,
            column_exchange_probability=0.0,
        ).cross(run_context, first, second, generator)
        == first.candidate.bicluster
    )
    assert (
        HalfUniformMembershipCrossover(
            application_probability=1.0,
            row_exchange_probability=0.0,
            column_exchange_probability=0.0,
        ).cross(run_context, first, second, generator)
        == first.candidate.bicluster
    )
    assert (
        BitFlipMembershipMutation(
            application_probability=1.0,
            bit_probability=1e-12,
        ).mutate(run_context, first, generator)
        == first.candidate.bicluster
    )

    assert _joint_groups(first) == ()
    assert _weighted_without_replacement((), 0, {}, generator) == ()
    assert _weighted_without_replacement((1, 2), 2, {}, generator) == (1, 2)


def test_generic_mutation_emitter_uses_configured_components_and_fallbacks(
    run_context: RunContext,
) -> None:
    parent = _repertoire_candidate((0, 1, 2), (0, 1), "parent")
    context = replace(
        run_context,
        mutation_operator=BitFlipMembershipMutation(
            application_probability=1.0,
            bit_probability=1.0,
        ),
    )

    children = MutationEmitter(max_attempts=2).emit(
        context,
        parent,
        2,
        start_sequence=7,
    )

    assert tuple(child.provenance.sequence for child in children if child.provenance) == (7, 8)
    assert all(
        child.provenance is not None
        and child.provenance.operation == "bit_flip_membership"
        and child.provenance.parent_identifiers == ("parent",)
        for child in children
    )

    fallback = MutationEmitter().emit(context, Repertoire(), 1)[0]
    assert fallback.provenance is not None
    assert fallback.provenance.operation == "restart_fallback"

    with pytest.raises(ValueError, match="non-negative"):
        MutationEmitter().emit(context, parent, -1)
    with pytest.raises(ComponentError, match="mutation operator"):
        MutationEmitter().emit(run_context, parent, 1)
    with pytest.raises(ComponentError, match="parent-selection"):
        MutationEmitter().emit(
            replace(context, parent_selection_policy=None),
            parent,
            1,
        )
    with pytest.raises(ComponentError, match="mate selection"):
        CrossoverEmitter().emit(run_context, parent, 1)


def test_guided_emitters_reuse_persisted_constant_fit(run_context: RunContext) -> None:
    evaluation = _scientific_evaluation(
        run_context,
        (0, 1, 2),
        (0, 1, 2),
        "constant-parent",
    )
    assert evaluation.pattern_fit is not None
    repertoire = Repertoire(evaluations=(evaluation,))

    for dimension, operation in (
        ("rows", "add"),
        ("rows", "remove"),
        ("rows", "swap"),
        ("columns", "remove"),
    ):
        child = MembershipMoveEmitter(
            dimension=cast(Dimension, dimension),
            operation=cast(MembershipOperation, operation),
            guided=True,
            component_name=f"{operation}_{dimension[:-1]}",
        ).emit(run_context, repertoire, 1)[0]
        run_context.candidate_validity_policy.validate(child, run_context.dataset)


@pytest.mark.parametrize(
    ("pattern", "columns"),
    [
        (PatternKind.ADDITIVE, (0, 1)),
        (PatternKind.MULTIPLICATIVE, (0, 2)),
    ],
)
def test_guided_row_moves_support_joint_pattern_fits(
    tmp_path: Path,
    pattern: PatternKind,
    columns: tuple[int, ...],
) -> None:
    context = _numeric_context(tmp_path, (pattern,))
    evaluation = _scientific_evaluation(
        context,
        (0, 1, 2, 3),
        columns,
        f"{pattern.value.lower()}-parent",
    )
    assert evaluation.pattern_fit is not None
    assert evaluation.pattern_fit.groups
    child = MembershipMoveEmitter(
        "rows",
        "add",
        guided=True,
        component_name="add_row",
    ).emit(context, Repertoire(evaluations=(evaluation,)), 1)[0]
    assert len(child.bicluster.row_indices) == 5


@pytest.mark.parametrize(
    ("patterns", "columns"),
    [
        ((PatternKind.CONSTANT,), (0, 1, 2)),
        ((PatternKind.ADDITIVE,), (0, 1, 3)),
        ((PatternKind.MULTIPLICATIVE,), (0, 2, 3)),
        (
            (
                PatternKind.CONSTANT,
                PatternKind.ADDITIVE,
                PatternKind.MULTIPLICATIVE,
            ),
            (0, 1, 2),
        ),
    ],
)
def test_all_guided_generation_paths_support_every_pattern_mode(
    tmp_path: Path,
    patterns: tuple[PatternKind, ...],
    columns: tuple[int, ...],
) -> None:
    context = _numeric_context(tmp_path, patterns)
    evaluation = _scientific_evaluation(
        context,
        (0, 1, 2, 3),
        columns,
        "pattern-parent",
    )
    assert evaluation.pattern_fit is not None
    repertoire = Repertoire(evaluations=(evaluation,))

    for dimension, operation in (
        ("rows", "add"),
        ("rows", "remove"),
        ("rows", "swap"),
        ("columns", "add"),
        ("columns", "remove"),
        ("columns", "swap"),
    ):
        child = MembershipMoveEmitter(
            dimension=cast(Dimension, dimension),
            operation=cast(MembershipOperation, operation),
            guided=True,
            candidate_pool_size=4,
            component_name=f"{operation}_{dimension[:-1]}",
        ).emit(context, repertoire, 1)[0]
        context.candidate_validity_policy.validate(child, context.dataset)

    local_child = AlternatingPatternLocalSearchEmitter(
        parent_pool_size=1,
        candidate_pool_size=4,
    ).emit(context, repertoire, 1)[0]
    context.candidate_validity_policy.validate(local_child, context.dataset)


def test_shape_emitter_guided_paths_and_fallback(run_context: RunContext) -> None:
    row_expand = _scientific_evaluation(
        run_context,
        (0, 1),
        (0, 1, 2),
        "row-expand",
    )
    row_child = ShapeMoveEmitter(guided=True).emit(
        run_context,
        Repertoire(evaluations=(row_expand,)),
        1,
    )[0]
    assert len(row_child.bicluster.row_indices) == 3
    assert len(row_child.bicluster.column_indices) == 2

    column_expand = _scientific_evaluation(
        run_context,
        (0, 1, 2, 3),
        (0, 1),
        "column-expand",
    )
    column_child = ShapeMoveEmitter(guided=True).emit(
        run_context,
        Repertoire(evaluations=(column_expand,)),
        1,
    )[0]
    assert len(column_child.bicluster.row_indices) == 3
    assert len(column_child.bicluster.column_indices) == 3

    fallback = ShapeMoveEmitter().emit(run_context, Repertoire(), 1)[0]
    assert fallback.provenance is not None
    assert fallback.provenance.operation == "restart_fallback"


def test_recombination_probability_and_restart_fallbacks(run_context: RunContext) -> None:
    generator = NamedRandomStreams(3).generator("recombination-test")
    first = (0, 1, 2)
    second = (1, 2, 3)
    assert _recombine_membership(first, second, 0.0, 2, 4, generator) == first
    assert len(_recombine_membership(first, second, 1.0, 2, 4, generator)) >= 2

    configured = replace(
        run_context,
        mate_selection_policy=RepertoireRandomMateSelection(),
        crossover_operator=MembershipRecombinationCrossover(),
    )
    fallback = CrossoverEmitter().emit(configured, Repertoire(), 1)[0]
    assert fallback.provenance is not None
    assert fallback.provenance.operation == "restart_fallback"

    pattern_restart = RestartEmitter(strategy="pattern_aware").emit(
        run_context,
        Repertoire(),
        2,
    )
    assert all(
        candidate.provenance is not None
        and candidate.provenance.operation == "pattern_aware_restart"
        for candidate in pattern_restart
    )


def test_cell_coverage_restart_targets_reachable_archive_cells(
    tmp_path: Path,
) -> None:
    context = replace(
        _numeric_context(tmp_path, tuple(PatternKind)),
        archive_cell_targets=(
            ArchiveCellTarget(
                coordinate=ArchiveCellCoordinate(indices=(0, 0)),
                row_count=3,
                column_count=2,
            ),
            ArchiveCellTarget(
                coordinate=ArchiveCellCoordinate(indices=(1, 1)),
                row_count=4,
                column_count=3,
            ),
        ),
    )
    emitter = CellCoverageRestartEmitter(joint_column_candidate_pool_size=4)

    generated = tuple(emitter.emit(context, Repertoire(), 3, start_sequence=20))

    assert len(generated) == 3
    assert tuple(
        candidate.provenance.sequence for candidate in generated if candidate.provenance is not None
    ) == (20, 21, 22)
    assert all(
        candidate.provenance is not None
        and candidate.provenance.operation == "cell_coverage_restart"
        for candidate in generated
    )
    assert {
        (len(candidate.bicluster.row_indices), len(candidate.bicluster.column_indices))
        for candidate in generated
    } <= {(3, 2), (4, 3)}
    assert emitter.emit(context, Repertoire(), 0) == ()
    with pytest.raises(ValueError, match="non-negative"):
        emitter.emit(context, Repertoire(), -1)
    with pytest.raises(ComponentError, match="archive cardinality targets"):
        emitter.emit(replace(context, archive_cell_targets=()), Repertoire(), 1)


def test_cell_first_mate_selection_never_crosses_its_configured_cell_radius(
    run_context: RunContext,
) -> None:
    first = _scientific_evaluation(
        run_context,
        (0, 1, 2),
        (0, 1),
        "first",
    ).model_copy(update={"archive_coordinate": (0, 0)})
    neighbor = _scientific_evaluation(
        run_context,
        (0, 1, 3),
        (0, 1, 2),
        "neighbor",
    ).model_copy(update={"archive_coordinate": (0, 1)})
    remote = _scientific_evaluation(
        run_context,
        (0, 1, 2),
        (0, 1, 2),
        "remote",
    ).model_copy(update={"archive_coordinate": (4, 4)})
    selector = CellFirstEvidenceCompatibleMateSelection(
        cell_neighborhood_radius=1,
        minimum_row_jaccard=0.1,
        minimum_column_jaccard=0.1,
    )

    pair = selector.select(
        Repertoire(evaluations=(first, neighbor, remote)),
        NamedRandomStreams(9).generator("cell-first-mates"),
    )

    assert pair is not None
    assert pair[0].archive_coordinate is not None
    assert pair[1].archive_coordinate is not None
    assert (
        max(
            abs(left - right)
            for left, right in zip(
                pair[0].archive_coordinate,
                pair[1].archive_coordinate,
                strict=True,
            )
        )
        <= 1
    )


def _continued_local_evaluation(
    evaluation: Evaluation,
    *,
    operation: str,
) -> Evaluation:
    provenance = CandidateProvenance(
        producer="alternating_pattern_local_search",
        operation=operation,
        sequence=1,
        parent_identifiers=("source",),
    )
    candidate = evaluation.candidate.model_copy(
        update={
            "identifier": f"continued-{operation}",
            "generation": 1,
            "provenance": provenance,
        }
    )
    return evaluation.model_copy(update={"candidate": candidate})


@pytest.mark.parametrize(
    ("previous_operation", "unchanged_dimension"),
    [
        ("swap_row", "rows"),
        ("swap_column", "columns"),
    ],
)
def test_alternating_local_search_continues_on_the_other_dimension(
    run_context: RunContext,
    previous_operation: str,
    unchanged_dimension: str,
) -> None:
    source = _scientific_evaluation(
        run_context,
        (0, 1, 2),
        (0, 1, 2),
        "source",
    )
    parent = _continued_local_evaluation(source, operation=previous_operation)
    child = AlternatingPatternLocalSearchEmitter(
        parent_pool_size=1,
        candidate_pool_size=8,
        cardinality_change_probability=0.0,
    ).emit(
        run_context,
        Repertoire(evaluations=(parent,)),
        1,
        start_sequence=10,
    )[0]

    if unchanged_dimension == "rows":
        assert child.bicluster.row_indices == parent.candidate.bicluster.row_indices
        assert child.bicluster.column_indices != parent.candidate.bicluster.column_indices
        expected_suffix = "_column"
    else:
        assert child.bicluster.column_indices == parent.candidate.bicluster.column_indices
        assert child.bicluster.row_indices != parent.candidate.bicluster.row_indices
        expected_suffix = "_row"
    assert child.provenance is not None
    assert child.provenance.operation.endswith(expected_suffix)
    assert child.provenance.parent_identifiers == (parent.candidate.identifier,)


def test_alternating_local_search_uses_explicit_fallback_and_validates_configuration(
    run_context: RunContext,
) -> None:
    fallback = AlternatingPatternLocalSearchEmitter().emit(
        run_context,
        Repertoire(),
        1,
    )[0]
    assert fallback.provenance is not None
    assert fallback.provenance.operation == "restart_fallback"

    with pytest.raises(ValueError):
        AlternatingPatternLocalSearchEmitter(cardinality_change_probability=1.1)
    with pytest.raises(ValueError, match="non-negative"):
        AlternatingPatternLocalSearchEmitter().emit(run_context, Repertoire(), -1)


def test_candidate_validity_policy_controls_generator_minima(run_context: RunContext) -> None:
    context = replace(
        run_context,
        random_streams=NamedRandomStreams(11),
        candidate_validity_policy=MinimumCardinality(min_rows=3, min_columns=3),
    )
    candidates = StratifiedInitializer().initialize(context, 10)
    assert all(len(candidate.bicluster.row_indices) >= 3 for candidate in candidates)
    assert all(len(candidate.bicluster.column_indices) >= 3 for candidate in candidates)


def test_generators_reject_invalid_requests_and_use_explicit_fallbacks(
    run_context: RunContext,
) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        StratifiedInitializer().initialize(run_context, -1)
    with pytest.raises(ValueError, match="non-negative"):
        PatternAwareInitializer().initialize(run_context, -1)
    with pytest.raises(ValueError, match="non-negative"):
        MembershipMoveEmitter("rows", "add").emit(run_context, Repertoire(), -1)
    with pytest.raises(ValueError, match="non-negative"):
        ShapeMoveEmitter().emit(run_context, Repertoire(), -1)
    with pytest.raises(ValueError, match="non-negative"):
        CrossoverEmitter().emit(run_context, Repertoire(), -1)

    fixed_context = replace(
        run_context,
        random_streams=NamedRandomStreams(13),
        candidate_validity_policy=MinimumCardinality(
            min_rows=run_context.dataset.row_count,
            min_columns=run_context.dataset.column_count,
        ),
    )
    full = _repertoire_candidate(
        tuple(range(run_context.dataset.row_count)),
        tuple(range(run_context.dataset.column_count)),
    )
    fallback = MembershipMoveEmitter(
        "rows",
        "remove",
        component_name="remove_row",
    ).emit(fixed_context, full, 1)[0]
    assert fallback.provenance is not None
    assert fallback.provenance.operation == "restart_fallback"

    additive_context = replace(
        run_context,
        random_streams=NamedRandomStreams(17),
        patterns=PatternConfiguration(allowed=(PatternKind.ADDITIVE,)),
    )
    with pytest.raises(ComponentError, match="numeric columns"):
        PatternAwareInitializer().initialize(additive_context, 1)

    generator = NamedRandomStreams(5).generator("shared-recombination")
    assert _recombine_membership((0, 1), (0, 1), 1.0, 2, 3, generator) == (0, 1)


def test_adaptive_scheduler_is_deterministic_and_checkpointable(
    run_context: RunContext,
) -> None:
    emitters = (
        MembershipMoveEmitter("rows", "add", component_name="add_row"),
        MembershipMoveEmitter("rows", "remove", component_name="remove_row"),
        MembershipMoveEmitter("columns", "swap", component_name="swap_column"),
    )
    scheduler = AdaptiveCreditScheduler(exploration_weight=0.25)
    allocations = scheduler.allocate(emitters, 3)
    assert tuple(item.emitter_name for item in allocations) == (
        "add_row",
        "remove_row",
        "swap_column",
    )
    scheduler.update(
        (
            EmitterFeedback(
                emitter_name="add_row",
                evaluated=1,
                accepted=1,
                created_cells=1,
                evictions=0,
                statuses=((ArchiveInsertionStatus.INSERTED, 1),),
            ),
            EmitterFeedback(
                emitter_name="remove_row",
                evaluated=1,
                accepted=0,
                created_cells=0,
                evictions=0,
                statuses=((ArchiveInsertionStatus.REJECTED_DOMINATED, 1),),
            ),
            EmitterFeedback(
                emitter_name="swap_column",
                evaluated=1,
                accepted=0,
                created_cells=0,
                evictions=0,
                statuses=((ArchiveInsertionStatus.REJECTED_DUPLICATE, 1),),
            ),
        )
    )
    assert scheduler.allocate(emitters, 1)[0].emitter_name == "add_row"

    restored = AdaptiveCreditScheduler(exploration_weight=0.25)
    restored.restore(scheduler.snapshot(), emitters)
    assert restored.reports(emitters) == scheduler.reports(emitters)
    assert restored.allocate(emitters, 5) == scheduler.allocate(emitters, 5)


def test_fixed_proportion_scheduler_tracks_exact_cumulative_shares() -> None:
    emitters = (
        MembershipMoveEmitter("rows", "add", component_name="add_row"),
        MembershipMoveEmitter("rows", "remove", component_name="remove_row"),
        MembershipMoveEmitter("columns", "swap", component_name="swap_column"),
    )
    scheduler = FixedProportionScheduler(
        shares={"add_row": 0.7, "remove_row": 0.2, "swap_column": 0.1}
    )

    allocations = scheduler.allocate(emitters, 100)

    assert {item.emitter_name: item.count for item in allocations} == {
        "add_row": 70,
        "remove_row": 20,
        "swap_column": 10,
    }
    restored = FixedProportionScheduler(
        shares={"add_row": 0.7, "remove_row": 0.2, "swap_column": 0.1}
    )
    restored.restore(scheduler.snapshot(), emitters)
    assert restored.allocate(emitters, 10) == scheduler.allocate(emitters, 10)

    with pytest.raises(ComponentError, match="must define every configured emitter"):
        FixedProportionScheduler(shares={"add_row": 1.0}).allocate(emitters, 1)


def test_cell_balanced_scheduler_rewards_useful_work_in_less_evaluated_cells(
    run_context: RunContext,
) -> None:
    del run_context
    emitters = (
        MembershipMoveEmitter("rows", "add", component_name="add_row"),
        MembershipMoveEmitter("rows", "remove", component_name="remove_row"),
    )
    scheduler = CellBalancedAdaptiveCreditScheduler(
        exploration_weight=0.0,
        underexplored_cell_weight=1.0,
    )
    scheduler.allocate(emitters, 2)
    scheduler.update(
        (
            EmitterFeedback(
                emitter_name="add_row",
                evaluated=8,
                accepted=1,
                created_cells=0,
                evictions=0,
                statuses=(
                    (ArchiveInsertionStatus.INSERTED, 1),
                    (ArchiveInsertionStatus.REJECTED_DOMINATED, 7),
                ),
                cells=(
                    EmitterCellFeedback(
                        coordinate=ArchiveCellCoordinate(indices=(0, 0)),
                        evaluated=8,
                        accepted=1,
                        created_cells=0,
                    ),
                ),
            ),
            EmitterFeedback(
                emitter_name="remove_row",
                evaluated=1,
                accepted=0,
                created_cells=0,
                evictions=0,
                statuses=((ArchiveInsertionStatus.REJECTED_DOMINATED, 1),),
                cells=(
                    EmitterCellFeedback(
                        coordinate=ArchiveCellCoordinate(indices=(2, 2)),
                        evaluated=1,
                        accepted=0,
                        created_cells=0,
                    ),
                ),
            ),
        )
    )
    before = {report.emitter_name: report.credit for report in scheduler.reports(emitters)}
    scheduler.update(
        (
            EmitterFeedback(
                emitter_name="add_row",
                evaluated=1,
                accepted=1,
                created_cells=0,
                evictions=0,
                statuses=((ArchiveInsertionStatus.INSERTED, 1),),
                cells=(
                    EmitterCellFeedback(
                        coordinate=ArchiveCellCoordinate(indices=(0, 0)),
                        evaluated=1,
                        accepted=1,
                        created_cells=0,
                    ),
                ),
            ),
            EmitterFeedback(
                emitter_name="remove_row",
                evaluated=1,
                accepted=1,
                created_cells=1,
                evictions=0,
                statuses=((ArchiveInsertionStatus.INSERTED, 1),),
                cells=(
                    EmitterCellFeedback(
                        coordinate=ArchiveCellCoordinate(indices=(3, 3)),
                        evaluated=1,
                        accepted=1,
                        created_cells=1,
                    ),
                ),
            ),
        )
    )
    after = {report.emitter_name: report.credit for report in scheduler.reports(emitters)}

    assert after["remove_row"] - before["remove_row"] > after["add_row"] - before["add_row"]
    restored = CellBalancedAdaptiveCreditScheduler()
    restored.restore(scheduler.snapshot(), emitters)
    assert restored.snapshot() == scheduler.snapshot()
