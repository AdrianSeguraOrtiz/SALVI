from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from pydantic import ValidationError

from salvi.application.context import NamedRandomStreams, RunContext
from salvi.components.candidate_initialization import UniformRandomInitializer
from salvi.components.evaluation_policies import MinimumCardinality
from salvi.components.membership_emitters import RandomMoveEmitter, _random_membership_move
from salvi.domain import (
    ArchiveCell,
    ArchiveCellCoordinate,
    ArchiveInsertionOutcome,
    ArchiveInsertionStatus,
    Bicluster,
    BinningStrategy,
    Candidate,
    CandidateBounds,
    DescriptorDomain,
    DescriptorValueKind,
    Evaluation,
    Repertoire,
    SearchProgress,
)
from salvi.engine.grid import ArchiveAxisConfiguration, AxisBinner
from salvi.exceptions import ComponentError


def test_uniform_initializer_is_deterministic_and_structurally_valid(
    run_context: RunContext,
) -> None:
    initializer = UniformRandomInitializer()
    first = initializer.initialize(run_context, 20)
    replay_context = replace(run_context, random_streams=NamedRandomStreams(7))
    second = initializer.initialize(replay_context, 20)
    assert first == second
    assert len(first) == 20
    assert len({candidate.bicluster.signature for candidate in first}) > 1
    for candidate in first:
        run_context.candidate_validity_policy.validate(candidate, run_context.dataset)
    with pytest.raises(ValueError, match="non-negative"):
        initializer.initialize(run_context, -1)


def test_random_move_emitter_covers_restart_and_parent_variation(
    run_context: RunContext,
) -> None:
    emitter = RandomMoveEmitter()
    restarts = emitter.emit(run_context, Repertoire(), 5)
    assert len(restarts) == 5
    parent = Evaluation(candidate=restarts[0], objectives=(), descriptors=())
    children = emitter.emit(run_context, Repertoire(evaluations=(parent,)), 50)
    assert len(children) == 50
    assert all(child.generation == 1 for child in children)
    assert any(
        child.bicluster.row_indices != parent.candidate.bicluster.row_indices for child in children
    )
    assert any(
        child.bicluster.column_indices != parent.candidate.bicluster.column_indices
        for child in children
    )
    with pytest.raises(ValueError, match="non-negative"):
        emitter.emit(run_context, Repertoire(), -1)


def test_random_move_handles_a_single_possible_full_dataset_candidate(
    run_context: RunContext,
) -> None:
    policy = MinimumCardinality(
        min_rows=run_context.dataset.row_count,
        min_columns=run_context.dataset.column_count,
    )
    fixed_context = replace(
        run_context,
        random_streams=NamedRandomStreams(19),
        candidate_validity_policy=policy,
    )
    full = Candidate(
        identifier="full",
        bicluster=Bicluster(
            row_indices=tuple(range(run_context.dataset.row_count)),
            column_indices=tuple(range(run_context.dataset.column_count)),
        ),
    )
    child = RandomMoveEmitter().emit(
        fixed_context,
        Repertoire(evaluations=(Evaluation(candidate=full, objectives=(), descriptors=()),)),
        1,
    )[0]
    assert child.bicluster == full.bicluster


def test_membership_move_helpers_cover_add_remove_swap_and_fixed_cases() -> None:
    generator = np.random.default_rng(5)
    assert len(_random_membership_move((0, 1), 5, 1, "add", generator) or ()) == 3
    assert len(_random_membership_move((0, 1), 5, 1, "remove", generator) or ()) == 1
    assert len(_random_membership_move((0, 1), 5, 1, "swap", generator) or ()) == 2
    assert _random_membership_move((0, 1), 2, 2, "add", generator) is None
    assert _random_membership_move((0, 1), 2, 2, "remove", generator) is None
    assert _random_membership_move((0, 1), 2, 2, "swap", generator) is None


def test_search_domain_models_reject_inconsistent_state() -> None:
    supported = (BinningStrategy.LINEAR,)
    with pytest.raises(ValidationError, match="minimum"):
        DescriptorDomain(
            value_kind=DescriptorValueKind.CONTINUOUS,
            minimum=2,
            maximum=1,
            supported_binnings=supported,
            recommended_binning=BinningStrategy.LINEAR,
        )
    with pytest.raises(ValidationError, match="at least one"):
        DescriptorDomain(
            value_kind=DescriptorValueKind.CONTINUOUS,
            minimum=0,
            maximum=1,
            supported_binnings=(),
            recommended_binning=BinningStrategy.LINEAR,
        )
    with pytest.raises(ValidationError, match="recommended"):
        DescriptorDomain(
            value_kind=DescriptorValueKind.CONTINUOUS,
            minimum=0,
            maximum=1,
            supported_binnings=supported,
            recommended_binning=BinningStrategy.GEOMETRIC,
        )
    with pytest.raises(ValidationError, match="integral"):
        DescriptorDomain(
            value_kind=DescriptorValueKind.INTEGER,
            minimum=1.5,
            maximum=2,
            supported_binnings=supported,
            recommended_binning=BinningStrategy.LINEAR,
        )
    with pytest.raises(ValidationError, match="minimum cardinality"):
        CandidateBounds(min_rows=3, max_rows=2, min_columns=1, max_columns=2)
    with pytest.raises(ValidationError, match="at least one axis"):
        ArchiveCellCoordinate(indices=())
    with pytest.raises(ValidationError, match="cannot be empty"):
        ArchiveCell(coordinate=ArchiveCellCoordinate(indices=(0,)), evaluations=())
    with pytest.raises(ValidationError, match="must equal"):
        SearchProgress(
            evaluations=2,
            accepted=2,
            rejected=1,
            occupied_cells=1,
            repertoire_size=1,
        )
    assert ArchiveInsertionOutcome(
        candidate_identifier="candidate",
        status=ArchiveInsertionStatus.INSERTED_WITH_EVICTIONS,
    ).accepted
    assert not ArchiveInsertionOutcome(
        candidate_identifier="candidate",
        status=ArchiveInsertionStatus.REJECTED_DOMINATED,
    ).accepted


def test_axis_configuration_and_binner_validation_is_explicit() -> None:
    with pytest.raises(ValidationError, match="does not accept boundaries"):
        ArchiveAxisConfiguration(
            descriptor="value",
            binning=BinningStrategy.LINEAR,
            bins=2,
            boundaries=(0.5,),
        )
    with pytest.raises(ValidationError, match="does not accept bins"):
        ArchiveAxisConfiguration(
            descriptor="value",
            binning=BinningStrategy.EXACT,
            bins=2,
        )
    with pytest.raises(ValidationError, match="requires boundaries"):
        ArchiveAxisConfiguration(
            descriptor="value",
            binning=BinningStrategy.CUSTOM,
        )
    with pytest.raises(ValidationError, match="cannot exceed"):
        ArchiveAxisConfiguration(
            descriptor="value",
            binning=BinningStrategy.EXACT,
            minimum=3,
            maximum=2,
        )

    continuous = DescriptorDomain(
        value_kind=DescriptorValueKind.CONTINUOUS,
        minimum=0,
        maximum=10,
        supported_binnings=(BinningStrategy.LINEAR, BinningStrategy.CUSTOM),
        recommended_binning=BinningStrategy.LINEAR,
    )
    with pytest.raises(ComponentError, match="does not support"):
        AxisBinner.create(
            ArchiveAxisConfiguration(
                descriptor="value",
                binning=BinningStrategy.GEOMETRIC,
                bins=2,
            ),
            continuous,
        )
    with pytest.raises(ComponentError, match="exceed"):
        AxisBinner.create(
            ArchiveAxisConfiguration(
                descriptor="value",
                binning=BinningStrategy.LINEAR,
                bins=2,
                maximum=11,
            ),
            continuous,
        )
    with pytest.raises(ComponentError, match="strictly inside"):
        AxisBinner.create(
            ArchiveAxisConfiguration(
                descriptor="value",
                binning=BinningStrategy.CUSTOM,
                boundaries=(10,),
            ),
            continuous,
        )
    exact_value = AxisBinner(
        descriptor="value",
        strategy=BinningStrategy.EXACT,
        minimum=1,
        maximum=3,
        bin_count=3,
    )
    assert exact_value.index(1.5) is None
    assert exact_value.index(float("nan")) is None
    one_bin = AxisBinner(
        descriptor="value",
        strategy=BinningStrategy.LINEAR,
        minimum=1,
        maximum=1,
        bin_count=1,
    )
    assert one_bin.index(1) == 0
