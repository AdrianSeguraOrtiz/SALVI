from __future__ import annotations

from itertools import combinations

from salvi.components.operators import _repair_joint_group_support
from salvi.domain import (
    Bicluster,
    Candidate,
    ColumnPatternFit,
    Evaluation,
    PatternCandidateFit,
    PatternFit,
    PatternGroupFit,
    PatternKind,
)


def _parent(identifier: str, columns: tuple[int, ...], group: tuple[int, ...]) -> Evaluation:
    bicluster = Bicluster(row_indices=(0, 1), column_indices=columns)
    fits = []
    for column in columns:
        pattern = PatternKind.ADDITIVE if column in group else PatternKind.CONSTANT
        fits.append(
            ColumnPatternFit(
                column_index=column,
                pattern=pattern,
                group_identifier="joint" if column in group else None,
                error=0.0,
                alternatives=(PatternCandidateFit(pattern=pattern, error=0.0),),
            )
        )
    return Evaluation(
        candidate=Candidate(identifier=identifier, bicluster=bicluster),
        objectives=(),
        descriptors=(),
        pattern_fit=PatternFit(
            candidate_signature=bicluster.signature,
            row_indices=bicluster.row_indices,
            column_indices=columns,
            columns=tuple(fits),
            groups=(
                PatternGroupFit(
                    identifier="joint", pattern=PatternKind.ADDITIVE, column_indices=group
                ),
            )
            if group
            else (),
        ),
    )


def test_repair_keeps_the_existing_group_member() -> None:
    first = _parent("first", (0, 1, 3), (0, 1))
    second = _parent("second", (2, 3), ())
    repaired = _repair_joint_group_support(
        (0, 2, 3),
        frozenset({3}),
        first,
        second,
        {0: 0.8, 1: 0.2, 2: 0.1, 3: 0.0},
    )
    assert repaired == (0, 1, 3)


def test_repair_does_not_break_a_complete_group_from_the_other_parent() -> None:
    first = _parent("first", (0, 1, 4), (0, 1))
    second = _parent("second", (2, 3, 4), (2, 3))
    repaired = _repair_joint_group_support(
        (0, 2, 3, 4),
        frozenset(),
        first,
        second,
        {0: 0.1, 1: 0.1, 2: 0.9, 3: 0.8, 4: 0.0},
    )
    assert repaired == (0, 1, 2, 3)


def test_unrepairable_groups_leave_membership_and_shared_columns_unchanged() -> None:
    first = _parent("first", (0, 1, 4), (0, 1))
    second = _parent("second", (2, 3, 4), (2, 3))
    selected = (0, 2, 3, 4)
    assert _repair_joint_group_support(selected, frozenset({4}), first, second, {}) == selected


def test_overlapping_group_repairs_preserve_invariants_and_reach_a_fixed_point() -> None:
    first = _parent("first", (0, 1, 2, 4), (0, 1, 2))
    second = _parent("second", (1, 2, 3, 4), (1, 2, 3))
    groups = ({0, 1, 2}, {1, 2, 3})
    for width in (2, 3, 4):
        for columns in combinations(range(5), width):
            shared = frozenset(set(columns) & {4})
            repaired = _repair_joint_group_support(columns, shared, first, second, {})
            assert len(repaired) == len(columns)
            assert shared <= set(repaired) <= set(range(5))
            for group in groups:
                if len(set(columns) & group) != 1:
                    assert len(set(repaired) & group) != 1
            assert sum(len(set(repaired) & group) == 1 for group in groups) <= sum(
                len(set(columns) & group) == 1 for group in groups
            )
            assert _repair_joint_group_support(repaired, shared, first, second, {}) == repaired
