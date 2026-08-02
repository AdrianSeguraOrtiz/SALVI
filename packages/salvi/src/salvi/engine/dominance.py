"""Direction-aware Pareto dominance utilities."""

from __future__ import annotations

from collections.abc import Sequence

from salvi.domain.enums import ObjectiveDirection
from salvi.domain.models import ConstraintValue, Evaluation, ObjectiveValue
from salvi.exceptions import ComponentError


def validate_objective_schema(
    evaluation: Evaluation,
    expected: Sequence[tuple[str, ObjectiveDirection]],
) -> None:
    actual = tuple((value.name, value.direction) for value in evaluation.objectives)
    if actual != tuple(expected):
        raise ComponentError(
            f"evaluation objective schema {actual!r} does not match archive schema "
            f"{tuple(expected)!r}"
        )


def objective_dominates(
    left: Sequence[ObjectiveValue],
    right: Sequence[ObjectiveValue],
) -> bool:
    if len(left) != len(right):
        raise ValueError("dominance requires the same number of objectives")
    strictly_better = False
    for left_value, right_value in zip(left, right, strict=True):
        if left_value.name != right_value.name or left_value.direction is not right_value.direction:
            raise ValueError("dominance requires matching objective names and directions")
        if left_value.direction is ObjectiveDirection.MINIMIZE:
            if left_value.value > right_value.value:
                return False
            strictly_better |= left_value.value < right_value.value
        else:
            if left_value.value < right_value.value:
                return False
            strictly_better |= left_value.value > right_value.value
    return strictly_better


def validate_constraint_schema(
    evaluation: Evaluation,
    expected: Sequence[str],
) -> None:
    actual = tuple(value.name for value in evaluation.constraints)
    if actual != tuple(expected):
        raise ComponentError(
            f"evaluation constraint schema {actual!r} does not match expected schema "
            f"{tuple(expected)!r}"
        )


def total_constraint_violation(constraints: Sequence[ConstraintValue]) -> float:
    return sum(constraint.violation for constraint in constraints)


def constrained_dominates(left: Evaluation, right: Evaluation) -> bool:
    """Apply feasibility first, aggregate violation second, and Pareto last."""

    left_violation = total_constraint_violation(left.constraints)
    right_violation = total_constraint_violation(right.constraints)
    if left_violation == 0.0 and right_violation > 0.0:
        return True
    if left_violation > 0.0 and right_violation == 0.0:
        return False
    if left_violation != right_violation:
        return left_violation < right_violation
    return objective_dominates(left.objectives, right.objectives)


__all__ = [
    "constrained_dominates",
    "objective_dominates",
    "total_constraint_violation",
    "validate_constraint_schema",
    "validate_objective_schema",
]
