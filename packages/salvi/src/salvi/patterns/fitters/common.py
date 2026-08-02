"""Shared support and failure handling for built-in pattern fitters."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from salvi.application.context import RunContext
from salvi.domain.enums import EvaluationIssueCode, PatternKind
from salvi.domain.models import EvaluationIssue, PatternCandidateFit
from salvi.patterns.contracts import GroupPatternProposal
from salvi.patterns.math import diagnostics


def support_counts(
    context: RunContext,
    rows: npt.NDArray[np.int64],
    column_index: int,
) -> tuple[int, int]:
    dataset = context.dataset
    return (
        int(np.count_nonzero(dataset.support_mask(column_index)[rows])),
        int(np.count_nonzero(dataset.available_mask(column_index)[rows])),
    )


def invalid_candidate(
    pattern: PatternKind,
    *,
    source_support: int = 0,
    available_support: int = 0,
    reason: str,
    issue_code: EvaluationIssueCode,
) -> PatternCandidateFit:
    return PatternCandidateFit(
        pattern=pattern,
        error=1.0,
        source_support=source_support,
        available_support=available_support,
        valid=False,
        issue_code=issue_code,
        diagnostics=diagnostics(reason=reason),
    )


def rejected_group(
    pattern: PatternKind,
    column_index: int,
    *,
    issue_code: EvaluationIssueCode,
    reason: str,
    fits: tuple[tuple[int, PatternCandidateFit], ...] = (),
) -> GroupPatternProposal:
    return GroupPatternProposal(
        pattern,
        fits,
        None,
        rejected_column=column_index,
        rejection_issue=EvaluationIssue(
            code=issue_code,
            message=reason,
            column_index=column_index,
            pattern=pattern,
        ),
    )


def maximum_parameter_change(
    row_parameters: npt.NDArray[np.float64],
    column_parameters: npt.NDArray[np.float64],
    next_row_parameters: npt.NDArray[np.float64],
    next_column_parameters: npt.NDArray[np.float64],
) -> float:
    if np.any(np.isfinite(row_parameters) != np.isfinite(next_row_parameters)):
        row_change = float("inf")
    else:
        finite = np.isfinite(row_parameters)
        row_change = (
            float(np.max(np.abs(row_parameters[finite] - next_row_parameters[finite])))
            if np.any(finite)
            else 0.0
        )
    return max(
        row_change,
        float(np.max(np.abs(column_parameters - next_column_parameters))),
    )


__all__ = [
    "invalid_candidate",
    "maximum_parameter_change",
    "rejected_group",
    "support_counts",
]
