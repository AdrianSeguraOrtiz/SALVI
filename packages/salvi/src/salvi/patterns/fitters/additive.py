"""Robust additive pattern fitting with alternating medians."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from salvi.application.context import RunContext
from salvi.domain.enums import (
    ColumnKind,
    EvaluationIssueCode,
    ParameterScale,
    PatternKind,
    PatternScope,
)
from salvi.domain.models import Bicluster, PatternCandidateFit, PatternGroupFit
from salvi.patterns.contracts import GroupPatternProposal, PatternDefinition
from salvi.patterns.fitters.common import (
    maximum_parameter_change,
    rejected_group,
)
from salvi.patterns.joint_models import (
    additive_row_effects,
    normalized_absolute_residuals,
    robust_column_scales,
)
from salvi.patterns.math import clamp01, diagnostics, nanmedian_2d


@dataclass(frozen=True, slots=True)
class AdditivePatternFitter:
    """Fit the strict additive model ``x_ij = alpha_i + beta_j``."""

    definition: PatternDefinition = field(
        default_factory=lambda: PatternDefinition(
            kind=PatternKind.ADDITIVE,
            scope=PatternScope.SUBSET,
            supported_column_kinds=frozenset({ColumnKind.NUMERIC}),
            minimum_columns=2,
            maximum_groups=1,
        )
    )

    def fit_group(
        self,
        context: RunContext,
        bicluster: Bicluster,
        column_indices: Sequence[int],
    ) -> GroupPatternProposal:
        columns = tuple(sorted(column_indices))
        if len(columns) < self.definition.minimum_columns:
            return GroupPatternProposal(PatternKind.ADDITIVE, (), None)
        dataset = context.dataset
        if not dataset.has_robust_scaling:
            raise ValueError("additive fitting requires robust numeric scaling")
        for column_index in columns:
            if not self.definition.supports(dataset.column_metadata(column_index).kind):
                return rejected_group(
                    PatternKind.ADDITIVE,
                    column_index,
                    issue_code=EvaluationIssueCode.UNSUPPORTED_COLUMN_KIND,
                    reason="additive patterns require numeric columns",
                )

        rows = np.asarray(bicluster.row_indices, dtype=np.int64)
        positions = np.asarray(
            [dataset.numeric_positions[column] for column in columns],
            dtype=np.int64,
        )
        column_array = np.asarray(columns, dtype=np.int64)
        matrix = dataset.numeric_matrix()[np.ix_(rows, positions)]
        scales = robust_column_scales(context, columns)
        source = dataset.support_matrix()[np.ix_(rows, column_array)]
        source_support = np.count_nonzero(source, axis=0)
        required_row_support = context.evaluation_support_policy.required_observations(len(rows))
        insufficient = np.flatnonzero(source_support < required_row_support)
        if insufficient.size:
            return rejected_group(
                PatternKind.ADDITIVE,
                columns[int(insufficient[0])],
                issue_code=EvaluationIssueCode.INSUFFICIENT_LOCAL_SUPPORT,
                reason="insufficient original observations for additive fitting",
            )

        finite_counts = np.count_nonzero(np.isfinite(matrix), axis=0)
        empty = np.flatnonzero(finite_counts == 0)
        if empty.size:
            return rejected_group(
                PatternKind.ADDITIVE,
                columns[int(empty[0])],
                issue_code=EvaluationIssueCode.PATTERN_FIT_FAILED,
                reason="no values are available for additive fitting",
            )
        beta = nanmedian_2d(matrix, axis=0)

        alpha = np.full(len(rows), np.nan, dtype=np.float64)
        iterations = 0
        converged = False
        for iteration in range(context.patterns.max_iterations):
            next_alpha = additive_row_effects(context, matrix, source, beta)
            next_beta, rejected = self._column_effects(context, matrix, source, next_alpha, columns)
            if rejected is not None:
                return rejected_group(
                    PatternKind.ADDITIVE,
                    rejected,
                    issue_code=EvaluationIssueCode.INSUFFICIENT_GROUP_SUPPORT,
                    reason="too few supported row effects for additive fitting",
                )
            finite_alpha = next_alpha[np.isfinite(next_alpha)]
            if finite_alpha.size:
                center = float(np.median(finite_alpha))
                next_alpha[np.isfinite(next_alpha)] -= center
                next_beta += center
            change = maximum_parameter_change(alpha, beta, next_alpha, next_beta) / float(
                np.median(scales)
            )
            alpha = next_alpha
            beta = next_beta
            iterations = iteration + 1
            if change <= context.patterns.convergence_tolerance:
                converged = True
                break

        fits: list[tuple[int, PatternCandidateFit]] = []
        usable_matrix = np.isfinite(matrix) & np.isfinite(alpha)[:, np.newaxis]
        available_counts = np.count_nonzero(usable_matrix, axis=0)
        source_counts = np.count_nonzero(usable_matrix & source, axis=0)
        residual_matrix = np.where(
            usable_matrix,
            normalized_absolute_residuals(
                matrix - alpha[:, np.newaxis] - beta[np.newaxis, :],
                scales,
            ),
            np.nan,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            errors = np.nanmean(residual_matrix, axis=0)
        insufficient = np.flatnonzero(source_counts < required_row_support)
        rejected_position = int(insufficient[0]) if insufficient.size else None
        for position, column_index in enumerate(columns):
            if position == rejected_position:
                return rejected_group(
                    PatternKind.ADDITIVE,
                    column_index,
                    issue_code=EvaluationIssueCode.INSUFFICIENT_GROUP_SUPPORT,
                    reason="insufficient fitted additive residual support",
                    fits=tuple(fits),
                )
            available_count = int(available_counts[position])
            source_count = int(source_counts[position])
            fits.append(
                (
                    column_index,
                    PatternCandidateFit.model_construct(
                        pattern=PatternKind.ADDITIVE,
                        error=clamp01(float(errors[position])),
                        parameter=float(beta[position]),
                        parameter_scale=ParameterScale.RAW,
                        source_support=source_count,
                        available_support=available_count,
                        diagnostics=diagnostics(
                            converged=converged,
                            iterations=iterations,
                            numeric_scale=float(scales[position]),
                            residual_normalization="global_robust_range",
                        ),
                    ),
                )
            )
        row_parameters = tuple(
            (int(rows[position]), float(value))
            for position, value in enumerate(alpha)
            if np.isfinite(value)
        )
        group = PatternGroupFit.model_construct(
            identifier="ADDITIVE-0",
            pattern=PatternKind.ADDITIVE,
            column_indices=columns,
            row_parameters=row_parameters,
            iterations=iterations,
            converged=converged,
            diagnostics=diagnostics(
                fit_space="raw",
                selected_rows=len(rows),
                fitted_row_effects=len(row_parameters),
                residual_normalization="per_column_global_robust_range",
            ),
        )
        return GroupPatternProposal(PatternKind.ADDITIVE, tuple(fits), group)

    @staticmethod
    def _column_effects(
        context: RunContext,
        matrix: npt.NDArray[np.float64],
        source: npt.NDArray[np.bool_],
        alpha: npt.NDArray[np.float64],
        columns: tuple[int, ...],
    ) -> tuple[npt.NDArray[np.float64], int | None]:
        usable = np.isfinite(matrix) & np.isfinite(alpha)[:, np.newaxis]
        original = np.count_nonzero(usable & source, axis=0)
        required = context.evaluation_support_policy.required_observations(matrix.shape[0])
        insufficient = np.flatnonzero(original < required)
        if insufficient.size:
            return (
                np.full(matrix.shape[1], np.nan, dtype=np.float64),
                columns[int(insufficient[0])],
            )
        residuals = np.where(usable, matrix - alpha[:, np.newaxis], np.nan)
        return nanmedian_2d(residuals, axis=0), None


__all__ = ["AdditivePatternFitter"]
