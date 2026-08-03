"""Robust multiplicative pattern fitting with alternating median ratios."""

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
    multiplicative_row_effects,
    normalized_absolute_residuals,
    robust_column_scales,
)
from salvi.patterns.math import (
    NUMERIC_TOLERANCE,
    clamp01,
    diagnostics,
    nanmedian_2d,
)


@dataclass(frozen=True, slots=True)
class MultiplicativePatternFitter:
    """Fit ``x_ij / scale_j = alpha_i * beta_j`` for one numeric subgroup."""

    definition: PatternDefinition = field(
        default_factory=lambda: PatternDefinition(
            kind=PatternKind.MULTIPLICATIVE,
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
            return GroupPatternProposal(PatternKind.MULTIPLICATIVE, (), None)
        dataset = context.dataset
        if not dataset.has_robust_scaling:
            raise ValueError("multiplicative fitting requires robust numeric scaling")
        for column_index in columns:
            if not self.definition.supports(dataset.column_metadata(column_index).kind):
                return rejected_group(
                    PatternKind.MULTIPLICATIVE,
                    column_index,
                    issue_code=EvaluationIssueCode.UNSUPPORTED_COLUMN_KIND,
                    reason="multiplicative patterns require numeric columns",
                )

        rows = np.asarray(bicluster.row_indices, dtype=np.int64)
        scales = robust_column_scales(context, columns)
        positions = np.asarray(
            [dataset.numeric_positions[column] for column in columns],
            dtype=np.int64,
        )
        column_array = np.asarray(columns, dtype=np.int64)
        raw_matrix = dataset.numeric_matrix()[np.ix_(rows, positions)]
        matrix = raw_matrix / scales[np.newaxis, :]
        source = dataset.support_matrix()[np.ix_(rows, column_array)]
        source_support = np.count_nonzero(source, axis=0)
        required_row_support = context.evaluation_support_policy.required_observations(len(rows))
        insufficient = np.flatnonzero(source_support < required_row_support)
        if insufficient.size:
            return rejected_group(
                PatternKind.MULTIPLICATIVE,
                columns[int(insufficient[0])],
                issue_code=EvaluationIssueCode.INSUFFICIENT_LOCAL_SUPPORT,
                reason=("insufficient original observations for multiplicative fitting"),
            )

        initialized = self._initialize(matrix)
        if initialized is None:
            return rejected_group(
                PatternKind.MULTIPLICATIVE,
                columns[0],
                issue_code=EvaluationIssueCode.PATTERN_FIT_FAILED,
                reason="multiplicative initialization has no non-zero rank-one signal",
            )
        alpha, beta = initialized
        iterations = 0
        converged = False
        for iteration in range(context.patterns.max_iterations):
            next_alpha = multiplicative_row_effects(context, matrix, source, beta)
            next_beta, rejected = self._column_effects(
                context,
                matrix,
                source,
                next_alpha,
                columns,
            )
            if rejected is not None:
                return rejected_group(
                    PatternKind.MULTIPLICATIVE,
                    rejected,
                    issue_code=EvaluationIssueCode.INSUFFICIENT_GROUP_SUPPORT,
                    reason="too few supported row effects for multiplicative fitting",
                )
            normalized = self._normalize(next_alpha, next_beta)
            if normalized is None:
                return rejected_group(
                    PatternKind.MULTIPLICATIVE,
                    columns[0],
                    issue_code=EvaluationIssueCode.PATTERN_FIT_FAILED,
                    reason="multiplicative effects collapsed to zero",
                )
            next_alpha, next_beta = normalized
            change = maximum_parameter_change(alpha, beta, next_alpha, next_beta)
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
                raw_matrix - scales[np.newaxis, :] * alpha[:, np.newaxis] * beta[np.newaxis, :],
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
                    PatternKind.MULTIPLICATIVE,
                    column_index,
                    issue_code=EvaluationIssueCode.INSUFFICIENT_GROUP_SUPPORT,
                    reason="insufficient fitted multiplicative residual support",
                    fits=tuple(fits),
                )
            available_count = int(available_counts[position])
            source_count = int(source_counts[position])
            fits.append(
                (
                    column_index,
                    PatternCandidateFit.model_construct(
                        pattern=PatternKind.MULTIPLICATIVE,
                        error=clamp01(float(errors[position])),
                        parameter=float(beta[position]),
                        parameter_scale=ParameterScale.ROBUST_SCALED,
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
            identifier="MULTIPLICATIVE-0",
            pattern=PatternKind.MULTIPLICATIVE,
            column_indices=columns,
            row_parameters=row_parameters,
            iterations=iterations,
            converged=converged,
            diagnostics=diagnostics(
                selected_rows=len(rows),
                fitted_row_effects=len(row_parameters),
                normalization="median_absolute_row_effect",
                residual_normalization="per_column_global_robust_range",
            ),
        )
        return GroupPatternProposal(
            PatternKind.MULTIPLICATIVE,
            tuple(fits),
            group,
        )

    @staticmethod
    def _column_effects(
        context: RunContext,
        matrix: npt.NDArray[np.float64],
        source: npt.NDArray[np.bool_],
        alpha: npt.NDArray[np.float64],
        columns: tuple[int, ...],
    ) -> tuple[npt.NDArray[np.float64], int | None]:
        usable_alpha = np.isfinite(alpha) & (np.abs(alpha) > NUMERIC_TOLERANCE)
        usable = np.isfinite(matrix) & usable_alpha[:, np.newaxis]
        original = np.count_nonzero(usable & source, axis=0)
        required = context.evaluation_support_policy.required_observations(matrix.shape[0])
        insufficient = np.flatnonzero(original < required)
        if insufficient.size:
            return (
                np.full(matrix.shape[1], np.nan, dtype=np.float64),
                columns[int(insufficient[0])],
            )
        ratios = np.full_like(matrix, np.nan)
        np.divide(
            matrix,
            alpha[:, np.newaxis],
            out=ratios,
            where=usable,
        )
        return nanmedian_2d(ratios, axis=0), None

    @classmethod
    def _initialize(
        cls,
        matrix: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
        finite_counts = np.count_nonzero(np.isfinite(matrix), axis=0)
        if np.any(finite_counts == 0):
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            percentiles = np.nanquantile(
                matrix,
                (0.05, 0.95),
                axis=0,
                method="linear",
            )
        ranges = percentiles[1] - percentiles[0]
        candidates = tuple(
            (float(ranges[position]), int(finite_counts[position]), -position)
            for position in range(matrix.shape[1])
        )
        anchor_position = -max(candidates)[2]
        anchor = matrix[:, anchor_position]
        nonzero_anchor = np.isfinite(anchor) & (np.abs(anchor) > NUMERIC_TOLERANCE)
        if not np.any(nonzero_anchor):
            return None
        anchor_scale = float(np.median(np.abs(anchor[nonzero_anchor])))
        alpha = np.full(matrix.shape[0], np.nan, dtype=np.float64)
        alpha[np.isfinite(anchor)] = anchor[np.isfinite(anchor)] / anchor_scale
        usable_alpha = np.isfinite(alpha) & (np.abs(alpha) > NUMERIC_TOLERANCE)
        usable = np.isfinite(matrix) & usable_alpha[:, np.newaxis]
        if np.any(np.count_nonzero(usable, axis=0) == 0):
            return None
        ratios = np.full_like(matrix, np.nan)
        np.divide(
            matrix,
            alpha[:, np.newaxis],
            out=ratios,
            where=usable,
        )
        beta = nanmedian_2d(ratios, axis=0)
        return cls._normalize(alpha, beta)

    @staticmethod
    def _normalize(
        alpha: npt.NDArray[np.float64],
        beta: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
        finite_nonzero = np.isfinite(alpha) & (np.abs(alpha) > NUMERIC_TOLERANCE)
        if not np.any(finite_nonzero):
            return None
        scale = float(np.median(np.abs(alpha[finite_nonzero])))
        if scale <= NUMERIC_TOLERANCE:
            return None
        normalized_alpha = alpha.copy()
        normalized_alpha[np.isfinite(normalized_alpha)] /= scale
        normalized_beta = beta * scale
        significant = np.flatnonzero(np.abs(normalized_beta) > NUMERIC_TOLERANCE)
        if significant.size and normalized_beta[int(significant[0])] < 0.0:
            normalized_alpha = -normalized_alpha
            normalized_beta = -normalized_beta
        return normalized_alpha, normalized_beta


__all__ = ["MultiplicativePatternFitter"]
