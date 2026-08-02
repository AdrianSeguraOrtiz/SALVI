"""Robust constant pattern fitting."""

from __future__ import annotations

import warnings
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
from salvi.domain.models import Bicluster, PatternCandidateFit
from salvi.patterns.contracts import PatternDefinition
from salvi.patterns.fitters.common import invalid_candidate, support_counts
from salvi.patterns.math import NUMERIC_TOLERANCE, clamp01, diagnostics


@dataclass(frozen=True, slots=True)
class ConstantPatternFitter:
    definition: PatternDefinition = field(
        default_factory=lambda: PatternDefinition(
            kind=PatternKind.CONSTANT,
            scope=PatternScope.COLUMN,
            supported_column_kinds=frozenset(ColumnKind),
            reference_model=True,
        )
    )

    def fit_column(
        self,
        context: RunContext,
        bicluster: Bicluster,
        column_index: int,
    ) -> PatternCandidateFit:
        dataset = context.dataset
        metadata = dataset.column_metadata(column_index)
        if not self.definition.supports(metadata.kind):
            return invalid_candidate(
                PatternKind.CONSTANT,
                reason=f"unsupported column kind {metadata.kind.value}",
                issue_code=EvaluationIssueCode.UNSUPPORTED_COLUMN_KIND,
            )
        rows = np.asarray(bicluster.row_indices, dtype=np.int64)
        source_support, available_support = support_counts(context, rows, column_index)
        if not context.evaluation_support_policy.is_sufficient(source_support, len(rows)):
            return invalid_candidate(
                PatternKind.CONSTANT,
                source_support=source_support,
                available_support=available_support,
                reason="insufficient original observations",
                issue_code=EvaluationIssueCode.INSUFFICIENT_LOCAL_SUPPORT,
            )
        if metadata.kind is ColumnKind.NUMERIC:
            return self._fit_numeric(
                context,
                rows,
                column_index,
                source_support,
                available_support,
            )
        return self._fit_discrete(
            context,
            rows,
            column_index,
            source_support,
            available_support,
        )

    def fit_columns(
        self,
        context: RunContext,
        bicluster: Bicluster,
    ) -> tuple[PatternCandidateFit, ...]:
        """Fit all selected columns while sharing masks and numeric matrix scans."""

        dataset = context.dataset
        rows = np.asarray(bicluster.row_indices, dtype=np.int64)
        columns = np.asarray(bicluster.column_indices, dtype=np.int64)
        source_supports = np.count_nonzero(
            dataset.support_matrix()[np.ix_(rows, columns)],
            axis=0,
        )
        available_supports = np.count_nonzero(
            dataset.available[np.ix_(rows, columns)],
            axis=0,
        )
        fits: list[PatternCandidateFit | None] = [None] * len(columns)

        numeric_entries = tuple(
            (offset, int(column), dataset.numeric_positions[int(column)])
            for offset, column in enumerate(columns)
            if dataset.numeric_positions[int(column)] >= 0
        )
        sufficient_numeric = tuple(
            entry
            for entry in numeric_entries
            if context.evaluation_support_policy.is_sufficient(
                int(source_supports[entry[0]]),
                len(rows),
            )
        )
        if sufficient_numeric:
            offsets = tuple(entry[0] for entry in sufficient_numeric)
            positions = np.asarray(
                [entry[2] for entry in sufficient_numeric],
                dtype=np.int64,
            )
            matrix = dataset.numeric_matrix()[np.ix_(rows, positions)]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                prototypes = np.nanmedian(matrix, axis=0)
            finite = np.isfinite(matrix)
            available_counts = np.count_nonzero(finite, axis=0)
            ranges = np.asarray(
                [dataset.numeric_statistics[int(position)].robust_range for position in positions],
                dtype=np.float64,
            )
            deviations = np.abs(matrix - prototypes[np.newaxis, :])
            with np.errstate(divide="ignore", invalid="ignore"):
                contributions = np.where(
                    ranges[np.newaxis, :] <= NUMERIC_TOLERANCE,
                    deviations > NUMERIC_TOLERANCE,
                    np.minimum(1.0, deviations / ranges[np.newaxis, :]),
                ).astype(np.float64)
            contributions[~finite] = np.nan
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                errors = np.nanmean(contributions, axis=0)
            for position, offset in enumerate(offsets):
                if available_counts[position] == 0 or not np.isfinite(prototypes[position]):
                    fits[offset] = invalid_candidate(
                        PatternKind.CONSTANT,
                        source_support=int(source_supports[offset]),
                        available_support=int(available_supports[offset]),
                        reason="no values available for fitting",
                        issue_code=EvaluationIssueCode.PATTERN_FIT_FAILED,
                    )
                    continue
                fits[offset] = PatternCandidateFit.model_construct(
                    pattern=PatternKind.CONSTANT,
                    error=clamp01(float(errors[position])),
                    parameter=float(prototypes[position]),
                    parameter_scale=ParameterScale.RAW,
                    source_support=int(source_supports[offset]),
                    available_support=int(available_counts[position]),
                    diagnostics=diagnostics(robust_range=float(ranges[position])),
                )

        for offset, raw_column in enumerate(columns):
            if fits[offset] is not None:
                continue
            column_index = int(raw_column)
            source_support = int(source_supports[offset])
            available_support = int(available_supports[offset])
            if not context.evaluation_support_policy.is_sufficient(source_support, len(rows)):
                fits[offset] = invalid_candidate(
                    PatternKind.CONSTANT,
                    source_support=source_support,
                    available_support=available_support,
                    reason="insufficient original observations",
                    issue_code=EvaluationIssueCode.INSUFFICIENT_LOCAL_SUPPORT,
                )
            elif dataset.numeric_positions[column_index] >= 0:
                fits[offset] = invalid_candidate(
                    PatternKind.CONSTANT,
                    source_support=source_support,
                    available_support=available_support,
                    reason="no values available for fitting",
                    issue_code=EvaluationIssueCode.PATTERN_FIT_FAILED,
                )
            else:
                fits[offset] = self._fit_discrete(
                    context,
                    rows,
                    column_index,
                    source_support,
                    available_support,
                )
        if any(fit is None for fit in fits):  # pragma: no cover - defensive invariant
            raise RuntimeError("constant batch fitting did not assign every selected column")
        return tuple(fit for fit in fits if fit is not None)

    @staticmethod
    def _fit_numeric(
        context: RunContext,
        rows: npt.NDArray[np.int64],
        column_index: int,
        source_support: int,
        available_support: int,
    ) -> PatternCandidateFit:
        dataset = context.dataset
        values = dataset.numeric_column(column_index)[rows]
        available_values = values[np.isfinite(values)]
        if available_values.size == 0:
            return invalid_candidate(
                PatternKind.CONSTANT,
                source_support=source_support,
                available_support=available_support,
                reason="no values available for fitting",
                issue_code=EvaluationIssueCode.PATTERN_FIT_FAILED,
            )
        prototype = float(np.median(available_values))
        position = dataset.numeric_positions[column_index]
        statistics = dataset.numeric_statistics[position]
        deviations = np.abs(available_values - prototype)
        if statistics.robust_range <= NUMERIC_TOLERANCE:
            contributions = (deviations > NUMERIC_TOLERANCE).astype(np.float64)
        else:
            contributions = np.minimum(1.0, deviations / statistics.robust_range)
        return PatternCandidateFit.model_construct(
            pattern=PatternKind.CONSTANT,
            error=clamp01(float(np.mean(contributions))),
            parameter=prototype,
            parameter_scale=ParameterScale.RAW,
            source_support=source_support,
            available_support=int(available_values.size),
            diagnostics=diagnostics(robust_range=statistics.robust_range),
        )

    @staticmethod
    def _fit_discrete(
        context: RunContext,
        rows: npt.NDArray[np.int64],
        column_index: int,
        source_support: int,
        available_support: int,
    ) -> PatternCandidateFit:
        dataset = context.dataset
        metadata = dataset.column_metadata(column_index)
        codes = dataset.discrete_column(column_index)[rows]
        observed_codes = codes[codes >= 0]
        if observed_codes.size == 0:
            return invalid_candidate(
                PatternKind.CONSTANT,
                source_support=source_support,
                available_support=available_support,
                reason="no values available for fitting",
                issue_code=EvaluationIssueCode.PATTERN_FIT_FAILED,
            )
        global_frequencies = dataset.discrete_global_frequencies(column_index)
        frequencies = np.bincount(
            observed_codes,
            minlength=len(global_frequencies),
        )
        prototype_code = int(np.argmax(frequencies))
        prototype = dataset.discrete_value(column_index, prototype_code)
        prototype_support = int(frequencies[prototype_code])
        global_cardinality = dataset.discrete_observed_cardinality(column_index)
        error = 0.0
        if global_cardinality > 1:
            mismatch = 1.0 - prototype_support / observed_codes.size
            error = mismatch / (1.0 - 1.0 / global_cardinality)
        return PatternCandidateFit.model_construct(
            pattern=PatternKind.CONSTANT,
            error=clamp01(error),
            parameter=(
                str(prototype) if metadata.kind is ColumnKind.CATEGORICAL else bool(prototype)
            ),
            parameter_scale=ParameterScale.CATEGORY_LABEL,
            source_support=source_support,
            available_support=int(observed_codes.size),
            prototype_support=prototype_support,
            diagnostics=diagnostics(global_observed_cardinality=global_cardinality),
        )


__all__ = ["ConstantPatternFitter"]
