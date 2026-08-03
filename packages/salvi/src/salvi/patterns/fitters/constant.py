"""Robust constant pattern fitting."""

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
from salvi.domain.models import Bicluster, PatternCandidateFit
from salvi.patterns.contracts import PatternDefinition
from salvi.patterns.fitters.common import invalid_candidate, support_counts
from salvi.patterns.math import NUMERIC_TOLERANCE, clamp01, diagnostics, nanmedian_2d


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
        column_indices: Sequence[int] | None = None,
    ) -> tuple[PatternCandidateFit, ...]:
        """Fit selected columns while sharing masks and numeric matrix scans."""

        dataset = context.dataset
        rows = np.asarray(bicluster.row_indices, dtype=np.int64)
        selected_columns = (
            bicluster.column_indices if column_indices is None else tuple(column_indices)
        )
        columns = np.asarray(selected_columns, dtype=np.int64)
        source_supports = np.count_nonzero(
            dataset.support_matrix()[np.ix_(rows, columns)],
            axis=0,
        )
        available_supports = np.count_nonzero(
            dataset.available[np.ix_(rows, columns)],
            axis=0,
        )
        fits: list[PatternCandidateFit | None] = [None] * len(columns)
        required_support = context.evaluation_support_policy.required_observations(len(rows))

        numeric_entries = tuple(
            (offset, int(column), dataset.numeric_positions[int(column)])
            for offset, column in enumerate(columns)
            if dataset.numeric_positions[int(column)] >= 0
        )
        sufficient_numeric = tuple(
            entry
            for entry in numeric_entries
            if source_supports[entry[0]] >= required_support
        )
        if sufficient_numeric:
            offsets = tuple(entry[0] for entry in sufficient_numeric)
            positions = np.asarray(
                [entry[2] for entry in sufficient_numeric],
                dtype=np.int64,
            )
            numeric_matrix = dataset.numeric_matrix()[np.ix_(rows, positions)]
            prototypes = nanmedian_2d(numeric_matrix, axis=0)
            finite = np.isfinite(numeric_matrix)
            available_counts = np.count_nonzero(finite, axis=0)
            ranges = np.asarray(
                [dataset.numeric_statistics[int(position)].robust_range for position in positions],
                dtype=np.float64,
            )
            deviations = np.abs(numeric_matrix - prototypes[np.newaxis, :])
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

        discrete_entries = tuple(
            (offset, int(column), dataset.discrete_positions[int(column)])
            for offset, column in enumerate(columns)
            if dataset.discrete_positions[int(column)] >= 0
            and source_supports[offset] >= required_support
        )
        if discrete_entries:
            offsets = tuple(entry[0] for entry in discrete_entries)
            positions = np.asarray([entry[2] for entry in discrete_entries], dtype=np.int64)
            discrete_matrix = dataset.discrete_matrix()[np.ix_(rows, positions)]
            for position, offset in enumerate(offsets):
                column_index = int(columns[offset])
                observed_codes = discrete_matrix[:, position]
                observed_codes = observed_codes[observed_codes >= 0]
                if observed_codes.size == 0:
                    fits[offset] = invalid_candidate(
                        PatternKind.CONSTANT,
                        source_support=int(source_supports[offset]),
                        available_support=int(available_supports[offset]),
                        reason="no values available for fitting",
                        issue_code=EvaluationIssueCode.PATTERN_FIT_FAILED,
                    )
                    continue
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
                metadata = dataset.columns[column_index]
                fits[offset] = PatternCandidateFit.model_construct(
                    pattern=PatternKind.CONSTANT,
                    error=clamp01(error),
                    parameter=(
                        str(prototype)
                        if metadata.kind is ColumnKind.CATEGORICAL
                        else bool(prototype)
                    ),
                    parameter_scale=ParameterScale.CATEGORY_LABEL,
                    source_support=int(source_supports[offset]),
                    available_support=int(observed_codes.size),
                    prototype_support=prototype_support,
                    diagnostics=diagnostics(global_observed_cardinality=global_cardinality),
                )

        for offset, raw_column in enumerate(columns):
            if fits[offset] is not None:
                continue
            column_index = int(raw_column)
            source_support = int(source_supports[offset])
            available_support = int(available_supports[offset])
            if source_support < required_support:
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
            else:  # pragma: no cover - every supported discrete column is batched above
                raise RuntimeError("constant batch fitting missed a supported column")
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
