from __future__ import annotations

import json
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from salvi.application.context import NamedRandomStreams, RunContext
from salvi.components.evaluation_policies import MinimumCardinality, MinimumObservedSupport
from salvi.components.execution import SerialEvaluationExecutor
from salvi.components.objectives import Contrast, InternalCoherence
from salvi.components.preprocessing import MedianModeImputation, RobustNumericScaling
from salvi.domain import (
    Bicluster,
    Candidate,
    ColumnKind,
    ColumnMetadata,
    Dataset,
    EvaluationIssueCode,
    ObjectiveDirection,
    ParameterScale,
    PatternKind,
    PatternScope,
    PreparedDataset,
)
from salvi.evaluation import EvaluationWorkspace
from salvi.patterns import PatternCatalog, PatternConfiguration, default_pattern_catalog
from salvi.patterns.contracts import (
    GroupPatternFitter,
    GroupPatternProposal,
    PatternImplementation,
)
from salvi.patterns.fitters.constant import ConstantPatternFitter
from salvi.patterns.math import nanmedian_2d


def _context(
    table: pa.Table,
    columns: tuple[ColumnMetadata, ...],
    *,
    patterns: tuple[PatternKind, ...] = (PatternKind.CONSTANT,),
    min_improvement: float = 0.1,
    min_observed_ratio: float = 0.8,
    max_iterations: int = 25,
) -> RunContext:
    dataset = Dataset(
        identifier="scientific-fixture",
        bundle_path="fixture",
        row_count=table.num_rows,
        column_count=table.num_columns,
        columns=columns,
    )
    prepared = PreparedDataset.from_arrow(
        dataset,
        table,
        pa.array([f"row-{index}" for index in range(table.num_rows)]),
    )
    prepared = RobustNumericScaling().transform(prepared)
    return _prepared_context(
        prepared,
        patterns=patterns,
        min_improvement=min_improvement,
        min_observed_ratio=min_observed_ratio,
        max_iterations=max_iterations,
    )


def _prepared_context(
    prepared: PreparedDataset,
    *,
    patterns: tuple[PatternKind, ...] = (PatternKind.CONSTANT,),
    min_improvement: float = 0.1,
    min_observed_ratio: float = 0.8,
    max_iterations: int = 25,
) -> RunContext:
    return RunContext(
        dataset=prepared,
        patterns=PatternConfiguration(
            allowed=patterns,
            min_improvement=min_improvement,
            max_iterations=max_iterations,
        ),
        random_streams=NamedRandomStreams(7),
        candidate_validity_policy=MinimumCardinality(),
        evaluation_support_policy=MinimumObservedSupport(
            min_observed_count=2,
            min_observed_ratio=min_observed_ratio,
        ),
    )


def _numeric_columns(*names: str) -> tuple[ColumnMetadata, ...]:
    return tuple(
        ColumnMetadata(index=index, name=name, kind=ColumnKind.NUMERIC)
        for index, name in enumerate(names)
    )


def _candidate(rows: tuple[int, ...], columns: tuple[int, ...]) -> Candidate:
    return Candidate(
        identifier="candidate",
        bicluster=Bicluster(row_indices=rows, column_indices=columns),
    )


def test_pattern_catalog_declares_column_eligibility_and_scope() -> None:
    catalog = default_pattern_catalog()
    definitions = {definition.kind: definition for definition in catalog.definitions()}
    assert definitions[PatternKind.CONSTANT].scope is PatternScope.COLUMN
    assert definitions[PatternKind.CONSTANT].supported_column_kinds == frozenset(ColumnKind)
    assert definitions[PatternKind.ADDITIVE].scope is PatternScope.SUBSET
    assert definitions[PatternKind.ADDITIVE].supported_column_kinds == frozenset(
        {ColumnKind.NUMERIC}
    )
    assert definitions[PatternKind.ADDITIVE].minimum_columns == 2
    assert definitions[PatternKind.ADDITIVE].maximum_groups == 1
    assert definitions[PatternKind.MULTIPLICATIVE].scope is PatternScope.SUBSET
    assert definitions[PatternKind.MULTIPLICATIVE].supported_column_kinds == frozenset(
        {ColumnKind.NUMERIC}
    )
    assert definitions[PatternKind.MULTIPLICATIVE].minimum_columns == 2
    assert definitions[PatternKind.MULTIPLICATIVE].maximum_groups == 1
    assert definitions[PatternKind.CONSTANT].reference_model
    assert not definitions[PatternKind.ADDITIVE].reference_model
    assert not definitions[PatternKind.MULTIPLICATIVE].reference_model
    assert catalog.implementation(PatternKind.CONSTANT).group_candidate_generator is None
    additive_generator = catalog.implementation(PatternKind.ADDITIVE).group_candidate_generator
    multiplicative_generator = catalog.implementation(
        PatternKind.MULTIPLICATIVE
    ).group_candidate_generator
    assert additive_generator is not None
    assert multiplicative_generator is not None
    assert additive_generator.pattern is PatternKind.ADDITIVE
    assert multiplicative_generator.pattern is PatternKind.MULTIPLICATIVE


def test_pattern_catalog_rejects_empty_duplicate_and_unknown_registrations() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        PatternCatalog(())
    constant = default_pattern_catalog((PatternKind.CONSTANT,)).implementation(PatternKind.CONSTANT)
    duplicate = PatternImplementation(
        definition=constant.definition,
        contrast_strategy=constant.contrast_strategy,
        column_fitter=constant.column_fitter,
    )
    with pytest.raises(ValueError, match="unique kinds"):
        PatternCatalog((constant, duplicate))
    catalog = PatternCatalog((constant,))
    assert catalog.implementations() == (constant,)
    with pytest.raises(ValueError, match="not registered"):
        catalog.implementation(PatternKind.ADDITIVE)
    with pytest.raises(ValueError, match="unregistered allowed"):
        catalog.implementations((PatternKind.ADDITIVE,))
    assert catalog.reference_implementation is constant

    assert constant.column_fitter is not None
    non_reference_definition = replace(
        constant.definition,
        reference_model=False,
    )
    non_reference_fitter = replace(
        constant.column_fitter,
        definition=non_reference_definition,
    )
    with pytest.raises(ValueError, match="exactly one reference"):
        PatternCatalog(
            (
                PatternImplementation(
                    definition=non_reference_definition,
                    contrast_strategy=constant.contrast_strategy,
                    column_fitter=non_reference_fitter,
                ),
            )
        )


def test_constant_coherence_preserves_reviewed_numeric_and_categorical_formula() -> None:
    table = pa.table(
        {
            "numeric": pa.array([0.0, 0.0, 10.0, 10.0]),
            "category": pa.array(["a", "b", "c", "c"]),
        }
    )
    columns = (
        ColumnMetadata(index=0, name="numeric", kind=ColumnKind.NUMERIC),
        ColumnMetadata(
            index=1,
            name="category",
            kind=ColumnKind.CATEGORICAL,
            categories=("a", "b", "c"),
        ),
    )
    context = _context(table, columns)
    candidate = _candidate((0, 1, 2, 3), (0, 1))

    result = InternalCoherence().evaluate(candidate, EvaluationWorkspace(context))

    assert tuple(column.value for column in result.columns) == pytest.approx((0.5, 0.75))
    assert result.value == pytest.approx(np.sqrt((0.5**2 + 0.75**2) / 2.0))
    assert all(column.valid for column in result.columns)


def test_constant_fit_preserves_original_boolean_and_category_parameters() -> None:
    table = pa.table(
        {
            "numeric": pa.array([1.0, 1.0, 8.0]),
            "boolean": pa.array([False, False, True]),
            "category": pa.array(["case", "case", "control"]),
        }
    )
    columns = (
        ColumnMetadata(index=0, name="numeric", kind=ColumnKind.NUMERIC),
        ColumnMetadata(index=1, name="boolean", kind=ColumnKind.BOOLEAN),
        ColumnMetadata(
            index=2,
            name="category",
            kind=ColumnKind.CATEGORICAL,
            categories=("case", "control"),
        ),
    )
    context = _context(table, columns)
    candidate = _candidate((0, 1), (0, 1, 2))
    workspace = EvaluationWorkspace(context)

    fit = workspace.infer(candidate)
    result = InternalCoherence().evaluate(candidate, workspace)

    assert result.value == pytest.approx(0.0)
    assert tuple(column.parameter for column in fit.columns) == (1.0, False, "case")
    assert tuple(column.pattern for column in fit.columns) == (PatternKind.CONSTANT,) * 3


def test_categorical_coherence_uses_global_not_local_observed_cardinality() -> None:
    table = pa.table({"category": pa.array(["a", "b", "c", "c"])})
    columns = (
        ColumnMetadata(
            index=0,
            name="category",
            kind=ColumnKind.CATEGORICAL,
            categories=("a", "b", "c"),
        ),
    )
    context = _context(table, columns)

    result = InternalCoherence().evaluate(_candidate((0, 1), (0,)), EvaluationWorkspace(context))

    assert result.value == pytest.approx(0.75)
    assert context.dataset.discrete_observed_cardinality(0) == 3


def test_perfect_additive_fit_is_joint_and_exposes_row_effects() -> None:
    table = pa.table(
        {
            "first": pa.array([0.0, 1.0, 2.0, 3.0]),
            "second": pa.array([10.0, 11.0, 12.0, 13.0]),
        }
    )
    context = _context(
        table,
        _numeric_columns("first", "second"),
        patterns=(PatternKind.ADDITIVE,),
    )
    candidate = _candidate((0, 1, 2, 3), (0, 1))
    workspace = EvaluationWorkspace(context)

    fit = workspace.infer(candidate)
    coherence = InternalCoherence().evaluate(candidate, workspace)

    assert fit.valid
    assert tuple(column.pattern for column in fit.columns) == (PatternKind.ADDITIVE,) * 2
    assert coherence.value == pytest.approx(0.0, abs=1e-12)
    assert len(fit.groups) == 1
    assert fit.groups[0].column_indices == (0, 1)
    assert len(fit.groups[0].row_parameters) == 4
    assert fit.groups[0].iterations >= 1


def test_mixed_inference_keeps_constant_columns_and_one_additive_subgroup() -> None:
    table = pa.table(
        {
            "constant": pa.array([5.0, 5.0, 5.0, 4.0, 6.0]),
            "additive-a": pa.array([0.0, 2.0, 4.0, 1.0, 3.0]),
            "additive-b": pa.array([10.0, 12.0, 14.0, 11.0, 13.0]),
            "boolean": pa.array([True, True, True, False, False]),
        }
    )
    columns = (
        *_numeric_columns("constant", "additive-a", "additive-b"),
        ColumnMetadata(index=3, name="boolean", kind=ColumnKind.BOOLEAN),
    )
    context = _context(
        table,
        columns,
        patterns=(PatternKind.CONSTANT, PatternKind.ADDITIVE),
    )
    candidate = _candidate((0, 1, 2), (0, 1, 2, 3))

    fit = EvaluationWorkspace(context).infer(candidate)

    assert tuple(column.pattern for column in fit.columns) == (
        PatternKind.CONSTANT,
        PatternKind.ADDITIVE,
        PatternKind.ADDITIVE,
        PatternKind.CONSTANT,
    )
    assert fit.groups[0].column_indices == (1, 2)
    assert tuple(column.group_identifier for column in fit.columns) == (
        None,
        "ADDITIVE-0",
        "ADDITIVE-0",
        None,
    )
    assert all(
        {alternative.pattern for alternative in fit.columns[index].alternatives}
        == {PatternKind.CONSTANT, PatternKind.ADDITIVE}
        for index in (1, 2)
    )


def test_additive_only_marks_incompatible_columns_as_unassigned() -> None:
    table = pa.table(
        {
            "first": pa.array([0.0, 1.0, 2.0]),
            "second": pa.array([10.0, 11.0, 12.0]),
            "category": pa.array(["a", "a", "a"]),
        }
    )
    columns = (
        *_numeric_columns("first", "second"),
        ColumnMetadata(
            index=2,
            name="category",
            kind=ColumnKind.CATEGORICAL,
            categories=("a",),
        ),
    )
    context = _context(table, columns, patterns=(PatternKind.ADDITIVE,))
    candidate = _candidate((0, 1, 2), (0, 1, 2))

    fit = EvaluationWorkspace(context).infer(candidate)

    assert fit.columns[2].pattern is None
    assert fit.columns[2].alternatives == ()
    assert not fit.valid
    assert fit.issues[0].code is EvaluationIssueCode.UNSUPPORTED_COLUMN_KIND


def test_constant_batch_fitting_matches_scalar_fitting_for_a_column_subset() -> None:
    table = pa.table(
        {
            "numeric": pa.array([1.0, None, 1.5, 2.0]),
            "category": pa.array(["a", "a", None, "b"]),
            "boolean": pa.array([True, True, False, None]),
        }
    )
    columns = (
        ColumnMetadata(index=0, name="numeric", kind=ColumnKind.NUMERIC),
        ColumnMetadata(
            index=1,
            name="category",
            kind=ColumnKind.CATEGORICAL,
            categories=("a", "b"),
        ),
        ColumnMetadata(index=2, name="boolean", kind=ColumnKind.BOOLEAN),
    )
    context = _context(table, columns, min_observed_ratio=0.5)
    bicluster = _candidate((0, 1, 2, 3), (0, 1, 2)).bicluster
    fitter = ConstantPatternFitter()

    batched = fitter.fit_columns(context, bicluster, (0, 2))
    scalar = tuple(fitter.fit_column(context, bicluster, column) for column in (0, 2))

    assert batched == scalar


@pytest.mark.parametrize(
    ("axis", "expected"),
    (
        (0, np.asarray([np.nan, 2.0, 7.0])),
        (1, np.asarray([np.inf, 4.0, 7.0])),
    ),
)
def test_nanmedian_2d_handles_missing_slices_without_warnings(
    axis: int,
    expected: np.ndarray,
) -> None:
    matrix = np.asarray(
        (
            (np.nan, 1.0, np.inf),
            (np.nan, 3.0, 5.0),
            (np.nan, np.nan, 7.0),
        )
    )

    with np.errstate(all="raise"):
        actual = nanmedian_2d(matrix, axis=axis)

    np.testing.assert_equal(actual, expected)


def test_additive_only_reports_insufficient_joint_cardinality_explicitly() -> None:
    table = pa.table(
        {
            "numeric": pa.array([0.0, 1.0, 2.0]),
            "category": pa.array(["a", "a", "a"]),
        }
    )
    columns = (
        ColumnMetadata(index=0, name="numeric", kind=ColumnKind.NUMERIC),
        ColumnMetadata(
            index=1,
            name="category",
            kind=ColumnKind.CATEGORICAL,
            categories=("a",),
        ),
    )
    context = _context(table, columns, patterns=(PatternKind.ADDITIVE,))

    fit = EvaluationWorkspace(context).infer(_candidate((0, 1, 2), (0, 1)))

    assert any(
        issue.code is EvaluationIssueCode.INSUFFICIENT_GROUP_SUPPORT and issue.column_index == 0
        for issue in fit.issues
    )


def test_mixed_inference_collapses_a_joint_pattern_below_two_columns() -> None:
    table = pa.table(
        {
            "varying": pa.array([0.0, 2.0, 4.0, 1.0, 3.0]),
            "constant": pa.array([5.0, 5.0, 5.0, 4.0, 6.0]),
        }
    )
    context = _context(
        table,
        _numeric_columns("varying", "constant"),
        patterns=(PatternKind.CONSTANT, PatternKind.ADDITIVE),
    )

    fit = EvaluationWorkspace(context).infer(_candidate((0, 1, 2), (0, 1)))

    assert tuple(column.pattern for column in fit.columns) == (PatternKind.CONSTANT,) * 2
    assert fit.groups == ()


def test_mixed_inference_accepts_an_exact_improvement_threshold() -> None:
    table = pa.table(
        {
            "first": pa.array([0.0, 2.0, 4.0, 1.0, 3.0]),
            "second": pa.array([10.0, 12.0, 14.0, 11.0, 13.0]),
        }
    )
    columns = _numeric_columns("first", "second")
    candidate = _candidate((0, 1, 2), (0, 1))
    exploratory = _context(
        table,
        columns,
        patterns=(PatternKind.CONSTANT, PatternKind.ADDITIVE),
        min_improvement=0.0,
    )
    initial_fit = EvaluationWorkspace(exploratory).infer(candidate)
    improvements = []
    for column in initial_fit.columns:
        errors = {alternative.pattern: alternative.error for alternative in column.alternatives}
        improvements.append(errors[PatternKind.CONSTANT] - errors[PatternKind.ADDITIVE])
    exact_threshold = min(improvements)
    threshold_context = _context(
        table,
        columns,
        patterns=(PatternKind.CONSTANT, PatternKind.ADDITIVE),
        min_improvement=exact_threshold,
    )

    threshold_fit = EvaluationWorkspace(threshold_context).infer(candidate)

    assert tuple(column.pattern for column in threshold_fit.columns) == (
        PatternKind.ADDITIVE,
        PatternKind.ADDITIVE,
    )


def test_competing_joint_patterns_partition_columns_before_final_refit() -> None:
    multiplicative_rows = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
    table = pa.table(
        {
            "additive-a": pa.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            "additive-b": pa.array([20.0, 21.0, 22.0, 23.0, 24.0, 25.0]),
            "multiplicative-a": pa.array(multiplicative_rows),
            "multiplicative-b": pa.array(
                [10.0 * value for value in multiplicative_rows]
            ),
        }
    )
    context = _context(
        table,
        _numeric_columns(
            "additive-a",
            "additive-b",
            "multiplicative-a",
            "multiplicative-b",
        ),
        patterns=(
            PatternKind.CONSTANT,
            PatternKind.ADDITIVE,
            PatternKind.MULTIPLICATIVE,
        ),
        min_improvement=0.05,
    )

    calls: dict[PatternKind, int] = {}

    class CountingGroupFitter:
        def __init__(self, delegate: GroupPatternFitter, pattern: PatternKind) -> None:
            self.definition = delegate.definition
            self._delegate = delegate
            self._pattern = pattern

        def fit_group(
            self,
            run_context: RunContext,
            bicluster: Bicluster,
            column_indices: Sequence[int],
        ) -> GroupPatternProposal:
            calls[self._pattern] = calls.get(self._pattern, 0) + 1
            return self._delegate.fit_group(
                run_context,
                bicluster,
                column_indices,
            )

    implementations: list[PatternImplementation] = []
    for implementation in default_pattern_catalog(context.patterns.allowed).implementations():
        if implementation.group_fitter is None:
            implementations.append(implementation)
            continue
        implementations.append(
            replace(
                implementation,
                group_fitter=CountingGroupFitter(
                    implementation.group_fitter,
                    implementation.definition.kind,
                ),
            )
        )
    fit = EvaluationWorkspace(
        context,
        pattern_catalog=PatternCatalog(implementations),
    ).infer(
        _candidate(tuple(range(6)), tuple(range(4)))
    )

    assert tuple(column.pattern for column in fit.columns) == (
        PatternKind.ADDITIVE,
        PatternKind.ADDITIVE,
        PatternKind.MULTIPLICATIVE,
        PatternKind.MULTIPLICATIVE,
    )
    assert {(group.pattern, group.column_indices) for group in fit.groups} == {
        (PatternKind.ADDITIVE, (0, 1)),
        (PatternKind.MULTIPLICATIVE, (2, 3)),
    }
    assert all(column.error == pytest.approx(0.0) for column in fit.columns)
    assert set(calls) == {PatternKind.ADDITIVE, PatternKind.MULTIPLICATIVE}
    assert all(2 <= count <= 17 for count in calls.values())


def test_positive_multiplicative_pattern_is_not_an_exact_additive_pattern() -> None:
    table = pa.table(
        {
            "first": pa.array([1.0, 2.0, 3.0, 4.0]),
            "second": pa.array([10.0, 20.0, 30.0, 40.0]),
        }
    )
    context = _context(
        table,
        _numeric_columns("first", "second"),
        patterns=(
            PatternKind.CONSTANT,
            PatternKind.ADDITIVE,
            PatternKind.MULTIPLICATIVE,
        ),
        min_improvement=0.0,
    )

    fit = EvaluationWorkspace(context).infer(_candidate((0, 1, 2, 3), (0, 1)))

    assert tuple(column.pattern for column in fit.columns) == (
        PatternKind.MULTIPLICATIVE,
        PatternKind.MULTIPLICATIVE,
    )
    for column in fit.columns:
        errors = {alternative.pattern: alternative.error for alternative in column.alternatives}
        assert errors[PatternKind.MULTIPLICATIVE] == pytest.approx(0.0, abs=1e-12)
        assert errors[PatternKind.ADDITIVE] > 0.0
        assert not dict(column.diagnostics)["model_ambiguous"]


def test_additive_fit_uses_raw_offsets_and_robustly_normalized_residuals() -> None:
    table = pa.table(
        {
            "first": pa.array([-2.0, 0.0, 2.0, 4.0]),
            "second": pa.array([98.0, 100.0, 102.0, 104.0]),
        }
    )
    context = _context(
        table,
        _numeric_columns("first", "second"),
        patterns=(PatternKind.ADDITIVE,),
        min_improvement=0.0,
    )

    fit = EvaluationWorkspace(context).infer(_candidate((0, 1, 2, 3), (0, 1)))

    assert tuple(column.pattern for column in fit.columns) == (PatternKind.ADDITIVE,) * 2
    assert all(column.error == pytest.approx(0.0, abs=1e-12) for column in fit.columns)
    assert all(column.parameter_scale is ParameterScale.RAW for column in fit.columns)
    assert all(
        dict(column.diagnostics)["residual_normalization"] == "global_robust_range"
        for column in fit.columns
    )


def test_observationally_equivalent_joint_patterns_are_reported_as_ambiguous() -> None:
    values = [1.0, 2.0, 4.0, 8.0]
    table = pa.table({"first": pa.array(values), "second": pa.array(values)})
    context = _context(
        table,
        _numeric_columns("first", "second"),
        patterns=(
            PatternKind.CONSTANT,
            PatternKind.ADDITIVE,
            PatternKind.MULTIPLICATIVE,
        ),
        min_improvement=0.0,
    )

    fit = EvaluationWorkspace(context).infer(_candidate((0, 1, 2, 3), (0, 1)))

    for column in fit.columns:
        details = dict(column.diagnostics)
        assert details["model_ambiguous"] is True
        assert details["model_error_margin"] == pytest.approx(0.0, abs=1e-12)
        assert details["model_equivalents"] == "ADDITIVE,MULTIPLICATIVE"


def test_positive_affine_relation_is_not_reported_as_an_exact_joint_model() -> None:
    first = np.asarray([1.0, 2.0, 3.0, 4.0])
    table = pa.table(
        {
            "first": pa.array(first),
            "second": pa.array(10.0 * first + 7.0),
        }
    )
    context = _context(
        table,
        _numeric_columns("first", "second"),
        patterns=(PatternKind.ADDITIVE, PatternKind.MULTIPLICATIVE),
        min_improvement=0.0,
    )

    fit = EvaluationWorkspace(context).infer(_candidate((0, 1, 2, 3), (0, 1)))

    assert all(
        alternative.error > 0.0
        for column in fit.columns
        for alternative in column.alternatives
        if alternative.pattern in {PatternKind.ADDITIVE, PatternKind.MULTIPLICATIVE}
    )


def test_noisy_additive_fit_retains_a_finite_nonzero_error() -> None:
    table = pa.table(
        {
            "first": pa.array([0.0, 1.0, 2.0, 3.0]),
            "second": pa.array([10.0, 11.2, 11.9, 13.1]),
        }
    )
    context = _context(
        table,
        _numeric_columns("first", "second"),
        patterns=(PatternKind.ADDITIVE,),
        min_improvement=0.0,
    )

    fit = EvaluationWorkspace(context).infer(_candidate((0, 1, 2, 3), (0, 1)))

    assert fit.valid
    assert all(0.0 < column.error < 1.0 for column in fit.columns)


def test_nonconverged_additive_fit_remains_explicit_and_finite() -> None:
    table = pa.table(
        {
            "first": pa.array([0.0, 1.0, 2.0, 3.0]),
            "second": pa.array([10.0, 11.0, 12.0, 13.0]),
        }
    )
    prepared = PreparedDataset.from_arrow(
        Dataset(
            identifier="nonconverged",
            bundle_path="fixture",
            row_count=4,
            column_count=2,
            columns=_numeric_columns("first", "second"),
        ),
        table,
        pa.array(["0", "1", "2", "3"]),
    )
    context = _prepared_context(
        RobustNumericScaling().transform(prepared),
        patterns=(PatternKind.ADDITIVE,),
        max_iterations=1,
    )

    fit = EvaluationWorkspace(context).infer(_candidate((0, 1, 2, 3), (0, 1)))

    assert fit.valid
    assert not fit.groups[0].converged
    assert fit.groups[0].iterations == 1
    assert all(np.isfinite(column.error) for column in fit.columns)


def test_perfect_multiplicative_fit_uses_a_joint_proportional_model() -> None:
    table = pa.table(
        {
            "first": pa.array([1.0, 2.0, 4.0, 8.0]),
            "second": pa.array([3.0, 6.0, 12.0, 24.0]),
        }
    )
    context = _context(
        table,
        _numeric_columns("first", "second"),
        patterns=(PatternKind.MULTIPLICATIVE,),
        min_improvement=0.0,
    )
    candidate = _candidate((0, 1, 2, 3), (0, 1))
    workspace = EvaluationWorkspace(context)

    fit = workspace.infer(candidate)
    coherence = InternalCoherence().evaluate(candidate, workspace)

    assert fit.valid
    assert tuple(column.pattern for column in fit.columns) == (
        PatternKind.MULTIPLICATIVE,
        PatternKind.MULTIPLICATIVE,
    )
    assert tuple(column.parameter_scale for column in fit.columns) == (
        ParameterScale.ROBUST_SCALED,
        ParameterScale.ROBUST_SCALED,
    )
    assert coherence.value == pytest.approx(0.0, abs=1e-12)
    assert fit.groups[0].pattern is PatternKind.MULTIPLICATIVE
    assert fit.groups[0].column_indices == (0, 1)
    assert len(fit.groups[0].row_parameters) == 4


def test_multiplicative_fit_supports_negative_values_and_zero_row_effects() -> None:
    table = pa.table(
        {
            "first": pa.array([-4.0, -2.0, 0.0, 2.0, 4.0]),
            "second": pa.array([6.0, 3.0, 0.0, -3.0, -6.0]),
        }
    )
    context = _context(
        table,
        _numeric_columns("first", "second"),
        patterns=(PatternKind.MULTIPLICATIVE,),
        min_improvement=0.0,
    )

    fit = EvaluationWorkspace(context).infer(_candidate((0, 1, 2, 3, 4), (0, 1)))

    assert fit.valid
    assert all(column.error == pytest.approx(0.0, abs=1e-12) for column in fit.columns)
    assert dict(fit.groups[0].row_parameters)[2] == pytest.approx(0.0)


def test_noisy_multiplicative_fit_reports_error_and_nonconvergence() -> None:
    table = pa.table(
        {
            "first": pa.array([1.0, 2.0, 4.0, 8.0]),
            "second": pa.array([3.0, 6.5, 11.0, 25.0]),
        }
    )
    context = _context(
        table,
        _numeric_columns("first", "second"),
        patterns=(PatternKind.MULTIPLICATIVE,),
        min_improvement=0.0,
        max_iterations=1,
    )

    fit = EvaluationWorkspace(context).infer(_candidate((0, 1, 2, 3), (0, 1)))

    assert fit.valid
    assert not fit.groups[0].converged
    assert fit.groups[0].iterations == 1
    assert all(0.0 < column.error < 1.0 for column in fit.columns)


def test_multiplicative_only_penalizes_a_merely_constant_rank_one_fit() -> None:
    table = pa.table(
        {
            "first": pa.array([2.0, 2.0, 2.0]),
            "second": pa.array([4.0, 4.0, 4.0]),
        }
    )
    context = _context(
        table,
        _numeric_columns("first", "second"),
        patterns=(PatternKind.MULTIPLICATIVE,),
        min_improvement=0.1,
    )

    fit = EvaluationWorkspace(context).infer(_candidate((0, 1, 2), (0, 1)))

    assert fit.valid
    assert all(column.error == pytest.approx(1.0) for column in fit.columns)
    assert all(
        dict(column.diagnostics)["evidence_deficit"] == pytest.approx(1.0) for column in fit.columns
    )


def test_multiplicative_reports_an_all_zero_group_as_unassigned() -> None:
    table = pa.table(
        {
            "first": pa.array([0.0, 0.0, 0.0]),
            "second": pa.array([0.0, 0.0, 0.0]),
        }
    )
    context = _context(
        table,
        _numeric_columns("first", "second"),
        patterns=(PatternKind.MULTIPLICATIVE,),
        min_improvement=0.0,
    )

    fit = EvaluationWorkspace(context).infer(_candidate((0, 1, 2), (0, 1)))

    assert not fit.valid
    assert all(column.pattern is None for column in fit.columns)
    assert any(
        issue.code is EvaluationIssueCode.PATTERN_FIT_FAILED
        and issue.pattern is PatternKind.MULTIPLICATIVE
        for issue in fit.issues
    )


def test_multiplicative_competes_with_more_than_one_other_pattern() -> None:
    table = pa.table(
        {
            "first": pa.array([1.0, 2.0, 4.0, 8.0, 5.0, 7.0]),
            "second": pa.array([3.0, 6.0, 12.0, 24.0, 2.0, 20.0]),
        }
    )
    context = _context(
        table,
        _numeric_columns("first", "second"),
        patterns=(
            PatternKind.CONSTANT,
            PatternKind.ADDITIVE,
            PatternKind.MULTIPLICATIVE,
        ),
        min_improvement=0.01,
    )

    fit = EvaluationWorkspace(context).infer(_candidate((0, 1, 2, 3), (0, 1)))

    assert tuple(column.pattern for column in fit.columns) == (
        PatternKind.MULTIPLICATIVE,
        PatternKind.MULTIPLICATIVE,
    )
    assert all(
        {alternative.pattern for alternative in column.alternatives}
        == {
            PatternKind.CONSTANT,
            PatternKind.ADDITIVE,
            PatternKind.MULTIPLICATIVE,
        }
        for column in fit.columns
    )


def test_mixed_multiplicative_inference_prunes_a_zero_constant_column() -> None:
    table = pa.table(
        {
            "first": pa.array([1.0, 2.0, 4.0, 8.0]),
            "second": pa.array([3.0, 6.0, 12.0, 24.0]),
            "zero": pa.array([0.0, 0.0, 0.0, 0.0]),
        }
    )
    context = _context(
        table,
        _numeric_columns("first", "second", "zero"),
        patterns=(PatternKind.CONSTANT, PatternKind.MULTIPLICATIVE),
        min_improvement=0.01,
    )

    fit = EvaluationWorkspace(context).infer(_candidate((0, 1, 2, 3), (0, 1, 2)))

    assert tuple(column.pattern for column in fit.columns) == (
        PatternKind.MULTIPLICATIVE,
        PatternKind.MULTIPLICATIVE,
        PatternKind.CONSTANT,
    )
    assert fit.groups[0].column_indices == (0, 1)


def test_multiplicative_only_rejects_non_numeric_columns_explicitly() -> None:
    table = pa.table(
        {
            "first": pa.array([1.0, 2.0, 4.0]),
            "second": pa.array([3.0, 6.0, 12.0]),
            "category": pa.array(["a", "a", "a"]),
        }
    )
    columns = (
        *_numeric_columns("first", "second"),
        ColumnMetadata(
            index=2,
            name="category",
            kind=ColumnKind.CATEGORICAL,
            categories=("a",),
        ),
    )
    context = _context(
        table,
        columns,
        patterns=(PatternKind.MULTIPLICATIVE,),
        min_improvement=0.0,
    )

    fit = EvaluationWorkspace(context).infer(_candidate((0, 1, 2), (0, 1, 2)))

    assert fit.columns[2].pattern is None
    assert any(
        issue.code is EvaluationIssueCode.UNSUPPORTED_COLUMN_KIND and issue.column_index == 2
        for issue in fit.issues
    )


@pytest.mark.parametrize(
    "local,background,expected",
    (
        ((0.0, 0.0), (-10.0, 10.0), 1.0),
        ((-1.0, 1.0), (-1.0, 1.0), 0.5),
        ((-10.0, 10.0), (0.0, 0.0), 0.0),
    ),
)
def test_numeric_constant_contrast_has_inverse_neutral_and_perfect_scale(
    local: tuple[float, float],
    background: tuple[float, float],
    expected: float,
) -> None:
    values = (*local, *background)
    table = pa.table({"first": values, "second": values})
    context = _context(
        table,
        _numeric_columns("first", "second"),
        min_observed_ratio=1.0,
    )
    candidate = _candidate((0, 1), (0, 1))

    result = Contrast().evaluate(candidate, EvaluationWorkspace(context))

    assert result.value == pytest.approx(expected)
    assert tuple(column.value for column in result.columns) == pytest.approx((expected, expected))


def test_categorical_contrast_uses_bilateral_prototype_frequency() -> None:
    table = pa.table(
        {
            "first": pa.array(["a", "a", "a", "b"]),
            "second": pa.array(["a", "a", "a", "b"]),
        }
    )
    columns = tuple(
        ColumnMetadata(
            index=index,
            name=name,
            kind=ColumnKind.CATEGORICAL,
            categories=("a", "b"),
        )
        for index, name in enumerate(("first", "second"))
    )
    context = _context(table, columns, min_observed_ratio=1.0)
    candidate = _candidate((0, 1), (0, 1))

    result = Contrast().evaluate(candidate, EvaluationWorkspace(context))

    assert result.value == pytest.approx(1.0)
    assert tuple(column.value for column in result.columns) == pytest.approx((1.0, 1.0))


@pytest.mark.parametrize(
    ("values", "expected"),
    (
        (("a", "a", "a", "a"), 0.5),
        (("a", "b", "a", "a"), 0.25),
    ),
)
def test_categorical_contrast_exposes_neutral_and_inverse_separation(
    values: tuple[str, ...], expected: float
) -> None:
    table = pa.table({"first": values, "second": values})
    columns = tuple(
        ColumnMetadata(
            index=index,
            name=name,
            kind=ColumnKind.CATEGORICAL,
            categories=("a", "b"),
        )
        for index, name in enumerate(("first", "second"))
    )
    context = _context(table, columns, min_observed_ratio=1.0)

    result = Contrast().evaluate(_candidate((0, 1), (0, 1)), EvaluationWorkspace(context))

    assert result.value == pytest.approx(expected)


@pytest.mark.parametrize("reproduced,expected", ((False, 1.0), (True, 0.5)))
def test_additive_contrast_detects_whether_the_profile_reappears_in_background(
    reproduced: bool,
    expected: float,
) -> None:
    second_background = (20.0, 0.0) if reproduced else (0.0, 20.0)
    table = pa.table(
        {
            "first": pa.array([0.0, 1.0, 2.0, 10.0, -10.0]),
            "second": pa.array([10.0, 11.0, 12.0, *second_background]),
        }
    )
    context = _context(
        table,
        _numeric_columns("first", "second"),
        patterns=(PatternKind.ADDITIVE,),
        min_observed_ratio=1.0,
    )
    candidate = _candidate((0, 1, 2), (0, 1))

    result = Contrast().evaluate(candidate, EvaluationWorkspace(context))

    assert result.value == pytest.approx(expected)


@pytest.mark.parametrize("reproduced,expected", ((False, 1.0), (True, 0.5)))
def test_multiplicative_contrast_detects_proportional_background_profiles(
    reproduced: bool,
    expected: float,
) -> None:
    second_background = (15.0, 21.0) if reproduced else (21.0, 15.0)
    table = pa.table(
        {
            "first": pa.array([1.0, 2.0, 4.0, 5.0, 7.0]),
            "second": pa.array([3.0, 6.0, 12.0, *second_background]),
        }
    )
    context = _context(
        table,
        _numeric_columns("first", "second"),
        patterns=(PatternKind.MULTIPLICATIVE,),
        min_improvement=0.0,
        min_observed_ratio=1.0,
    )
    candidate = _candidate((0, 1, 2), (0, 1))

    result = Contrast().evaluate(candidate, EvaluationWorkspace(context))

    assert result.value == pytest.approx(expected)


def test_multiplicative_fit_ignores_missing_values_with_original_support() -> None:
    table = pa.table(
        {
            "first": pa.array([1.0, 2.0, 4.0, 8.0]),
            "second": pa.array([3.0, 6.0, None, 24.0], type=pa.float64()),
            "third": pa.array([5.0, 10.0, 20.0, 40.0]),
        }
    )
    context = _context(
        table,
        _numeric_columns("first", "second", "third"),
        patterns=(PatternKind.MULTIPLICATIVE,),
        min_improvement=0.0,
        min_observed_ratio=0.75,
    )

    fit = EvaluationWorkspace(context).infer(_candidate((0, 1, 2, 3), (0, 1, 2)))

    assert fit.valid
    assert fit.columns[1].source_support == 3
    assert fit.columns[1].available_support == 3
    assert all(column.error == pytest.approx(0.0, abs=1e-12) for column in fit.columns)


def test_contrast_explains_columns_that_cannot_use_an_allowed_pattern() -> None:
    table = pa.table(
        {
            "first": [0.0, 1.0, 2.0, 3.0, 4.0],
            "second": [10.0, 11.0, 12.0, 13.0, 14.0],
            "category": ["a", "a", "a", "b", "b"],
        }
    )
    columns = (
        *_numeric_columns("first", "second"),
        ColumnMetadata(
            index=2,
            name="category",
            kind=ColumnKind.CATEGORICAL,
            categories=("a", "b"),
        ),
    )
    context = _context(table, columns, patterns=(PatternKind.ADDITIVE,))

    result = Contrast().evaluate(_candidate((0, 1, 2), (0, 1, 2)), EvaluationWorkspace(context))

    assert not result.columns[2].valid
    assert result.columns[2].value == 0.0
    assert any(
        issue.code is EvaluationIssueCode.PATTERN_UNASSIGNED and issue.column_index == 2
        for issue in result.issues
    )


@pytest.mark.parametrize("ratio", (-0.1, 1.0))
def test_contrast_rejects_invalid_background_ratios(ratio: float) -> None:
    with pytest.raises(ValueError, match="min_background_ratio"):
        Contrast(min_background_ratio=ratio)


def test_missing_support_is_not_inflated_and_is_explained_per_column() -> None:
    table = pa.table(
        {
            "sparse": pa.array([1.0, None, None, None], type=pa.float64()),
            "complete": pa.array([0.0, 0.0, 0.0, 0.0]),
        }
    )
    context = _context(table, _numeric_columns("sparse", "complete"))
    candidate = _candidate((0, 1, 2, 3), (0, 1))
    workspace = EvaluationWorkspace(context)

    fit = workspace.infer(candidate)
    result = InternalCoherence().evaluate(candidate, workspace)

    assert fit.columns[0].pattern is None
    assert result.columns[0].value == 1.0
    assert not result.columns[0].valid
    assert not fit.valid
    assert any(
        issue.code is EvaluationIssueCode.INSUFFICIENT_LOCAL_SUPPORT and issue.column_index == 0
        for issue in fit.issues
    )


def test_imputation_is_consumed_but_does_not_inflate_original_support() -> None:
    table = pa.table(
        {
            "first": pa.array([1.0, None, 3.0], type=pa.float64()),
            "second": pa.array([2.0, 2.0, 2.0]),
        }
    )
    metadata = Dataset(
        identifier="imputed",
        bundle_path="fixture",
        row_count=3,
        column_count=2,
        columns=_numeric_columns("first", "second"),
    )
    raw = PreparedDataset.from_arrow(metadata, table, pa.array(["0", "1", "2"]))
    prepared = RobustNumericScaling().transform(MedianModeImputation().apply(raw))
    candidate = _candidate((0, 1, 2), (0, 1))

    permissive = _prepared_context(prepared, min_observed_ratio=2.0 / 3.0)
    permissive_fit = EvaluationWorkspace(permissive).infer(candidate)
    assert permissive_fit.columns[0].source_support == 2
    assert permissive_fit.columns[0].available_support == 3
    assert permissive_fit.columns[0].error == pytest.approx(10.0 / 27.0)

    strict = _prepared_context(prepared, min_observed_ratio=1.0)
    strict_fit = EvaluationWorkspace(strict).infer(candidate)
    assert strict_fit.columns[0].pattern is None
    assert not strict_fit.valid


def test_insufficient_background_is_typed_instead_of_becoming_neutral() -> None:
    table = pa.table({"first": [1.0, 1.0, 2.0], "second": [1.0, 1.0, 2.0]})
    context = _context(
        table,
        _numeric_columns("first", "second"),
        min_observed_ratio=1.0,
    )
    candidate = _candidate((0, 1), (0, 1))

    result = Contrast().evaluate(candidate, EvaluationWorkspace(context))

    assert result.value == 0.0
    assert result.issues[0].code is EvaluationIssueCode.INSUFFICIENT_BACKGROUND
    assert all(not column.valid for column in result.columns)


def test_executor_persists_one_shared_fit_and_column_objective_values() -> None:
    table = pa.table(
        {
            "first": [0.0, 0.0, -10.0, 10.0],
            "second": [1.0, 1.0, -20.0, 20.0],
        }
    )
    context = _context(
        table,
        _numeric_columns("first", "second"),
        min_observed_ratio=1.0,
    )
    candidate = _candidate((0, 1), (0, 1))
    workspace = EvaluationWorkspace(context)

    evaluations = SerialEvaluationExecutor().evaluate(
        (candidate,),
        (InternalCoherence(), Contrast()),
        (),
        workspace,
    )

    evaluation = evaluations[0]
    assert evaluation.valid
    assert evaluation.pattern_fit is workspace.infer(candidate)
    assert tuple(objective.direction for objective in evaluation.objectives) == (
        ObjectiveDirection.MINIMIZE,
        ObjectiveDirection.MAXIMIZE,
    )
    assert all(len(objective.columns) == 2 for objective in evaluation.objectives)


def test_workspace_shares_inference_across_concurrent_objectives() -> None:
    table = pa.table({"first": [0.0, 1.0, 2.0], "second": [10.0, 11.0, 12.0]})
    context = _context(
        table,
        _numeric_columns("first", "second"),
        patterns=(PatternKind.ADDITIVE,),
    )
    candidate = _candidate((0, 1, 2), (0, 1))
    workspace = EvaluationWorkspace(context)

    with ThreadPoolExecutor(max_workers=2) as executor:
        coherence_future = executor.submit(InternalCoherence().evaluate, candidate, workspace)
        contrast_future = executor.submit(Contrast().evaluate, candidate, workspace)
        coherence_future.result()
        contrast_future.result()

    assert len(workspace) == 1


def test_constant_only_workspace_does_not_register_joint_fitting() -> None:
    table = pa.table({"first": [1.0, 1.0], "second": [2.0, 2.0]})
    context = _context(table, _numeric_columns("first", "second"))
    workspace = EvaluationWorkspace(context)

    assert tuple(definition.kind for definition in workspace.pattern_catalog.definitions()) == (
        PatternKind.CONSTANT,
    )


def test_multiplicative_only_workspace_registers_only_its_reference_and_requested_pattern() -> None:
    table = pa.table({"first": [1.0, 2.0], "second": [3.0, 6.0]})
    context = _context(
        table,
        _numeric_columns("first", "second"),
        patterns=(PatternKind.MULTIPLICATIVE,),
        min_improvement=0.0,
    )
    workspace = EvaluationWorkspace(context)

    assert tuple(definition.kind for definition in workspace.pattern_catalog.definitions()) == (
        PatternKind.CONSTANT,
        PatternKind.MULTIPLICATIVE,
    )


def test_versioned_scientific_regression_fixture() -> None:
    fixture_path = Path(__file__).with_name("fixtures") / "scientific-evaluation-v1.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    columns = tuple(
        ColumnMetadata(
            index=index,
            name=column["name"],
            kind=ColumnKind(column["kind"]),
            categories=tuple(column.get("categories", ())),
        )
        for index, column in enumerate(fixture["columns"])
    )
    table = pa.table({column["name"]: pa.array(column["values"]) for column in fixture["columns"]})
    context = _context(
        table,
        columns,
        patterns=tuple(PatternKind(value) for value in fixture["patterns"]),
        min_observed_ratio=1.0,
    )
    candidate = _candidate(
        tuple(fixture["candidate"]["rows"]),
        tuple(fixture["candidate"]["columns"]),
    )
    workspace = EvaluationWorkspace(context)

    coherence = InternalCoherence().evaluate(candidate, workspace)
    contrast = Contrast().evaluate(candidate, workspace)
    fit = workspace.infer(candidate)

    assert [column.pattern.value for column in fit.columns] == fixture["expected"]["patterns"]
    assert [column.error for column in fit.columns] == pytest.approx(
        fixture["expected"]["coherence_by_column"]
    )
    assert coherence.value == pytest.approx(fixture["expected"]["coherence"])
    assert [column.value for column in contrast.columns] == pytest.approx(
        fixture["expected"]["contrast_by_column"]
    )
    assert contrast.value == pytest.approx(fixture["expected"]["contrast"])
