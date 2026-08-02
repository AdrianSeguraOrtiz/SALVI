from __future__ import annotations

import math
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import ValidationError

from salvi.application.context import RunContext
from salvi.domain import (
    Bicluster,
    Candidate,
    ColumnKind,
    ColumnMetadata,
    ColumnPatternFit,
    Dataset,
    NamedValue,
    ObjectiveDirection,
    ObjectiveValue,
    PatternFit,
)
from salvi.evaluation import EvaluationWorkspace


def _fit(candidate: Candidate, *, signature: str | None = None) -> PatternFit:
    return PatternFit(
        candidate_signature=signature or candidate.bicluster.signature,
        row_indices=candidate.bicluster.row_indices,
        column_indices=candidate.bicluster.column_indices,
        columns=tuple(
            ColumnPatternFit(column_index=column, pattern=None, error=1.0)
            for column in candidate.bicluster.column_indices
        ),
    )


def test_bicluster_has_stable_signature_and_rejects_invalid_indices() -> None:
    first = Bicluster(row_indices=(0, 2), column_indices=(1, 3))
    second = Bicluster(row_indices=(0, 2), column_indices=(1, 3))
    assert first.signature == second.signature
    assert len(first.signature) == 64

    for rows in ((), (1, 0), (0, 0), (-1, 0)):
        with pytest.raises(ValidationError):
            Bicluster(row_indices=rows, column_indices=(0,))


def test_dataset_and_column_metadata_are_semantically_validated() -> None:
    categorical = ColumnMetadata(
        index=0,
        name="group",
        kind=ColumnKind.CATEGORICAL,
        categories=("a", "b"),
    )
    dataset = Dataset(
        identifier="dataset",
        bundle_path=Path("dataset"),
        row_count=2,
        column_count=1,
        columns=(categorical,),
    )
    assert dataset.columns == (categorical,)
    with pytest.raises(ValidationError):
        ColumnMetadata(index=0, name="group", kind=ColumnKind.CATEGORICAL)
    with pytest.raises(ValidationError):
        ColumnMetadata(
            index=0,
            name="numeric",
            kind=ColumnKind.NUMERIC,
            categories=("invalid",),
        )
    with pytest.raises(ValidationError):
        Dataset(
            identifier="dataset",
            bundle_path=Path("dataset"),
            row_count=2,
            column_count=2,
            columns=(categorical,),
        )


def test_named_values_must_be_finite() -> None:
    assert NamedValue(name="quality", value=0.5).value == 0.5
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValidationError):
            NamedValue(name="quality", value=value)
    assert (
        ObjectiveValue(
            name="contrast",
            value=0.5,
            direction=ObjectiveDirection.MAXIMIZE,
        ).direction
        is ObjectiveDirection.MAXIMIZE
    )


def test_workspace_computes_one_exact_fit_per_signature(run_context: RunContext) -> None:
    candidate = Candidate(bicluster=Bicluster(row_indices=(0,), column_indices=(0,)))
    fit = _fit(candidate)
    calls = 0

    def compute() -> PatternFit:
        nonlocal calls
        calls += 1
        return fit

    workspace = EvaluationWorkspace(run_context)
    assert workspace.pattern_fit(candidate, compute) is fit
    assert workspace.pattern_fit(candidate, compute) is fit
    assert calls == 1
    assert len(workspace) == 1

    wrong = _fit(candidate, signature="0" * 64)
    with pytest.raises(ValueError, match="does not match"):
        EvaluationWorkspace(run_context).pattern_fit(candidate, lambda: wrong)


def test_workspace_deduplicates_equal_candidates_without_serializing_different_ones(
    run_context: RunContext,
) -> None:
    workspace = EvaluationWorkspace(run_context)
    first = Candidate(bicluster=Bicluster(row_indices=(0,), column_indices=(0,)))
    second = Candidate(bicluster=Bicluster(row_indices=(1,), column_indices=(0,)))
    barrier = threading.Barrier(2)

    def compute(candidate: Candidate) -> PatternFit:
        barrier.wait(timeout=2)
        return _fit(candidate)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(
            executor.map(
                lambda candidate: workspace.pattern_fit(
                    candidate,
                    lambda candidate=candidate: compute(candidate),
                ),
                (first, second),
            )
        )
    assert {fit.candidate_signature for fit in results} == {
        first.bicluster.signature,
        second.bicluster.signature,
    }

    calls = 0
    started = threading.Event()
    release = threading.Event()
    duplicate_workspace = EvaluationWorkspace(run_context)

    def compute_once() -> PatternFit:
        nonlocal calls
        calls += 1
        started.set()
        release.wait(timeout=2)
        return _fit(first)

    with ThreadPoolExecutor(max_workers=2) as executor:
        owner = executor.submit(duplicate_workspace.pattern_fit, first, compute_once)
        assert started.wait(timeout=2)
        waiter = executor.submit(duplicate_workspace.pattern_fit, first, compute_once)
        release.set()
        assert owner.result() is waiter.result()
    assert calls == 1
