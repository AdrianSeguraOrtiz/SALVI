from __future__ import annotations

import pytest

from salvi.components.evaluation_policies import MinimumCardinality, MinimumObservedSupport
from salvi.domain import Bicluster, Candidate
from salvi.exceptions import ComponentError


def test_minimum_cardinality_accepts_valid_candidates_and_rejects_invalid_ones(
    run_context,  # type: ignore[no-untyped-def]
) -> None:
    policy = MinimumCardinality(min_rows=2, min_columns=2)
    policy.validate_dataset(run_context.dataset)
    policy.validate(
        Candidate(bicluster=Bicluster(row_indices=(0, 1), column_indices=(0, 1))),
        run_context.dataset,
    )

    with pytest.raises(ComponentError, match="at least 3"):
        MinimumCardinality(min_rows=3).validate(
            Candidate(bicluster=Bicluster(row_indices=(0, 1), column_indices=(0, 1))),
            run_context.dataset,
        )
    with pytest.raises(ComponentError, match="row outside"):
        policy.validate(
            Candidate(bicluster=Bicluster(row_indices=(0, 99), column_indices=(0, 1))),
            run_context.dataset,
        )
    with pytest.raises(ComponentError, match="column outside"):
        policy.validate(
            Candidate(bicluster=Bicluster(row_indices=(0, 1), column_indices=(0, 99))),
            run_context.dataset,
        )
    with pytest.raises(ComponentError, match="exceeds the dataset row count"):
        MinimumCardinality(min_rows=99).validate_dataset(run_context.dataset)
    with pytest.raises(ComponentError, match="exceeds the prepared column count"):
        MinimumCardinality(min_columns=99).validate_dataset(run_context.dataset)


def test_observed_support_combines_absolute_and_relative_thresholds() -> None:
    policy = MinimumObservedSupport(min_observed_count=2, min_observed_ratio=0.8)
    assert policy.required_observations(10) == 8
    assert policy.required_observations(1) == 2
    assert policy.is_sufficient(8, 10)
    assert not policy.is_sufficient(7, 10)
    assert not policy.is_sufficient(1, 1)
    with pytest.raises(ValueError, match="cannot exceed"):
        policy.is_sufficient(2, 1)
    with pytest.raises(ValueError, match="non-negative"):
        policy.required_observations(-1)


def test_observed_support_rejects_an_impossible_dataset_threshold(
    run_context,  # type: ignore[no-untyped-def]
) -> None:
    MinimumObservedSupport(min_observed_count=4).validate_dataset(run_context.dataset)
    with pytest.raises(ComponentError, match="exceeds the dataset row count"):
        MinimumObservedSupport(min_observed_count=5).validate_dataset(run_context.dataset)
