from __future__ import annotations

import pytest

from salvi_experiments.configuration import UncertaintyConfiguration
from salvi_experiments.metrics import BiclusterMembership, calculate_accuracy


def _membership(
    identifier: str,
    rows: tuple[int, ...],
    columns: tuple[int, ...],
) -> BiclusterMembership:
    return BiclusterMembership(
        identifier=identifier,
        row_indices=rows,
        column_indices=columns,
    )


def test_perfect_accuracy_is_one() -> None:
    truth = (_membership("truth", (0, 1), (0, 1)),)
    inferred = (_membership("inferred", (0, 1), (0, 1)),)
    result = calculate_accuracy(
        inferred,
        truth,
        uncertainty=UncertaintyConfiguration(bootstrap_samples=0),
        coverage_thresholds=(0.25, 0.5, 0.75),
    )
    assert result.relevance == pytest.approx(1.0)
    assert result.recovery == pytest.approx(1.0)
    assert result.biclustering_error == pytest.approx(1.0)
    assert result.biclustering_error_interval.estimate == pytest.approx(1.0)
    assert dict(result.coverage) == {0.25: 1.0, 0.5: 1.0, 0.75: 1.0}


def test_recovery_does_not_penalize_extra_biclusters_but_relevance_does() -> None:
    truth = (_membership("truth", (0, 1), (0, 1)),)
    inferred = (
        _membership("correct", (0, 1), (0, 1)),
        _membership("extra", (2, 3), (2, 3)),
    )
    result = calculate_accuracy(
        inferred,
        truth,
        uncertainty=UncertaintyConfiguration(bootstrap_samples=0),
        coverage_thresholds=(0.5,),
    )
    assert result.recovery == pytest.approx(1.0)
    assert result.relevance == pytest.approx(0.5)
    assert result.biclustering_error < 1.0


def test_biclustering_error_matches_moeba_bio_disjoint_fixture() -> None:
    truth = (
        _membership("t0", (0, 1), (2, 3)),
        _membership("t1", (2, 3), (2, 3)),
        _membership("t2", (5, 6), (4, 5, 6, 7)),
    )
    inferred = (
        _membership("i0", (0, 1, 2, 3), (2, 3)),
        _membership("i1", (5, 6), (3, 4)),
        _membership("i2", (3, 4, 5), (6, 7, 8)),
    )
    result = calculate_accuracy(
        inferred,
        truth,
        uncertainty=UncertaintyConfiguration(bootstrap_samples=0),
        coverage_thresholds=(0.5,),
    )
    assert result.biclustering_error == pytest.approx(1.0 - 19.0 / 25.0)


def test_empty_detection_has_zero_accuracy_and_explicit_unmatched_targets() -> None:
    truth = (_membership("truth", (0, 1), (0, 1)),)
    result = calculate_accuracy(
        (),
        truth,
        uncertainty=UncertaintyConfiguration(bootstrap_samples=10),
        coverage_thresholds=(0.5,),
    )
    assert result.relevance == result.recovery == result.biclustering_error == 0.0
    assert result.matches[0].perspective == "GROUND_TRUTH"
    assert result.matches[0].best_match_id is None
