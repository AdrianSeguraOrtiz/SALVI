"""Structural similarity utilities shared by search, monitoring, and selection."""

from __future__ import annotations

import math
from collections.abc import Set


def jaccard_distance(first: Set[int], second: Set[int]) -> float:
    union_size = len(first | second)
    return 0.0 if union_size == 0 else 1.0 - len(first & second) / union_size


def jaccard_similarity(first: Set[int], second: Set[int]) -> float:
    """Return exact set overlap on a zero-to-one scale."""

    return 1.0 - jaccard_distance(first, second)


def geometric_structural_similarity(
    first_rows: Set[int],
    first_columns: Set[int],
    second_rows: Set[int],
    second_columns: Set[int],
) -> float:
    """Require simultaneous row and column overlap through a geometric mean."""

    row_similarity = jaccard_similarity(first_rows, second_rows)
    column_similarity = jaccard_similarity(first_columns, second_columns)
    return math.sqrt(row_similarity * column_similarity)


def structural_distance(
    first_rows: Set[int],
    first_columns: Set[int],
    second_rows: Set[int],
    second_columns: Set[int],
    *,
    row_weight: float,
) -> float:
    if not 0.0 <= row_weight <= 1.0:
        raise ValueError("row_weight must be between 0 and 1")
    return row_weight * jaccard_distance(first_rows, second_rows) + (
        1.0 - row_weight
    ) * jaccard_distance(first_columns, second_columns)


__all__ = [
    "geometric_structural_similarity",
    "jaccard_distance",
    "jaccard_similarity",
    "structural_distance",
]
