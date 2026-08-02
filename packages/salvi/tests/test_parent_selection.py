from __future__ import annotations

import numpy as np

from salvi.components.parent_selection import (
    CellUniformQualityParentSelection,
    RepertoireUniformParentSelection,
)
from salvi.domain import (
    Bicluster,
    Candidate,
    Evaluation,
    ObjectiveDirection,
    ObjectiveValue,
    Repertoire,
)


def _evaluation(identifier: str, coordinate: tuple[int, ...], value: float) -> Evaluation:
    index = int(identifier.split("-")[-1])
    return Evaluation(
        candidate=Candidate(
            identifier=identifier,
            bicluster=Bicluster(
                row_indices=(index, index + 1),
                column_indices=(0, 1),
            ),
        ),
        objectives=(
            ObjectiveValue(
                name="loss",
                value=value,
                direction=ObjectiveDirection.MINIMIZE,
            ),
        ),
        descriptors=(),
        archive_coordinate=coordinate,
    )


def test_cell_uniform_selection_does_not_overweight_large_cells() -> None:
    repertoire = Repertoire(
        evaluations=(
            *tuple(_evaluation(f"large-{index}", (0, 0), 0.5) for index in range(9)),
            _evaluation("small-20", (1, 1), 0.5),
        )
    )
    uniform = RepertoireUniformParentSelection()
    cell_uniform = CellUniformQualityParentSelection()
    uniform_generator = np.random.default_rng(17)
    cell_generator = np.random.default_rng(17)

    uniform_small = sum(
        uniform.select(
            repertoire,
            uniform_generator,
            pool_size=1,
            eligible=lambda _: True,
            guided=False,
        ).archive_coordinate
        == (1, 1)
        for _ in range(1000)
    )
    cell_small = sum(
        cell_uniform.select(
            repertoire,
            cell_generator,
            pool_size=1,
            eligible=lambda _: True,
            guided=False,
        ).archive_coordinate
        == (1, 1)
        for _ in range(1000)
    )

    assert 50 <= uniform_small <= 150
    assert 450 <= cell_small <= 550


def test_parent_selection_respects_eligibility_and_quality_guidance() -> None:
    repertoire = Repertoire(
        evaluations=(
            _evaluation("candidate-0", (0, 0), 0.8),
            _evaluation("candidate-1", (0, 0), 0.1),
            _evaluation("candidate-2", (0, 0), 0.5),
        )
    )

    selected = RepertoireUniformParentSelection().select(
        repertoire,
        np.random.default_rng(3),
        pool_size=3,
        eligible=lambda evaluation: evaluation.candidate.identifier != "candidate-2",
        guided=True,
    )

    assert selected is not None
    assert selected.candidate.identifier == "candidate-1"
