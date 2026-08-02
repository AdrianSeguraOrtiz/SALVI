from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import salvi.engine as engine_package
from salvi.application.configuration import load_configuration
from salvi.application.factory import build_specification, prepare_run
from salvi.application.run_service import RunService
from salvi.components.backend_operators import BackendOperatorSpec
from salvi.components.defaults import default_component_registry
from salvi.components.protocols import ComponentKind
from salvi.domain.enums import EvaluationIssueCode, ObjectiveDirection
from salvi.domain.models import EvaluationIssue
from salvi.engine import PymooNsga2SearchEngine
from salvi.engine import pymoo as pymoo_module
from salvi.evaluation.workspace import EvaluationWorkspace
from salvi.exceptions import ComponentError

from .conftest import configuration_mapping, create_dataset_bundle, write_configuration

PYMOO_AVAILABLE = importlib.util.find_spec("pymoo") is not None


@dataclass(frozen=True, slots=True)
class _ExternalPymooOperator:
    component_name: str = "external_operator"

    def backend_operator_spec(self, backend: str) -> BackendOperatorSpec:
        assert backend == "pymoo"
        return BackendOperatorSpec(
            factory_path="pymoo.operators.mutation.bitflip:BitflipMutation",
            keyword_arguments=(("prob", 0.25),),
        )


def _pymoo_mapping(
    dataset: Path,
    output: Path,
    *,
    evaluations: int = 12,
    crossover: str = "half_uniform_membership",
) -> dict[str, object]:
    mapping = configuration_mapping(dataset, output, overwrite=True)
    mapping["search"] = {
        "engine": {
            "name": "pymoo_nsga2",
            "parameters": {
                "population_size": 4,
                "eliminate_duplicates": True,
            },
        },
        "objectives": [
            {"name": "internal_coherence", "parameters": {}},
            {"name": "contrast", "parameters": {"min_background_ratio": 0.1}},
        ],
        "constraints": [
            {
                "name": "balanced_bicluster_size_range",
                "parameters": {"minimum": 0.0, "maximum": 1.0},
            }
        ],
        "descriptors": [],
        "crossover": {
            "name": crossover,
            "parameters": {
                "application_probability": 0.9,
                "row_exchange_probability": 0.5,
                "column_exchange_probability": 0.5,
            },
        },
        "mutation": {
            "name": "bit_flip_membership",
            "parameters": {
                "application_probability": 1.0,
                "bit_probability": None,
            },
        },
        "initialization": {"name": "stratified", "parameters": {"cardinality_levels": 3}},
        "termination": {
            "name": "evaluation_budget",
            "parameters": {"max_evaluations": evaluations},
        },
    }
    mapping["monitoring"] = {
        "queue_capacity": 64,
        "checkpoint_interval_evaluations": None,
        "observers": [{"name": "search_progress", "parameters": {}}],
    }
    mapping["final_selection"] = None
    return mapping


def _scientific_state(result: object) -> tuple[tuple[object, ...], ...]:
    from salvi.domain import RunResult

    assert isinstance(result, RunResult)
    return tuple(
        (
            evaluation.candidate.identifier,
            evaluation.candidate.bicluster.row_indices,
            evaluation.candidate.bicluster.column_indices,
            tuple((value.name, value.value) for value in evaluation.objectives),
        )
        for evaluation in result.repertoire.evaluations
    )


def test_pymoo_nsga2_is_described_as_a_bundled_non_resumable_engine() -> None:
    registration = default_component_registry().get(
        ComponentKind.SEARCH_ENGINE,
        "pymoo_nsga2",
    )

    assert registration.provides == frozenset({"search-engine", "search-result"})
    assert any("bundled" in note for note in registration.compatibility_notes)
    assert any("resumption" in note for note in registration.compatibility_notes)
    assert any("forbidden" in note for note in registration.compatibility_notes)


def test_pymoo_nsga2_reports_a_missing_runtime_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_module(name: str) -> object:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(pymoo_module.importlib, "import_module", missing_module)

    with pytest.raises(ComponentError, match="pymoo runtime"):
        pymoo_module._load_pymoo()


def test_search_engine_package_exposes_all_lazy_engine_types() -> None:
    assert engine_package.DeepGridMomeArchive.__name__ == "DeepGridMomeArchive"
    assert engine_package.SerialMomeSearchEngine.__name__ == "SerialMomeSearchEngine"
    assert engine_package.PymooNsga2SearchEngine is PymooNsga2SearchEngine
    with pytest.raises(AttributeError):
        engine_package.__getattr__("unknown_engine")


@pytest.mark.skipif(not PYMOO_AVAILABLE, reason="requires the pymoo runtime")
def test_pymoo_operator_adaptation_uses_a_lazy_component_specification() -> None:
    operator = pymoo_module._build_backend_operator(_ExternalPymooOperator())

    assert type(operator).__name__ == "BitflipMutation"
    assert float(operator.prob.value) == pytest.approx(0.25)


def test_backend_operator_spec_rejects_invalid_factories_and_arguments() -> None:
    with pytest.raises(ValueError, match="module:attribute"):
        BackendOperatorSpec(factory_path="invalid")
    with pytest.raises(ValueError, match="unique"):
        BackendOperatorSpec(
            factory_path="package.module:Factory",
            keyword_arguments=(("prob", 0.1), ("prob", 0.2)),
        )


def test_non_resumable_engine_rejects_resume_and_periodic_checkpoints(tmp_path: Path) -> None:
    dataset = create_dataset_bundle(tmp_path / "dataset", rows=8, columns=3)
    mapping = _pymoo_mapping(dataset, tmp_path / "output")
    mapping["run"]["resume_from_checkpoint"] = str(tmp_path / "checkpoint.json")
    loaded = load_configuration(write_configuration(tmp_path / "resume.yaml", mapping))
    with pytest.raises(ComponentError, match="does not support checkpoint resumption"):
        build_specification(loaded.configuration)

    mapping = _pymoo_mapping(dataset, tmp_path / "output")
    mapping["monitoring"]["checkpoint_interval_evaluations"] = 4
    loaded = load_configuration(write_configuration(tmp_path / "periodic.yaml", mapping))
    with pytest.raises(ComponentError, match="does not support resumable periodic"):
        build_specification(loaded.configuration)


def test_pymoo_nsga2_requires_explicit_variation_operators(tmp_path: Path) -> None:
    dataset = create_dataset_bundle(tmp_path / "dataset", rows=8, columns=3)
    mapping = _pymoo_mapping(dataset, tmp_path / "output")
    search = mapping["search"]
    assert isinstance(search, dict)

    search.pop("crossover")
    loaded = load_configuration(write_configuration(tmp_path / "missing-crossover.yaml", mapping))
    with pytest.raises(ComponentError, match="crossover_operator expects exactly 1"):
        build_specification(loaded.configuration)

    mapping = _pymoo_mapping(dataset, tmp_path / "output")
    search = mapping["search"]
    assert isinstance(search, dict)
    search.pop("mutation")
    loaded = load_configuration(write_configuration(tmp_path / "missing-mutation.yaml", mapping))
    with pytest.raises(ComponentError, match="mutation_operator expects exactly 1"):
        build_specification(loaded.configuration)


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    (
        (
            "emitters",
            [{"name": "random_move", "parameters": {}}],
            "emitter expects exactly 0",
        ),
        (
            "scheduler",
            {"name": "first", "parameters": {}},
            "scheduler expects exactly 0",
        ),
    ),
)
def test_pymoo_nsga2_rejects_qd_emitters_and_scheduler(
    tmp_path: Path,
    key: str,
    value: object,
    expected: str,
) -> None:
    dataset = create_dataset_bundle(tmp_path / "dataset", rows=8, columns=3)
    mapping = _pymoo_mapping(dataset, tmp_path / "output")
    search = mapping["search"]
    assert isinstance(search, dict)
    search[key] = value
    loaded = load_configuration(write_configuration(tmp_path / f"{key}.yaml", mapping))

    with pytest.raises(ComponentError, match=expected):
        build_specification(loaded.configuration)


@pytest.mark.skipif(not PYMOO_AVAILABLE, reason="requires the pymoo runtime")
def test_pymoo_nsga2_uses_salvi_evaluation_reproducibly(
    tmp_path: Path,
) -> None:
    dataset = create_dataset_bundle(tmp_path / "dataset", rows=8, columns=3)
    first_path = write_configuration(
        tmp_path / "first.yaml",
        _pymoo_mapping(dataset, tmp_path / "first-output"),
    )
    second_path = write_configuration(
        tmp_path / "second.yaml",
        _pymoo_mapping(dataset, tmp_path / "second-output"),
    )

    first = RunService().run(first_path)
    second = RunService().run(second_path)

    assert first.repertoire.evaluations
    assert _scientific_state(first) == _scientific_state(second)
    assert all(
        evaluation.candidate.provenance is not None
        and evaluation.candidate.provenance.producer == "pymoo_nsga2"
        for evaluation in first.repertoire.evaluations
    )
    assert not tuple((first.output_directory / "checkpoints").iterdir())


@pytest.mark.parametrize(
    "crossover",
    (
        "membership_recombination",
        "evidence_weighted_recombination",
    ),
)
@pytest.mark.skipif(not PYMOO_AVAILABLE, reason="requires the pymoo runtime")
def test_pymoo_nsga2_accepts_salvi_crossover_operators(
    tmp_path: Path,
    crossover: str,
) -> None:
    dataset = create_dataset_bundle(tmp_path / "dataset", rows=8, columns=3)
    path = write_configuration(
        tmp_path / f"{crossover}.yaml",
        _pymoo_mapping(
            dataset,
            tmp_path / f"{crossover}-output",
            crossover=crossover,
        ),
    )

    result = RunService().run(path)

    assert result.repertoire.evaluations
    assert all(
        evaluation.candidate.provenance is not None
        and crossover in evaluation.candidate.provenance.operation
        for evaluation in result.repertoire.evaluations
        if evaluation.candidate.generation > 0
    )


@pytest.mark.skipif(not PYMOO_AVAILABLE, reason="requires the pymoo runtime")
def test_pymoo_nsga2_requires_budget_for_its_initial_population(tmp_path: Path) -> None:
    dataset = create_dataset_bundle(tmp_path / "dataset", rows=8, columns=3)
    path = write_configuration(
        tmp_path / "configuration.yaml",
        _pymoo_mapping(dataset, tmp_path / "output", evaluations=3),
    )

    with pytest.raises(ComponentError, match="evaluation budget at least as large"):
        RunService().run(path)


@pytest.mark.skipif(not PYMOO_AVAILABLE, reason="requires the pymoo runtime")
def test_pymoo_nsga2_does_not_write_misleading_pending_checkpoints(
    tmp_path: Path,
) -> None:
    dataset = create_dataset_bundle(tmp_path / "dataset", rows=8, columns=3)
    path = write_configuration(
        tmp_path / "configuration.yaml",
        _pymoo_mapping(dataset, tmp_path / "output"),
    )
    loaded = load_configuration(path)
    specification = build_specification(loaded.configuration)
    prepared = prepare_run(specification)
    engine = specification.search_engine
    assert isinstance(engine, PymooNsga2SearchEngine)
    engine.initialize(specification, prepared.context)

    assert engine.result().evaluations == ()
    with pytest.raises(ValueError, match="positive"):
        engine.ask(0)
    with pytest.raises(ComponentError, match="ask must be called"):
        engine.tell(())
    engine.ask(engine.batch_size)

    with pytest.raises(ComponentError, match="before asking"):
        engine.ask(engine.batch_size)
    with pytest.raises(ComponentError, match="awaiting evaluation"):
        engine.result()
    with pytest.raises(ComponentError, match="match the preceding"):
        engine.tell(())
    with pytest.raises(ComponentError, match="does not support checkpoints"):
        engine.checkpoint()
    with pytest.raises(ComponentError, match="invalid shape"):
        engine._repair_vector(
            np.asarray([[True]], dtype=np.bool_),
            prepared.context,
            np.random.default_rng(0),
        )
    with pytest.raises(ComponentError, match="invalid shape"):
        engine._validate_vector_shape(
            np.asarray([[True]], dtype=np.bool_),
            prepared.context,
        )
    with pytest.raises(ComponentError, match="minimum cardinality"):
        engine._activate_minimum(
            np.asarray([False], dtype=np.bool_),
            2,
            np.random.default_rng(0),
        )


@pytest.mark.skipif(not PYMOO_AVAILABLE, reason="requires the pymoo runtime")
def test_pymoo_nsga2_rejects_an_initializer_that_underproduces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = create_dataset_bundle(tmp_path / "dataset", rows=8, columns=3)
    path = write_configuration(
        tmp_path / "configuration.yaml",
        _pymoo_mapping(dataset, tmp_path / "output"),
    )
    loaded = load_configuration(path)
    specification = build_specification(loaded.configuration)
    prepared = prepare_run(specification)
    engine = specification.search_engine
    assert isinstance(engine, PymooNsga2SearchEngine)
    monkeypatch.setattr(
        type(specification.initializer),
        "initialize",
        lambda *_args, **_kwargs: (),
    )

    with pytest.raises(ComponentError, match="did not produce the population"):
        engine.initialize(specification, prepared.context)


@pytest.mark.skipif(not PYMOO_AVAILABLE, reason="requires the pymoo runtime")
def test_pymoo_nsga2_rejects_qd_components(
    tmp_path: Path,
) -> None:
    dataset = create_dataset_bundle(tmp_path / "dataset", rows=8, columns=3)
    mapping = _pymoo_mapping(dataset, tmp_path / "output", evaluations=4)
    mapping["search"]["archive"] = {
        "name": "deep_grid_mome",
        "parameters": {
            "axes": [
                {"descriptor": "row_cardinality", "binning": "EXACT"},
                {"descriptor": "column_cardinality", "binning": "EXACT"},
            ]
        },
    }
    path = write_configuration(tmp_path / "configuration.yaml", mapping)
    loaded = load_configuration(path)
    with pytest.raises(ComponentError, match="archive expects exactly 0"):
        build_specification(loaded.configuration)


@pytest.mark.skipif(not PYMOO_AVAILABLE, reason="requires the pymoo runtime")
def test_pymoo_nsga2_maps_salvi_directions_and_returns_its_own_front(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = create_dataset_bundle(tmp_path / "dataset", rows=8, columns=3)
    path = write_configuration(
        tmp_path / "configuration.yaml",
        _pymoo_mapping(dataset, tmp_path / "output", evaluations=4),
    )
    loaded = load_configuration(path)
    specification = build_specification(loaded.configuration)
    prepared = prepare_run(specification)
    engine = specification.search_engine
    assert isinstance(engine, PymooNsga2SearchEngine)
    engine.initialize(specification, prepared.context)
    candidates = tuple(engine.ask(engine.batch_size))
    population = engine._pending_population
    assert population is not None
    workspace = EvaluationWorkspace(prepared.context)
    batch = specification.executor.evaluate(
        candidates,
        specification.objectives,
        specification.descriptors,
        workspace,
        constraints=specification.constraints,
        worker_count=1,
    )
    evaluations = list(batch.evaluations)
    evaluations[0] = evaluations[0].model_copy(
        update={
            "issues": (
                EvaluationIssue(
                    code=EvaluationIssueCode.PATTERN_FIT_FAILED,
                    message="forced invalid candidate",
                ),
            )
        }
    )

    engine.tell(evaluations)

    assert engine.finished()
    assert engine.ask(engine.batch_size) == ()
    values = np.asarray(population.get("F"), dtype=np.float64)
    constraints = np.asarray(population.get("G"), dtype=np.float64)
    for row, evaluation in enumerate(evaluations):
        for column, objective in enumerate(evaluation.objectives):
            expected = (
                objective.value
                if objective.direction is ObjectiveDirection.MINIMIZE
                else -objective.value
            )
            assert values[row, column] == pytest.approx(expected)
    for row, evaluation in enumerate(evaluations):
        assert bool(constraints[row, 0] <= 0.0) is evaluation.valid
        assert constraints[row, 1] == pytest.approx(evaluation.constraints[0].value)

    result = engine.result()
    assert result.evaluations
    assert all(evaluation.valid for evaluation in result.evaluations)
    assert all(evaluation.archive_coordinate is None for evaluation in result.evaluations)
    specification.executor.close()


def test_pymoo_nsga2_rejects_resume_even_before_initialization() -> None:
    engine = PymooNsga2SearchEngine()

    with pytest.raises(ComponentError, match="checkpoint resumption"):
        engine.restore(object())  # type: ignore[arg-type]
    with pytest.raises(ComponentError, match="not initialized"):
        engine.progress()


@pytest.mark.skipif(not PYMOO_AVAILABLE, reason="requires the pymoo runtime")
def test_pymoo_nsga2_honors_a_non_divisible_evaluation_budget(tmp_path: Path) -> None:
    dataset = create_dataset_bundle(tmp_path / "dataset", rows=8, columns=3)
    path = write_configuration(
        tmp_path / "configuration.yaml",
        _pymoo_mapping(dataset, tmp_path / "output", evaluations=10),
    )

    result = RunService().run(path)

    metadata = json.loads((result.output_directory / "run-metadata.json").read_text())
    assert metadata["search"]["evaluations"] == 10
    assert metadata["final_selection"] is None
    assert not tuple((result.output_directory / "checkpoints").iterdir())
