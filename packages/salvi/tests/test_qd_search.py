from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pytest
from pydantic import ValidationError

from salvi.application.configuration import load_configuration
from salvi.application.factory import build_specification, prepare_run
from salvi.application.run_service import RunService
from salvi.components.descriptors import ColumnCardinality, RowCardinality
from salvi.components.protocols import Objective
from salvi.domain import (
    ArchiveInsertionStatus,
    Bicluster,
    BinningStrategy,
    Candidate,
    ColumnKind,
    ColumnMetadata,
    Evaluation,
    EvaluationIssue,
    EvaluationIssueCode,
    EventType,
    NamedValue,
    ObjectiveDirection,
    ObjectiveResult,
    ObjectiveValue,
)
from salvi.engine.archive import DeepGridMomeArchive
from salvi.engine.dominance import objective_dominates
from salvi.engine.grid import ArchiveAxisConfiguration, AxisBinner
from salvi.engine.mome import SerialMomeSearchEngine
from salvi.evaluation.workspace import EvaluationWorkspace
from salvi.exceptions import ComponentError
from salvi.infrastructure.bicluster_set import BiclusterSetReader
from salvi.infrastructure.dataset_bundle import DatasetBundleWriter
from salvi.infrastructure.events import SQLiteRunEventSource
from salvi.infrastructure.files import sha256_file

from .conftest import configuration_mapping, create_dataset_bundle, write_configuration


@dataclass(frozen=True, slots=True)
class StaticObjective:
    component_name: str
    direction: ObjectiveDirection
    provides: frozenset[str] = frozenset({"objective"})
    requires: frozenset[str] = frozenset()

    def evaluate(self, candidate: Candidate, workspace: EvaluationWorkspace) -> ObjectiveResult:
        del candidate, workspace
        return ObjectiveResult(value=0.0, columns=())


def _evaluation(
    identifier: str,
    rows: tuple[int, ...],
    columns: tuple[int, ...],
    coherence: float,
    contrast: float,
) -> Evaluation:
    candidate = Candidate(
        identifier=identifier,
        bicluster=Bicluster(row_indices=rows, column_indices=columns),
    )
    return Evaluation(
        candidate=candidate,
        objectives=(
            ObjectiveValue(
                name="coherence",
                value=coherence,
                direction=ObjectiveDirection.MINIMIZE,
            ),
            ObjectiveValue(
                name="contrast",
                value=contrast,
                direction=ObjectiveDirection.MAXIMIZE,
            ),
        ),
        descriptors=(
            NamedValue(name="row_cardinality", value=float(len(rows))),
            NamedValue(name="column_cardinality", value=float(len(columns))),
        ),
    )


def _archive(run_context: object, capacity: int = 4) -> DeepGridMomeArchive:
    from salvi.application.context import RunContext

    assert isinstance(run_context, RunContext)
    archive = DeepGridMomeArchive(
        axes=(
            ArchiveAxisConfiguration(
                descriptor="row_cardinality",
                binning=BinningStrategy.EXACT,
            ),
            ArchiveAxisConfiguration(
                descriptor="column_cardinality",
                binning=BinningStrategy.EXACT,
            ),
        ),
        cell_capacity=capacity,
    )
    objectives: tuple[Objective, ...] = (
        StaticObjective("coherence", ObjectiveDirection.MINIMIZE),
        StaticObjective("contrast", ObjectiveDirection.MAXIMIZE),
    )
    archive.initialize(
        run_context,
        objectives,
        (RowCardinality(), ColumnCardinality()),
    )
    return archive


def test_descriptor_domains_allow_axis_specific_discretization(run_context: object) -> None:
    from salvi.application.context import RunContext

    assert isinstance(run_context, RunContext)
    row_domain = RowCardinality().domain(run_context)
    column_domain = ColumnCardinality().domain(run_context)
    row_axis = AxisBinner.create(
        ArchiveAxisConfiguration(
            descriptor="row_cardinality",
            binning=BinningStrategy.GEOMETRIC,
            bins=2,
        ),
        row_domain,
    )
    column_axis = AxisBinner.create(
        ArchiveAxisConfiguration(
            descriptor="column_cardinality",
            binning=BinningStrategy.LINEAR,
            bins=2,
        ),
        column_domain,
    )
    assert row_axis.index(row_domain.minimum) == 0
    assert row_axis.index(row_domain.maximum) == 1
    assert column_axis.index(column_domain.minimum) == 0
    assert column_axis.index(column_domain.maximum) == 1
    assert row_axis.index(row_domain.maximum + 1) is None

    custom = AxisBinner.create(
        ArchiveAxisConfiguration(
            descriptor="row_cardinality",
            binning=BinningStrategy.CUSTOM,
            boundaries=(3.0,),
        ),
        row_domain,
    )
    assert custom.index(3.0) == 1


def test_archive_enumerates_every_reachable_cardinality_cell(run_context: object) -> None:
    archive = _archive(run_context)

    targets = archive.cell_targets()

    assert {(target.row_count, target.column_count) for target in targets} == {
        (rows, columns) for rows in range(2, 5) for columns in range(2, 4)
    }
    assert len({target.coordinate.indices for target in targets}) == len(targets)


def test_axis_configuration_rejects_ambiguous_shapes() -> None:
    with pytest.raises(ValidationError, match="requires bins"):
        ArchiveAxisConfiguration(
            descriptor="rows",
            binning=BinningStrategy.LINEAR,
        )
    with pytest.raises(ValidationError, match="strictly increasing"):
        ArchiveAxisConfiguration(
            descriptor="rows",
            binning=BinningStrategy.CUSTOM,
            boundaries=(3.0, 2.0),
        )


def test_direction_aware_dominance_handles_mixed_objectives() -> None:
    better = _evaluation("better", (0, 1), (0, 1), 0.1, 0.9)
    worse = _evaluation("worse", (0, 1), (0, 1), 0.2, 0.8)
    tradeoff = _evaluation("tradeoff", (0, 1), (0, 1), 0.05, 0.7)
    assert objective_dominates(better.objectives, worse.objectives)
    assert not objective_dominates(worse.objectives, better.objectives)
    assert not objective_dominates(better.objectives, tradeoff.objectives)
    assert not objective_dominates(tradeoff.objectives, better.objectives)


def test_sparse_archive_is_lazy_and_retains_local_pareto_front(run_context: object) -> None:
    archive = _archive(run_context, capacity=3)
    assert archive.occupied_cell_count == 0

    first = _evaluation("first", (0, 1), (0, 1), 0.2, 0.8)
    dominated = _evaluation("dominated", (0, 1), (0, 2), 0.3, 0.7)
    tradeoff = _evaluation("tradeoff", (0, 1), (1, 2), 0.1, 0.6)
    other_cell = _evaluation("other", (0, 1, 2), (0, 1), 0.4, 0.9)

    outcomes = archive.add((first, dominated, tradeoff, other_cell))
    assert tuple(outcome.status for outcome in outcomes) == (
        ArchiveInsertionStatus.INSERTED,
        ArchiveInsertionStatus.REJECTED_DOMINATED,
        ArchiveInsertionStatus.INSERTED,
        ArchiveInsertionStatus.INSERTED,
    )
    assert archive.occupied_cell_count == 2
    assert {item.candidate.identifier for item in archive.repertoire().evaluations} == {
        "first",
        "tradeoff",
        "other",
    }

    duplicate = _evaluation("duplicate", (0, 1), (0, 1), 0.1, 1.0)
    assert archive.add((duplicate,))[0].status is ArchiveInsertionStatus.REJECTED_DUPLICATE


def test_archive_reuses_repertoire_until_retained_state_changes(run_context: object) -> None:
    archive = _archive(run_context, capacity=3)
    first = _evaluation("first", (0, 1), (0, 1), 0.2, 0.8)
    dominated = _evaluation("dominated", (0, 1), (0, 2), 0.3, 0.7)
    replacement = _evaluation("replacement", (0, 1), (0, 2), 0.1, 0.9)

    archive.add((first,))
    initial = archive.repertoire()
    assert archive.repertoire() is initial

    archive.add((dominated,))
    assert archive.repertoire() is initial

    archive.add((replacement,))
    updated = archive.repertoire()
    assert updated is not initial
    assert archive.repertoire() is updated


def test_archive_depth_is_bounded_and_deterministic(run_context: object) -> None:
    archive = _archive(run_context, capacity=2)
    evaluations = (
        _evaluation("left", (0, 1), (0, 1), 0.1, 0.4),
        _evaluation("middle", (0, 1), (0, 2), 0.2, 0.6),
        _evaluation("right", (0, 1), (1, 2), 0.3, 0.9),
    )
    outcomes = archive.add(evaluations)
    assert outcomes[-1].status is ArchiveInsertionStatus.INSERTED_WITH_EVICTIONS
    assert outcomes[-1].evicted_candidate_identifiers == ("middle",)
    assert {item.candidate.identifier for item in archive.repertoire().evaluations} == {
        "left",
        "right",
    }


def test_archive_rejects_invalid_and_explicitly_out_of_bounds_candidates(
    run_context: object,
) -> None:
    from salvi.application.context import RunContext

    assert isinstance(run_context, RunContext)
    invalid_candidate = Candidate(
        identifier="invalid",
        bicluster=Bicluster(row_indices=(0, 1), column_indices=(0, 1)),
    )
    invalid = Evaluation(
        candidate=invalid_candidate,
        objectives=(),
        descriptors=(),
        issues=(
            EvaluationIssue(
                code=EvaluationIssueCode.PATTERN_FIT_FAILED,
                message="invalid test fit",
            ),
        ),
    )
    archive = _archive(run_context)
    assert archive.add((invalid,))[0].status is ArchiveInsertionStatus.REJECTED_INVALID
    assert archive.occupied_cell_count == 0

    narrowed = DeepGridMomeArchive(
        axes=(
            ArchiveAxisConfiguration(
                descriptor="row_cardinality",
                binning=BinningStrategy.EXACT,
                maximum=2,
            ),
            ArchiveAxisConfiguration(
                descriptor="column_cardinality",
                binning=BinningStrategy.EXACT,
            ),
        )
    )
    narrowed.initialize(
        run_context,
        (
            StaticObjective("coherence", ObjectiveDirection.MINIMIZE),
            StaticObjective("contrast", ObjectiveDirection.MAXIMIZE),
        ),
        (RowCardinality(), ColumnCardinality()),
    )
    outside = _evaluation("outside", (0, 1, 2), (0, 1), 0.1, 0.9)
    outcome = narrowed.add((outside,))[0]
    assert outcome.status is ArchiveInsertionStatus.REJECTED_OUT_OF_BOUNDS
    assert narrowed.occupied_cell_count == 0


def test_equal_objective_vectors_preserve_distinct_structures(run_context: object) -> None:
    archive = _archive(run_context, capacity=2)
    equal = (
        _evaluation("one", (0, 1), (0, 1), 0.2, 0.8),
        _evaluation("two", (0, 1), (0, 2), 0.2, 0.8),
    )
    assert all(outcome.accepted for outcome in archive.add(equal))
    assert len(archive.repertoire().evaluations) == 2


def _scientific_mapping(
    dataset: Path,
    output: Path,
    *,
    evaluations: int,
    resume_from: Path | None = None,
    patterns: list[str] | None = None,
    checkpoint_interval: int = 4,
) -> dict[str, object]:
    mapping = configuration_mapping(dataset, output, overwrite=True, patterns=patterns)
    mapping["run"]["resume_from_checkpoint"] = None if resume_from is None else str(resume_from)
    mapping["search"] = {
        "engine": {
            "name": "serial_mome",
            "parameters": {"initial_population_size": 4, "batch_size": 2},
        },
        "objectives": [
            {"name": "internal_coherence", "parameters": {}},
            {"name": "contrast", "parameters": {"min_background_ratio": 0.1}},
        ],
        "descriptors": [
            {"name": "row_cardinality", "parameters": {}},
            {"name": "column_cardinality", "parameters": {}},
        ],
        "archive": {
            "name": "deep_grid_mome",
            "parameters": {
                "axes": [
                    {
                        "descriptor": "row_cardinality",
                        "binning": "GEOMETRIC",
                        "bins": 3,
                    },
                    {
                        "descriptor": "column_cardinality",
                        "binning": "EXACT",
                    },
                ],
                "cell_capacity": 3,
            },
        },
        "parent_selection": {"name": "repertoire_uniform", "parameters": {}},
        "initialization": {"name": "uniform_random", "parameters": {}},
        "emitters": [{"name": "random_move", "parameters": {}}],
        "scheduler": {"name": "first", "parameters": {}},
        "termination": {
            "name": "evaluation_budget",
            "parameters": {"max_evaluations": evaluations},
        },
    }
    mapping["monitoring"]["checkpoint_interval_evaluations"] = checkpoint_interval
    mapping["final_selection"] = None
    return mapping


def _full_qd_mapping(dataset: Path, output: Path) -> dict[str, object]:
    mapping = _scientific_mapping(dataset, output, evaluations=27, checkpoint_interval=9)
    search = mapping["search"]
    assert isinstance(search, dict)
    search["engine"] = {
        "name": "serial_mome",
        "parameters": {"initial_population_size": 9, "batch_size": 9},
    }
    search["initialization"] = {
        "name": "pattern_aware",
        "parameters": {"cardinality_levels": 4, "joint_column_candidate_pool_size": 8},
    }
    search["mate_selection"] = {
        "name": "repertoire_random",
        "parameters": {},
    }
    search["crossover"] = {
        "name": "membership_recombination",
        "parameters": {
            "application_probability": 1.0,
            "row_exchange_probability": 0.5,
            "column_exchange_probability": 0.5,
        },
    }
    search["emitters"] = [
        {"name": name, "parameters": parameters}
        for name, parameters in (
            ("add_row", {"guided": True, "parent_pool_size": 8}),
            ("remove_row", {"guided": True, "parent_pool_size": 8}),
            ("swap_row", {"guided": True, "parent_pool_size": 8}),
            ("add_column", {"guided": True, "parent_pool_size": 8}),
            ("remove_column", {"guided": True, "parent_pool_size": 8}),
            ("swap_column", {"guided": True, "parent_pool_size": 8}),
            ("shape_move", {"guided": True, "parent_pool_size": 8}),
            ("crossover", {"max_attempts": 8}),
            (
                "restart",
                {
                    "strategy": "pattern_aware",
                    "cardinality_levels": 4,
                    "joint_column_candidate_pool_size": 8,
                },
            ),
        )
    ]
    search["scheduler"] = {
        "name": "adaptive_credit",
        "parameters": {
            "exploration_weight": 0.5,
            "new_cell_reward": 1.0,
            "insertion_reward": 0.25,
        },
    }
    mapping["monitoring"]["observers"] = [
        {"name": "search_progress", "parameters": {}},
        {"name": "archive_coverage", "parameters": {}},
        {"name": "descriptor_distribution", "parameters": {"every_evaluations": 9}},
        {"name": "objective_distribution", "parameters": {"every_evaluations": 9}},
        {"name": "emitter_credit", "parameters": {}},
        {
            "name": "candidate_diversity",
            "parameters": {
                "window_size": 27,
                "row_weight": 0.5,
                "every_evaluations": 9,
            },
        },
    ]
    return mapping


def _create_numeric_dataset(destination: Path) -> Path:
    row_count = 20
    table = pa.table(
        {
            "x": pa.array([float(index + 1) for index in range(row_count)]),
            "y": pa.array([float(2 * index + 3) for index in range(row_count)]),
            "z": pa.array([float(index % 5 + 1) for index in range(row_count)]),
        }
    )
    columns = tuple(
        ColumnMetadata(index=index, name=name, kind=ColumnKind.NUMERIC)
        for index, name in enumerate(table.column_names)
    )
    DatasetBundleWriter().write(
        destination,
        identifier="numeric-dataset",
        table=table,
        columns=columns,
    )
    return destination


def _scientific_state(result: object) -> tuple[tuple[object, ...], ...]:
    from salvi.domain import RunResult

    assert isinstance(result, RunResult)
    return tuple(
        (
            evaluation.candidate.identifier,
            evaluation.candidate.bicluster.row_indices,
            evaluation.candidate.bicluster.column_indices,
            tuple((value.name, value.value) for value in evaluation.objectives),
            tuple((value.name, value.value) for value in evaluation.descriptors),
        )
        for evaluation in result.repertoire.evaluations
    )


def test_serial_mome_is_reproducible_and_writes_canonical_output(tmp_path: Path) -> None:
    dataset = create_dataset_bundle(tmp_path / "dataset", rows=8, columns=3)
    first_path = write_configuration(
        tmp_path / "first.yaml",
        _scientific_mapping(dataset, tmp_path / "first-output", evaluations=12),
    )
    second_path = write_configuration(
        tmp_path / "second.yaml",
        _scientific_mapping(dataset, tmp_path / "second-output", evaluations=12),
    )
    first = RunService().run(first_path)
    second = RunService().run(second_path)
    assert first.repertoire.evaluations
    assert _scientific_state(first) == _scientific_state(second)
    persisted = BiclusterSetReader().read(first.output_directory / "artifacts" / "repertoire")
    assert _scientific_state(
        first.model_copy(update={"repertoire": persisted})
    ) == _scientific_state(first)
    assert (first.output_directory / "checkpoints" / "checkpoint-000000000012.json").is_file()


def test_containment_final_selection_is_traced_to_archive_and_checkpoint(
    tmp_path: Path,
) -> None:
    dataset = create_dataset_bundle(tmp_path / "dataset", rows=8, columns=3)
    mapping = _scientific_mapping(dataset, tmp_path / "output", evaluations=12)
    mapping["final_selection"] = {
        "name": "containment_marginal_quality",
        "parameters": {
            "max_objective_degradation": 0.15,
            "max_degradation_per_log_area_gain": 0.20,
        },
    }
    path = write_configuration(tmp_path / "configuration.yaml", mapping)

    result = RunService().run(path)
    search_contents = BiclusterSetReader().read_contents(
        result.output_directory / "artifacts" / "search-repertoire"
    )
    contents = BiclusterSetReader().read_contents(
        result.output_directory / "artifacts" / "repertoire"
    )
    manifest = contents.manifest
    assert all(
        evaluation.archive_coordinate is not None and evaluation.final_selection is not None
        for evaluation in contents.repertoire.evaluations
    )
    assert manifest.source_checkpoint is not None
    checkpoint = result.output_directory / manifest.source_checkpoint
    assert checkpoint.is_file()
    assert sha256_file(checkpoint) == manifest.source_checkpoint_sha256
    assert manifest.source_checkpoint_evaluations == 12
    events = SQLiteRunEventSource(result.event_store).poll(limit=1000)
    selection_event = next(
        event for event in events if event.event_type is EventType.FINAL_SELECTION_COMPLETED
    )
    assert len(search_contents.repertoire.evaluations) == selection_event.payload["input_count"]
    assert search_contents.manifest.source_checkpoint == manifest.source_checkpoint
    assert selection_event.payload["input_count"] >= selection_event.payload["output_count"]
    artifact_identifiers = {
        artifact.identifier for artifact in SQLiteRunEventSource(result.event_store).artifacts()
    }
    assert artifact_identifiers == {
        "search-repertoire",
        "final-repertoire",
    }


def test_full_qd_components_complete_the_scientific_run_and_persist_feedback(
    tmp_path: Path,
) -> None:
    dataset = _create_numeric_dataset(tmp_path / "dataset")
    path = write_configuration(
        tmp_path / "configuration.yaml",
        _full_qd_mapping(dataset, tmp_path / "output"),
    )

    result = RunService().run(path)
    persisted = BiclusterSetReader().read(result.output_directory / "artifacts" / "repertoire")
    source = SQLiteRunEventSource(result.event_store)
    events = source.poll(limit=1000)
    metrics = source.poll_metrics(limit=10_000)

    assert persisted.evaluations
    assert all(evaluation.candidate.provenance is not None for evaluation in persisted.evaluations)
    assert any(
        evaluation.candidate.provenance is not None
        and evaluation.candidate.provenance.operation != "constant_anchor"
        for evaluation in persisted.evaluations
    )
    assert any(event.event_type.value == "scheduler.allocation.updated" for event in events)
    assert any(event.event_type.value == "emitter.credit.updated" for event in events)
    metric_names = {metric.name for metric in metrics}
    assert {
        "archive.occupied_cells",
        "descriptor.row_cardinality.mean",
        "objective.internal_coherence.mean",
        "diversity.cumulative_unique",
    } <= metric_names
    assert any(name.startswith("emitter.") and name.endswith(".credit") for name in metric_names)


@pytest.mark.parametrize(
    "patterns",
    [
        ["CONSTANT"],
        ["ADDITIVE"],
        ["MULTIPLICATIVE"],
        ["CONSTANT", "ADDITIVE", "MULTIPLICATIVE"],
    ],
)
def test_serial_mome_runs_every_registered_pattern_mode(
    tmp_path: Path,
    patterns: list[str],
) -> None:
    dataset = _create_numeric_dataset(tmp_path / "dataset")
    path = write_configuration(
        tmp_path / "configuration.yaml",
        _scientific_mapping(
            dataset,
            tmp_path / "output",
            evaluations=32,
            patterns=patterns,
        ),
    )

    result = RunService().run(path)
    contents = BiclusterSetReader().read_contents(
        result.output_directory / "artifacts" / "repertoire"
    )

    assert result.repertoire.evaluations
    assert contents.repertoire == result.repertoire
    assert tuple(column.name for column in contents.columns) == ("x", "y", "z")


@pytest.mark.parametrize("checkpoint_evaluations", [2, 6])
def test_checkpoint_resume_matches_uninterrupted_serial_state(
    tmp_path: Path,
    checkpoint_evaluations: int,
) -> None:
    dataset = create_dataset_bundle(tmp_path / "dataset", rows=8, columns=3)
    partial_path = write_configuration(
        tmp_path / "partial.yaml",
        _scientific_mapping(
            dataset,
            tmp_path / "partial-output",
            evaluations=8,
            checkpoint_interval=checkpoint_evaluations,
        ),
    )
    partial = RunService().run(partial_path)
    checkpoint = (
        partial.output_directory / "checkpoints" / f"checkpoint-{checkpoint_evaluations:012d}.json"
    )

    resumed_path = write_configuration(
        tmp_path / "resumed.yaml",
        _scientific_mapping(
            dataset,
            tmp_path / "resumed-output",
            evaluations=12,
            resume_from=checkpoint,
            checkpoint_interval=checkpoint_evaluations,
        ),
    )
    uninterrupted_path = write_configuration(
        tmp_path / "uninterrupted.yaml",
        _scientific_mapping(
            dataset,
            tmp_path / "uninterrupted-output",
            evaluations=12,
            checkpoint_interval=checkpoint_evaluations,
        ),
    )
    resumed = RunService().run(resumed_path)
    uninterrupted = RunService().run(uninterrupted_path)
    assert _scientific_state(resumed) == _scientific_state(uninterrupted)


def test_serial_mome_rejects_an_unconsumed_crossover_operator(tmp_path: Path) -> None:
    dataset = create_dataset_bundle(tmp_path / "dataset", rows=8, columns=3)
    mapping = _scientific_mapping(dataset, tmp_path / "output", evaluations=4)
    search = mapping["search"]
    assert isinstance(search, dict)
    search["crossover"] = {
        "name": "membership_recombination",
        "parameters": {},
    }
    loaded = load_configuration(write_configuration(tmp_path / "configuration.yaml", mapping))

    with pytest.raises(ComponentError, match="no active component consumes capability"):
        build_specification(loaded.configuration)


def test_mutation_emitter_requires_a_mutation_operator(tmp_path: Path) -> None:
    dataset = create_dataset_bundle(tmp_path / "dataset", rows=8, columns=3)
    mapping = _scientific_mapping(dataset, tmp_path / "output", evaluations=4)
    search = mapping["search"]
    assert isinstance(search, dict)
    search["emitters"] = [{"name": "mutation", "parameters": {}}]
    loaded = load_configuration(write_configuration(tmp_path / "configuration.yaml", mapping))

    with pytest.raises(ComponentError, match="requires unavailable capabilities"):
        build_specification(loaded.configuration)


def test_checkpoint_rejects_a_changed_scientific_configuration(tmp_path: Path) -> None:
    dataset = create_dataset_bundle(tmp_path / "dataset", rows=8, columns=3)
    partial_path = write_configuration(
        tmp_path / "partial.yaml",
        _scientific_mapping(dataset, tmp_path / "partial-output", evaluations=4),
    )
    partial = RunService().run(partial_path)
    checkpoint = partial.output_directory / "checkpoints" / "checkpoint-000000000004.json"
    changed = _scientific_mapping(
        dataset,
        tmp_path / "changed-output",
        evaluations=8,
        resume_from=checkpoint,
    )
    changed["search"]["archive"]["parameters"]["cell_capacity"] = 2
    changed_path = write_configuration(tmp_path / "changed.yaml", changed)
    with pytest.raises(ComponentError, match="checkpoint does not match"):
        RunService().run(changed_path)


def test_serial_mome_enforces_ask_tell_lifecycle(tmp_path: Path) -> None:
    dataset = create_dataset_bundle(tmp_path / "dataset", rows=8, columns=3)
    path = write_configuration(
        tmp_path / "configuration.yaml",
        _scientific_mapping(dataset, tmp_path / "output", evaluations=4),
    )
    specification = build_specification(load_configuration(path).configuration)
    prepared = prepare_run(specification)
    engine = SerialMomeSearchEngine(initial_population_size=2, configured_batch_size=2)
    assert not engine.finished()
    with pytest.raises(ComponentError, match="not initialized"):
        engine.progress()

    engine.initialize(specification, prepared.context)
    with pytest.raises(ValueError, match="positive"):
        engine.ask(0)
    with pytest.raises(ComponentError, match="ask must"):
        engine.tell(())
    checkpoint = engine.checkpoint()
    candidates = engine.ask(2)
    assert len(candidates) == 2
    with pytest.raises(ComponentError, match="tell must"):
        engine.ask(1)
    with pytest.raises(ComponentError, match="awaiting"):
        engine.result()
    pending_checkpoint = engine.checkpoint()
    assert pending_checkpoint.pending_candidates == candidates
    with pytest.raises(ComponentError, match="preceding asked batch"):
        engine.tell(())
    forged = (
        Evaluation(
            candidate=candidates[0].model_copy(update={"generation": 99}),
            objectives=(),
            descriptors=(),
        ),
        Evaluation(candidate=candidates[1], objectives=(), descriptors=()),
    )
    with pytest.raises(ComponentError, match="preceding asked batch"):
        engine.tell(forged)
    with pytest.raises(ComponentError, match="awaiting evaluation"):
        engine.restore(checkpoint)


def test_serial_mome_bootstrap_visits_distinct_target_cells_before_retrying(
    tmp_path: Path,
) -> None:
    dataset = create_dataset_bundle(tmp_path / "dataset", rows=8, columns=3)
    mapping = _scientific_mapping(dataset, tmp_path / "output", evaluations=12)
    search = mapping["search"]
    assert isinstance(search, dict)
    search["archive"] = {
        "name": "deep_grid_mome",
        "parameters": {
            "axes": [
                {
                    "descriptor": "row_cardinality",
                    "binning": "EXACT",
                },
                {
                    "descriptor": "column_cardinality",
                    "binning": "EXACT",
                },
            ],
            "cell_capacity": 4,
        },
    }
    search["initialization"] = {
        "name": "cell_coverage_pattern_aware",
        "parameters": {
            "seeds_per_cell": 2,
            "max_attempts_per_cell": 4,
            "joint_column_candidate_pool_size": 8,
        },
    }
    path = write_configuration(tmp_path / "configuration.yaml", mapping)
    specification = build_specification(load_configuration(path).configuration)
    prepared = prepare_run(specification)
    engine = SerialMomeSearchEngine(initial_population_size=2, configured_batch_size=3)

    engine.initialize(specification, prepared.context)
    candidates = tuple(engine.ask(3))
    targets = tuple(
        candidate.provenance.target_archive_coordinate
        for candidate in candidates
        if candidate.provenance is not None
    )
    checkpoint = engine.checkpoint()

    assert len(set(targets)) == 3
    assert all(target is not None for target in targets)
    assert sum(state.attempts for state in checkpoint.bootstrap_cells) == 3
    assert checkpoint.schema_version == 4


def test_pending_emitter_batch_restores_without_rescheduling(tmp_path: Path) -> None:
    dataset = create_dataset_bundle(tmp_path / "dataset", rows=8, columns=3)
    path = write_configuration(
        tmp_path / "configuration.yaml",
        _scientific_mapping(dataset, tmp_path / "output", evaluations=4),
    )
    loaded = load_configuration(path).configuration
    specification = build_specification(loaded)
    prepared = prepare_run(specification)
    engine = SerialMomeSearchEngine(initial_population_size=2, configured_batch_size=2)
    engine.initialize(specification, prepared.context)

    initial = tuple(engine.ask(2))
    initial_batch = specification.executor.evaluate(
        initial,
        specification.objectives,
        specification.descriptors,
        EvaluationWorkspace(prepared.context),
    )
    engine.tell(initial_batch.evaluations)
    pending = tuple(engine.ask(2))
    checkpoint = engine.checkpoint()
    assert checkpoint.pending_emitter_names == ("random_move", "random_move")

    restored_specification = build_specification(loaded)
    restored_prepared = prepare_run(restored_specification)
    restored = SerialMomeSearchEngine(initial_population_size=2, configured_batch_size=2)
    restored.initialize(restored_specification, restored_prepared.context)
    restored.restore(checkpoint)

    assert tuple(restored.ask(2)) == pending
    replayed_batch = restored_specification.executor.evaluate(
        pending,
        restored_specification.objectives,
        restored_specification.descriptors,
        EvaluationWorkspace(restored_prepared.context),
    )
    update = restored.tell(replayed_batch.evaluations)
    assert update.emitter_feedback[0].emitter_name == "random_move"
    assert update.emitter_feedback[0].evaluated == 2
