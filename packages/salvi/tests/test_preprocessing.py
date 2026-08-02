from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from salvi.application.configuration import load_configuration
from salvi.application.context import NamedRandomStreams, RunContext
from salvi.application.factory import build_specification, prepare_run
from salvi.components.preprocessing import (
    DropAllMissingColumns,
    MedianModeImputation,
    MissingnessIndicators,
    PreserveMissingValues,
    RejectMissingValues,
    RobustNumericScaling,
)
from salvi.domain import (
    ColumnKind,
    ColumnMetadata,
    Dataset,
    NumericColumnStatistics,
    PreparedDataset,
)
from salvi.exceptions import ArtifactError, ComponentError
from salvi.infrastructure.dataset_bundle import DatasetBundleReader, DatasetBundleWriter

from .conftest import configuration_mapping, write_configuration


def _prepared_dataset(tmp_path: Path, *, row_identifiers: bool = True) -> PreparedDataset:
    table = pa.table(
        {
            "varying": pa.array([0.0, None, 5.0, 10.0], type=pa.float64()),
            "constant": pa.array([3, 3, 3, None], type=pa.int64()),
            "all_missing": pa.array([None, None, None, None], type=pa.float64()),
            "flag": pa.array([True, None, False, True], type=pa.bool_()),
            "group": pa.array(["case", None, "control", "case"], type=pa.string()),
            "empty_group": pa.array([None, None, None, None], type=pa.string()),
        }
    )
    columns = (
        ColumnMetadata(index=0, name="varying", kind=ColumnKind.NUMERIC),
        ColumnMetadata(index=1, name="constant", kind=ColumnKind.NUMERIC),
        ColumnMetadata(index=2, name="all_missing", kind=ColumnKind.NUMERIC),
        ColumnMetadata(index=3, name="flag", kind=ColumnKind.BOOLEAN),
        ColumnMetadata(
            index=4,
            name="group",
            kind=ColumnKind.CATEGORICAL,
            categories=("case", "control"),
        ),
        ColumnMetadata(
            index=5,
            name="empty_group",
            kind=ColumnKind.CATEGORICAL,
            categories=("known-label",),
        ),
    )
    identifiers = ("p-3", "p-8", "p-13", "p-21") if row_identifiers else None
    bundle = tmp_path / "prepared-dataset"
    DatasetBundleWriter().write(
        bundle,
        identifier="prepared-dataset",
        table=table,
        columns=columns,
        row_identifiers=identifiers,
    )
    loaded = DatasetBundleReader().load(bundle)
    return PreparedDataset.from_arrow(loaded.dataset, loaded.table, loaded.row_identifiers)


def test_prepared_dataset_preserves_semantics_masks_and_original_identifiers(
    tmp_path: Path,
) -> None:
    dataset = _prepared_dataset(tmp_path)
    assert dataset.numeric_column_indices == (0, 1, 2)
    assert dataset.boolean_column_indices == (3,)
    assert dataset.categorical_column_indices == (4, 5)
    assert dataset.numeric_positions == (0, 1, 2, -1, -1, -1)
    assert dataset.discrete_column_indices == (3, 4, 5)
    assert dataset.discrete_positions == (-1, -1, -1, 0, 1, 2)
    assert dataset.discrete_column(3).tolist() == [1, -1, 0, 1]
    assert dataset.discrete_column(4).tolist() == [0, -1, 1, 0]
    assert dataset.discrete_global_frequencies(3) == (1, 2)
    assert dataset.discrete_global_frequencies(4) == (2, 1)
    assert dataset.discrete_observed_cardinality(5) == 0
    assert dataset.discrete_value(4, 1) == "control"
    assert tuple(dataset.row_identifier(index) for index in range(4)) == (
        "p-3",
        "p-8",
        "p-13",
        "p-21",
    )
    assert dataset.metadata.columns[4].categories == ("case", "control")
    assert dataset.metadata.columns[5].categories == ("known-label",)
    assert dataset.source_observed_mask(0).tolist() == [True, False, True, True]
    assert dataset.source_observed_mask(5).tolist() == [False, False, False, False]
    assert dataset.available_mask(0).tolist() == [True, False, True, True]
    assert dataset.available is dataset.source_observed
    assert dataset.support_matrix() is dataset.source_observed
    assert np.shares_memory(dataset.support_mask(0), dataset.support_matrix())
    assert np.isnan(dataset.numeric_column(0)[1])
    assert dataset.raw_table.column("group").to_pylist() == ["case", None, "control", "case"]
    assert dataset.missing_count == 12
    assert dataset.memory_bytes > dataset.raw_table.nbytes

    with pytest.raises(ValueError, match="read-only"):
        dataset.numeric_values[0, 0] = 99.0
    with pytest.raises(ValueError, match="read-only"):
        dataset.source_observed[0, 0] = False
    with pytest.raises(ValueError, match="read-only"):
        dataset.discrete_codes[0, 0] = 0
    with pytest.raises(ValueError, match="read-only"):
        dataset.support_matrix()[0, 0] = False
    with pytest.raises(ValueError, match="not Boolean or categorical"):
        dataset.discrete_column(0)
    with pytest.raises(ValueError, match="not Boolean or categorical"):
        dataset.discrete_code(0, False)
    with pytest.raises(IndexError, match="discrete code out of range"):
        dataset.discrete_value(3, 2)
    with pytest.raises(ValueError, match="is not declared"):
        dataset.discrete_code(4, "unknown")


def test_generated_row_identifiers_preserve_original_positions(tmp_path: Path) -> None:
    dataset = _prepared_dataset(tmp_path, row_identifiers=False)
    assert tuple(dataset.row_identifier(index) for index in range(4)) == ("0", "1", "2", "3")


def test_robust_scaling_handles_partial_missing_zero_scale_and_all_missing(
    tmp_path: Path,
) -> None:
    raw = _prepared_dataset(tmp_path)
    prepared = RobustNumericScaling().transform(PreserveMissingValues().apply(raw))
    varying, constant, all_missing = prepared.numeric_statistics
    assert varying.observed_count == 3
    assert varying.percentile_05 == pytest.approx(0.5)
    assert varying.median == pytest.approx(5.0)
    assert varying.percentile_95 == pytest.approx(9.5)
    assert varying.robust_range == pytest.approx(9.0)
    assert varying.zero_scale is False
    assert prepared.numeric_column(0, standardized=True)[[0, 2, 3]].tolist() == pytest.approx(
        [-5.0 / 9.0, 0.0, 5.0 / 9.0]
    )
    assert np.isnan(prepared.numeric_column(0, standardized=True)[1])

    assert constant.zero_scale is True
    assert prepared.numeric_column(1, standardized=True)[:3].tolist() == [0.0, 0.0, 0.0]
    assert all_missing.observed_count == 0
    assert all_missing.median is None
    assert np.isnan(prepared.numeric_column(2, standardized=True)).all()
    assert raw.standardized_numeric_values is None
    assert prepared.raw_table is raw.raw_table
    with pytest.raises(ComponentError, match="more than once"):
        RobustNumericScaling().transform(prepared)


def test_robust_scaling_preserves_rare_zero_scale_deviations() -> None:
    values = [0.0] * 20 + [10.0]
    table = pa.table({"rare": values})
    metadata = Dataset(
        identifier="rare-zero-scale",
        bundle_path="fixture",
        row_count=len(values),
        column_count=1,
        columns=(ColumnMetadata(index=0, name="rare", kind=ColumnKind.NUMERIC),),
    )
    prepared = RobustNumericScaling().transform(
        PreparedDataset.from_arrow(
            metadata,
            table,
            pa.array([str(index) for index in range(len(values))]),
        )
    )

    assert prepared.numeric_statistics[0].zero_scale
    assert prepared.numeric_column(0, standardized=True)[:-1].tolist() == [0.0] * 20
    assert prepared.numeric_column(0, standardized=True)[-1] == 1.0


def test_missing_value_policies_are_explicit_components(tmp_path: Path) -> None:
    dataset = _prepared_dataset(tmp_path)
    assert PreserveMissingValues().apply(dataset) is dataset
    with pytest.raises(ComponentError, match="found 12 missing"):
        RejectMissingValues().apply(dataset)


def test_complete_dataset_is_accepted(run_context: RunContext) -> None:
    dataset = run_context.dataset
    assert dataset.missing_count == 0
    assert RejectMissingValues().apply(dataset) is dataset


def test_median_mode_imputation_preserves_original_support(tmp_path: Path) -> None:
    dataset = _prepared_dataset(tmp_path)
    filtered = DropAllMissingColumns().transform(dataset)
    imputed = MedianModeImputation().apply(filtered)

    assert imputed.missing_count == 4
    assert imputed.unavailable_count == 0
    assert imputed.imputed_count == 4
    assert imputed.source_observed_mask(0).tolist() == [True, False, True, True]
    assert imputed.available_mask(0).tolist() == [True, True, True, True]
    assert imputed.available is not imputed.source_observed
    assert imputed.numeric_column(0).tolist() == [0.0, 5.0, 5.0, 10.0]
    assert imputed.column(2).to_pylist() == [True, True, False, True]
    assert imputed.column(3).to_pylist() == ["case", "case", "control", "case"]

    scaled = RobustNumericScaling().transform(imputed)
    assert scaled.numeric_statistics[0].observed_count == 3
    assert scaled.numeric_statistics[0].median == pytest.approx(5.0)
    assert scaled.numeric_column(0, standardized=True)[1] == pytest.approx(0.0)


def test_imputation_rejects_columns_without_an_observed_prototype(tmp_path: Path) -> None:
    with pytest.raises(ComponentError, match="all-missing column 'all_missing'"):
        MedianModeImputation().apply(_prepared_dataset(tmp_path))


def test_numeric_imputation_promotes_integer_columns_for_fractional_medians(
    tmp_path: Path,
) -> None:
    table = pa.table({"integer": pa.array([1, None, 4, None], type=pa.int64())})
    bundle = tmp_path / "integer-imputation"
    DatasetBundleWriter().write(
        bundle,
        identifier="integer-imputation",
        table=table,
        columns=(ColumnMetadata(index=0, name="integer", kind=ColumnKind.NUMERIC),),
    )
    loaded = DatasetBundleReader().load(bundle)
    prepared = PreparedDataset.from_arrow(loaded.dataset, loaded.table, loaded.row_identifiers)
    imputed = MedianModeImputation().apply(prepared)
    assert imputed.column(0).type == pa.float64()
    assert imputed.numeric_column(0).tolist() == [1.0, 2.5, 4.0, 2.5]


def test_missingness_indicators_use_source_masks_after_imputation(tmp_path: Path) -> None:
    dataset = DropAllMissingColumns().transform(_prepared_dataset(tmp_path))
    imputed = MedianModeImputation().apply(dataset)
    augmented = MissingnessIndicators(min_missing_ratio=0.25).transform(imputed)

    assert augmented.column_count == 8
    assert augmented.source_column_count == 6
    indicator = augmented.column_metadata(4)
    assert indicator.name == "varying__is_missing"
    assert indicator.source_column_index == 0
    assert indicator.derivation == "missingness_indicators"
    assert augmented.column(4).to_pylist() == [False, True, False, False]
    assert augmented.source_observed_mask(4).tolist() == [True, True, True, True]
    assert augmented.boolean_column_indices == (2, 4, 5, 6, 7)
    with pytest.raises(ComponentError, match="more than once"):
        MissingnessIndicators().transform(augmented)


def test_missingness_indicators_respect_the_maximum_missing_ratio(tmp_path: Path) -> None:
    dataset = DropAllMissingColumns().transform(_prepared_dataset(tmp_path))
    augmented = MissingnessIndicators(
        min_missing_ratio=0.20,
        max_missing_ratio=0.25,
    ).transform(dataset)

    derived = tuple(
        column.name for column in augmented.columns if column.derivation == "missingness_indicators"
    )
    assert derived == (
        "varying__is_missing",
        "constant__is_missing",
        "flag__is_missing",
        "group__is_missing",
    )


def test_missingness_indicator_bounds_must_be_ordered() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        MissingnessIndicators(min_missing_ratio=0.8, max_missing_ratio=0.2)


def test_missingness_indicator_threshold_and_name_collision_are_explicit(tmp_path: Path) -> None:
    dataset = DropAllMissingColumns().transform(_prepared_dataset(tmp_path))
    assert MissingnessIndicators(min_missing_ratio=0.3).transform(dataset) is dataset

    conflicting = dataset.with_appended_boolean_columns(
        (
            (
                "varying__is_missing",
                np.zeros(dataset.row_count, dtype=np.bool_),
                0,
                "external",
            ),
        )
    )
    with pytest.raises(ComponentError, match="already exists"):
        MissingnessIndicators(min_missing_ratio=0.25).transform(conflicting)


def test_drop_all_missing_columns_preserves_source_mapping(tmp_path: Path) -> None:
    filtered = DropAllMissingColumns().transform(_prepared_dataset(tmp_path))
    assert filtered.column_count == 4
    assert tuple(column.name for column in filtered.columns) == (
        "varying",
        "constant",
        "flag",
        "group",
    )
    assert tuple(column.source_column_index for column in filtered.columns) == (0, 1, 3, 4)
    assert filtered.numeric_column_indices == (0, 1)
    assert filtered.boolean_column_indices == (2,)
    assert filtered.categorical_column_indices == (3,)


def test_configured_preprocessing_families_compose_in_semantic_order(tmp_path: Path) -> None:
    raw = _prepared_dataset(tmp_path)
    mapping = configuration_mapping(raw.metadata.bundle_path, tmp_path / "output")
    mapping["preprocessing"] = {
        "source_column_filters": [{"name": "drop_all_missing_columns", "parameters": {}}],
        "missing_values": {"name": "median_mode_imputation", "parameters": {}},
        "column_augmentations": [
            {
                "name": "missingness_indicators",
                "parameters": {"min_missing_ratio": 0.25},
            }
        ],
        "numeric_transformations": [{"name": "robust_numeric_scaling", "parameters": {}}],
    }
    configuration = load_configuration(
        write_configuration(tmp_path / "pipeline.yaml", mapping)
    ).configuration
    prepared = prepare_run(build_specification(configuration))

    assert tuple(step.component_name for step in prepared.preprocessing.steps) == (
        "drop_all_missing_columns",
        "median_mode_imputation",
        "missingness_indicators",
        "robust_numeric_scaling",
    )
    assert prepared.context.dataset.column_count == 8
    assert prepared.context.dataset.unavailable_count == 0
    assert prepared.context.dataset.imputed_count == 4
    assert prepared.context.dataset.has_robust_scaling


@pytest.mark.parametrize(
    "updates, message",
    (
        ({"column_index": -1}, "non-negative"),
        ({"median": float("inf")}, "finite"),
        ({"robust_range": -1.0}, "robust_range"),
        ({"observed_count": 0}, "cannot have"),
        ({"median": None}, "complete statistics"),
    ),
)
def test_numeric_statistics_reject_inconsistent_states(
    updates: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "column_index": 0,
        "observed_count": 2,
        "median": 1.0,
        "percentile_05": 0.0,
        "percentile_95": 2.0,
        "robust_range": 2.0,
        "zero_scale": False,
    }
    values.update(updates)
    with pytest.raises(ValueError, match=message):
        NumericColumnStatistics(**values)  # type: ignore[arg-type]


def test_prepared_dataset_accessors_and_shape_guards(tmp_path: Path) -> None:
    dataset = _prepared_dataset(tmp_path)
    with pytest.raises(ValueError, match="not numeric"):
        dataset.numeric_column(3)
    with pytest.raises(ValueError, match="has not been applied"):
        dataset.numeric_column(0, standardized=True)
    with pytest.raises(IndexError, match="column index"):
        dataset.column(99)
    with pytest.raises(IndexError, match="row index"):
        dataset.row_identifier(-1)
    with pytest.raises(ValueError, match="observed mask dimensions"):
        replace(dataset, source_observed=np.ones((1, 1), dtype=np.bool_))
    with pytest.raises(ValueError, match="available mask dimensions"):
        replace(dataset, available=np.ones((1, 1), dtype=np.bool_))
    with pytest.raises(ValueError, match="numeric view dimensions"):
        replace(dataset, numeric_values=np.ones((1, 1), dtype=np.float64))
    with pytest.raises(ValueError, match="numeric position map"):
        replace(dataset, numeric_positions=())
    with pytest.raises(ValueError, match="inconsistent with metadata"):
        replace(dataset, boolean_column_indices=())


def test_dataset_bundle_rejects_non_finite_observed_numeric_values(tmp_path: Path) -> None:
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ArtifactError, match="non-finite"):
            DatasetBundleWriter().write(
                tmp_path / str(value),
                identifier="invalid",
                table=pa.table({"value": pa.array([1.0, value], type=pa.float64())}),
                columns=(ColumnMetadata(index=0, name="value", kind=ColumnKind.NUMERIC),),
            )


def test_named_random_streams_are_stable_and_independent() -> None:
    first = NamedRandomStreams(91)
    second = NamedRandomStreams(91)
    assert (
        first.generator("initializer").integers(0, 1_000, size=8).tolist()
        == second.generator("initializer").integers(0, 1_000, size=8).tolist()
    )
    assert (
        NamedRandomStreams(91).generator("initializer").integers(0, 1_000, size=8).tolist()
        != NamedRandomStreams(91).generator("emitter").integers(0, 1_000, size=8).tolist()
    )
    assert first.generator("shared") is first.generator("shared")
    with pytest.raises(ValueError, match="blank"):
        first.generator(" ")
    with pytest.raises(ValueError, match="non-negative"):
        NamedRandomStreams(-1)


def test_named_random_stream_state_round_trips_exactly() -> None:
    streams = NamedRandomStreams(91)
    generator = streams.generator("emitter")
    generator.integers(0, 1_000, size=5)
    state = streams.snapshot()
    expected = generator.integers(0, 1_000, size=8).tolist()

    restored = NamedRandomStreams(0)
    restored.restore(state)
    assert restored.generator("emitter").integers(0, 1_000, size=8).tolist() == expected


def test_prepare_run_loads_once_and_shares_one_prepared_object(
    configuration_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specification = build_specification(load_configuration(configuration_path).configuration)
    calls = 0
    original = DatasetBundleReader.load

    def counted(reader: DatasetBundleReader, path: Path):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return original(reader, path)

    monkeypatch.setattr(DatasetBundleReader, "load", counted)
    prepared = prepare_run(specification)
    assert calls == 1
    assert prepared.context.dataset.has_robust_scaling
    assert prepared.context.patterns is specification.patterns
    assert not hasattr(prepared.context, "ground_truth")
    assert not hasattr(prepared.context, "archive_cell_targets")
    assert not hasattr(prepared.context, "parent_selection_policy")
    assert not hasattr(prepared.context, "crossover_operator")
    assert tuple(step.component_name for step in prepared.preprocessing.steps) == (
        "preserve",
        "robust_numeric_scaling",
    )
    assert prepared.preprocessing.loading_seconds >= 0.0
    assert prepared.preprocessing.final_memory_bytes >= prepared.preprocessing.initial_memory_bytes


def test_ground_truth_is_outside_the_scientific_loading_boundary(
    configuration_path: Path,
) -> None:
    configuration = load_configuration(configuration_path).configuration
    ground_truth = configuration.dataset.bundle / "ground-truth.json"
    ground_truth.write_text("corrupted experimental payload", encoding="utf-8")

    prepared = prepare_run(build_specification(configuration))
    assert prepared.context.dataset.row_count == 4
    with pytest.raises(ArtifactError, match="checksum mismatch"):
        DatasetBundleReader().read_ground_truth(configuration.dataset.bundle)
