from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from pydantic import ValidationError

from salvi.domain import (
    Bicluster,
    Candidate,
    CandidateProvenance,
    ColumnKind,
    ColumnMetadata,
    ColumnObjectiveValue,
    ColumnPatternFit,
    ConstraintValue,
    Evaluation,
    NamedValue,
    ObjectiveDirection,
    ObjectiveValue,
    ParameterScale,
    PatternCandidateFit,
    PatternFit,
    PatternGroupFit,
    PatternKind,
    PreparedColumnMetadata,
    Repertoire,
)
from salvi.exceptions import ArtifactError
from salvi.infrastructure.bicluster_set import (
    BiclusterSetReader,
    BiclusterSetWriter,
    ColumnPatternRecord,
    PagedBiclusterSetReader,
    PatternRowParameterRecord,
)
from salvi.infrastructure.dataset_bundle import DatasetBundleReader, DatasetBundleWriter
from salvi.infrastructure.files import sha256_file
from salvi.infrastructure.ground_truth import (
    GroundTruth,
    GroundTruthBicluster,
    GroundTruthColumnPattern,
)


def _prepared_columns(count: int) -> tuple[PreparedColumnMetadata, ...]:
    return tuple(
        PreparedColumnMetadata(
            index=index,
            name=f"column-{index}",
            kind=ColumnKind.NUMERIC,
            categories=(),
            source_column_index=index,
        )
        for index in range(count)
    )


def test_dataset_bundle_round_trip_and_checksum_validation(tmp_path: Path) -> None:
    destination = tmp_path / "dataset"
    table = pa.table({"value": pa.array([1.0, None, 3.0], type=pa.float64())})
    columns = (ColumnMetadata(index=0, name="value", kind=ColumnKind.NUMERIC),)
    dataset = DatasetBundleWriter().write(
        destination,
        identifier="dataset",
        table=table,
        columns=columns,
        row_identifiers=("patient-a", "patient-b", "patient-c"),
        ground_truth=GroundTruth(
            dataset_identifier="dataset",
            row_count=3,
            column_count=1,
            biclusters=(
                GroundTruthBicluster(
                    identifier="truth-0",
                    row_indices=(0,),
                    column_indices=(0,),
                    column_patterns=(
                        GroundTruthColumnPattern(
                            column_index=0,
                            pattern=PatternKind.CONSTANT,
                        ),
                    ),
                ),
            ),
        ),
    )
    assert dataset.row_count == 3
    assert DatasetBundleReader().read_table(destination).equals(table)
    loaded = DatasetBundleReader().load(destination)
    assert loaded.row_identifiers.to_pylist() == ["patient-a", "patient-b", "patient-c"]
    assert DatasetBundleReader().read_ground_truth(destination) is not None
    assert json.loads((destination / "ground-truth.json").read_text())["biclusters"]

    with pytest.raises(ArtifactError, match="already exists"):
        DatasetBundleWriter().write(
            destination,
            identifier="dataset",
            table=table,
            columns=columns,
        )

    with (destination / "data.parquet").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ArtifactError, match="checksum mismatch"):
        DatasetBundleReader().read(destination)


def test_dataset_bundle_rejects_mismatched_columns(tmp_path: Path) -> None:
    table = pa.table({"actual": [1, 2]})
    metadata = (ColumnMetadata(index=0, name="expected", kind=ColumnKind.NUMERIC),)
    with pytest.raises(ArtifactError, match="exactly match"):
        DatasetBundleWriter().write(
            tmp_path / "dataset",
            identifier="dataset",
            table=table,
            columns=metadata,
        )

    with pytest.raises(ArtifactError, match="at least one row"):
        DatasetBundleWriter().write(
            tmp_path / "empty",
            identifier="dataset",
            table=pa.table({"actual": pa.array([], type=pa.float64())}),
            columns=(ColumnMetadata(index=0, name="actual", kind=ColumnKind.NUMERIC),),
        )


def test_dataset_bundle_overwrite_and_missing_artifact(tmp_path: Path) -> None:
    destination = tmp_path / "dataset"
    table = pa.table({"value": [1.0, 2.0]})
    columns = (ColumnMetadata(index=0, name="value", kind=ColumnKind.NUMERIC),)
    writer = DatasetBundleWriter()
    writer.write(destination, identifier="first", table=table, columns=columns)
    replaced = writer.write(
        destination,
        identifier="second",
        table=table,
        columns=columns,
        overwrite=True,
    )
    assert replaced.identifier == "second"
    (destination / "data.parquet").unlink()
    with pytest.raises(ArtifactError, match="missing"):
        DatasetBundleReader().read(destination)


def test_dataset_bundle_validates_semantic_types_and_categories(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="Arrow type"):
        DatasetBundleWriter().write(
            tmp_path / "wrong-type",
            identifier="dataset",
            table=pa.table({"value": ["not-numeric"]}),
            columns=(ColumnMetadata(index=0, name="value", kind=ColumnKind.NUMERIC),),
        )

    with pytest.raises(ArtifactError, match="undeclared categories"):
        DatasetBundleWriter().write(
            tmp_path / "category",
            identifier="dataset",
            table=pa.table({"group": ["A", "C"]}),
            columns=(
                ColumnMetadata(
                    index=0,
                    name="group",
                    kind=ColumnKind.CATEGORICAL,
                    categories=("A", "B"),
                ),
            ),
        )


@pytest.mark.parametrize(
    "identifiers, message",
    (
        (("only-one",), "count"),
        (("same", "same"), "unique"),
        (("valid", " "), "blank"),
        (("valid", 2), "strings"),
    ),
)
def test_dataset_bundle_validates_row_identifiers(
    tmp_path: Path,
    identifiers: tuple[object, ...],
    message: str,
) -> None:
    with pytest.raises(ArtifactError, match=message):
        DatasetBundleWriter().write(
            tmp_path / message,
            identifier="dataset",
            table=pa.table({"value": [1.0, 2.0]}),
            columns=(ColumnMetadata(index=0, name="value", kind=ColumnKind.NUMERIC),),
            row_identifiers=identifiers,  # type: ignore[arg-type]
        )


def test_dataset_bundle_rejects_mismatched_ground_truth(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="ground truth"):
        DatasetBundleWriter().write(
            tmp_path / "dataset",
            identifier="dataset",
            table=pa.table({"value": [1.0, 2.0]}),
            columns=(ColumnMetadata(index=0, name="value", kind=ColumnKind.NUMERIC),),
            ground_truth=GroundTruth(
                dataset_identifier="different",
                row_count=2,
                column_count=1,
                biclusters=(),
            ),
        )


@pytest.mark.parametrize(
    "table, message",
    (
        (pa.table({"wrong": ["a", "b"]}), "only 'row_identifier'"),
        (pa.table({"row_identifier": ["a"]}), "count"),
        (pa.table({"row_identifier": ["a", None]}), "non-null"),
        (pa.table({"row_identifier": ["a", "a"]}), "unique"),
        (pa.table({"row_identifier": ["a", " "]}), "blank"),
    ),
)
def test_dataset_bundle_reader_validates_persisted_row_identifiers(
    tmp_path: Path,
    table: pa.Table,
    message: str,
) -> None:
    destination = tmp_path / message.replace(" ", "-").replace("'", "")
    DatasetBundleWriter().write(
        destination,
        identifier="dataset",
        table=pa.table({"value": [1.0, 2.0]}),
        columns=(ColumnMetadata(index=0, name="value", kind=ColumnKind.NUMERIC),),
        row_identifiers=("a", "b"),
    )
    row_path = destination / "row-identifiers.parquet"
    pq.write_table(table, row_path)
    manifest_path = destination / "dataset.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["checksums"]["row-identifiers.parquet"] = sha256_file(row_path)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    with pytest.raises(ArtifactError, match=message):
        DatasetBundleReader().load(destination)


def test_ground_truth_models_reject_invalid_coordinates() -> None:
    pattern = GroundTruthColumnPattern(column_index=0, pattern=PatternKind.CONSTANT)
    with pytest.raises(ValidationError, match="must not be empty"):
        GroundTruthBicluster(
            identifier="empty",
            row_indices=(),
            column_indices=(0,),
            column_patterns=(pattern,),
        )
    with pytest.raises(ValidationError, match="sorted"):
        GroundTruthBicluster(
            identifier="unsorted",
            row_indices=(1, 0),
            column_indices=(0,),
            column_patterns=(pattern,),
        )
    with pytest.raises(ValidationError, match="cover"):
        GroundTruthBicluster(
            identifier="patterns",
            row_indices=(0,),
            column_indices=(0,),
            column_patterns=(),
        )

    valid = GroundTruthBicluster(
        identifier="valid",
        row_indices=(0,),
        column_indices=(0,),
        column_patterns=(pattern,),
    )
    with pytest.raises(ValidationError, match="identifiers"):
        GroundTruth(
            dataset_identifier="dataset",
            row_count=1,
            column_count=1,
            biclusters=(valid, valid),
        )
    with pytest.raises(ValidationError, match="row index"):
        GroundTruth(
            dataset_identifier="dataset",
            row_count=1,
            column_count=1,
            biclusters=(valid.model_copy(update={"row_indices": (1,)}),),
        )
    with pytest.raises(ValidationError, match="column index"):
        GroundTruth(
            dataset_identifier="dataset",
            row_count=1,
            column_count=1,
            biclusters=(
                valid.model_copy(
                    update={
                        "column_indices": (1,),
                        "column_patterns": (
                            GroundTruthColumnPattern(
                                column_index=1,
                                pattern=PatternKind.CONSTANT,
                            ),
                        ),
                    }
                ),
            ),
        )


def test_bicluster_set_round_trip_with_pattern_details(tmp_path: Path) -> None:
    candidate = Candidate(
        identifier="bicluster-1",
        generation=3,
        bicluster=Bicluster(row_indices=(0, 2), column_indices=(0, 1, 2)),
        provenance=CandidateProvenance(
            producer="swap_row",
            operation="swap_row",
            sequence=17,
            parent_identifiers=("parent-1",),
            pattern_hint=PatternKind.ADDITIVE,
        ),
    )
    constant = PatternCandidateFit(
        pattern=PatternKind.CONSTANT,
        error=0.0,
        parameter="case",
        parameter_scale=ParameterScale.CATEGORY_LABEL,
        source_support=2,
        available_support=2,
        prototype_support=2,
    )
    additive_one = PatternCandidateFit(
        pattern=PatternKind.ADDITIVE,
        error=0.1,
        parameter=0.25,
        parameter_scale=ParameterScale.ROBUST_STANDARDIZED,
        source_support=2,
        available_support=2,
    )
    additive_two = additive_one.model_copy(update={"parameter": 0.5})
    pattern_fit = PatternFit(
        candidate_signature=candidate.bicluster.signature,
        row_indices=candidate.bicluster.row_indices,
        column_indices=candidate.bicluster.column_indices,
        columns=(
            ColumnPatternFit(
                column_index=0,
                pattern=PatternKind.CONSTANT,
                error=0.0,
                parameter="case",
                parameter_scale=ParameterScale.CATEGORY_LABEL,
                source_support=2,
                available_support=2,
                prototype_support=2,
                alternatives=(constant,),
            ),
            ColumnPatternFit(
                column_index=1,
                pattern=PatternKind.ADDITIVE,
                group_identifier="ADDITIVE-0",
                error=0.1,
                parameter=0.25,
                parameter_scale=ParameterScale.ROBUST_STANDARDIZED,
                source_support=2,
                available_support=2,
                alternatives=(additive_one,),
            ),
            ColumnPatternFit(
                column_index=2,
                pattern=PatternKind.ADDITIVE,
                group_identifier="ADDITIVE-0",
                error=0.1,
                parameter=0.5,
                parameter_scale=ParameterScale.ROBUST_STANDARDIZED,
                source_support=2,
                available_support=2,
                alternatives=(additive_two,),
            ),
        ),
        groups=(
            PatternGroupFit(
                identifier="ADDITIVE-0",
                pattern=PatternKind.ADDITIVE,
                column_indices=(1, 2),
                row_parameters=((0, -0.5), (2, 0.5)),
                iterations=2,
                converged=True,
            ),
        ),
    )
    explained_columns = tuple(
        ColumnObjectiveValue(column_index=column, value=value)
        for column, value in ((0, 0.0), (1, 0.1), (2, 0.1))
    )
    repertoire = Repertoire(
        evaluations=(
            Evaluation(
                candidate=candidate,
                objectives=(
                    ObjectiveValue(
                        name="coherence",
                        value=0.1,
                        direction=ObjectiveDirection.MINIMIZE,
                        columns=explained_columns,
                    ),
                    ObjectiveValue(
                        name="contrast",
                        value=0.9,
                        direction=ObjectiveDirection.MAXIMIZE,
                        columns=tuple(
                            ColumnObjectiveValue(column_index=column, value=0.9)
                            for column in candidate.bicluster.column_indices
                        ),
                    ),
                ),
                descriptors=(NamedValue(name="rows", value=2),),
                constraints=(ConstraintValue(name="size_limit", value=-0.1),),
                pattern_fit=pattern_fit,
            ),
        )
    )
    destination = tmp_path / "results"
    manifest = BiclusterSetWriter().write(
        destination,
        identifier="results",
        dataset_identifier="dataset",
        row_count=4,
        source_column_count=3,
        columns=_prepared_columns(3),
        repertoire=repertoire,
        source_run="run-1",
    )
    assert manifest.column_patterns_file == "column-patterns.parquet"
    assert manifest.column_objectives_file == "column-objectives.parquet"
    contents = BiclusterSetReader().read_contents(destination)
    assert contents.columns == _prepared_columns(3)
    assert contents.repertoire.evaluations[0].candidate == candidate
    assert contents.repertoire.evaluations[0].objectives[0].value == pytest.approx(0.1)
    assert contents.repertoire.evaluations[0].constraints[0].name == "size_limit"
    assert contents.repertoire.evaluations[0].feasible
    assert tuple(item.name for item in contents.repertoire.evaluations[0].objectives) == (
        "coherence",
        "contrast",
    )
    assert contents.repertoire.evaluations[0].objectives[1].direction is (
        ObjectiveDirection.MAXIMIZE
    )
    assert contents.column_patterns[0].parameter == "case"
    assert contents.column_patterns[1].group_identifier == "ADDITIVE-0"
    assert contents.column_objectives[0].objective_name == "coherence"
    assert contents.pattern_groups[0].column_indices == (1, 2)
    assert contents.row_parameters[0].value == pytest.approx(-0.5)
    assert contents.repertoire.evaluations[0].pattern_fit == pattern_fit
    paged = PagedBiclusterSetReader(destination)
    assert paged.row_count == 1
    assert paged.columns == _prepared_columns(3)
    assert paged.read_page(0, limit=1) == contents.repertoire.evaluations
    assert paged.read_page(1) == ()
    with pytest.raises(ValueError):
        paged.read_page(-1)


def test_bicluster_set_round_trips_multiplicative_explanations(
    tmp_path: Path,
) -> None:
    candidate = Candidate(
        identifier="multiplicative",
        bicluster=Bicluster(row_indices=(0, 1), column_indices=(0, 1)),
    )
    alternatives = tuple(
        PatternCandidateFit(
            pattern=PatternKind.MULTIPLICATIVE,
            error=0.0,
            parameter=parameter,
            parameter_scale=ParameterScale.ROBUST_SCALED,
            source_support=2,
            available_support=2,
        )
        for parameter in (0.5, 1.5)
    )
    pattern_fit = PatternFit(
        candidate_signature=candidate.bicluster.signature,
        row_indices=candidate.bicluster.row_indices,
        column_indices=candidate.bicluster.column_indices,
        columns=tuple(
            ColumnPatternFit(
                column_index=index,
                pattern=PatternKind.MULTIPLICATIVE,
                group_identifier="MULTIPLICATIVE-0",
                error=0.0,
                parameter=alternative.parameter,
                parameter_scale=ParameterScale.ROBUST_SCALED,
                source_support=2,
                available_support=2,
                alternatives=(alternative,),
            )
            for index, alternative in enumerate(alternatives)
        ),
        groups=(
            PatternGroupFit(
                identifier="MULTIPLICATIVE-0",
                pattern=PatternKind.MULTIPLICATIVE,
                column_indices=(0, 1),
                row_parameters=((0, 0.5), (1, 1.5)),
                iterations=2,
                converged=True,
            ),
        ),
    )
    repertoire = Repertoire(
        evaluations=(
            Evaluation(
                candidate=candidate,
                objectives=(
                    ObjectiveValue(
                        name="coherence",
                        value=0.0,
                        direction=ObjectiveDirection.MINIMIZE,
                        columns=tuple(
                            ColumnObjectiveValue(column_index=index, value=0.0) for index in (0, 1)
                        ),
                    ),
                ),
                descriptors=(),
                pattern_fit=pattern_fit,
            ),
        )
    )
    destination = tmp_path / "multiplicative-results"

    BiclusterSetWriter().write(
        destination,
        identifier="multiplicative-results",
        dataset_identifier="dataset",
        row_count=2,
        source_column_count=2,
        columns=_prepared_columns(2),
        repertoire=repertoire,
    )
    restored = BiclusterSetReader().read_contents(destination)

    assert restored.repertoire == repertoire
    assert restored.column_patterns[0].parameter_scale is ParameterScale.ROBUST_SCALED
    assert restored.pattern_groups[0].pattern is PatternKind.MULTIPLICATIVE


def test_bicluster_set_supports_empty_repertoire(tmp_path: Path) -> None:
    destination = tmp_path / "results"
    BiclusterSetWriter().write(
        destination,
        identifier="empty",
        dataset_identifier="dataset",
        row_count=2,
        source_column_count=2,
        columns=_prepared_columns(2),
        repertoire=Repertoire(),
    )
    assert BiclusterSetReader().read(destination) == Repertoire()


def test_bicluster_set_preserves_prepared_column_provenance(tmp_path: Path) -> None:
    columns = (
        PreparedColumnMetadata(
            index=0,
            name="retained-source",
            kind=ColumnKind.NUMERIC,
            categories=(),
            source_column_index=2,
        ),
        PreparedColumnMetadata(
            index=1,
            name="missing__is_missing",
            kind=ColumnKind.BOOLEAN,
            categories=(),
            source_column_index=0,
            derivation="missingness_indicator",
        ),
    )
    destination = tmp_path / "prepared-results"
    manifest = BiclusterSetWriter().write(
        destination,
        identifier="prepared-results",
        dataset_identifier="dataset",
        row_count=3,
        source_column_count=4,
        columns=columns,
        repertoire=Repertoire(),
    )

    contents = BiclusterSetReader().read_contents(destination)

    assert manifest.schema_version == 7
    assert manifest.source_column_count == 4
    assert contents.columns == columns


def test_bicluster_set_rejects_invalid_provenance_and_indices(tmp_path: Path) -> None:
    candidate = Candidate(
        identifier="bicluster",
        bicluster=Bicluster(row_indices=(0, 3), column_indices=(0,)),
    )
    evaluation = Evaluation(candidate=candidate, objectives=(), descriptors=())
    with pytest.raises(ArtifactError, match="row index"):
        BiclusterSetWriter().write(
            tmp_path / "range",
            identifier="results",
            dataset_identifier="dataset",
            row_count=3,
            source_column_count=1,
            columns=_prepared_columns(1),
            repertoire=Repertoire(evaluations=(evaluation,)),
        )

    valid = Candidate(
        identifier="known",
        bicluster=Bicluster(row_indices=(0,), column_indices=(0,)),
    )
    duplicate = Evaluation(candidate=valid, objectives=(), descriptors=())
    with pytest.raises(ArtifactError, match="identifiers must be unique"):
        BiclusterSetWriter().write(
            tmp_path / "duplicate",
            identifier="results",
            dataset_identifier="dataset",
            row_count=1,
            source_column_count=1,
            columns=_prepared_columns(1),
            repertoire=Repertoire(evaluations=(duplicate, duplicate)),
        )

    with pytest.raises(ArtifactError, match="dimensions must be positive"):
        BiclusterSetWriter().write(
            tmp_path / "dimensions",
            identifier="results",
            dataset_identifier="dataset",
            row_count=0,
            source_column_count=1,
            columns=_prepared_columns(1),
            repertoire=Repertoire(),
        )


def test_pattern_details_require_finite_parameters() -> None:
    with pytest.raises(ValidationError, match="finite"):
        ColumnPatternRecord(
            bicluster_id="bicluster",
            column_index=0,
            pattern=PatternKind.ADDITIVE,
            error=0,
            parameter=float("inf"),
            source_support=2,
            available_support=2,
        )
    with pytest.raises(ValidationError, match="finite"):
        PatternRowParameterRecord(
            bicluster_id="bicluster",
            group_identifier="ADDITIVE-0",
            row_index=0,
            value=float("nan"),
        )
