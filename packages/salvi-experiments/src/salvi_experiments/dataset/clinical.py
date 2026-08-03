"""Clinical characterization, association, and repertoire-stability analyses."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Self

import numpy as np
import numpy.typing as npt
from pydantic import Field, ValidationError, model_validator
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.stats import chi2, chi2_contingency, fisher_exact, mannwhitneyu

from salvi import BiclusterSetReader, DatasetBundleReader
from salvi.domain.enums import PatternKind
from salvi.infrastructure.yaml import load_strict_yaml
from salvi_experiments.artifacts import (
    atomic_experiment_directory,
    sha256_file,
    write_json,
    write_manifest,
    write_table,
)
from salvi_experiments.configuration import FrozenExperimentModel
from salvi_experiments.exceptions import ExperimentArtifactError, ExperimentConfigurationError
from salvi_experiments.interop.uci import (
    ClinicalAnnotationKind,
    ClinicalColumnRole,
    ClinicalDatasetBundleReader,
)
from salvi_experiments.progress import ProgressReporter, progress_or_null


class ClinicalTestingConfiguration(FrozenExperimentModel):
    minimum_members: Annotated[int, Field(ge=1)] = 10
    minimum_nonmembers: Annotated[int, Field(ge=1)] = 20
    minimum_events: Annotated[int, Field(ge=1)] = 10
    fdr_alpha: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.05
    fdr_scope: Literal["ALL", "ROLE", "ANNOTATION"] = "ANNOTATION"


class ClinicalValidationConfiguration(FrozenExperimentModel):
    schema_version: Literal[1] = 1
    identifier: str = Field(min_length=1)
    clinical_dataset_bundle: Path
    bicluster_set: Path
    output_directory: Path
    artifact: Literal["SEARCH", "FINAL"] = "FINAL"
    testing: ClinicalTestingConfiguration = Field(default_factory=ClinicalTestingConfiguration)
    overwrite: bool = False

    def resolved(self, base: Path) -> Self:
        def resolve(path: Path) -> Path:
            expanded = path.expanduser()
            return expanded.resolve() if expanded.is_absolute() else (base / expanded).resolve()

        return self.model_copy(
            update={
                "clinical_dataset_bundle": resolve(self.clinical_dataset_bundle),
                "bicluster_set": resolve(self.bicluster_set),
                "output_directory": resolve(self.output_directory),
            }
        )


class RepertoireReference(FrozenExperimentModel):
    identifier: str = Field(min_length=1)
    dataset_bundle: Path
    bicluster_set: Path


class StabilityConfiguration(FrozenExperimentModel):
    thresholds: tuple[Annotated[float, Field(gt=0.0, le=1.0)], ...] = (0.25, 0.5, 0.75)

    @model_validator(mode="after")
    def validate_thresholds(self) -> Self:
        if not self.thresholds or tuple(sorted(set(self.thresholds))) != self.thresholds:
            raise ValueError("stability thresholds must be sorted and unique")
        return self


@dataclass(frozen=True, slots=True)
class ReferenceStabilityAnalysis:
    repertoire_comparisons: tuple[dict[str, object], ...]
    biclusters: tuple[dict[str, object], ...]


def load_clinical_validation_configuration(path: str | Path) -> ClinicalValidationConfiguration:
    source = Path(path).expanduser().resolve()
    try:
        return ClinicalValidationConfiguration.model_validate(load_strict_yaml(source)).resolved(
            source.parent
        )
    except ValidationError as error:
        raise ExperimentConfigurationError(
            f"invalid clinical validation configuration {source}: {error}"
        ) from error


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _benjamini_hochberg(records: list[dict[str, object]]) -> None:
    indexed = [
        (index, float(p_value))
        for index, record in enumerate(records)
        if isinstance((p_value := record.get("p_value")), (int, float))
    ]
    indexed.sort(key=lambda item: item[1])
    count = len(indexed)
    adjusted = 1.0
    for reverse_rank, (index, p_value) in enumerate(reversed(indexed), start=1):
        rank = count - reverse_rank + 1
        adjusted = min(adjusted, p_value * count / rank)
        records[index]["q_value"] = adjusted


def adjust_clinical_association_fdr(
    records: Sequence[Mapping[str, object]],
    *,
    alpha: float = 0.05,
    scope: Literal["ALL", "ROLE", "ANNOTATION"] = "ANNOTATION",
) -> tuple[dict[str, object], ...]:
    adjusted = [dict(record) for record in records]
    groups: dict[str, list[dict[str, object]]] = {}
    for record in adjusted:
        record["q_value"] = None
        key = {
            "ALL": "ALL",
            "ROLE": str(record.get("role")),
            "ANNOTATION": f"{record.get('role')}::{record.get('annotation')}",
        }[scope]
        groups.setdefault(key, []).append(record)
    for key, group in groups.items():
        _benjamini_hochberg(group)
        for record in group:
            record["fdr_family"] = key
    for record in adjusted:
        q_value = record.get("q_value")
        record["fdr_significant"] = isinstance(q_value, int | float) and q_value <= alpha
    return tuple(adjusted)


def _binary_association(
    values: Sequence[bool | None],
    selected: np.ndarray,
) -> tuple[str, float, str, float, dict[str, float | None]]:
    observed = np.asarray([value is not None for value in values], dtype=np.bool_)
    event = np.asarray([bool(value) if value is not None else False for value in values])
    inside = selected & observed
    outside = ~selected & observed
    a = int(np.count_nonzero(event & inside))
    b = int(np.count_nonzero(~event & inside))
    c = int(np.count_nonzero(event & outside))
    d = int(np.count_nonzero(~event & outside))
    odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative="two-sided")
    inside_risk = _safe_ratio(a, a + b)
    outside_risk = _safe_ratio(c, c + d)
    risk_difference = (
        None if inside_risk is None or outside_risk is None else inside_risk - outside_risk
    )
    return (
        "fisher_exact",
        float(p_value),
        "odds_ratio",
        float(odds_ratio),
        {
            "risk_inside": float(inside_risk or 0.0),
            "risk_outside": float(outside_risk or 0.0),
            "risk_difference": float(risk_difference or 0.0),
        },
    )


def _numeric_association(
    values: Sequence[float | None],
    selected: np.ndarray,
) -> tuple[str, float, str, float, dict[str, float | None]]:
    inside = np.asarray(
        [
            float(value)
            for value, member in zip(values, selected, strict=True)
            if member and value is not None
        ]
    )
    outside = np.asarray(
        [
            float(value)
            for value, member in zip(values, selected, strict=True)
            if not member and value is not None
        ]
    )
    result = mannwhitneyu(inside, outside, alternative="two-sided")
    effect = 2.0 * float(result.statistic) / (inside.size * outside.size) - 1.0
    return (
        "mann_whitney_u",
        float(result.pvalue),
        "rank_biserial_correlation",
        effect,
        {
            "median_inside": float(np.median(inside)),
            "median_outside": float(np.median(outside)),
        },
    )


def _categorical_association(
    values: Sequence[str | None],
    selected: np.ndarray,
    categories: Sequence[str],
) -> tuple[str, float, str, float, dict[str, float | None]]:
    table = np.zeros((2, len(categories)), dtype=np.int64)
    category_index = {category: index for index, category in enumerate(categories)}
    for value, member in zip(values, selected, strict=True):
        if value is not None:
            table[0 if member else 1, category_index[str(value)]] += 1
    nonempty = table.sum(axis=0) > 0
    reduced = table[:, nonempty]
    statistic, p_value, _, _ = chi2_contingency(reduced, correction=False)
    total = float(reduced.sum())
    denominator = min(reduced.shape[0] - 1, reduced.shape[1] - 1)
    cramer = math.sqrt(float(statistic) / (total * denominator)) if denominator > 0 else 0.0
    return (
        "chi_square",
        float(p_value),
        "cramers_v",
        cramer,
        {"chi_square": float(statistic), "observed_categories": float(reduced.shape[1])},
    )


def _logrank(
    times: np.ndarray,
    events: np.ndarray,
    selected: np.ndarray,
) -> tuple[float, float]:
    observed_inside = 0.0
    expected_inside = 0.0
    variance = 0.0
    for time in np.unique(times[events]):
        at_risk = times >= time
        n_total = int(np.count_nonzero(at_risk))
        n_inside = int(np.count_nonzero(at_risk & selected))
        event_at_time = (times == time) & events
        d_total = int(np.count_nonzero(event_at_time))
        d_inside = int(np.count_nonzero(event_at_time & selected))
        if n_total <= 1 or d_total == 0:
            continue
        observed_inside += d_inside
        expected_inside += d_total * n_inside / n_total
        variance += (
            n_inside
            * (n_total - n_inside)
            * d_total
            * (n_total - d_total)
            / (n_total * n_total * (n_total - 1))
        )
    statistic = 0.0 if variance <= 0.0 else (observed_inside - expected_inside) ** 2 / variance
    return statistic, float(chi2.sf(statistic, 1))


def _cox_binary_hazard_ratio(
    times: np.ndarray,
    events: np.ndarray,
    selected: np.ndarray,
) -> float:
    covariate = selected.astype(np.float64)
    beta = 0.0
    for _ in range(50):
        score = 0.0
        information = 0.0
        weights = np.exp(np.clip(beta * covariate, -40.0, 40.0))
        for time in np.unique(times[events]):
            event_at_time = (times == time) & events
            event_count = int(np.count_nonzero(event_at_time))
            at_risk = times >= time
            risk_weights = weights[at_risk]
            risk_covariate = covariate[at_risk]
            denominator = float(risk_weights.sum())
            if denominator <= 0.0:
                continue
            mean = float(np.dot(risk_weights, risk_covariate) / denominator)
            second = float(np.dot(risk_weights, risk_covariate**2) / denominator)
            score += float(covariate[event_at_time].sum()) - event_count * mean
            information += event_count * max(0.0, second - mean * mean)
        if information <= 1e-12:
            return 1.0
        step = score / information
        beta += step
        if abs(step) < 1e-9:
            break
    return float(math.exp(max(-40.0, min(40.0, beta))))


def _kaplan_meier_median(times: np.ndarray, events: np.ndarray) -> float | None:
    survival = 1.0
    for time in np.unique(times[events]):
        at_risk = int(np.count_nonzero(times >= time))
        occurred = int(np.count_nonzero((times == time) & events))
        if at_risk:
            survival *= 1.0 - occurred / at_risk
        if survival <= 0.5:
            return float(time)
    return None


def _survival_association(
    times_raw: Sequence[float | None],
    events_raw: Sequence[bool | None],
    selected_raw: np.ndarray,
) -> tuple[str, float, str, float, dict[str, float | None]]:
    observed = np.asarray(
        [
            time is not None and event is not None
            for time, event in zip(times_raw, events_raw, strict=True)
        ]
    )
    times = np.asarray([float(value or 0.0) for value in times_raw])[observed]
    events = np.asarray([bool(value) for value in events_raw])[observed]
    selected = selected_raw[observed]
    statistic, p_value = _logrank(times, events, selected)
    return (
        "log_rank",
        p_value,
        "hazard_ratio",
        _cox_binary_hazard_ratio(times, events, selected),
        {
            "log_rank_statistic": statistic,
            "median_survival_inside": _kaplan_meier_median(times[selected], events[selected]),
            "median_survival_outside": _kaplan_meier_median(times[~selected], events[~selected]),
        },
    )


def _evaluable_counts(
    values: Sequence[object | None],
    selected: np.ndarray,
) -> tuple[int, int]:
    observed = np.asarray([value is not None for value in values], dtype=np.bool_)
    return (
        int(np.count_nonzero(selected & observed)),
        int(np.count_nonzero(~selected & observed)),
    )


def calculate_clinical_associations(
    clinical_dataset_bundle: Path,
    bicluster_set: Path,
    *,
    testing: ClinicalTestingConfiguration | None = None,
) -> tuple[dict[str, object], ...]:
    configuration = testing or ClinicalTestingConfiguration()
    clinical = ClinicalDatasetBundleReader().load(clinical_dataset_bundle)
    contents = BiclusterSetReader().read_contents(bicluster_set)
    if contents.manifest.dataset_identifier != clinical.manifest.identifier:
        raise ExperimentArtifactError("BiclusterSet and ClinicalDatasetBundle identities differ")
    annotation_values = {
        name: clinical.annotations.column(name).to_pylist()
        for name in clinical.annotations.column_names
        if name != "row_identifier"
    }
    associations = tuple(
        annotation
        for annotation in clinical.manifest.annotations
        if annotation.role in {ClinicalColumnRole.OUTCOME, ClinicalColumnRole.COVARIATE}
    )
    records: list[dict[str, object]] = []
    for evaluation in contents.repertoire.evaluations:
        selected = np.zeros(clinical.manifest.row_count, dtype=np.bool_)
        selected[list(evaluation.candidate.bicluster.row_indices)] = True
        for annotation in associations:
            values = annotation_values[annotation.name]
            members, nonmembers = _evaluable_counts(values, selected)
            base: dict[str, object] = {
                "bicluster_id": evaluation.candidate.identifier,
                "annotation": annotation.name,
                "role": annotation.role.value,
                "annotation_kind": annotation.kind.value,
                "members": members,
                "nonmembers": nonmembers,
                "evaluable": False,
                "reason": None,
                "test": None,
                "p_value": None,
                "q_value": None,
                "effect_type": None,
                "effect_value": None,
                "diagnostics": None,
            }
            if members < configuration.minimum_members:
                base["reason"] = "insufficient_members"
            elif nonmembers < configuration.minimum_nonmembers:
                base["reason"] = "insufficient_nonmembers"
            else:
                try:
                    if annotation.kind is ClinicalAnnotationKind.BOOLEAN:
                        result = _binary_association(values, selected)
                    elif annotation.kind is ClinicalAnnotationKind.SURVIVAL_TIME:
                        event_name = annotation.survival_event_column
                        if event_name is None or event_name not in annotation_values:
                            raise ValueError("survival annotation has no event column")
                        event_values = annotation_values[event_name]
                        event_count = sum(
                            bool(event)
                            for time, event in zip(values, event_values, strict=True)
                            if time is not None and event is not None
                        )
                        if event_count < configuration.minimum_events:
                            base["reason"] = "insufficient_events"
                            records.append(base)
                            continue
                        result = _survival_association(values, event_values, selected)
                    elif annotation.kind is ClinicalAnnotationKind.NUMERIC:
                        result = _numeric_association(values, selected)
                    elif annotation.kind is ClinicalAnnotationKind.ORDINAL:
                        rank = {
                            category: index for index, category in enumerate(annotation.categories)
                        }
                        ordinal = [
                            None if value is None else float(rank[str(value)]) for value in values
                        ]
                        result = _numeric_association(ordinal, selected)
                    else:
                        result = _categorical_association(values, selected, annotation.categories)
                except (ValueError, ZeroDivisionError) as error:
                    base["reason"] = f"statistical_test_failed:{error}"
                else:
                    test, p_value, effect_type, effect_value, diagnostics = result
                    base.update(
                        {
                            "evaluable": True,
                            "test": test,
                            "p_value": p_value,
                            "effect_type": effect_type,
                            "effect_value": effect_value,
                            "diagnostics": json.dumps(diagnostics, sort_keys=True),
                        }
                    )
            records.append(base)
    return adjust_clinical_association_fdr(
        records,
        alpha=configuration.fdr_alpha,
        scope=configuration.fdr_scope,
    )


def characterize_biclusters(
    clinical_dataset_bundle: Path,
    bicluster_set: Path,
) -> tuple[dict[str, object], ...]:
    clinical = ClinicalDatasetBundleReader().load(clinical_dataset_bundle)
    contents = BiclusterSetReader().read_contents(bicluster_set)
    if contents.manifest.dataset_identifier != clinical.manifest.identifier:
        raise ExperimentArtifactError("BiclusterSet and ClinicalDatasetBundle identities differ")
    records: list[dict[str, object]] = []
    for evaluation in contents.repertoire.evaluations:
        selected_columns = tuple(
            contents.columns[index] for index in evaluation.candidate.bicluster.column_indices
        )
        patterns = (
            ()
            if evaluation.pattern_fit is None
            else tuple(
                column.pattern.value
                for column in evaluation.pattern_fit.columns
                if column.pattern is not None
            )
        )
        records.append(
            {
                "bicluster_id": evaluation.candidate.identifier,
                "rows": len(evaluation.candidate.bicluster.row_indices),
                "columns": len(selected_columns),
                "area": len(evaluation.candidate.bicluster.row_indices) * len(selected_columns),
                "patterns": sorted(set(patterns)),
                "constant_columns": patterns.count(PatternKind.CONSTANT.value),
                "additive_columns": patterns.count(PatternKind.ADDITIVE.value),
                "multiplicative_columns": patterns.count(PatternKind.MULTIPLICATIVE.value),
                "missingness_indicator_columns": sum(
                    column.derivation == "missingness_indicators" for column in selected_columns
                ),
                "missingness_indicator_share": (
                    sum(
                        column.derivation == "missingness_indicators" for column in selected_columns
                    )
                    / len(selected_columns)
                ),
                "objectives": json.dumps(
                    {item.name: item.value for item in evaluation.objectives},
                    sort_keys=True,
                ),
                "constraints": json.dumps(
                    {item.name: item.value for item in evaluation.constraints},
                    sort_keys=True,
                ),
                "feasible": evaluation.feasible,
                "archive_coordinate": (
                    None
                    if evaluation.archive_coordinate is None
                    else list(evaluation.archive_coordinate)
                ),
            }
        )
    return tuple(records)


@dataclass(frozen=True, slots=True)
class _Structure:
    identifier: str
    rows: frozenset[str]
    columns: frozenset[str]
    patterns: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class _EncodedRepertoire:
    identifiers: tuple[str, ...]
    rows: csr_matrix
    columns: csr_matrix
    pattern_columns: tuple[csr_matrix, ...]


def _structures(reference: RepertoireReference) -> tuple[_Structure, ...]:
    dataset = DatasetBundleReader().load(reference.dataset_bundle)
    contents = BiclusterSetReader().read_structures(reference.bicluster_set)
    row_ids = tuple(str(value) for value in dataset.row_identifiers.to_pylist())
    structures = []
    for record in contents.biclusters:
        rows = frozenset(row_ids[index] for index in record.bicluster.row_indices)
        columns = frozenset(
            contents.columns[index].name for index in record.bicluster.column_indices
        )
        patterns = {
            contents.columns[index].name: (
                "UNASSIGNED" if pattern is None else pattern.value
            )
            for index, pattern in record.column_patterns
        }
        structures.append(
            _Structure(
                identifier=record.identifier,
                rows=rows,
                columns=columns,
                patterns=patterns,
            )
        )
    return tuple(structures)


def _encode_structures(
    groups: Mapping[str, Sequence[_Structure]],
) -> dict[str, _EncodedRepertoire]:
    all_structures = tuple(structure for values in groups.values() for structure in values)
    row_positions = {
        value: index
        for index, value in enumerate(
            sorted({row for structure in all_structures for row in structure.rows})
        )
    }
    column_positions = {
        value: index
        for index, value in enumerate(
            sorted({column for structure in all_structures for column in structure.columns})
        )
    }
    missing_pattern = "__MISSING_PATTERN__"
    pattern_positions = {
        value: index
        for index, value in enumerate(
            sorted(
                {
                    structure.patterns.get(column, missing_pattern)
                    for structure in all_structures
                    for column in structure.columns
                }
            )
        )
    }

    def incidence(
        structures: Sequence[_Structure],
        positions: Mapping[str, int],
        values: Sequence[frozenset[str]],
    ) -> csr_matrix:
        row_indices: list[int] = []
        column_indices: list[int] = []
        for row, selected in enumerate(values):
            row_indices.extend([row] * len(selected))
            column_indices.extend(positions[value] for value in selected)
        data = np.ones(len(row_indices), dtype=np.int32)
        return csr_matrix(
            (data, (row_indices, column_indices)),
            shape=(len(structures), len(positions)),
            dtype=np.int32,
        )

    encoded: dict[str, _EncodedRepertoire] = {}
    for identifier, structures in groups.items():
        pattern_matrices = []
        for pattern in pattern_positions:
            selected = tuple(
                frozenset(
                    column
                    for column in structure.columns
                    if structure.patterns.get(column, missing_pattern) == pattern
                )
                for structure in structures
            )
            pattern_matrices.append(incidence(structures, column_positions, selected))
        encoded[identifier] = _EncodedRepertoire(
            identifiers=tuple(structure.identifier for structure in structures),
            rows=incidence(
                structures,
                row_positions,
                tuple(structure.rows for structure in structures),
            ),
            columns=incidence(
                structures,
                column_positions,
                tuple(structure.columns for structure in structures),
            ),
            pattern_columns=tuple(pattern_matrices),
        )
    return encoded


def _intersection_counts(
    left: csr_matrix,
    right: csr_matrix,
) -> npt.NDArray[np.float64]:
    return np.asarray((left @ right.transpose()).toarray(), dtype=np.float64)


def _jaccard_scores(
    left: csr_matrix,
    right: csr_matrix,
    intersections: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    left_sizes = np.asarray(left.sum(axis=1), dtype=np.float64).reshape(-1, 1)
    right_sizes = np.asarray(right.sum(axis=1), dtype=np.float64).reshape(1, -1)
    unions = left_sizes + right_sizes - intersections
    result = np.ones_like(intersections)
    np.divide(
        intersections,
        unions,
        out=result,
        where=unions > 0,
    )
    return result


def _stability_scores(
    left: _EncodedRepertoire,
    right: _EncodedRepertoire,
) -> npt.NDArray[np.float64]:
    row_intersections = _intersection_counts(left.rows, right.rows)
    column_intersections = _intersection_counts(left.columns, right.columns)
    row_scores = _jaccard_scores(left.rows, right.rows, row_intersections)
    column_scores = _jaccard_scores(left.columns, right.columns, column_intersections)
    matching_patterns = np.zeros_like(column_intersections)
    for left_pattern, right_pattern in zip(
        left.pattern_columns,
        right.pattern_columns,
        strict=True,
    ):
        matching_patterns += _intersection_counts(left_pattern, right_pattern)
    pattern_scores = np.divide(
        matching_patterns,
        column_intersections,
        out=np.zeros_like(matching_patterns),
        where=column_intersections > 0,
    )
    scores: npt.NDArray[np.float64] = np.sqrt(row_scores * column_scores)
    scores *= pattern_scores
    return scores


def _repertoire_stability_record(
    left_identifier: str,
    right_identifier: str,
    left_count: int,
    right_count: int,
    scores: np.ndarray,
    settings: StabilityConfiguration,
) -> dict[str, object]:
    if scores.size:
        row_indices, column_indices = linear_sum_assignment(-scores)
        matched = scores[row_indices, column_indices]
    else:
        matched = np.asarray([], dtype=np.float64)
    record: dict[str, object] = {
        "left": left_identifier,
        "right": right_identifier,
        "left_count": left_count,
        "right_count": right_count,
        "matched_count": int(matched.size),
        "mean_matched_stability": float(matched.mean()) if matched.size else 0.0,
        "median_matched_stability": float(np.median(matched)) if matched.size else 0.0,
    }
    denominator = max(left_count, right_count, 1)
    for threshold in settings.thresholds:
        record[f"coverage_at_{threshold:g}".replace(".", "_")] = (
            int(np.count_nonzero(matched >= threshold)) / denominator
        )
    return record


def calculate_repertoire_stability(
    references: Sequence[RepertoireReference],
    *,
    configuration: StabilityConfiguration | None = None,
) -> tuple[dict[str, object], ...]:
    settings = configuration or StabilityConfiguration()
    if len(references) < 2:
        raise ValueError("repertoire stability requires at least two results")
    loaded = _encode_structures(
        {reference.identifier: _structures(reference) for reference in references}
    )
    records: list[dict[str, object]] = []
    for left_index, left_reference in enumerate(references):
        for right_reference in references[left_index + 1 :]:
            left = loaded[left_reference.identifier]
            right = loaded[right_reference.identifier]
            scores = _stability_scores(left, right)
            records.append(
                _repertoire_stability_record(
                    left_reference.identifier,
                    right_reference.identifier,
                    len(left.identifiers),
                    len(right.identifiers),
                    scores,
                    settings,
                )
            )
    return tuple(records)


def calculate_reference_stability(
    reference: RepertoireReference,
    comparisons: Sequence[RepertoireReference],
    *,
    configuration: StabilityConfiguration | None = None,
    bicluster_threshold: float = 0.5,
) -> ReferenceStabilityAnalysis:
    if not comparisons:
        raise ValueError("reference stability requires at least one comparison")
    if not 0.0 < bicluster_threshold <= 1.0:
        raise ValueError("reference stability threshold must be in (0, 1]")
    settings = configuration or StabilityConfiguration()
    raw = {
        reference.identifier: _structures(reference),
        **{comparison.identifier: _structures(comparison) for comparison in comparisons},
    }
    loaded = _encode_structures(raw)
    reference_structures = loaded[reference.identifier]
    repertoire_records: list[dict[str, object]] = []
    matched_rows = []
    for comparison in comparisons:
        comparison_structures = loaded[comparison.identifier]
        scores = _stability_scores(reference_structures, comparison_structures)
        repertoire_records.append(
            _repertoire_stability_record(
                reference.identifier,
                comparison.identifier,
                len(reference_structures.identifiers),
                len(comparison_structures.identifiers),
                scores,
                settings,
            )
        )
        matched = np.zeros(len(reference_structures.identifiers), dtype=np.float64)
        if scores.size:
            row_indices, column_indices = linear_sum_assignment(-scores)
            matched[row_indices] = scores[row_indices, column_indices]
        matched_rows.append(matched)
    matched_by_comparison = np.vstack(matched_rows)
    bicluster_records: list[dict[str, object]] = []
    for index, identifier in enumerate(reference_structures.identifiers):
        scores = matched_by_comparison[:, index]
        bicluster_records.append(
            {
                "reference": reference.identifier,
                "bicluster_id": identifier,
                "comparison_count": len(comparisons),
                "match_threshold": bicluster_threshold,
                "matched_count": int(np.count_nonzero(scores >= bicluster_threshold)),
                "support_fraction": float(np.mean(scores >= bicluster_threshold)),
                "mean_matched_stability": float(np.mean(scores)),
                "median_matched_stability": float(np.median(scores)),
                "minimum_matched_stability": float(np.min(scores)),
                "maximum_matched_stability": float(np.max(scores)),
            }
        )
    return ReferenceStabilityAnalysis(
        repertoire_comparisons=tuple(repertoire_records),
        biclusters=tuple(bicluster_records),
    )


def calculate_reference_bicluster_stability(
    reference: RepertoireReference,
    comparisons: Sequence[RepertoireReference],
    *,
    threshold: float = 0.5,
) -> tuple[dict[str, object], ...]:
    return calculate_reference_stability(
        reference,
        comparisons,
        bicluster_threshold=threshold,
    ).biclusters


def run_clinical_validation(
    configuration: ClinicalValidationConfiguration,
    *,
    progress: ProgressReporter | None = None,
) -> Path:
    reporter = progress_or_null(progress)
    reporter.stage("loading clinical dataset and bicluster artifacts")
    clinical = ClinicalDatasetBundleReader().load(configuration.clinical_dataset_bundle)
    reporter.stage("characterizing detected biclusters")
    characterization = characterize_biclusters(
        configuration.clinical_dataset_bundle,
        configuration.bicluster_set,
    )
    reporter.stage("calculating clinical associations")
    associations = calculate_clinical_associations(
        configuration.clinical_dataset_bundle,
        configuration.bicluster_set,
        testing=configuration.testing,
    )
    reporter.stage(f"writing clinical validation to {configuration.output_directory}")
    with atomic_experiment_directory(
        configuration.output_directory,
        overwrite=configuration.overwrite,
    ) as temporary:
        write_table(temporary, "bicluster-characterization", characterization)
        write_table(temporary, "outcome-associations", associations)
        report = {
            "schema_version": 1,
            "experiment_type": "dataset.clinical-validation",
            "identifier": configuration.identifier,
            "artifact": configuration.artifact,
            "clinical_dataset_bundle": str(configuration.clinical_dataset_bundle),
            "clinical_manifest_sha256": sha256_file(
                configuration.clinical_dataset_bundle / "clinical-dataset.yaml"
            ),
            "bicluster_set": str(configuration.bicluster_set),
            "bicluster_set_manifest_sha256": sha256_file(
                configuration.bicluster_set / "manifest.json"
            ),
            "dataset_identifier": clinical.manifest.identifier,
            "bicluster_count": len(characterization),
            "association_count": len(associations),
            "evaluable_association_count": sum(
                bool(record["evaluable"]) for record in associations
            ),
            "testing": configuration.testing.model_dump(mode="json"),
        }
        write_json(temporary / "report.json", report)
        write_manifest(
            temporary,
            experiment_type="dataset.clinical-validation",
            identifier=configuration.identifier,
            metadata={
                "dataset_identifier": clinical.manifest.identifier,
                "artifact": configuration.artifact,
            },
        )
    return configuration.output_directory.resolve()


__all__ = [
    "ClinicalTestingConfiguration",
    "ClinicalValidationConfiguration",
    "ReferenceStabilityAnalysis",
    "RepertoireReference",
    "StabilityConfiguration",
    "adjust_clinical_association_fdr",
    "calculate_clinical_associations",
    "calculate_reference_bicluster_stability",
    "calculate_reference_stability",
    "calculate_repertoire_stability",
    "characterize_biclusters",
    "load_clinical_validation_configuration",
    "run_clinical_validation",
]
