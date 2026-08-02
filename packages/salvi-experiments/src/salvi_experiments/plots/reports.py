"""Publication-oriented plots for experiment reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

plt.switch_backend("Agg")


COLORS = ("#16697A", "#DB6400", "#5C946E", "#8B5E83", "#3D5A80", "#A23E48")


def _number(value: object) -> float:
    if not isinstance(value, int | float):
        raise TypeError(f"expected numeric plot value, received {type(value).__name__}")
    return float(value)


def _save(figure: Figure, root: Path, stem: str) -> None:
    figure.savefig(root / f"{stem}.png", dpi=220, bbox_inches="tight")
    figure.savefig(root / f"{stem}.svg", bbox_inches="tight")
    plt.close(figure)


def plot_objective_alignment(
    root: Path,
    *,
    objective_names: Sequence[str],
    candidate_records: Sequence[Mapping[str, object]],
    summary_records: Sequence[Mapping[str, object]],
) -> None:
    candidate_types = ("GROUND_TRUTH", "RANDOM_MATCHED", "REMOVED", "ADDED")
    figure, axes = plt.subplots(
        len(objective_names),
        1,
        figsize=(8.2, max(3.2, 2.8 * len(objective_names))),
        squeeze=False,
    )
    for index, objective_name in enumerate(objective_names):
        values = [
            [
                _number(record[objective_name])
                for record in candidate_records
                if record["candidate_type"] == candidate_type and bool(record["valid"])
            ]
            for candidate_type in candidate_types
        ]
        axis = axes[index, 0]
        present = [
            (label, data) for label, data in zip(candidate_types, values, strict=True) if data
        ]
        if present:
            box_labels, samples = zip(*present, strict=True)
            boxes = axis.boxplot(samples, tick_labels=box_labels, patch_artist=True)
            for patch, color in zip(boxes["boxes"], COLORS, strict=False):
                patch.set_facecolor(color)
                patch.set_alpha(0.72)
        axis.set_title(objective_name.replace("_", " ").title())
        axis.set_ylabel("Objective value")
        axis.grid(axis="y", alpha=0.22)
    figure.suptitle("Ground truth and matched controls", fontsize=14)
    figure.tight_layout()
    _save(figure, root, "objective-distributions")

    controls = ("random_matched", "removed", "added")
    rows: list[list[float]] = []
    labels: list[str] = []
    for objective_name in objective_names:
        labels.append(objective_name.replace("_", " ").title())
        rows.append(
            [
                float(
                    np.mean(
                        [
                            _number(record[f"{objective_name}_{control}_favorable_fraction"])
                            for record in summary_records
                            if record[f"{objective_name}_{control}_favorable_fraction"] is not None
                        ]
                    )
                )
                if any(
                    record[f"{objective_name}_{control}_favorable_fraction"] is not None
                    for record in summary_records
                )
                else float("nan")
                for control in controls
            ]
        )
    figure, axis = plt.subplots(figsize=(7.5, max(2.8, 0.8 * len(rows) + 1.8)))
    image = axis.imshow(np.asarray(rows), vmin=0.0, vmax=1.0, cmap="YlGnBu")
    axis.set_xticks(range(len(controls)), [label.replace("_", " ").title() for label in controls])
    axis.set_yticks(range(len(labels)), labels)
    for row_index, row in enumerate(rows):
        for column_index, value in enumerate(row):
            if np.isfinite(value):
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value > 0.62 else "#1D2935",
                )
    axis.set_title("Fraction of controls no better than ground truth")
    figure.colorbar(image, ax=axis, label="Favorable fraction")
    figure.tight_layout()
    _save(figure, root, "objective-alignment")


def plot_accuracy(
    root: Path,
    *,
    relevance: float,
    recovery: float,
    biclustering_error: float,
    coverage: Sequence[tuple[float, float]],
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
    names = ("REL", "REC", "BE")
    values = (relevance, recovery, biclustering_error)
    axes[0].bar(names, values, color=COLORS[:3])
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Biclustering accuracy")
    axes[0].grid(axis="y", alpha=0.22)
    for index, value in enumerate(values):
        axes[0].text(index, min(value + 0.035, 0.97), f"{value:.3f}", ha="center")

    thresholds = tuple(item[0] for item in coverage)
    coverage_values = tuple(item[1] for item in coverage)
    axes[1].plot(thresholds, coverage_values, marker="o", color=COLORS[0], linewidth=2)
    axes[1].fill_between(thresholds, coverage_values, alpha=0.15, color=COLORS[0])
    axes[1].set_xlim(min(thresholds), max(thresholds))
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xlabel("Structural similarity threshold")
    axes[1].set_ylabel("Ground-truth coverage")
    axes[1].set_title("Target coverage")
    axes[1].grid(alpha=0.22)
    figure.tight_layout()
    _save(figure, root, "accuracy-summary")


def plot_benchmark_metrics(
    root: Path,
    *,
    records: Sequence[Mapping[str, object]],
    stem: str,
) -> None:
    metrics = ("relevance", "recovery", "biclustering_error")
    labels = ("REL", "REC", "BE")
    values = [np.asarray([_number(record[metric]) for record in records]) for metric in metrics]
    figure, axis = plt.subplots(figsize=(7.4, 4.2))
    boxes = axis.boxplot(values, tick_labels=labels, patch_artist=True)
    for patch, color in zip(boxes["boxes"], COLORS, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Score")
    axis.set_title("Accuracy across datasets")
    axis.grid(axis="y", alpha=0.22)
    figure.tight_layout()
    _save(figure, root, stem)


def plot_algorithm_comparison(
    root: Path,
    *,
    summaries: Sequence[Mapping[str, object]],
) -> None:
    algorithms = tuple(str(record["algorithm"]) for record in summaries)
    metrics = ("relevance_mean", "recovery_mean", "biclustering_error_mean")
    labels = ("REL", "REC", "BE")
    positions = np.arange(len(algorithms))
    width = 0.24
    figure, axis = plt.subplots(figsize=(max(8.0, len(algorithms) * 1.5), 4.5))
    for metric_index, (metric, label) in enumerate(zip(metrics, labels, strict=True)):
        values = tuple(_number(record[metric]) for record in summaries)
        axis.bar(
            positions + (metric_index - 1) * width,
            values,
            width,
            label=label,
            color=COLORS[metric_index],
        )
    axis.set_xticks(positions, algorithms)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Mean score")
    axis.set_title("Algorithm comparison")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.22)
    figure.tight_layout()
    _save(figure, root, "algorithm-comparison")


def plot_alignment_benchmark(
    root: Path,
    *,
    records: Sequence[Mapping[str, object]],
) -> None:
    objectives = tuple(sorted({str(record["objective"]) for record in records}))
    controls = ("RANDOM_MATCHED", "REMOVED", "ADDED")
    positions = np.arange(len(objectives))
    width = 0.24
    figure, axis = plt.subplots(figsize=(max(8.0, len(objectives) * 1.7), 4.5))
    for control_index, control in enumerate(controls):
        values: list[float] = []
        for objective in objectives:
            samples = [
                _number(record["favorable_fraction"])
                for record in records
                if record["objective"] == objective and record["control_type"] == control
            ]
            values.append(float(np.mean(samples)) if samples else float("nan"))
        axis.bar(
            positions + (control_index - 1) * width,
            values,
            width,
            label=control.replace("_", " ").title(),
            color=COLORS[control_index],
        )
    axis.set_xticks(
        positions,
        [objective.replace("_", " ").title() for objective in objectives],
    )
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Favorable fraction")
    axis.set_title("Objective alignment across datasets")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.22)
    figure.tight_layout()
    _save(figure, root, "objective-alignment-benchmark")


def plot_salvi_ablation(
    root: Path,
    *,
    repertoire_records: Sequence[Mapping[str, object]],
    run_records: Sequence[Mapping[str, object]],
) -> None:
    """Plot accuracy, runtime, and retained structural diversity by pipeline."""

    groups = tuple(
        sorted(
            {(str(record["pipeline_id"]), str(record["artifact"])) for record in repertoire_records}
        )
    )
    labels = tuple(f"{pipeline}\n{artifact.lower()}" for pipeline, artifact in groups)
    metrics = (
        ("relevance", "REL"),
        ("recovery", "REC"),
        ("biclustering_error", "BE"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(max(11.0, len(groups) * 1.5), 4.2))
    for axis, (metric, label) in zip(axes, metrics, strict=True):
        samples = [
            [
                _number(record[metric])
                for record in repertoire_records
                if (
                    str(record["pipeline_id"]),
                    str(record["artifact"]),
                )
                == group
            ]
            for group in groups
        ]
        boxes = axis.boxplot(samples, tick_labels=labels, patch_artist=True)
        for index, patch in enumerate(boxes["boxes"]):
            patch.set_facecolor(COLORS[index % len(COLORS)])
            patch.set_alpha(0.72)
        axis.set_ylim(0.0, 1.0)
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.22)
        axis.tick_params(axis="x", rotation=30)
    figure.suptitle("SALVI ablation accuracy")
    figure.tight_layout()
    _save(figure, root, "ablation-accuracy")

    pipeline_ids = tuple(sorted({str(record["pipeline_id"]) for record in run_records}))
    runtime_samples = [
        [
            _number(record["wall_time_seconds"])
            for record in run_records
            if str(record["pipeline_id"]) == pipeline_id
        ]
        for pipeline_id in pipeline_ids
    ]
    figure, axis = plt.subplots(figsize=(max(7.2, len(pipeline_ids) * 1.4), 4.2))
    boxes = axis.boxplot(runtime_samples, tick_labels=pipeline_ids, patch_artist=True)
    for index, patch in enumerate(boxes["boxes"]):
        patch.set_facecolor(COLORS[index % len(COLORS)])
        patch.set_alpha(0.72)
    axis.set_ylabel("Wall time (seconds)")
    axis.set_title("Search cost by pipeline")
    axis.grid(axis="y", alpha=0.22)
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    _save(figure, root, "ablation-runtime")

    diversity_samples = [
        [
            _number(record["repertoire_nearest_distance_mean"])
            for record in repertoire_records
            if (
                str(record["pipeline_id"]),
                str(record["artifact"]),
            )
            == group
            and record.get("repertoire_nearest_distance_mean") is not None
        ]
        for group in groups
    ]
    figure, axis = plt.subplots(figsize=(max(7.2, len(groups) * 1.4), 4.2))
    boxes = axis.boxplot(diversity_samples, tick_labels=labels, patch_artist=True)
    for index, patch in enumerate(boxes["boxes"]):
        patch.set_facecolor(COLORS[index % len(COLORS)])
        patch.set_alpha(0.72)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Nearest structural distance")
    axis.set_title("Retained repertoire diversity")
    axis.grid(axis="y", alpha=0.22)
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    _save(figure, root, "ablation-diversity")


__all__ = [
    "plot_accuracy",
    "plot_algorithm_comparison",
    "plot_alignment_benchmark",
    "plot_benchmark_metrics",
    "plot_objective_alignment",
    "plot_salvi_ablation",
]
