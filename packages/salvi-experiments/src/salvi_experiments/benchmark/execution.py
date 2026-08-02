"""CPU-allocation checks shared by benchmark orchestration protocols."""

from __future__ import annotations

import os

from salvi_experiments.configuration import BenchmarkExecutionConfiguration
from salvi_experiments.exceptions import ExperimentArtifactError


def validate_benchmark_parallelism(
    execution: BenchmarkExecutionConfiguration,
    internal_workers: tuple[int, ...],
) -> int:
    """Return the effective outer worker count after validating nested parallelism."""

    workers = min(execution.workers, max(len(internal_workers), 1))
    available = os.cpu_count() or 1
    active_internal = tuple(sorted(internal_workers, reverse=True)[:workers])
    required = sum(active_internal) if active_internal else workers
    if (
        workers > 1
        and max(active_internal, default=1) > 1
        and not execution.allow_nested_parallelism
    ):
        raise ExperimentArtifactError(
            "benchmark execution would create nested parallelism: "
            f"{workers} benchmark workers and SALVI execution.workers up to "
            f"{max(active_internal)}. Set execution.allow_nested_parallelism to true "
            "to run this configuration explicitly."
        )
    if required > available and not execution.allow_cpu_oversubscription:
        raise ExperimentArtifactError(
            "benchmark execution would oversubscribe available CPUs: "
            f"requires up to {required} workers but only {available} CPUs were detected. "
            "Set execution.allow_cpu_oversubscription to true to run this "
            "configuration explicitly."
        )
    return workers


__all__ = ["validate_benchmark_parallelism"]
