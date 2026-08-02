"""Reproducible end-to-end profiling for complete SALVI configurations."""

from __future__ import annotations

import cProfile
import gc
import json
import os
import platform
import pstats
import statistics
import sys
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from salvi.application.configuration import (
    RunBinding,
    load_bound_configuration,
    load_configuration,
)
from salvi.application.run_service import RunService
from salvi.components.defaults import default_component_registry
from salvi.components.protocols import ComponentKind, EvaluationExecutor
from salvi.exceptions import RunError
from salvi.infrastructure.files import atomic_directory, sha256_file
from salvi.versioning import package_version


@dataclass(frozen=True, slots=True)
class _ResourceSnapshot:
    process_cpu_seconds: float
    child_cpu_seconds: float | None
    peak_rss_bytes: int | None
    child_peak_rss_bytes: int | None
    io_read_bytes: int | None
    io_write_bytes: int | None


def _rusage() -> tuple[float | None, int | None, float | None, int | None]:
    try:
        import resource
    except ImportError:  # pragma: no cover - unavailable on Windows
        return None, None, None, None

    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    scale = 1024 if sys.platform.startswith("linux") else 1
    return (
        own.ru_utime + own.ru_stime,
        int(own.ru_maxrss * scale),
        children.ru_utime + children.ru_stime,
        int(children.ru_maxrss * scale),
    )


def _proc_io() -> tuple[int | None, int | None]:
    path = Path("/proc/self/io")
    if not path.is_file():
        return None, None
    try:
        values = {
            key.rstrip(":"): int(value)
            for key, value in (line.split(maxsplit=1) for line in path.read_text().splitlines())
        }
    except (OSError, ValueError):
        return None, None
    return values.get("read_bytes"), values.get("write_bytes")


def _snapshot() -> _ResourceSnapshot:
    resource_cpu, peak_rss, child_cpu, child_peak_rss = _rusage()
    read_bytes, write_bytes = _proc_io()
    return _ResourceSnapshot(
        process_cpu_seconds=resource_cpu if resource_cpu is not None else time.process_time(),
        child_cpu_seconds=child_cpu,
        peak_rss_bytes=peak_rss,
        child_peak_rss_bytes=child_peak_rss,
        io_read_bytes=read_bytes,
        io_write_bytes=write_bytes,
    )


def _difference(after: int | float | None, before: int | float | None) -> int | float | None:
    if after is None or before is None:
        return None
    return max(0, after - before)


def _directory_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _top_functions(profile: cProfile.Profile, *, limit: int = 30) -> list[dict[str, object]]:
    stats = pstats.Stats(profile)
    raw_stats = cast(
        dict[tuple[str, int, str], tuple[int, int, float, float, dict[object, object]]],
        vars(stats)["stats"],
    )
    ranked = sorted(
        raw_stats.items(),
        key=lambda item: (-item[1][3], item[0][0], item[0][1], item[0][2]),
    )
    return [
        {
            "function": f"{Path(filename).name}:{line}:{name}",
            "primitive_calls": primitive_calls,
            "total_calls": total_calls,
            "self_seconds": self_seconds,
            "cumulative_seconds": cumulative_seconds,
        }
        for (filename, line, name), (
            primitive_calls,
            total_calls,
            self_seconds,
            cumulative_seconds,
            _callers,
        ) in ranked[:limit]
    ]


def _median(records: list[dict[str, Any]], key: str) -> float | int | None:
    values = [record[key] for record in records if record.get(key) is not None]
    return None if not values else statistics.median(values)


def profile_configuration(
    configuration_path: Path,
    destination: Path,
    *,
    binding: RunBinding | None = None,
    repetitions: int = 1,
    overwrite: bool = False,
    instrument: bool = True,
) -> Path:
    """Profile a bound pipeline or an already persisted effective configuration.

    The command-line interface always supplies ``binding``.  Supporting an
    effective configuration here preserves the artifact-oriented programmatic API.
    """

    if repetitions < 1:
        raise ValueError("profiling repetitions must be positive")
    effective_loaded = load_configuration(configuration_path) if binding is None else None
    bound_loaded = (
        None if binding is None else load_bound_configuration(configuration_path, binding)
    )
    loaded = bound_loaded or effective_loaded
    assert loaded is not None
    configuration = loaded.configuration
    executor = cast(
        EvaluationExecutor,
        default_component_registry().create(
            ComponentKind.EVALUATION_EXECUTOR,
            configuration.execution.executor.name,
            configuration.execution.executor.parameters,
        ),
    )
    executor.validate_worker_count(configuration.execution.workers)
    uses_child_workers = executor.uses_child_processes
    run_output = configuration.output.directory.resolve()
    destination = destination.resolve()
    if destination == run_output or destination.is_relative_to(run_output):
        raise RunError("profile output must be outside the configured run output directory")
    if repetitions > 1 and not configuration.output.overwrite:
        raise RunError("multiple profiling repetitions require run-output overwrite")

    records: list[dict[str, Any]] = []
    with atomic_directory(destination, overwrite=overwrite) as temporary:
        for repetition in range(1, repetitions + 1):
            gc.collect()
            before = _snapshot()
            profiler = cProfile.Profile() if instrument else None
            if instrument:
                tracemalloc.start()
            wall_started = time.perf_counter()
            if profiler is not None:
                profiler.enable()
            peak_python_bytes: int | None
            try:
                if bound_loaded is None:
                    assert effective_loaded is not None
                    result = RunService().run(effective_loaded.source)
                else:
                    result = RunService().run_pipeline(bound_loaded.source, bound_loaded.binding)
            finally:
                if profiler is not None:
                    profiler.disable()
                wall_seconds = time.perf_counter() - wall_started
                if instrument:
                    _current_python_bytes, peak_python_bytes = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                else:
                    peak_python_bytes = None
            after = _snapshot()
            profile_name = None if profiler is None else f"profile-{repetition:03d}.pstats"
            if profiler is not None and profile_name is not None:
                profiler.dump_stats(temporary / profile_name)
            metadata_path = result.output_directory / "run-metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            evaluations = int(metadata.get("search", {}).get("evaluations", 0))
            records.append(
                {
                    "repetition": repetition,
                    "status": result.status.value,
                    "evaluations": evaluations,
                    "result_count": len(result.repertoire.evaluations),
                    "wall_seconds": wall_seconds,
                    "process_cpu_seconds": _difference(
                        after.process_cpu_seconds,
                        before.process_cpu_seconds,
                    ),
                    "child_cpu_seconds": (
                        _difference(
                            after.child_cpu_seconds,
                            before.child_cpu_seconds,
                        )
                        if uses_child_workers
                        else None
                    ),
                    "peak_rss_bytes": after.peak_rss_bytes,
                    "child_peak_rss_bytes": (
                        after.child_peak_rss_bytes if uses_child_workers else None
                    ),
                    "peak_traced_python_bytes": peak_python_bytes,
                    "io_read_bytes": _difference(after.io_read_bytes, before.io_read_bytes),
                    "io_write_bytes": _difference(after.io_write_bytes, before.io_write_bytes),
                    "input_bytes": _directory_size(configuration.dataset.bundle),
                    "output_bytes": _directory_size(result.output_directory),
                    "evaluations_per_second": (
                        evaluations / wall_seconds if wall_seconds > 0 else None
                    ),
                    "preprocessing": metadata.get("preprocessing"),
                    "profile_file": profile_name,
                    "top_cumulative_functions": (
                        [] if profiler is None else _top_functions(profiler)
                    ),
                }
            )

        summary_keys = (
            "wall_seconds",
            "process_cpu_seconds",
            "child_cpu_seconds",
            "peak_rss_bytes",
            "child_peak_rss_bytes",
            "peak_traced_python_bytes",
            "io_read_bytes",
            "io_write_bytes",
            "input_bytes",
            "output_bytes",
            "evaluations_per_second",
        )
        report = {
            "schema_version": 1,
            "salvi_version": package_version(),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "configuration": str(loaded.source),
            "configuration_sha256": sha256_file(loaded.source),
            "instrumentation": {
                "cpu_profile": instrument,
                "python_allocation_tracing": instrument,
            },
            "environment": {
                "python": platform.python_version(),
                "implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "logical_cpu_count": os.cpu_count(),
            },
            "scientific_scope": {
                "patterns": [pattern.value for pattern in configuration.patterns.allowed],
                "objectives": [objective.name for objective in configuration.search.objectives],
                "constraints": [constraint.name for constraint in configuration.search.constraints],
                "executor": configuration.execution.executor.name,
                "workers": configuration.execution.workers,
            },
            "repetitions": records,
            "median": {key: _median(records, key) for key in summary_keys},
            "notes": {
                "peak_rss": "process-lifetime high-water mark when the platform exposes it",
                "traced_memory": "Python allocations only; NumPy and Arrow buffers may be external",
                "cpu_profile": (
                    "profiles the coordinator; use the serial executor for kernel detail"
                ),
                "lightweight": (
                    "disables CPU call-stack and Python allocation instrumentation "
                    "for fair executor timing"
                ),
            },
        }
        (temporary / "profile-report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return destination


__all__ = ["profile_configuration"]
