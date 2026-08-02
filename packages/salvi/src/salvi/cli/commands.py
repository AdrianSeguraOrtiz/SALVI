"""Command implementations kept independent from argument parser construction."""

from __future__ import annotations

import argparse
import json
import queue
import signal
import sys
import threading
import time
from pathlib import Path

from salvi.application.configuration import (
    RunBinding,
    load_pipeline_configuration,
    serialize_pipeline_configuration,
)
from salvi.application.inspection import inspect_pipeline
from salvi.application.profiling import profile_configuration
from salvi.application.run_service import CancellationToken, RunService
from salvi.application.selection_service import (
    FinalSelectionService,
    selection_manifest_sha256,
)
from salvi.cli.monitor import ConsoleRunMonitor
from salvi.components.defaults import default_component_registry
from salvi.components.protocols import ComponentKind
from salvi.domain.models import RunResult
from salvi.infrastructure.files import atomic_write_text
from salvi.versioning import public_version_info


def _binding(
    namespace: argparse.Namespace,
    *,
    overwrite_attribute: str = "overwrite",
) -> RunBinding:
    return RunBinding(
        identifier=namespace.identifier or namespace.configuration.stem,
        dataset_bundle=namespace.dataset,
        output_directory=namespace.output,
        seed=namespace.seed,
        resume_from_checkpoint=namespace.resume_from_checkpoint,
        overwrite=bool(getattr(namespace, overwrite_attribute)),
    )


def _validation_binding(namespace: argparse.Namespace) -> RunBinding:
    return RunBinding(
        identifier=f"validate-{namespace.configuration.stem}",
        dataset_bundle=namespace.dataset,
        output_directory=Path.cwd() / ".salvi-validation-unused",
        seed=namespace.seed,
    )


def _print_run_summary(result: RunResult) -> None:
    print(
        json.dumps(
            {
                "status": result.status.value,
                "output_directory": str(result.output_directory),
                "event_store": str(result.event_store),
                "result_count": len(result.repertoire.evaluations),
                "message": result.message,
            },
            sort_keys=True,
        )
    )


def _run_with_progress(
    service: RunService,
    namespace: argparse.Namespace,
    token: CancellationToken,
) -> RunResult:
    binding = _binding(namespace)
    if (
        namespace.quiet
        or namespace.progress == "never"
        or (namespace.progress == "auto" and not sys.stderr.isatty())
    ):
        return service.run_pipeline(namespace.configuration, binding, cancellation=token)

    event_store = binding.output_directory.resolve() / "run.sqlite"
    started_mtime_ns = time.time_ns()
    results: queue.Queue[RunResult | BaseException] = queue.Queue(maxsize=1)

    def execute() -> None:
        try:
            results.put(service.run_pipeline(namespace.configuration, binding, cancellation=token))
        except BaseException as error:
            results.put(error)

    thread = threading.Thread(target=execute, name="salvi-run", daemon=False)
    thread.start()
    ConsoleRunMonitor(
        event_store,
        interval_seconds=namespace.monitor_interval,
        minimum_mtime_ns=started_mtime_ns,
    ).monitor_until_finished(lambda: not thread.is_alive())
    thread.join()
    result = results.get()
    if isinstance(result, BaseException):
        raise result
    return result


def _install_cancellation_handlers(token: CancellationToken) -> None:
    def cancel(_signal: int, _frame: object) -> None:
        token.cancel()

    signal.signal(signal.SIGINT, cancel)
    signal.signal(signal.SIGTERM, cancel)


def dispatch(namespace: argparse.Namespace) -> int:
    service = RunService()
    if namespace.command == "validate":
        validated_run = service.validate_pipeline(
            namespace.configuration,
            _validation_binding(namespace),
        )
        print(
            json.dumps(
                {
                    "valid": True,
                    "configuration": str(validated_run.source),
                    "dataset": str(validated_run.binding.dataset_bundle),
                },
                sort_keys=True,
            )
        )
        return 0
    if namespace.command == "inspect":
        report = inspect_pipeline(
            namespace.configuration,
            dataset_bundle=namespace.dataset,
            seed=namespace.seed,
        )
        print(report.model_dump_json(indent=2))
        return 0
    if namespace.command == "run":
        token = CancellationToken()
        _install_cancellation_handlers(token)
        _print_run_summary(_run_with_progress(service, namespace, token))
        return 0
    if namespace.command == "select":
        result = FinalSelectionService().select(
            namespace.configuration,
            dataset_bundle=namespace.dataset,
            repertoire=namespace.repertoire,
            output=namespace.output,
            identifier=namespace.identifier,
            overwrite=namespace.overwrite,
        )
        print(
            json.dumps(
                {
                    "selector": result.selector,
                    "input_count": result.input_count,
                    "output_count": result.output_count,
                    "output": str(result.output_directory),
                    "manifest_sha256": selection_manifest_sha256(result),
                },
                sort_keys=True,
            )
        )
        return 0
    if namespace.command == "components":
        _print_components(namespace)
        return 0
    if namespace.command == "config" and namespace.config_command == "format":
        loaded_pipeline = load_pipeline_configuration(namespace.configuration)
        content = serialize_pipeline_configuration(
            loaded_pipeline.pipeline,
            compact=not namespace.expanded,
        )
        if namespace.output is None:
            print(content, end="")
        else:
            atomic_write_text(namespace.output.expanduser().resolve(), content)
        return 0
    if namespace.command == "schemas":
        print(json.dumps(public_version_info(), indent=2, sort_keys=True))
        return 0
    if namespace.command == "profile":
        destination = profile_configuration(
            namespace.configuration,
            namespace.destination,
            binding=_binding(namespace, overwrite_attribute="run_overwrite"),
            repetitions=namespace.repetitions,
            overwrite=namespace.overwrite,
            instrument=not namespace.lightweight,
        )
        print(json.dumps({"profile": str(destination)}, sort_keys=True))
        return 0
    if namespace.command == "gui":
        from salvi.web.main import launch_web_gui

        return launch_web_gui(
            host=namespace.host,
            port=namespace.port,
            open_browser=not namespace.no_open,
            data_directory=namespace.data_directory,
            max_upload_mib=namespace.max_upload_mib,
        )
    raise ValueError(f"unsupported command: {namespace.command}")


def _print_components(namespace: argparse.Namespace) -> None:
    kind = None if namespace.kind is None else ComponentKind(namespace.kind)
    descriptions = default_component_registry().catalog(kind)
    if namespace.format == "json":
        print(json.dumps([item.model_dump(mode="json") for item in descriptions], indent=2))
        return
    if namespace.format == "markdown":
        for item in descriptions:
            print(f"### `{item.kind.value}:{item.name}`")
            print()
            print(item.description)
            print()
            print(f"Maturity: `{item.maturity.value}`")
            if item.parameters:
                print()
                print("| Parameter | Default | Description |")
                print("| --- | --- | --- |")
                for parameter in item.parameters:
                    default = "required" if parameter.required else repr(parameter.default)
                    print(f"| `{parameter.name}` | `{default}` | {parameter.description} |")
            print()
        return
    current_kind: ComponentKind | None = None
    for item in descriptions:
        if item.kind is not current_kind:
            current_kind = item.kind
            print(f"{item.kind.value}:")
        print(f"  {item.name:<36} [{item.maturity.value}] {item.description}")


__all__ = ["dispatch"]
