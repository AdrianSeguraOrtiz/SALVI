"""Argument parser for the SALVI command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from salvi.components.protocols import ComponentKind
from salvi.versioning import package_version


def _bounded_integer(value: str, *, minimum: int, maximum: int | None = None) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if parsed < minimum or (maximum is not None and parsed > maximum):
        bounds = f"between {minimum} and {maximum}" if maximum is not None else f">= {minimum}"
        raise argparse.ArgumentTypeError(f"value must be {bounds}")
    return parsed


def _tcp_port(value: str) -> int:
    return _bounded_integer(value, minimum=1, maximum=65535)


def _positive_integer(value: str) -> int:
    return _bounded_integer(value, minimum=1)


def _non_negative_integer(value: str) -> int:
    return _bounded_integer(value, minimum=0)


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be a number") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _add_dataset_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="canonical DatasetBundle used to resolve this pipeline",
    )
    parser.add_argument(
        "--seed",
        type=_non_negative_integer,
        default=0,
        help="random seed used for dataset-dependent component setup",
    )


def _add_run_binding_arguments(
    parser: argparse.ArgumentParser,
    *,
    overwrite_option: str = "--overwrite",
    overwrite_destination: str = "overwrite",
) -> None:
    _add_dataset_argument(parser)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="directory created for this run's artifacts and event store",
    )
    parser.add_argument(
        "--identifier",
        help="run identifier recorded in metadata; defaults to the pipeline file stem",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        help="checkpoint to resume; it must match the pipeline and dataset",
    )
    parser.add_argument(
        overwrite_option,
        dest=overwrite_destination,
        action="store_true",
        help="replace an existing output directory",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="salvi", description="SALVI biclustering framework")
    parser.add_argument("--version", action="version", version=f"%(prog)s {package_version()}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate",
        help="validate a reusable pipeline against one DatasetBundle",
    )
    validate.add_argument("configuration", type=Path, help="pipeline YAML configuration")
    _add_dataset_argument(validate)

    inspect = subparsers.add_parser(
        "inspect",
        help="inspect preprocessing, components, descriptor domains, and archive shape",
    )
    inspect.add_argument("configuration", type=Path, help="pipeline YAML configuration")
    _add_dataset_argument(inspect)

    run = subparsers.add_parser("run", help="execute a pipeline on one DatasetBundle")
    run.add_argument("configuration", type=Path, help="pipeline YAML configuration")
    _add_run_binding_arguments(run)
    run.add_argument(
        "--progress",
        choices=("auto", "always", "never"),
        default="auto",
        help="report live progress from run.sqlite on stderr",
    )
    run.add_argument(
        "--monitor-interval",
        type=_positive_float,
        default=0.5,
        help="seconds between console progress refreshes",
    )
    run.add_argument(
        "--quiet",
        action="store_true",
        help="suppress progress output while retaining the final JSON summary",
    )

    select = subparsers.add_parser(
        "select",
        help="apply a pipeline's final selector to an existing search repertoire",
    )
    select.add_argument("configuration", type=Path, help="pipeline YAML configuration")
    select.add_argument("--dataset", required=True, type=Path)
    select.add_argument("--repertoire", required=True, type=Path)
    select.add_argument("--output", required=True, type=Path)
    select.add_argument("--identifier")
    select.add_argument("--overwrite", action="store_true")

    components = subparsers.add_parser(
        "components",
        help="list component contracts from the installed registry",
    )
    components.add_argument("--kind", choices=tuple(kind.value for kind in ComponentKind))
    components.add_argument("--format", choices=("text", "json", "markdown"), default="text")

    config = subparsers.add_parser("config", help="configuration document utilities")
    config_subparsers = config.add_subparsers(dest="config_command", required=True)
    format_parser = config_subparsers.add_parser(
        "format",
        help="validate and normalize one reusable pipeline YAML",
    )
    format_parser.add_argument("configuration", type=Path)
    format_parser.add_argument("--output", type=Path)
    format_parser.add_argument(
        "--expanded",
        action="store_true",
        help="write every default and empty optional collection",
    )

    subparsers.add_parser("schemas", help="print package and public schema versions")

    profile = subparsers.add_parser(
        "profile",
        help="profile one pipeline bound to a concrete DatasetBundle",
    )
    profile.add_argument("configuration", type=Path)
    profile.add_argument("destination", type=Path)
    _add_run_binding_arguments(
        profile,
        overwrite_option="--run-overwrite",
        overwrite_destination="run_overwrite",
    )
    profile.add_argument("--repetitions", type=_positive_integer, default=1)
    profile.add_argument("--overwrite", action="store_true")
    profile.add_argument("--lightweight", action="store_true")

    gui = subparsers.add_parser("gui", help="launch the optional local web interface")
    gui.add_argument("--host", default="127.0.0.1")
    gui.add_argument("--port", type=_tcp_port, default=8765)
    gui.add_argument("--no-open", action="store_true")
    gui.add_argument("--data-directory", type=Path)
    gui.add_argument("--max-upload-mib", type=_positive_integer, default=2048)
    return parser


__all__ = ["build_parser"]
