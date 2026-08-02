"""Command-line entry point for SALVI scientific experiments."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from salvi import SalviError
from salvi_experiments import __version__
from salvi_experiments.benchmark import (
    run_accuracy_benchmark,
    run_comparison,
    run_objective_alignment_benchmark,
    run_salvi_ablation,
)
from salvi_experiments.configuration import (
    AccuracyBenchmarkConfiguration,
    AccuracyConfiguration,
    ComparisonConfiguration,
    ObjectiveAlignmentBenchmarkConfiguration,
    ObjectiveAlignmentConfiguration,
    SalviAblationConfiguration,
    load_experiment_configuration,
)
from salvi_experiments.dataset import (
    load_clinical_validation_configuration,
    run_accuracy,
    run_clinical_validation,
    run_objective_alignment,
)
from salvi_experiments.exceptions import ExperimentError
from salvi_experiments.interop import (
    CsvBiclusterSetExporter,
    GbicConverter,
    HbicConverter,
    UciConverter,
    load_uci_import_recipe,
)
from salvi_experiments.progress import ConsoleProgressReporter
from salvi_experiments.versioning import public_version_info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="salvi-exp", description="SALVI experiments")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress progress messages and print only the final output path",
    )
    levels = parser.add_subparsers(dest="level", required=True)
    levels.add_parser("schemas", help="print experiment package and schema versions")
    convert = levels.add_parser("convert", help="convert external artifacts")
    converters = convert.add_subparsers(dest="converter", required=True)
    gbic = converters.add_parser("gbic", help="convert G-Bic data to DatasetBundles")
    gbic.add_argument("source", type=Path)
    gbic.add_argument("destination", type=Path)
    gbic.add_argument("--overwrite", action="store_true")
    hbic = converters.add_parser("hbic", help="convert HBIC results to a BiclusterSet")
    hbic.add_argument("source", type=Path)
    hbic.add_argument("destination", type=Path)
    hbic.add_argument("--dataset-bundle", required=True, type=Path)
    hbic.add_argument("--overwrite", action="store_true")
    uci = converters.add_parser("uci", help="convert an official UCI dataset using a recipe")
    uci.add_argument("recipe", type=Path)
    uci.add_argument("destination", type=Path)
    export = levels.add_parser("export", help="export canonical artifacts")
    exporters = export.add_subparsers(dest="exporter", required=True)
    csv_export = exporters.add_parser("csv", help="export a BiclusterSet as CSV tables")
    csv_export.add_argument("source", type=Path)
    csv_export.add_argument("destination", type=Path)
    csv_export.add_argument("--overwrite", action="store_true")
    dataset = levels.add_parser("dataset", help="dataset-level studies")
    dataset_experiments = dataset.add_subparsers(dest="experiment", required=True)
    for name in ("objective-alignment", "accuracy", "clinical-validation"):
        command = dataset_experiments.add_parser(name)
        command.add_argument("configuration", help="self-contained experiment YAML")
    benchmark = levels.add_parser("benchmark", help="benchmark-level studies")
    benchmark_experiments = benchmark.add_subparsers(dest="experiment", required=True)
    for name in (
        "objective-alignment",
        "accuracy",
        "compare",
        "ablation",
    ):
        command = benchmark_experiments.add_parser(name)
        command.add_argument("configuration", help="self-contained experiment YAML")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    namespace = build_parser().parse_args(arguments)
    if namespace.level == "schemas":
        print(json.dumps(public_version_info(), indent=2, sort_keys=True))
        return 0
    if namespace.level == "convert" and namespace.converter == "gbic":
        try:
            destination = GbicConverter(overwrite=namespace.overwrite).convert(
                namespace.source,
                namespace.destination,
            )
        except (ExperimentError, SalviError, ValueError) as error:
            print(f"salvi-exp: {error}", file=sys.stderr)
            return 2
        print(json.dumps({"converter": "gbic", "destination": str(destination)}))
        return 0
    if namespace.level == "convert" and namespace.converter == "hbic":
        try:
            destination = HbicConverter(
                dataset_bundle=namespace.dataset_bundle,
                overwrite=namespace.overwrite,
            ).convert(namespace.source, namespace.destination)
        except (ExperimentError, SalviError, ValueError) as error:
            print(f"salvi-exp: {error}", file=sys.stderr)
            return 2
        print(json.dumps({"converter": "hbic", "destination": str(destination)}))
        return 0
    if namespace.level == "convert" and namespace.converter == "uci":
        try:
            destination = UciConverter().convert(
                load_uci_import_recipe(namespace.recipe),
                namespace.destination,
            )
        except (ExperimentError, SalviError, ValueError) as error:
            print(f"salvi-exp: {error}", file=sys.stderr)
            return 2
        print(json.dumps({"converter": "uci", "destination": str(destination)}))
        return 0
    if namespace.level == "export" and namespace.exporter == "csv":
        try:
            destination = CsvBiclusterSetExporter(overwrite=namespace.overwrite).export(
                namespace.source,
                namespace.destination,
            )
        except (ExperimentError, SalviError, ValueError) as error:
            print(f"salvi-exp: {error}", file=sys.stderr)
            return 2
        print(json.dumps({"exporter": "csv", "destination": str(destination)}))
        return 0
    command = (namespace.level, namespace.experiment)
    progress = None if namespace.quiet else ConsoleProgressReporter()
    if progress is not None:
        progress.begin(" ".join(command))
    try:
        if command == ("dataset", "objective-alignment"):
            output = run_objective_alignment(
                load_experiment_configuration(
                    namespace.configuration,
                    ObjectiveAlignmentConfiguration,
                ),
                progress=progress,
            )
        elif command == ("dataset", "accuracy"):
            output = run_accuracy(
                load_experiment_configuration(
                    namespace.configuration,
                    AccuracyConfiguration,
                ),
                progress=progress,
            )
        elif command == ("dataset", "clinical-validation"):
            output = run_clinical_validation(
                load_clinical_validation_configuration(namespace.configuration),
                progress=progress,
            )
        elif command == ("benchmark", "objective-alignment"):
            output = run_objective_alignment_benchmark(
                load_experiment_configuration(
                    namespace.configuration,
                    ObjectiveAlignmentBenchmarkConfiguration,
                ),
                progress=progress,
            )
        elif command == ("benchmark", "accuracy"):
            output = run_accuracy_benchmark(
                load_experiment_configuration(
                    namespace.configuration,
                    AccuracyBenchmarkConfiguration,
                ),
                progress=progress,
            )
        elif command == ("benchmark", "compare"):
            output = run_comparison(
                load_experiment_configuration(
                    namespace.configuration,
                    ComparisonConfiguration,
                ),
                progress=progress,
            )
        else:
            output = run_salvi_ablation(
                load_experiment_configuration(
                    namespace.configuration,
                    SalviAblationConfiguration,
                ),
                progress=progress,
            )
    except (ExperimentError, SalviError, ValueError) as error:
        print(f"salvi-exp: {error}", file=sys.stderr)
        return 2
    if progress is not None:
        progress.done(str(output))
    print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
