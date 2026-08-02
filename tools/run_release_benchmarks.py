"""Run all release profiles and write their compact published baseline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from salvi.infrastructure.files import atomic_write_text
from salvi.versioning import package_version


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configurations",
        type=Path,
        default=root / "benchmarks" / "configurations",
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=root / "benchmarks" / "generated" / "profiles",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "benchmarks" / "performance-baseline-v0.1.0.json",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    arguments = parser.parse_args()

    summaries: list[dict[str, object]] = []
    dataset = root / "benchmarks" / "generated" / "release-dataset"
    for configuration in sorted(arguments.configurations.glob("*.yaml")):
        destination = arguments.profiles / configuration.stem
        run_output = arguments.profiles / f"{configuration.stem}-run"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "salvi.cli.main",
                "profile",
                str(configuration),
                str(destination),
                "--dataset",
                str(dataset),
                "--output",
                str(run_output),
                "--identifier",
                f"release-{configuration.stem}",
                "--seed",
                "20260724",
                "--repetitions",
                str(arguments.repetitions),
                "--lightweight",
                "--overwrite",
                "--run-overwrite",
            ],
            check=True,
        )
        report = json.loads((destination / "profile-report.json").read_text(encoding="utf-8"))
        summaries.append(
            {
                "configuration": configuration.name,
                "configuration_sha256": report["configuration_sha256"],
                "scientific_scope": report["scientific_scope"],
                "median": report["median"],
            }
        )

    if not summaries:
        parser.error(f"no YAML configurations found in {arguments.configurations}")
    environment = json.loads(
        (
            arguments.profiles
            / Path(str(summaries[0]["configuration"])).stem
            / "profile-report.json"
        ).read_text(encoding="utf-8")
    )["environment"]
    output = {
        "schema_version": 1,
        "salvi_version": package_version(),
        "fixture": {
            "identifier": "salvi-release-profile",
            "rows": 240,
            "columns": 12,
            "generator": "tools/generate_release_fixture.py",
            "seed": 20260724,
        },
        "environment": environment,
        "repetitions": arguments.repetitions,
        "profiles": summaries,
    }
    atomic_write_text(
        arguments.output,
        json.dumps(output, indent=2, sort_keys=True) + "\n",
    )
    print(arguments.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
