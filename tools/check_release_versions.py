"""Verify that publishable package versions and internal pins remain synchronized."""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path


def _project(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        return tomllib.load(stream)["project"]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--tag")
    arguments = parser.parse_args()
    core = _project(arguments.root / "packages" / "salvi" / "pyproject.toml")
    experiments = _project(arguments.root / "packages" / "salvi-experiments" / "pyproject.toml")
    core_version = str(core["version"])
    experiments_version = str(experiments["version"])
    if core_version != experiments_version:
        raise SystemExit(
            f"package versions differ: salvi={core_version}, "
            f"salvi-experiments={experiments_version}"
        )
    dependencies = tuple(str(value) for value in experiments["dependencies"])
    expected_pin = f"salvi=={core_version}"
    if expected_pin not in dependencies:
        raise SystemExit(f"salvi-experiments must depend on {expected_pin}")
    if arguments.tag is not None:
        match = re.fullmatch(r"v(.+)", arguments.tag)
        if match is None or match.group(1) != core_version:
            raise SystemExit(
                f"release tag {arguments.tag!r} does not match package version {core_version}"
            )
    print(core_version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
