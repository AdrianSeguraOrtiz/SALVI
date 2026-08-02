"""Verify that the publishable SALVI version matches an optional release tag."""

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
    project = _project(arguments.root / "pyproject.toml")
    project_name = str(project["name"])
    if project_name != "salvi":
        raise SystemExit(f"expected the root distribution to be 'salvi', found {project_name!r}")
    project_version = str(project["version"])
    if arguments.tag is not None:
        match = re.fullmatch(r"v(.+)", arguments.tag)
        if match is None or match.group(1) != project_version:
            raise SystemExit(
                f"release tag {arguments.tag!r} does not match package version {project_version}"
            )
    print(project_version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
