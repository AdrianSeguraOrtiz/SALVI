"""Build every publishable package twice and compare distribution bytes."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build(root: Path, destination: Path, uv: str, epoch: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment["SOURCE_DATE_EPOCH"] = epoch
    subprocess.run(
        [
            uv,
            "build",
            "--all-packages",
            "--out-dir",
            str(destination),
            "--clear",
            "--no-create-gitignore",
        ],
        cwd=root,
        env=environment,
        check=True,
    )
    return {
        artifact.name: _digest(artifact)
        for artifact in sorted(destination.iterdir())
        if artifact.is_file()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--uv", default=shutil.which("uv"))
    parser.add_argument("--source-date-epoch", default="1704067200")
    arguments = parser.parse_args()
    if arguments.uv is None:
        parser.error("uv is required")

    with tempfile.TemporaryDirectory(prefix="salvi-reproducible-build-") as temporary:
        first = _build(
            arguments.root.resolve(),
            Path(temporary) / "first",
            arguments.uv,
            arguments.source_date_epoch,
        )
        second = _build(
            arguments.root.resolve(),
            Path(temporary) / "second",
            arguments.uv,
            arguments.source_date_epoch,
        )
    if first != second:
        names = sorted(set(first) | set(second))
        differences = [
            f"{name}: {first.get(name, '<missing>')} != {second.get(name, '<missing>')}"
            for name in names
            if first.get(name) != second.get(name)
        ]
        raise SystemExit("distribution builds are not reproducible:\n" + "\n".join(differences))
    for name, digest in first.items():
        print(f"{digest}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
