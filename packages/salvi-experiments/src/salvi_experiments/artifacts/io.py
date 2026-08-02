"""Atomic, checksummed experiment output helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from salvi_experiments.exceptions import ExperimentArtifactError


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def atomic_experiment_directory(
    destination: Path,
    *,
    overwrite: bool,
) -> Iterator[Path]:
    target = destination.expanduser().resolve()
    if target == target.parent:
        raise ExperimentArtifactError("experiment output cannot be a filesystem root")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent)).resolve()
    try:
        yield temporary
        if target.exists():
            if not overwrite:
                raise ExperimentArtifactError(f"experiment output already exists: {target}")
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        os.replace(temporary, target)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_table(
    root: Path,
    stem: str,
    records: Sequence[Mapping[str, object]],
) -> tuple[Path, Path]:
    parquet_path = root / f"{stem}.parquet"
    csv_path = root / f"{stem}.csv"
    table = pa.Table.from_pylist([dict(record) for record in records])
    pq.write_table(table, parquet_path, compression="zstd")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=table.column_names)
        writer.writeheader()
        for record in table.to_pylist():
            writer.writerow(
                {
                    key: (
                        json.dumps(value, sort_keys=True, separators=(",", ":"))
                        if isinstance(value, list | dict)
                        else value
                    )
                    for key, value in record.items()
                }
            )
    return parquet_path, csv_path


def write_manifest(
    root: Path,
    *,
    experiment_type: str,
    identifier: str,
    metadata: Mapping[str, object],
) -> Path:
    files = tuple(
        sorted(
            path.relative_to(root)
            for path in root.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        )
    )
    manifest = {
        "schema_version": 1,
        "experiment_type": experiment_type,
        "identifier": identifier,
        "created_at": datetime.now(UTC).isoformat(),
        "metadata": dict(metadata),
        "checksums": {path.as_posix(): sha256_file(root / path) for path in files},
    }
    path = root / "manifest.json"
    write_json(path, manifest)
    return path


def read_report(directory: Path) -> dict[str, object]:
    path = directory.expanduser().resolve() / "report.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExperimentArtifactError(f"invalid experiment report {path}: {error}") from error
    if not isinstance(raw, dict):
        raise ExperimentArtifactError(f"experiment report must be an object: {path}")
    return raw


__all__ = [
    "atomic_experiment_directory",
    "read_report",
    "sha256_file",
    "write_json",
    "write_manifest",
    "write_table",
]
