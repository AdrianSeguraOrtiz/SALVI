"""Filesystem helpers shared by canonical artifact stores."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from salvi.exceptions import ArtifactError


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ArtifactError(f"cannot hash artifact file {path}: {error}") from error
    return digest.hexdigest()


def atomic_write_text(path: Path, content: str) -> None:
    """Durably replace one text file without exposing a partial checkpoint."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        try:
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise ArtifactError(f"cannot atomically write {path}: {error}") from error


@contextmanager
def atomic_directory(destination: Path, *, overwrite: bool) -> Iterator[Path]:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not destination.is_dir():
        raise ArtifactError(f"destination must be a directory path: {destination}")
    if destination.exists() and not overwrite:
        raise ArtifactError(f"destination already exists: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        yield temporary
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
