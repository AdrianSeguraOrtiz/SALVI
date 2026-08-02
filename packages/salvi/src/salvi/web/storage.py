"""Private persistent state for the local single-user web application."""

from __future__ import annotations

import shutil
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from platformdirs import user_data_path

from salvi.domain.enums import RunStatus
from salvi.exceptions import ArtifactError
from salvi.web.models import WebDatasetRecord, WebImportRecord, WebRunRecord


@dataclass(frozen=True, slots=True)
class WebApplicationPaths:
    root: Path
    uploads: Path
    datasets: Path
    runs: Path
    database: Path

    @classmethod
    def create(cls, root: Path | None = None) -> WebApplicationPaths:
        resolved = (
            Path(user_data_path("salvi", appauthor=False)) / "web"
            if root is None
            else root.expanduser().resolve()
        )
        paths = cls(
            root=resolved,
            uploads=resolved / "uploads",
            datasets=resolved / "datasets",
            runs=resolved / "runs",
            database=resolved / "web.sqlite",
        )
        for directory in (paths.root, paths.uploads, paths.datasets, paths.runs):
            directory.mkdir(parents=True, exist_ok=True)
        return paths


class WebStateStore:
    """Small durable index; scientific events remain in each run's run.sqlite."""

    def __init__(self, paths: WebApplicationPaths) -> None:
        self.paths = paths
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.paths.database, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS imports (
                    identifier TEXT PRIMARY KEY,
                    record_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS datasets (
                    identifier TEXT PRIMARY KEY,
                    record_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    identifier TEXT PRIMARY KEY,
                    record_json TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def new_identifier(prefix: str) -> str:
        return f"{prefix}-{uuid4().hex}"

    @staticmethod
    def _serialize(record: WebImportRecord | WebDatasetRecord | WebRunRecord) -> str:
        return record.model_dump_json()

    def put_import(self, record: WebImportRecord) -> None:
        self._put("imports", record.identifier, self._serialize(record))

    def get_import(self, identifier: str) -> WebImportRecord | None:
        raw = self._get("imports", identifier)
        return None if raw is None else WebImportRecord.model_validate_json(raw)

    def delete_import(self, identifier: str) -> None:
        record = self.get_import(identifier)
        self._delete("imports", identifier)
        if record is not None:
            self._remove_owned(record.upload_directory, self.paths.uploads)

    def put_dataset(self, record: WebDatasetRecord) -> None:
        self._put("datasets", record.identifier, self._serialize(record))

    def get_dataset(self, identifier: str) -> WebDatasetRecord | None:
        raw = self._get("datasets", identifier)
        return None if raw is None else WebDatasetRecord.model_validate_json(raw)

    def datasets(self) -> tuple[WebDatasetRecord, ...]:
        return tuple(WebDatasetRecord.model_validate_json(raw) for raw in self._list("datasets"))

    def delete_dataset(self, identifier: str) -> None:
        record = self.get_dataset(identifier)
        if record is None:
            return
        if any(run.dataset_identifier == identifier for run in self.runs()):
            raise ArtifactError("dataset is referenced by one or more retained runs")
        self._delete("datasets", identifier)
        self._remove_owned(record.storage_path or record.bundle_path, self.paths.datasets)

    def put_run(self, record: WebRunRecord) -> None:
        self._put("runs", record.identifier, self._serialize(record))

    def get_run(self, identifier: str) -> WebRunRecord | None:
        raw = self._get("runs", identifier)
        return None if raw is None else WebRunRecord.model_validate_json(raw)

    def runs(self) -> tuple[WebRunRecord, ...]:
        records = tuple(WebRunRecord.model_validate_json(raw) for raw in self._list("runs"))
        return tuple(sorted(records, key=lambda item: item.created_at, reverse=True))

    def active_runs(self) -> tuple[WebRunRecord, ...]:
        return tuple(run for run in self.runs() if run.status is RunStatus.RUNNING)

    def delete_run(self, identifier: str) -> None:
        record = self.get_run(identifier)
        if record is None:
            return
        if record.status is RunStatus.RUNNING:
            raise ArtifactError("an active run cannot be deleted")
        self._delete("runs", identifier)
        self._remove_owned(record.pipeline_path.parent, self.paths.runs)

    def mark_interrupted_runs(self) -> None:
        for record in self.active_runs():
            self.put_run(
                record.model_copy(
                    update={
                        "status": RunStatus.FAILED,
                        "error": "The web server stopped while this run was active.",
                        "finished_at": datetime.now(UTC),
                    }
                )
            )

    def _put(self, table: str, identifier: str, record_json: str) -> None:
        with self._connection() as connection:
            connection.execute(
                f"""
                INSERT INTO {table}(identifier, record_json) VALUES (?, ?)
                ON CONFLICT(identifier) DO UPDATE SET record_json = excluded.record_json
                """,
                (identifier, record_json),
            )

    def _get(self, table: str, identifier: str) -> str | None:
        with self._connection() as connection:
            row = connection.execute(
                f"SELECT record_json FROM {table} WHERE identifier = ?",
                (identifier,),
            ).fetchone()
        return None if row is None else str(row["record_json"])

    def _list(self, table: str) -> tuple[str, ...]:
        with self._connection() as connection:
            rows = connection.execute(f"SELECT record_json FROM {table}").fetchall()
        return tuple(str(row["record_json"]) for row in rows)

    def _delete(self, table: str, identifier: str) -> None:
        with self._connection() as connection:
            connection.execute(
                f"DELETE FROM {table} WHERE identifier = ?",
                (identifier,),
            )

    @staticmethod
    def _remove_owned(path: Path, owner: Path) -> None:
        resolved = path.resolve()
        if not resolved.is_relative_to(owner.resolve()):
            raise ArtifactError(f"refusing to remove a path outside managed storage: {resolved}")
        if resolved.is_dir():
            shutil.rmtree(resolved)
        elif resolved.exists():
            resolved.unlink()


__all__ = ["WebApplicationPaths", "WebStateStore"]
