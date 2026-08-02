"""Small progress reporting primitives for experiment command-line runs."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from time import perf_counter
from typing import Protocol, TextIO


class ProgressReporter(Protocol):
    """Receives concise progress notifications from experiment protocols."""

    def begin(self, label: str) -> None: ...

    def stage(self, message: str) -> None: ...

    def step(self, message: str, current: int, total: int) -> None: ...

    def done(self, message: str) -> None: ...


class NullProgressReporter:
    """Progress sink used by programmatic calls."""

    def begin(self, label: str) -> None:
        return None

    def stage(self, message: str) -> None:
        return None

    def step(self, message: str, current: int, total: int) -> None:
        return None

    def done(self, message: str) -> None:
        return None


@dataclass(slots=True)
class ConsoleProgressReporter:
    """Writes human-readable progress to a text stream."""

    stream: TextIO | None = None
    prefix: str = "salvi-exp"
    _started_at: float = field(default_factory=perf_counter, init=False)

    def begin(self, label: str) -> None:
        self._started_at = perf_counter()
        self._write(f"starting {label}")

    def stage(self, message: str) -> None:
        self._write(message)

    def step(self, message: str, current: int, total: int) -> None:
        if total <= 0:
            self._write(message)
            return
        bounded = min(max(current, 0), total)
        percent = 100.0 * bounded / total
        self._write(f"[{bounded}/{total} {percent:5.1f}%] {message}")

    def done(self, message: str) -> None:
        elapsed = perf_counter() - self._started_at
        self._write(f"completed in {elapsed:.1f}s: {message}")

    def _write(self, message: str) -> None:
        stream = self.stream if self.stream is not None else sys.stderr
        print(f"{self.prefix}: {message}", file=stream, flush=True)


def progress_or_null(progress: ProgressReporter | None) -> ProgressReporter:
    return progress if progress is not None else NullProgressReporter()


__all__ = [
    "ConsoleProgressReporter",
    "NullProgressReporter",
    "ProgressReporter",
    "progress_or_null",
]
