"""Search engines and quality-diversity archive implementations."""

from __future__ import annotations

from typing import Any

__all__ = [
    "DeepGridMomeArchive",
    "PymooNsga2SearchEngine",
    "SerialMomeSearchEngine",
]


def __getattr__(name: str) -> Any:
    if name == "DeepGridMomeArchive":
        from salvi.engine.archive import DeepGridMomeArchive

        return DeepGridMomeArchive
    if name == "SerialMomeSearchEngine":
        from salvi.engine.mome import SerialMomeSearchEngine

        return SerialMomeSearchEngine
    if name == "PymooNsga2SearchEngine":
        from salvi.engine.pymoo import PymooNsga2SearchEngine

        return PymooNsga2SearchEngine
    raise AttributeError(name)
