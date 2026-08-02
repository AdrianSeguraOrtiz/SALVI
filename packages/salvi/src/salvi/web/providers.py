"""Provider contracts for optional web-facing interoperability."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Protocol

from salvi.web.models import (
    AccuracySummary,
    AnalysisDescription,
    DatasetImportPreview,
    InputAdapterDescription,
    WebColumnProposal,
)


class InputAdapter(Protocol):
    @property
    def description(self) -> InputAdapterDescription: ...

    def inspect(
        self,
        files: Mapping[str, Path],
        *,
        parameters: Mapping[str, str | int | float | bool] | None = None,
        identifier: str,
        workspace: Path,
    ) -> DatasetImportPreview: ...

    def convert(
        self,
        files: Mapping[str, Path],
        *,
        identifier: str,
        columns: Sequence[WebColumnProposal],
        parameters: Mapping[str, str | int | float | bool] | None = None,
        adapter_configuration: Mapping[str, object] | None = None,
        destination: Path,
        workspace: Path,
    ) -> Path: ...


class ResultAnalysis(Protocol):
    @property
    def description(self) -> AnalysisDescription: ...

    def calculate(
        self,
        *,
        dataset_bundle: Path,
        bicluster_set: Path,
    ) -> AccuracySummary: ...


@dataclass(frozen=True, slots=True)
class WebExtensionProvider:
    adapters: tuple[InputAdapter, ...] = ()
    analyses: tuple[ResultAnalysis, ...] = ()


class WebProviderRegistry:
    """Collect built-ins and installed optional providers deterministically."""

    def __init__(
        self,
        *,
        adapters: Sequence[InputAdapter] = (),
        analyses: Sequence[ResultAnalysis] = (),
    ) -> None:
        self._adapters = {adapter.description.name: adapter for adapter in adapters}
        self._analyses = {analysis.description.name: analysis for analysis in analyses}

    def register(self, provider: WebExtensionProvider) -> None:
        for adapter in provider.adapters:
            name = adapter.description.name
            if name in self._adapters:
                raise ValueError(f"duplicate web input adapter {name!r}")
            self._adapters[name] = adapter
        for analysis in provider.analyses:
            name = analysis.description.name
            if name in self._analyses:
                raise ValueError(f"duplicate web analysis {name!r}")
            self._analyses[name] = analysis

    def load_entry_points(self) -> None:
        selected = entry_points(group="salvi.web.providers")
        for entry_point in sorted(selected, key=lambda item: item.name):
            factory = entry_point.load()
            provider = factory()
            if not isinstance(provider, WebExtensionProvider):
                raise TypeError(
                    f"web provider {entry_point.name!r} did not return WebExtensionProvider"
                )
            self.register(provider)

    def adapter(self, name: str) -> InputAdapter:
        try:
            return self._adapters[name]
        except KeyError as error:
            raise KeyError(f"unknown input adapter {name!r}") from error

    def analysis(self, name: str) -> ResultAnalysis:
        try:
            return self._analyses[name]
        except KeyError as error:
            raise KeyError(f"unknown result analysis {name!r}") from error

    @property
    def adapter_descriptions(self) -> tuple[InputAdapterDescription, ...]:
        return tuple(self._adapters[name].description for name in sorted(self._adapters))

    @property
    def analysis_descriptions(self) -> tuple[AnalysisDescription, ...]:
        return tuple(self._analyses[name].description for name in sorted(self._analyses))


ProviderFactory = Callable[[], WebExtensionProvider]


__all__ = [
    "InputAdapter",
    "ProviderFactory",
    "ResultAnalysis",
    "WebExtensionProvider",
    "WebProviderRegistry",
]
