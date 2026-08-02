"""Convert the versioned JSON emitted by SALVI's BicPAMS bridge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from salvi.exceptions import ConversionError
from salvi_experiments.interop.external import (
    ExternalBiclusterSetConverter,
    ExternalResultDocument,
)


@dataclass(frozen=True, slots=True)
class BicPamsConverter:
    dataset_bundle: Path
    overwrite: bool = False

    def convert(self, source: Path, destination: Path) -> Path:
        try:
            document = ExternalResultDocument.model_validate_json(
                source.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as error:
            raise ConversionError(f"invalid BicPAMS result document {source}: {error}") from error
        if document.algorithm != "BicPAMS":
            raise ConversionError("BicPAMS result document has an unexpected algorithm")
        return ExternalBiclusterSetConverter(
            dataset_bundle=self.dataset_bundle,
            overwrite=self.overwrite,
        ).write(document, destination)


__all__ = ["BicPamsConverter"]
