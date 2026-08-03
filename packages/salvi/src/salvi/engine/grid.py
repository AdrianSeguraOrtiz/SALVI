"""Per-axis discretization for sparse quality-diversity archives."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from salvi.domain.enums import BinningStrategy, DescriptorValueKind
from salvi.domain.search import DescriptorDomain
from salvi.exceptions import ComponentError


class ArchiveAxisConfiguration(BaseModel):
    """Discretization selected by one archive axis for one descriptor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    descriptor: str = Field(min_length=1)
    binning: BinningStrategy
    bins: Annotated[int, Field(ge=1)] | None = None
    boundaries: tuple[float, ...] = ()
    minimum: float | None = None
    maximum: float | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        if self.binning in {BinningStrategy.LINEAR, BinningStrategy.GEOMETRIC}:
            if self.bins is None:
                raise ValueError(f"{self.binning.value} binning requires bins")
            if self.boundaries:
                raise ValueError(f"{self.binning.value} binning does not accept boundaries")
        elif self.binning is BinningStrategy.EXACT:
            if self.bins is not None or self.boundaries:
                raise ValueError("EXACT binning does not accept bins or boundaries")
        elif self.binning is BinningStrategy.CUSTOM:
            if self.bins is not None or not self.boundaries:
                raise ValueError("CUSTOM binning requires boundaries and does not accept bins")
            if tuple(sorted(set(self.boundaries))) != self.boundaries:
                raise ValueError("custom boundaries must be strictly increasing")
            if any(not math.isfinite(value) for value in self.boundaries):
                raise ValueError("custom boundaries must be finite")
        if self.minimum is not None and not math.isfinite(self.minimum):
            raise ValueError("axis minimum must be finite")
        if self.maximum is not None and not math.isfinite(self.maximum):
            raise ValueError("axis maximum must be finite")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("axis minimum cannot exceed its maximum")
        return self


@dataclass(frozen=True, slots=True)
class AxisBinner:
    descriptor: str
    strategy: BinningStrategy
    minimum: float
    maximum: float
    bin_count: int
    boundaries: tuple[float, ...] = ()

    @classmethod
    def create(
        cls,
        configuration: ArchiveAxisConfiguration,
        domain: DescriptorDomain,
    ) -> AxisBinner:
        if configuration.binning not in domain.supported_binnings:
            raise ComponentError(
                f"descriptor {configuration.descriptor!r} does not support "
                f"{configuration.binning.value} binning"
            )
        minimum = domain.minimum if configuration.minimum is None else configuration.minimum
        maximum = domain.maximum if configuration.maximum is None else configuration.maximum
        if minimum < domain.minimum or maximum > domain.maximum:
            raise ComponentError(
                f"axis {configuration.descriptor!r} bounds [{minimum}, {maximum}] exceed "
                f"descriptor domain [{domain.minimum}, {domain.maximum}]"
            )
        if minimum > maximum:
            raise ComponentError(
                f"axis {configuration.descriptor!r} minimum cannot exceed its maximum"
            )
        if configuration.binning is BinningStrategy.GEOMETRIC and minimum <= 0.0:
            raise ComponentError("GEOMETRIC binning requires a strictly positive minimum")
        if configuration.binning is BinningStrategy.EXACT:
            if domain.value_kind is not DescriptorValueKind.INTEGER:
                raise ComponentError("EXACT binning requires an integer descriptor")
            if not minimum.is_integer() or not maximum.is_integer():
                raise ComponentError("EXACT axis bounds must be integral")
            bin_count = int(maximum - minimum) + 1
        elif configuration.binning is BinningStrategy.CUSTOM:
            if any(value <= minimum or value >= maximum for value in configuration.boundaries):
                raise ComponentError(
                    "custom boundaries must lie strictly inside the configured axis bounds"
                )
            bin_count = len(configuration.boundaries) + 1
        else:
            assert configuration.bins is not None
            bin_count = configuration.bins
        return cls(
            descriptor=configuration.descriptor,
            strategy=configuration.binning,
            minimum=minimum,
            maximum=maximum,
            bin_count=bin_count,
            boundaries=configuration.boundaries,
        )

    def index(self, value: float) -> int | None:
        if not math.isfinite(value) or value < self.minimum or value > self.maximum:
            return None
        if self.bin_count == 1 or self.minimum == self.maximum:
            return 0
        if self.strategy is BinningStrategy.EXACT:
            rounded = round(value)
            if not math.isclose(value, rounded, abs_tol=1e-12):
                return None
            return int(rounded - self.minimum)
        if self.strategy is BinningStrategy.CUSTOM:
            return bisect.bisect_right(self.boundaries, value)
        if self.strategy is BinningStrategy.LINEAR:
            position = (value - self.minimum) / (self.maximum - self.minimum)
        else:
            position = math.log(value / self.minimum) / math.log(self.maximum / self.minimum)
        return min(self.bin_count - 1, math.floor(position * self.bin_count))

    def representative_integer(self, index: int) -> int | None:
        """Return a central reachable integer for one bin, if it has one."""

        if index < 0 or index >= self.bin_count:
            raise IndexError("archive bin index is out of range")
        if self.strategy is BinningStrategy.EXACT:
            value = int(self.minimum) + index
            return value if self.index(float(value)) == index else None

        if self.strategy is BinningStrategy.CUSTOM:
            lower = self.minimum if index == 0 else self.boundaries[index - 1]
            upper = self.maximum if index == self.bin_count - 1 else self.boundaries[index]
            center = (lower + upper) / 2.0
        elif self.strategy is BinningStrategy.LINEAR:
            span = self.maximum - self.minimum
            lower = self.minimum + span * index / self.bin_count
            upper = self.minimum + span * (index + 1) / self.bin_count
            center = (lower + upper) / 2.0
        else:
            ratio = self.maximum / self.minimum
            lower = self.minimum * ratio ** (index / self.bin_count)
            upper = self.minimum * ratio ** ((index + 1) / self.bin_count)
            center = math.sqrt(lower * upper)

        candidates = {
            round(center),
            math.floor(center),
            math.ceil(center),
            math.ceil(lower),
            math.floor(upper),
        }
        valid = tuple(
            value
            for value in candidates
            if self.minimum <= value <= self.maximum and self.index(float(value)) == index
        )
        if not valid:
            return None
        return min(valid, key=lambda value: (abs(value - center), value))

    def integer_bounds(self, index: int) -> tuple[int, int] | None:
        """Return the exact inclusive integer interval mapped to one bin."""

        if index < 0 or index >= self.bin_count:
            raise IndexError("archive bin index is out of range")
        minimum = math.ceil(self.minimum)
        maximum = math.floor(self.maximum)
        if minimum > maximum:
            return None

        def first_integer_at_or_after_bin(target: int) -> int:
            lower = minimum
            upper = maximum + 1
            while lower < upper:
                midpoint = (lower + upper) // 2
                mapped = self.index(float(midpoint))
                if mapped is not None and mapped >= target:
                    upper = midpoint
                else:
                    lower = midpoint + 1
            return lower

        lower = first_integer_at_or_after_bin(index)
        upper = first_integer_at_or_after_bin(index + 1) - 1
        if lower > maximum or lower > upper or self.index(float(lower)) != index:
            return None
        return lower, min(upper, maximum)


__all__ = ["ArchiveAxisConfiguration", "AxisBinner"]
