"""Pattern-selection settings shared by configuration and programmatic runs."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from salvi.domain.enums import PatternKind


class PatternConfiguration(BaseModel):
    """Immutable configuration for one or more registered pattern families."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed: tuple[PatternKind, ...] = (PatternKind.CONSTANT,)
    min_improvement: Annotated[float, Field(ge=0.0, le=1.0)] = 0.10
    max_iterations: Annotated[int, Field(ge=1)] = 25
    convergence_tolerance: Annotated[float, Field(gt=0.0)] = 1e-6

    @field_validator("allowed")
    @classmethod
    def validate_allowed_patterns(
        cls,
        value: tuple[PatternKind, ...],
    ) -> tuple[PatternKind, ...]:
        if not value:
            raise ValueError("at least one pattern must be allowed")
        if len(set(value)) != len(value):
            raise ValueError("allowed patterns must not contain duplicates")
        return value
