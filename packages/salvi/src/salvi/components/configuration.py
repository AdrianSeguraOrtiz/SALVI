"""Shared component configuration models."""

from pydantic import BaseModel, ConfigDict


class EmptyConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


__all__ = ["EmptyConfiguration"]
