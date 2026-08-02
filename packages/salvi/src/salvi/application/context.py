"""Explicit runtime data shared by scientific components."""

from __future__ import annotations

import hashlib
import threading
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.random as npr

from salvi.domain.prepared import PreparedDataset
from salvi.domain.search import ArchiveCellTarget
from salvi.exceptions import ComponentError
from salvi.patterns.configuration import PatternConfiguration

if TYPE_CHECKING:
    from salvi.api.run import RunSpecification
    from salvi.components.protocols import (
        CandidateValidityPolicy,
        CrossoverOperator,
        EvaluationSupportPolicy,
        MateSelectionPolicy,
        MutationOperator,
        ParentSelectionPolicy,
    )


class NamedRandomStreams:
    """Lazily create stable, independent random streams from one run seed."""

    def __init__(self, seed: int) -> None:
        if seed < 0:
            raise ValueError("random seed must be non-negative")
        self._seed = seed
        self._generators: dict[str, npr.Generator] = {}
        self._lock = threading.Lock()

    @property
    def seed(self) -> int:
        return self._seed

    def generator(self, name: str) -> npr.Generator:
        normalized = name.strip()
        if not normalized:
            raise ValueError("random stream name must not be blank")
        with self._lock:
            generator = self._generators.get(normalized)
            if generator is None:
                digest = hashlib.sha256(normalized.encode("utf-8")).digest()
                words = np.frombuffer(digest, dtype="<u4").astype(np.uint32).tolist()
                generator = npr.default_rng(npr.SeedSequence([self._seed, *words]))
                self._generators[normalized] = generator
            return generator

    def snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {
                name: deepcopy(dict(generator.bit_generator.state))
                for name, generator in sorted(self._generators.items())
            }

    def restore(self, states: dict[str, dict[str, Any]]) -> None:
        restored: dict[str, npr.Generator] = {}
        for name, state in sorted(states.items()):
            if not name.strip():
                raise ValueError("random stream names must not be blank")
            bit_generator = npr.PCG64()
            try:
                bit_generator.state = cast(Any, deepcopy(state))
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid state for random stream {name!r}") from error
            restored[name] = npr.Generator(bit_generator)
        with self._lock:
            self._generators = restored


@dataclass(frozen=True, slots=True)
class PreprocessingStepReport:
    component_name: str
    duration_seconds: float
    memory_before_bytes: int
    memory_after_bytes: int


@dataclass(frozen=True, slots=True)
class PreprocessingReport:
    loading_seconds: float
    initial_memory_bytes: int
    final_memory_bytes: int
    steps: tuple[PreprocessingStepReport, ...]


@dataclass(frozen=True, slots=True)
class RunContext:
    """Scientific runtime context, deliberately free of UI and experiment state."""

    dataset: PreparedDataset
    patterns: PatternConfiguration
    random_streams: NamedRandomStreams
    candidate_validity_policy: CandidateValidityPolicy
    evaluation_support_policy: EvaluationSupportPolicy

    def random_generator(self, name: str) -> npr.Generator:
        """Resolve a deterministic named random stream."""

        return self.random_streams.generator(name)


@dataclass(frozen=True, slots=True)
class QdRunContext(RunContext):
    """Search-only dependencies shared by QD emitters."""

    parent_selection_policy: ParentSelectionPolicy | None = None
    mate_selection_policy: MateSelectionPolicy | None = None
    crossover_operator: CrossoverOperator | None = None
    mutation_operator: MutationOperator | None = None
    archive_cell_targets: tuple[ArchiveCellTarget, ...] = ()

    @classmethod
    def from_run_context(
        cls,
        context: RunContext,
        *,
        parent_selection_policy: ParentSelectionPolicy | None,
        mate_selection_policy: MateSelectionPolicy | None,
        crossover_operator: CrossoverOperator | None,
        mutation_operator: MutationOperator | None,
        archive_cell_targets: tuple[ArchiveCellTarget, ...] = (),
    ) -> QdRunContext:
        return cls(
            dataset=context.dataset,
            patterns=context.patterns,
            random_streams=context.random_streams,
            candidate_validity_policy=context.candidate_validity_policy,
            evaluation_support_policy=context.evaluation_support_policy,
            parent_selection_policy=parent_selection_policy,
            mate_selection_policy=mate_selection_policy,
            crossover_operator=crossover_operator,
            mutation_operator=mutation_operator,
            archive_cell_targets=archive_cell_targets,
        )


def require_qd_run_context(context: RunContext) -> QdRunContext:
    """Narrow a base context for components consumed only by QD engines."""

    if not isinstance(context, QdRunContext):
        raise ComponentError("the configured component requires a QD search context")
    return context


@dataclass(frozen=True, slots=True)
class PreparedRun:
    specification: RunSpecification
    context: RunContext
    preprocessing: PreprocessingReport


__all__ = [
    "NamedRandomStreams",
    "PreparedRun",
    "PreprocessingReport",
    "PreprocessingStepReport",
    "QdRunContext",
    "RunContext",
    "require_qd_run_context",
]
