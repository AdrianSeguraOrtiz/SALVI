"""Direct in-memory execution for programmatically composed SALVI runs."""

from __future__ import annotations

from dataclasses import dataclass

from salvi.api.run import RunSpecification
from salvi.application.factory import prepare_run
from salvi.components.protocols import CancellationSignal
from salvi.domain.models import Repertoire
from salvi.evaluation.workspace import EvaluationWorkspace
from salvi.exceptions import RunError


@dataclass(frozen=True, slots=True)
class InMemoryRunResult:
    """Search and selected repertoires produced without durable run artifacts."""

    search_repertoire: Repertoire
    repertoire: Repertoire
    evaluations: int


def execute_in_memory(
    specification: RunSpecification,
    *,
    cancellation: CancellationSignal | None = None,
) -> InMemoryRunResult:
    """Execute one validated specification and return its repertoires.

    This entry point is intended for scripts that compose concrete component
    instances directly. It deliberately omits SQLite events, observers,
    checkpoints, and artifact persistence; use :class:`salvi.RunService` or the
    CLI when those facilities are required.
    """

    prepared = prepare_run(specification)
    engine = specification.search_engine
    try:
        engine.initialize(specification, prepared.context)
        while not engine.finished():
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            candidates = tuple(engine.ask(engine.batch_size))
            if not candidates:
                raise RunError("search engine returned no candidates before termination")
            batch = specification.executor.evaluate(
                candidates,
                specification.objectives,
                specification.descriptors,
                EvaluationWorkspace(prepared.context),
                constraints=specification.constraints,
                worker_count=specification.worker_count,
                cancellation=cancellation,
                collect_timings=False,
            )
            engine.tell(batch.evaluations)

        search_repertoire = engine.result()
        repertoire = (
            search_repertoire
            if specification.final_selector is None
            else specification.final_selector.select(prepared.context, search_repertoire)
        )
        return InMemoryRunResult(
            search_repertoire=search_repertoire,
            repertoire=repertoire,
            evaluations=engine.progress().evaluations,
        )
    finally:
        specification.executor.close()


__all__ = ["InMemoryRunResult", "execute_in_memory"]
