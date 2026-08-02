"""Batch-scoped sharing of exact scientific evaluation results."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from threading import RLock
from typing import TYPE_CHECKING

from salvi.application.context import RunContext
from salvi.domain.models import Candidate, ObjectiveResult, PatternFit
from salvi.patterns.catalog import PatternCatalog, default_pattern_catalog
from salvi.patterns.contracts import (
    PatternContrastEvaluator,
    PatternInferenceEngine,
)
from salvi.patterns.inference import DefaultPatternInferenceEngine

if TYPE_CHECKING:
    from salvi.components.protocols import Objective


class EvaluationWorkspace:
    """Share one inference and derived calculation per exact candidate signature."""

    def __init__(
        self,
        context: RunContext,
        *,
        pattern_catalog: PatternCatalog | None = None,
        inference_engine: PatternInferenceEngine | None = None,
    ) -> None:
        self._context = context
        self._catalog = pattern_catalog or default_pattern_catalog(context.patterns.allowed)
        self._inference_engine = inference_engine or DefaultPatternInferenceEngine.create(
            self._catalog
        )
        self._fits: dict[str, PatternFit] = {}
        self._fit_inflight: dict[str, Future[PatternFit]] = {}
        self._contrasts: dict[tuple[str, str], ObjectiveResult] = {}
        self._contrast_inflight: dict[tuple[str, str], Future[ObjectiveResult]] = {}
        self._objectives: dict[tuple[str, str], ObjectiveResult] = {}
        self._objective_inflight: dict[tuple[str, str], Future[ObjectiveResult]] = {}
        self._lock = RLock()

    @property
    def context(self) -> RunContext:
        return self._context

    @property
    def pattern_catalog(self) -> PatternCatalog:
        return self._catalog

    def infer(self, candidate: Candidate) -> PatternFit:
        return self.pattern_fit(
            candidate,
            lambda: self._inference_engine.infer(self._context, candidate.bicluster),
        )

    def pattern_fit(
        self,
        candidate: Candidate,
        compute: Callable[[], PatternFit],
    ) -> PatternFit:
        signature = candidate.bicluster.signature
        with self._lock:
            existing = self._fits.get(signature)
            if existing is not None:
                return existing
            pending = self._fit_inflight.get(signature)
            owner = pending is None
            if pending is None:
                pending = Future()
                self._fit_inflight[signature] = pending

        if not owner:
            return pending.result()
        try:
            fit = compute()
            if fit.candidate_signature != signature:
                raise ValueError("computed PatternFit does not match the candidate signature")
        except BaseException as error:
            pending.set_exception(error)
            with self._lock:
                self._fit_inflight.pop(signature, None)
            raise
        with self._lock:
            self._fits[signature] = fit
            self._fit_inflight.pop(signature, None)
        pending.set_result(fit)
        return fit

    def cached_pattern_fit(self, candidate: Candidate) -> PatternFit | None:
        with self._lock:
            return self._fits.get(candidate.bicluster.signature)

    def contrast(
        self,
        candidate: Candidate,
        evaluator: PatternContrastEvaluator,
    ) -> ObjectiveResult:
        fit = self.infer(candidate)
        key = (candidate.bicluster.signature, evaluator.cache_key)
        with self._lock:
            existing = self._contrasts.get(key)
            if existing is not None:
                return existing
            pending = self._contrast_inflight.get(key)
            owner = pending is None
            if pending is None:
                pending = Future()
                self._contrast_inflight[key] = pending
        if not owner:
            return pending.result()
        try:
            result = evaluator.evaluate(
                self._context,
                candidate.bicluster,
                fit,
                self._catalog,
            )
            if not isinstance(result, ObjectiveResult):
                raise TypeError("contrast evaluators must return ObjectiveResult")
        except BaseException as error:
            pending.set_exception(error)
            with self._lock:
                self._contrast_inflight.pop(key, None)
            raise
        with self._lock:
            self._contrasts[key] = result
            self._contrast_inflight.pop(key, None)
        pending.set_result(result)
        return result

    def objective(self, candidate: Candidate, objective: Objective) -> ObjectiveResult:
        """Evaluate one configured objective once for an exact candidate."""

        key = (candidate.bicluster.signature, objective.component_name)
        with self._lock:
            existing = self._objectives.get(key)
            if existing is not None:
                return existing
            pending = self._objective_inflight.get(key)
            owner = pending is None
            if pending is None:
                pending = Future()
                self._objective_inflight[key] = pending
        if not owner:
            return pending.result()
        try:
            result = objective.evaluate(candidate, self)
            if not isinstance(result, ObjectiveResult):
                raise TypeError("objectives must return ObjectiveResult")
        except BaseException as error:
            pending.set_exception(error)
            with self._lock:
                self._objective_inflight.pop(key, None)
            raise
        with self._lock:
            self._objectives[key] = result
            self._objective_inflight.pop(key, None)
        pending.set_result(result)
        return result

    def cached_objective(
        self,
        candidate: Candidate,
        objective_name: str,
    ) -> ObjectiveResult | None:
        with self._lock:
            return self._objectives.get((candidate.bicluster.signature, objective_name))

    def __len__(self) -> int:
        with self._lock:
            return len(self._fits)
