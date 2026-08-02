"""Optional pymoo-backed evolutionary search engines."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Annotated, Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from salvi.api.run import RunSpecification
from salvi.application.context import RunContext
from salvi.components.backend_operators import BackendOperatorProvider, BackendOperatorSpec
from salvi.components.contracts import EngineCompositionContract, nsga2_engine_contract
from salvi.components.protocols import CrossoverOperator
from salvi.domain.enums import ObjectiveDirection
from salvi.domain.models import (
    Bicluster,
    Candidate,
    CandidateProvenance,
    Evaluation,
    Repertoire,
)
from salvi.domain.search import SearchCheckpoint, SearchProgress, SearchUpdate
from salvi.engine.dominance import (
    validate_constraint_schema,
    validate_objective_schema,
)
from salvi.exceptions import ComponentError

BooleanVector = npt.NDArray[np.bool_]


class PymooNsga2Configuration(BaseModel):
    """Strict configuration for the binary NSGA-II adapter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    population_size: Annotated[int, Field(ge=2)] = 64
    eliminate_duplicates: bool = True


@dataclass(frozen=True, slots=True)
class _PymooApi:
    nsga2: type[Any]
    problem: type[Any]
    crossover: type[Any]
    no_termination: type[Any]


@dataclass(slots=True)
class _CandidateRepair:
    """Apply SALVI cardinality rules before pymoo eliminates duplicates."""

    engine: PymooNsga2SearchEngine
    context: RunContext
    generator: np.random.Generator

    def __call__(self, problem: Any, population: Any, **kwargs: object) -> Any:
        return self.do(problem, population, **kwargs)

    def do(self, problem: Any, population: Any, **kwargs: object) -> Any:
        del problem, kwargs
        started = perf_counter()
        vectors = np.asarray(population.get("X"), dtype=np.bool_)
        repaired = np.stack(
            tuple(
                self.engine._repair_vector(vector, self.context, self.generator)
                for vector in vectors
            )
        )
        population.set("X", repaired)
        self.engine._record_timing(
            f"validity.{self.context.candidate_validity_policy.component_name}.repair",
            perf_counter() - started,
        )
        return population


@dataclass(slots=True)
class _TimedBackendOperator:
    """Time a callable pymoo operator without changing its scientific behavior."""

    operator: Any
    engine: PymooNsga2SearchEngine
    metric_name: str

    def __call__(self, *args: object, **kwargs: object) -> Any:
        started = perf_counter()
        try:
            return self.operator(*args, **kwargs)
        finally:
            self.engine._record_timing(self.metric_name, perf_counter() - started)


def _load_pymoo() -> _PymooApi:
    try:
        nsga2_module = importlib.import_module("pymoo.algorithms.moo.nsga2")
        crossover_module = importlib.import_module("pymoo.core.crossover")
        problem_module = importlib.import_module("pymoo.core.problem")
        termination_module = importlib.import_module("pymoo.core.termination")
    except ModuleNotFoundError as error:
        raise ComponentError(
            "pymoo_nsga2 requires the optional evolutionary backend; "
            "install SALVI with the 'evolution' extra"
        ) from error
    return _PymooApi(
        nsga2=nsga2_module.NSGA2,
        problem=problem_module.Problem,
        crossover=crossover_module.Crossover,
        no_termination=termination_module.NoTermination,
    )


def _build_backend_operator(component: object) -> Any:
    component_name = getattr(component, "component_name", type(component).__name__)
    if not isinstance(component, BackendOperatorProvider):
        raise ComponentError(
            f"component {component_name!r} does not expose a pymoo operator specification"
        )
    try:
        specification = component.backend_operator_spec("pymoo")
        if not isinstance(specification, BackendOperatorSpec):
            raise TypeError("backend_operator_spec did not return BackendOperatorSpec")
        module_name, _, attribute_name = specification.factory_path.partition(":")
        factory = getattr(importlib.import_module(module_name), attribute_name)
        return factory(**dict(specification.keyword_arguments))
    except (AttributeError, ImportError, TypeError, ValueError) as error:
        raise ComponentError(
            f"cannot adapt component {component_name!r} to pymoo: {error}"
        ) from error


def _build_salvi_crossover(
    crossover_type: type[Any],
    component: CrossoverOperator,
    engine: PymooNsga2SearchEngine,
    context: RunContext,
) -> Any:
    """Adapt an evaluated-parent SALVI crossover to pymoo's vector interface."""

    def cross(
        _adapter: object,
        _problem: object,
        vectors: npt.NDArray[np.bool_],
        *_args: object,
        random_state: np.random.Generator | None = None,
        **_kwargs: object,
    ) -> npt.NDArray[np.bool_]:
        if random_state is None:
            raise ComponentError("pymoo did not provide a crossover random stream")
        if vectors.ndim != 3 or vectors.shape[0] != 2:
            raise ComponentError("pymoo supplied an invalid crossover parent matrix")
        started = perf_counter()
        try:
            offspring = np.empty_like(vectors)
            for mating in range(vectors.shape[1]):
                first = engine._evaluation_for_vector(vectors[0, mating], context)
                second = engine._evaluation_for_vector(vectors[1, mating], context)
                offspring[0, mating] = engine._encode_bicluster(
                    component.cross(context, first, second, random_state),
                    context,
                )
                offspring[1, mating] = engine._encode_bicluster(
                    component.cross(context, second, first, random_state),
                    context,
                )
            return offspring
        finally:
            engine._record_timing(
                f"crossover.{component.component_name}",
                perf_counter() - started,
            )

    adapter_type = type(
        f"_Salvi{type(component).__name__}Adapter",
        (crossover_type,),
        {"_do": cross},
    )
    # The SALVI operator owns its application probability.
    return adapter_type(n_parents=2, n_offsprings=2, prob=1.0)


@dataclass(slots=True)
class PymooNsga2SearchEngine:
    """Adapt pymoo NSGA-II to SALVI evaluation and variation components."""

    population_size: int = 64
    eliminate_duplicates: bool = True
    component_name: str = "pymoo_nsga2"
    composition_contract: EngineCompositionContract = field(
        default_factory=lambda: nsga2_engine_contract("pymoo_nsga2")
    )
    provides: frozenset[str] = frozenset({"search-engine", "search-result"})
    requires: frozenset[str] = frozenset(
        {
            "crossover-operator",
            "initialization",
            "evaluation",
            "mutation-operator",
            "termination",
            "pymoo-mutation",
        }
    )
    _specification: RunSpecification | None = None
    _context: RunContext | None = None
    _algorithm: Any | None = None
    _pending_population: Any | None = None
    _awaiting_candidates: tuple[Candidate, ...] = ()
    _objective_schema: tuple[tuple[str, ObjectiveDirection], ...] = ()
    _constraint_schema: tuple[str, ...] = ()
    _variation_operation: str = ""
    _evaluation_by_vector: dict[bytes, Evaluation] = field(default_factory=dict)
    _evaluation_count: int = 0
    _accepted_count: int = 0
    _next_candidate_sequence: int = 0
    _component_timings: dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        configuration = PymooNsga2Configuration(
            population_size=self.population_size,
            eliminate_duplicates=self.eliminate_duplicates,
        )
        self.population_size = configuration.population_size
        self.eliminate_duplicates = configuration.eliminate_duplicates

    @property
    def batch_size(self) -> int:
        return self.population_size

    @property
    def pending_candidates(self) -> Sequence[Candidate]:
        return self._awaiting_candidates

    def initialize(self, specification: RunSpecification, context: RunContext) -> None:
        api = _load_pymoo()
        remaining = specification.termination.remaining(0)
        if remaining is not None and remaining < self.population_size:
            raise ComponentError(
                "pymoo_nsga2 requires an evaluation budget at least as large as its "
                f"population_size ({self.population_size})"
            )
        crossover = specification.require_crossover_operator()
        mutation = specification.require_mutation_operator()
        pymoo_crossover = _build_salvi_crossover(
            api.crossover,
            crossover,
            self,
            context,
        )
        pymoo_mutation = _TimedBackendOperator(
            operator=_build_backend_operator(mutation),
            engine=self,
            metric_name=f"mutation.{mutation.component_name}",
        )
        initializer_started = perf_counter()
        initial_candidates = tuple(
            specification.initializer.initialize(
                context,
                self.population_size,
                start_sequence=0,
            )
        )
        self._record_timing(
            f"initializer.{specification.initializer.component_name}",
            perf_counter() - initializer_started,
        )
        if len(initial_candidates) != self.population_size:
            raise ComponentError(
                "initializer did not produce the population requested by pymoo_nsga2"
            )
        sampling = np.stack(
            tuple(self._encode(candidate, context) for candidate in initial_candidates)
        )
        variable_count = context.dataset.row_count + context.dataset.column_count
        problem = api.problem(
            n_var=variable_count,
            n_obj=len(specification.objectives),
            n_ieq_constr=1 + len(specification.constraints),
            xl=0,
            xu=1,
            vtype=bool,
        )
        algorithm = api.nsga2(
            pop_size=self.population_size,
            sampling=sampling,
            crossover=pymoo_crossover,
            mutation=pymoo_mutation,
            repair=_CandidateRepair(
                engine=self,
                context=context,
                generator=context.random_generator("search.pymoo_nsga2.repair"),
            ),
            eliminate_duplicates=self.eliminate_duplicates,
            seed=context.random_streams.seed,
            termination=api.no_termination(),
        )
        setup_started = perf_counter()
        algorithm.setup(problem)
        self._record_timing(
            "engine.pymoo_nsga2.setup",
            perf_counter() - setup_started,
        )

        self._specification = specification
        self._context = context
        self._algorithm = algorithm
        self._pending_population = None
        self._awaiting_candidates = ()
        self._objective_schema = tuple(
            (objective.component_name, objective.direction)
            for objective in specification.objectives
        )
        self._constraint_schema = tuple(
            constraint.component_name for constraint in specification.constraints
        )
        self._variation_operation = f"{crossover.component_name}+{mutation.component_name}"
        self._evaluation_by_vector = {}
        self._evaluation_count = 0
        self._accepted_count = 0
        self._next_candidate_sequence = 0

    def ask(self, count: int) -> Sequence[Candidate]:
        specification, context, algorithm = self._require_initialized()
        if count < 1:
            raise ValueError("ask count must be positive")
        if self._awaiting_candidates:
            raise ComponentError("tell must be called before asking for another batch")
        remaining = specification.termination.remaining(self._evaluation_count)
        if remaining == 0:
            return ()
        requested = min(count, self.population_size)
        if remaining is not None:
            requested = min(requested, remaining)
        if algorithm.is_initialized:
            algorithm.n_offsprings = requested
        generation_started = perf_counter()
        population = algorithm.ask()
        self._record_timing(
            "engine.pymoo_nsga2.candidate_generation",
            perf_counter() - generation_started,
        )
        if population is None or len(population) == 0:
            raise ComponentError("pymoo_nsga2 could not generate an offspring batch")
        if len(population) > requested:
            raise ComponentError(
                "pymoo_nsga2 generated more candidates than the active evaluation budget permits"
            )

        generation = 0 if not algorithm.is_initialized else int(algorithm.n_iter or 1)
        candidates: list[Candidate] = []
        validity_seconds = 0.0
        for individual in population:
            vector = np.asarray(individual.X, dtype=np.bool_)
            self._validate_vector_shape(vector, context)
            sequence = self._next_candidate_sequence
            self._next_candidate_sequence += 1
            operation = "initial_population" if generation == 0 else self._variation_operation
            candidate = self._decode(
                vector,
                context,
                sequence=sequence,
                generation=generation,
                operation=operation,
            )
            validity_started = perf_counter()
            context.candidate_validity_policy.validate(candidate, context.dataset)
            validity_seconds += perf_counter() - validity_started
            candidates.append(candidate)
        self._record_timing(
            f"validity.{context.candidate_validity_policy.component_name}",
            validity_seconds,
        )

        self._pending_population = population
        self._awaiting_candidates = tuple(candidates)
        return self._awaiting_candidates

    def tell(self, evaluations: Sequence[Evaluation]) -> SearchUpdate:
        _, context, algorithm = self._require_initialized()
        population = self._pending_population
        if population is None or not self._awaiting_candidates:
            raise ComponentError("ask must be called before tell")
        awaiting_by_identifier = {
            candidate.identifier: candidate for candidate in self._awaiting_candidates
        }
        evaluation_by_identifier = {
            evaluation.candidate.identifier: evaluation for evaluation in evaluations
        }
        if (
            len(evaluation_by_identifier) != len(evaluations)
            or set(evaluation_by_identifier) != set(awaiting_by_identifier)
            or any(
                evaluation_by_identifier[identifier].candidate != candidate
                for identifier, candidate in awaiting_by_identifier.items()
            )
        ):
            raise ComponentError("tell evaluations must match the preceding asked batch exactly")

        ordered = tuple(
            evaluation_by_identifier[candidate.identifier]
            for candidate in self._awaiting_candidates
        )
        objective_values = np.empty(
            (len(ordered), len(self._objective_schema)),
            dtype=np.float64,
        )
        constraints = np.zeros(
            (len(ordered), 1 + len(self._constraint_schema)),
            dtype=np.float64,
        )
        for row, evaluation in enumerate(ordered):
            validate_objective_schema(evaluation, self._objective_schema)
            validate_constraint_schema(evaluation, self._constraint_schema)
            for column, objective in enumerate(evaluation.objectives):
                objective_values[row, column] = (
                    objective.value
                    if objective.direction is ObjectiveDirection.MINIMIZE
                    else -objective.value
                )
            constraints[row, 0] = 0.0 if evaluation.valid else 1.0
            for column, constraint in enumerate(evaluation.constraints, start=1):
                constraints[row, column] = constraint.value
        population.set("F", objective_values)
        population.set("G", constraints)
        for individual, evaluation in zip(
            population,
            ordered,
            strict=True,
        ):
            individual.set("salvi_evaluation", evaluation)
        algorithm.evaluator.n_eval += len(ordered)
        survivor_started = perf_counter()
        algorithm.tell(infills=population)
        self._record_timing(
            "engine.pymoo_nsga2.survivor_selection",
            perf_counter() - survivor_started,
        )
        evaluation_by_vector: dict[bytes, Evaluation] = {}
        for individual in algorithm.pop:
            evaluation = individual.get("salvi_evaluation")
            if not isinstance(evaluation, Evaluation):
                raise ComponentError(
                    "pymoo_nsga2 lost the SALVI evaluation associated with a survivor"
                )
            vector = np.asarray(individual.X, dtype=np.bool_)
            self._validate_vector_shape(vector, context)
            evaluation_by_vector[self._vector_key(vector)] = evaluation
        self._evaluation_by_vector = evaluation_by_vector

        self._evaluation_count += len(evaluations)
        self._accepted_count += sum(evaluation.feasible for evaluation in evaluations)
        self._pending_population = None
        self._awaiting_candidates = ()
        return SearchUpdate(outcomes=())

    def drain_component_timings(self) -> tuple[tuple[str, float], ...]:
        timings = tuple(sorted(self._component_timings.items()))
        self._component_timings.clear()
        return timings

    def _record_timing(self, name: str, duration: float) -> None:
        self._component_timings[name] = self._component_timings.get(name, 0.0) + duration

    def finished(self) -> bool:
        return self._specification is not None and self._specification.termination.should_stop(
            self._evaluation_count
        )

    def result(self) -> Repertoire:
        _, _, algorithm = self._require_initialized()
        if self._awaiting_candidates:
            raise ComponentError("cannot obtain a result while a batch is awaiting evaluation")
        if algorithm.opt is None:
            return Repertoire()
        evaluations: dict[str, Evaluation] = {}
        for individual in algorithm.opt:
            evaluation = individual.get("salvi_evaluation")
            if not isinstance(evaluation, Evaluation):
                raise ComponentError(
                    "pymoo_nsga2 lost the SALVI evaluation associated with its Pareto front"
                )
            evaluations[evaluation.candidate.bicluster.signature] = evaluation
        return Repertoire(
            evaluations=tuple(
                sorted(
                    evaluations.values(),
                    key=lambda evaluation: evaluation.candidate.identifier,
                )
            )
        )

    def progress(self) -> SearchProgress:
        _, _, algorithm = self._require_initialized()
        repertoire_size = 0 if algorithm.opt is None else len(algorithm.opt)
        return SearchProgress(
            evaluations=self._evaluation_count,
            accepted=self._accepted_count,
            rejected=self._evaluation_count - self._accepted_count,
            occupied_cells=0,
            repertoire_size=repertoire_size,
        )

    def checkpoint(self) -> SearchCheckpoint:
        raise ComponentError("pymoo_nsga2 does not support checkpoints")

    def restore(self, checkpoint: SearchCheckpoint) -> None:
        del checkpoint
        raise ComponentError("pymoo_nsga2 does not support checkpoint resumption")

    def _encode(self, candidate: Candidate, context: RunContext) -> BooleanVector:
        return self._encode_bicluster(candidate.bicluster, context)

    @staticmethod
    def _encode_bicluster(bicluster: Bicluster, context: RunContext) -> BooleanVector:
        vector = np.zeros(
            context.dataset.row_count + context.dataset.column_count,
            dtype=np.bool_,
        )
        vector[np.asarray(bicluster.row_indices, dtype=np.int64)] = True
        column_offset = context.dataset.row_count
        vector[column_offset + np.asarray(bicluster.column_indices, dtype=np.int64)] = True
        return vector

    @staticmethod
    def _vector_key(vector: BooleanVector) -> bytes:
        return np.packbits(vector, bitorder="little").tobytes()

    def _evaluation_for_vector(
        self,
        vector: BooleanVector,
        context: RunContext,
    ) -> Evaluation:
        self._validate_vector_shape(vector, context)
        evaluation = self._evaluation_by_vector.get(self._vector_key(vector))
        if evaluation is None:
            raise ComponentError(
                "pymoo_nsga2 could not recover the SALVI evaluation for a crossover parent"
            )
        return evaluation

    def _decode(
        self,
        vector: BooleanVector,
        context: RunContext,
        *,
        sequence: int,
        generation: int,
        operation: str,
    ) -> Candidate:
        row_count = context.dataset.row_count
        rows = tuple(int(index) for index in np.flatnonzero(vector[:row_count]))
        columns = tuple(int(index) for index in np.flatnonzero(vector[row_count:]))
        return Candidate(
            identifier=f"{self.component_name}-{sequence:012d}",
            generation=generation,
            bicluster=Bicluster(
                row_indices=rows,
                column_indices=columns,
            ),
            provenance=CandidateProvenance(
                producer=self.component_name,
                operation=operation,
                sequence=sequence,
            ),
        )

    def _repair_vector(
        self,
        raw_vector: BooleanVector,
        context: RunContext,
        generator: np.random.Generator,
    ) -> BooleanVector:
        expected = context.dataset.row_count + context.dataset.column_count
        if raw_vector.ndim != 1 or raw_vector.size != expected:
            raise ComponentError("pymoo_nsga2 produced a vector with an invalid shape")
        vector = raw_vector.copy()
        bounds = context.candidate_validity_policy.bounds(context.dataset)
        row_count = context.dataset.row_count
        self._activate_minimum(
            vector[:row_count],
            bounds.min_rows,
            generator,
        )
        self._activate_minimum(
            vector[row_count:],
            bounds.min_columns,
            generator,
        )
        return vector

    @staticmethod
    def _validate_vector_shape(vector: BooleanVector, context: RunContext) -> None:
        expected = context.dataset.row_count + context.dataset.column_count
        if vector.ndim != 1 or vector.size != expected:
            raise ComponentError("pymoo_nsga2 produced a vector with an invalid shape")

    @staticmethod
    def _activate_minimum(
        mask: BooleanVector,
        minimum: int,
        generator: np.random.Generator,
    ) -> None:
        selected = int(np.count_nonzero(mask))
        missing = minimum - selected
        if missing <= 0:
            return
        available = np.flatnonzero(~mask)
        if available.size < missing:
            raise ComponentError("candidate repair cannot satisfy minimum cardinality")
        additions = generator.choice(available, size=missing, replace=False)
        mask[np.asarray(additions, dtype=np.int64)] = True

    def _require_initialized(self) -> tuple[RunSpecification, RunContext, Any]:
        if self._specification is None or self._context is None or self._algorithm is None:
            raise ComponentError("search engine is not initialized")
        return self._specification, self._context, self._algorithm


__all__ = [
    "PymooNsga2Configuration",
    "PymooNsga2SearchEngine",
]
