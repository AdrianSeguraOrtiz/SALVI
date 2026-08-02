"""Public composition API."""

from salvi.api.evaluation import ScientificEvaluationBatch, ScientificEvaluationService
from salvi.api.execution import InMemoryRunResult, execute_in_memory
from salvi.api.run import RunSpecification, SalviRun, SalviRunBuilder

__all__ = [
    "InMemoryRunResult",
    "RunSpecification",
    "SalviRun",
    "SalviRunBuilder",
    "ScientificEvaluationBatch",
    "ScientificEvaluationService",
    "execute_in_memory",
]
