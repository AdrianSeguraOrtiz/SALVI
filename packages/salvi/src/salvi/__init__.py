"""Public API for the SALVI biclustering framework."""

from salvi.api.evaluation import ScientificEvaluationBatch, ScientificEvaluationService
from salvi.api.execution import InMemoryRunResult, execute_in_memory
from salvi.api.run import SalviRun
from salvi.application.configuration import (
    PipelineConfiguration,
    RunBinding,
    SalviConfiguration,
    bind_pipeline,
    load_configuration,
    load_pipeline_configuration,
)
from salvi.application.run_service import RunService
from salvi.application.selection_service import FinalSelectionResult, FinalSelectionService
from salvi.domain.enums import ObjectiveDirection, PatternKind, SearchFamily
from salvi.domain.models import (
    Bicluster,
    Candidate,
    ConstraintValue,
    Dataset,
    Evaluation,
    ObjectiveValue,
    Repertoire,
    RunArtifact,
    RunMetric,
    RunResult,
)
from salvi.domain.prepared import PreparedDataset
from salvi.evaluation.structure import jaccard_distance, structural_distance
from salvi.exceptions import SalviError
from salvi.infrastructure.bicluster_set import (
    BiclusterSetContents,
    BiclusterSetManifest,
    BiclusterSetReader,
    BiclusterSetWriter,
)
from salvi.infrastructure.dataset_bundle import (
    DatasetBundleReader,
    DatasetBundleWriter,
    DatasetManifest,
)
from salvi.infrastructure.ground_truth import (
    GroundTruth,
    GroundTruthBicluster,
    GroundTruthColumnPattern,
)
from salvi.patterns.configuration import PatternConfiguration
from salvi.versioning import package_version

__version__ = package_version()

__all__ = [
    "Bicluster",
    "BiclusterSetContents",
    "BiclusterSetManifest",
    "BiclusterSetReader",
    "BiclusterSetWriter",
    "Candidate",
    "ConstraintValue",
    "Dataset",
    "DatasetBundleReader",
    "DatasetBundleWriter",
    "DatasetManifest",
    "Evaluation",
    "FinalSelectionResult",
    "FinalSelectionService",
    "GroundTruth",
    "GroundTruthBicluster",
    "GroundTruthColumnPattern",
    "InMemoryRunResult",
    "ObjectiveDirection",
    "ObjectiveValue",
    "PatternConfiguration",
    "PatternKind",
    "PipelineConfiguration",
    "PreparedDataset",
    "Repertoire",
    "RunArtifact",
    "RunBinding",
    "RunMetric",
    "RunResult",
    "RunService",
    "SalviConfiguration",
    "SalviError",
    "SalviRun",
    "ScientificEvaluationBatch",
    "ScientificEvaluationService",
    "SearchFamily",
    "__version__",
    "bind_pipeline",
    "execute_in_memory",
    "jaccard_distance",
    "load_configuration",
    "load_pipeline_configuration",
    "structural_distance",
]
