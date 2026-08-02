"""Stable domain enumerations."""

from enum import StrEnum


class PatternKind(StrEnum):
    CONSTANT = "CONSTANT"
    ADDITIVE = "ADDITIVE"
    MULTIPLICATIVE = "MULTIPLICATIVE"


class PatternScope(StrEnum):
    COLUMN = "COLUMN"
    SUBSET = "SUBSET"


class ParameterScale(StrEnum):
    RAW = "RAW"
    CATEGORY_LABEL = "CATEGORY_LABEL"
    ROBUST_STANDARDIZED = "ROBUST_STANDARDIZED"
    ROBUST_SCALED = "ROBUST_SCALED"
    NONE = "NONE"


class ObjectiveDirection(StrEnum):
    MINIMIZE = "MINIMIZE"
    MAXIMIZE = "MAXIMIZE"


class DescriptorValueKind(StrEnum):
    INTEGER = "INTEGER"
    CONTINUOUS = "CONTINUOUS"


class BinningStrategy(StrEnum):
    LINEAR = "LINEAR"
    GEOMETRIC = "GEOMETRIC"
    EXACT = "EXACT"
    CUSTOM = "CUSTOM"


class ArchiveInsertionStatus(StrEnum):
    INSERTED = "INSERTED"
    INSERTED_WITH_EVICTIONS = "INSERTED_WITH_EVICTIONS"
    REJECTED_INVALID = "REJECTED_INVALID"
    REJECTED_DUPLICATE = "REJECTED_DUPLICATE"
    REJECTED_DOMINATED = "REJECTED_DOMINATED"
    REJECTED_CAPACITY = "REJECTED_CAPACITY"
    REJECTED_OUT_OF_BOUNDS = "REJECTED_OUT_OF_BOUNDS"


class ColumnKind(StrEnum):
    NUMERIC = "NUMERIC"
    BOOLEAN = "BOOLEAN"
    CATEGORICAL = "CATEGORICAL"


class RunStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EvaluationIntegrationMode(StrEnum):
    """Order in which completed evaluations enter the search engine."""

    DETERMINISTIC = "DETERMINISTIC"
    THROUGHPUT = "THROUGHPUT"


class EvaluationIssueCode(StrEnum):
    INVALID_STRUCTURE = "INVALID_STRUCTURE"
    UNSUPPORTED_COLUMN_KIND = "UNSUPPORTED_COLUMN_KIND"
    INSUFFICIENT_LOCAL_SUPPORT = "INSUFFICIENT_LOCAL_SUPPORT"
    INSUFFICIENT_GROUP_SUPPORT = "INSUFFICIENT_GROUP_SUPPORT"
    INSUFFICIENT_BACKGROUND = "INSUFFICIENT_BACKGROUND"
    PATTERN_FIT_FAILED = "PATTERN_FIT_FAILED"
    PATTERN_UNASSIGNED = "PATTERN_UNASSIGNED"


class EventType(StrEnum):
    RUN_STARTED = "run.started"
    CONFIGURATION_VALIDATED = "configuration.validated"
    DATASET_VALIDATED = "dataset.validated"
    COMPONENTS_BUILT = "components.built"
    DATASET_PREPARED = "dataset.prepared"
    ENGINE_INITIALIZED = "engine.initialized"
    CANDIDATES_ASKED = "candidates.asked"
    EVALUATION_BATCH_STARTED = "evaluation.batch.started"
    CANDIDATES_EVALUATED = "candidates.evaluated"
    ARCHIVE_UPDATED = "archive.updated"
    EMITTER_CREDIT_UPDATED = "emitter.credit.updated"
    SCHEDULER_ALLOCATION_UPDATED = "scheduler.allocation.updated"
    ENGINE_UPDATED = "engine.updated"
    CHECKPOINT_WRITTEN = "checkpoint.written"
    FINAL_SELECTION_COMPLETED = "final_selection.completed"
    ARTIFACT_WRITTEN = "artifact.written"
    PROGRESS = "run.progress"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    RUN_CANCELLED = "run.cancelled"
    WARNING = "run.warning"
