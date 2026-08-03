"""Public pattern inference API."""

from salvi.patterns.catalog import PatternCatalog, default_pattern_catalog
from salvi.patterns.configuration import PatternConfiguration
from salvi.patterns.contracts import (
    BatchColumnPatternFitter,
    ColumnPatternFitter,
    GroupPatternCandidateGenerator,
    GroupPatternFitter,
    GroupPatternProposal,
    MixedPatternAssignmentStrategy,
    PatternContrastEvaluator,
    PatternContrastStrategy,
    PatternDefinition,
    PatternImplementation,
    PatternInferenceEngine,
    PatternSeed,
    PatternSeedShape,
    PatternSeedStrategy,
)
from salvi.patterns.fitters import (
    AdditivePatternFitter,
    ConstantPatternFitter,
    MultiplicativePatternFitter,
)
from salvi.patterns.inference import DefaultPatternInferenceEngine, IterativeBestFitAssignment

__all__ = [
    "AdditivePatternFitter",
    "BatchColumnPatternFitter",
    "ColumnPatternFitter",
    "ConstantPatternFitter",
    "DefaultPatternInferenceEngine",
    "GroupPatternCandidateGenerator",
    "GroupPatternFitter",
    "GroupPatternProposal",
    "IterativeBestFitAssignment",
    "MixedPatternAssignmentStrategy",
    "MultiplicativePatternFitter",
    "PatternCatalog",
    "PatternConfiguration",
    "PatternContrastEvaluator",
    "PatternContrastStrategy",
    "PatternDefinition",
    "PatternImplementation",
    "PatternInferenceEngine",
    "PatternSeed",
    "PatternSeedShape",
    "PatternSeedStrategy",
    "default_pattern_catalog",
]
