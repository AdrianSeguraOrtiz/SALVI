"""Component protocols and the explicit component registry."""

from salvi.components.advanced_emitters import (
    AlternatingPatternLocalSearchEmitter,
)
from salvi.components.candidate_initialization import (
    CellCoveragePatternAwareInitializer,
    PatternAwareInitializer,
    StratifiedInitializer,
    UniformRandomInitializer,
)
from salvi.components.catalog import (
    ComponentDescription,
    ComponentMaturity,
    ComponentParameterDescription,
    ComponentReference,
    MetricPopulation,
    MetricTemporalScope,
    MetricValueKind,
    ObserverMetricGroupPresentation,
    ObserverMetricPresentation,
)
from salvi.components.constraints import (
    BalancedBiclusterSizeRange,
    MaximumInternalCoherence,
)
from salvi.components.descriptors import ColumnCardinality, RowCardinality
from salvi.components.evaluation_policies import MinimumCardinality, MinimumObservedSupport
from salvi.components.execution import (
    ProcessPoolEvaluationExecutor,
    SerialEvaluationExecutor,
    ThreadPoolEvaluationExecutor,
)
from salvi.components.final_selection import ContainmentMarginalQualitySelector
from salvi.components.mate_selection import (
    CellFirstEvidenceCompatibleMateSelection,
    RepertoireRandomMateSelection,
)
from salvi.components.membership_emitters import (
    CellCoverageRestartEmitter,
    MembershipMoveEmitter,
    RandomMoveEmitter,
    RestartEmitter,
    ShapeMoveEmitter,
)
from salvi.components.objectives import (
    BalancedBiclusterSize,
    Contrast,
    InternalCoherence,
)
from salvi.components.observers import (
    ArchiveCoverageObserver,
    ArchiveDescriptorDistributionObserver,
    CandidateDiversityObserver,
    CandidateOutcomesObserver,
    ComponentTimingObserver,
    DescriptorDistributionObserver,
    EmitterCreditObserver,
    ObjectiveDistributionObserver,
    ResourceUsageObserver,
    RuntimeThroughputObserver,
    SearchProgressObserver,
)
from salvi.components.operators import (
    BitFlipMembershipMutation,
    EvidenceWeightedRecombinationCrossover,
    HalfUniformMembershipCrossover,
    MembershipRecombinationCrossover,
)
from salvi.components.parent_selection import (
    CellUniformQualityParentSelection,
    RepertoireUniformParentSelection,
)
from salvi.components.preprocessing import (
    DropAllMissingColumns,
    MedianModeImputation,
    MissingnessIndicators,
    PreserveMissingValues,
    RejectMissingValues,
    RobustNumericScaling,
)
from salvi.components.protocols import Component, ComponentKind, ComponentTimingSource, Constraint
from salvi.components.registry import ComponentRegistration, ComponentRegistry
from salvi.components.schedulers import (
    AdaptiveCreditScheduler,
    CellBalancedAdaptiveCreditScheduler,
    FirstEmitterScheduler,
    FixedProportionScheduler,
)
from salvi.components.termination import EvaluationBudget
from salvi.components.variation_emitters import CrossoverEmitter, MutationEmitter

__all__ = [
    "AdaptiveCreditScheduler",
    "AlternatingPatternLocalSearchEmitter",
    "ArchiveCoverageObserver",
    "ArchiveDescriptorDistributionObserver",
    "BalancedBiclusterSize",
    "BalancedBiclusterSizeRange",
    "BitFlipMembershipMutation",
    "CandidateDiversityObserver",
    "CandidateOutcomesObserver",
    "CellBalancedAdaptiveCreditScheduler",
    "CellCoveragePatternAwareInitializer",
    "CellCoverageRestartEmitter",
    "CellFirstEvidenceCompatibleMateSelection",
    "CellUniformQualityParentSelection",
    "ColumnCardinality",
    "Component",
    "ComponentDescription",
    "ComponentKind",
    "ComponentMaturity",
    "ComponentParameterDescription",
    "ComponentReference",
    "ComponentRegistration",
    "ComponentRegistry",
    "ComponentTimingObserver",
    "ComponentTimingSource",
    "Constraint",
    "ContainmentMarginalQualitySelector",
    "Contrast",
    "CrossoverEmitter",
    "DescriptorDistributionObserver",
    "DropAllMissingColumns",
    "EmitterCreditObserver",
    "EvaluationBudget",
    "EvidenceWeightedRecombinationCrossover",
    "FirstEmitterScheduler",
    "FixedProportionScheduler",
    "HalfUniformMembershipCrossover",
    "InternalCoherence",
    "MaximumInternalCoherence",
    "MedianModeImputation",
    "MembershipMoveEmitter",
    "MembershipRecombinationCrossover",
    "MetricPopulation",
    "MetricTemporalScope",
    "MetricValueKind",
    "MinimumCardinality",
    "MinimumObservedSupport",
    "MissingnessIndicators",
    "MutationEmitter",
    "ObjectiveDistributionObserver",
    "ObserverMetricGroupPresentation",
    "ObserverMetricPresentation",
    "PatternAwareInitializer",
    "PreserveMissingValues",
    "ProcessPoolEvaluationExecutor",
    "RandomMoveEmitter",
    "RejectMissingValues",
    "RepertoireRandomMateSelection",
    "RepertoireUniformParentSelection",
    "ResourceUsageObserver",
    "RestartEmitter",
    "RobustNumericScaling",
    "RowCardinality",
    "RuntimeThroughputObserver",
    "SearchProgressObserver",
    "SerialEvaluationExecutor",
    "ShapeMoveEmitter",
    "StratifiedInitializer",
    "ThreadPoolEvaluationExecutor",
    "UniformRandomInitializer",
]
