"""Public passive-observer facade."""

from salvi.components.diagnostic_observers import (
    CandidateDiversityObserver,
    EvaluationIssuesObserver,
    QDArchiveDiagnosticsObserver,
    ResourceUsageObserver,
)
from salvi.components.runtime_observers import (
    ArchiveCoverageObserver,
    ArchiveDescriptorDistributionObserver,
    CandidateDiversityObserverConfiguration,
    CandidateOutcomesObserver,
    ComponentTimingObserver,
    DescriptorDistributionObserver,
    DistributionObserverConfiguration,
    EmitterCreditObserver,
    ObjectiveDistributionObserver,
    QDArchiveDiagnosticsObserverConfiguration,
    ResourceUsageObserverConfiguration,
    RuntimeThroughputObserver,
    SearchProgressObserver,
)

__all__ = [
    "ArchiveCoverageObserver",
    "ArchiveDescriptorDistributionObserver",
    "CandidateDiversityObserver",
    "CandidateDiversityObserverConfiguration",
    "CandidateOutcomesObserver",
    "ComponentTimingObserver",
    "DescriptorDistributionObserver",
    "DistributionObserverConfiguration",
    "EmitterCreditObserver",
    "EvaluationIssuesObserver",
    "ObjectiveDistributionObserver",
    "QDArchiveDiagnosticsObserver",
    "QDArchiveDiagnosticsObserverConfiguration",
    "ResourceUsageObserver",
    "ResourceUsageObserverConfiguration",
    "RuntimeThroughputObserver",
    "SearchProgressObserver",
]
