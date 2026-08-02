"""Composition of the built-in component registry."""

from __future__ import annotations

from typing import cast

from pydantic import BaseModel

from salvi.components.catalog import ComponentMaturity
from salvi.components.configuration import EmptyConfiguration
from salvi.components.default_registration import (
    _registration,
)
from salvi.components.execution import (
    ProcessPoolEvaluationConfiguration,
    ProcessPoolEvaluationExecutor,
    SerialEvaluationExecutor,
    ThreadPoolEvaluationConfiguration,
    ThreadPoolEvaluationExecutor,
)
from salvi.components.final_selection import (
    ContainmentMarginalQualityConfiguration,
    ContainmentMarginalQualitySelector,
)
from salvi.components.observers import (
    ArchiveCoverageObserver,
    ArchiveDescriptorDistributionObserver,
    CandidateDiversityObserver,
    CandidateDiversityObserverConfiguration,
    CandidateOutcomesObserver,
    ComponentTimingObserver,
    DescriptorDistributionObserver,
    DistributionObserverConfiguration,
    EmitterCreditObserver,
    EvaluationIssuesObserver,
    ObjectiveDistributionObserver,
    QDArchiveDiagnosticsObserver,
    QDArchiveDiagnosticsObserverConfiguration,
    ResourceUsageObserver,
    ResourceUsageObserverConfiguration,
    RuntimeThroughputObserver,
    SearchProgressObserver,
)
from salvi.components.protocols import ComponentKind
from salvi.components.registry import ComponentRegistration
from salvi.components.residual_selection import (
    AdaptiveResidualEvidenceCoverConfiguration,
    AdaptiveResidualEvidenceCoverSelector,
)
from salvi.components.termination import EvaluationBudget, TerminationConfiguration
from salvi.engine.mome import SerialMomeConfiguration, SerialMomeSearchEngine
from salvi.engine.pymoo import PymooNsga2Configuration, PymooNsga2SearchEngine


def _adaptive_residual_selector(config: BaseModel) -> AdaptiveResidualEvidenceCoverSelector:
    typed = cast(AdaptiveResidualEvidenceCoverConfiguration, config)
    return AdaptiveResidualEvidenceCoverSelector(
        objective_names=typed.objective_names,
        quality_scale=typed.quality_scale,
        overlap_penalty=typed.overlap_penalty,
        low_quality_penalty=typed.low_quality_penalty,
        complexity_penalty=typed.complexity_penalty,
        minimum_marginal_evidence=typed.minimum_marginal_evidence,
        maximum_dense_cells=typed.maximum_dense_cells,
        minimum_quality_floor=typed.minimum_quality_floor,
        maximum_quality_floor=typed.maximum_quality_floor,
        minimum_candidates_for_knee=typed.minimum_candidates_for_knee,
        minimum_knee_prominence=typed.minimum_knee_prominence,
        fallback_quality_quantile=typed.fallback_quality_quantile,
    )


def default_runtime_registrations() -> tuple[ComponentRegistration, ...]:
    return (
        _registration(
            ComponentKind.EVALUATION_EXECUTOR,
            "serial",
            EmptyConfiguration,
            lambda _: SerialEvaluationExecutor(),
        ),
        _registration(
            ComponentKind.EVALUATION_EXECUTOR,
            "thread_pool",
            ThreadPoolEvaluationConfiguration,
            lambda config: ThreadPoolEvaluationExecutor(
                configured_integration_mode=cast(
                    ThreadPoolEvaluationConfiguration, config
                ).integration_mode,
                configured_max_in_flight=cast(
                    ThreadPoolEvaluationConfiguration, config
                ).max_in_flight,
            ),
        ),
        _registration(
            ComponentKind.EVALUATION_EXECUTOR,
            "process_pool",
            ProcessPoolEvaluationConfiguration,
            lambda config: ProcessPoolEvaluationExecutor(
                configured_integration_mode=cast(
                    ProcessPoolEvaluationConfiguration, config
                ).integration_mode,
                configured_max_in_flight=cast(
                    ProcessPoolEvaluationConfiguration, config
                ).max_in_flight,
            ),
        ),
        _registration(
            ComponentKind.OBSERVER,
            "search_progress",
            EmptyConfiguration,
            lambda _: SearchProgressObserver(),
        ),
        _registration(
            ComponentKind.OBSERVER,
            "archive_coverage",
            EmptyConfiguration,
            lambda _: ArchiveCoverageObserver(),
        ),
        _registration(
            ComponentKind.OBSERVER,
            "candidate_outcomes",
            DistributionObserverConfiguration,
            lambda config: CandidateOutcomesObserver(
                every_evaluations=cast(DistributionObserverConfiguration, config).every_evaluations
            ),
        ),
        _registration(
            ComponentKind.OBSERVER,
            "descriptor_distribution",
            DistributionObserverConfiguration,
            lambda config: DescriptorDistributionObserver(
                every_evaluations=cast(DistributionObserverConfiguration, config).every_evaluations
            ),
        ),
        _registration(
            ComponentKind.OBSERVER,
            "archive_descriptor_distribution",
            DistributionObserverConfiguration,
            lambda config: ArchiveDescriptorDistributionObserver(
                every_evaluations=cast(DistributionObserverConfiguration, config).every_evaluations
            ),
        ),
        _registration(
            ComponentKind.OBSERVER,
            "objective_distribution",
            DistributionObserverConfiguration,
            lambda config: ObjectiveDistributionObserver(
                every_evaluations=cast(DistributionObserverConfiguration, config).every_evaluations
            ),
        ),
        _registration(
            ComponentKind.OBSERVER,
            "emitter_credit",
            DistributionObserverConfiguration,
            lambda config: EmitterCreditObserver(
                every_evaluations=cast(DistributionObserverConfiguration, config).every_evaluations
            ),
        ),
        _registration(
            ComponentKind.OBSERVER,
            "candidate_diversity",
            CandidateDiversityObserverConfiguration,
            lambda config: CandidateDiversityObserver(
                window_size=cast(CandidateDiversityObserverConfiguration, config).window_size,
                distance_sample_size=cast(
                    CandidateDiversityObserverConfiguration, config
                ).distance_sample_size,
                row_weight=cast(CandidateDiversityObserverConfiguration, config).row_weight,
                every_evaluations=cast(
                    CandidateDiversityObserverConfiguration, config
                ).every_evaluations,
            ),
        ),
        _registration(
            ComponentKind.OBSERVER,
            "runtime_throughput",
            EmptyConfiguration,
            lambda _: RuntimeThroughputObserver(),
        ),
        _registration(
            ComponentKind.OBSERVER,
            "component_timing",
            DistributionObserverConfiguration,
            lambda config: ComponentTimingObserver(
                every_evaluations=cast(DistributionObserverConfiguration, config).every_evaluations
            ),
        ),
        _registration(
            ComponentKind.OBSERVER,
            "evaluation_issues",
            DistributionObserverConfiguration,
            lambda config: EvaluationIssuesObserver(
                every_evaluations=cast(DistributionObserverConfiguration, config).every_evaluations
            ),
        ),
        _registration(
            ComponentKind.OBSERVER,
            "qd_archive_diagnostics",
            QDArchiveDiagnosticsObserverConfiguration,
            lambda config: QDArchiveDiagnosticsObserver(
                every_evaluations=cast(
                    QDArchiveDiagnosticsObserverConfiguration, config
                ).every_evaluations,
                include_cell_metrics=cast(
                    QDArchiveDiagnosticsObserverConfiguration, config
                ).include_cell_metrics,
            ),
        ),
        _registration(
            ComponentKind.OBSERVER,
            "resource_usage",
            ResourceUsageObserverConfiguration,
            lambda config: ResourceUsageObserver(
                every_evaluations=cast(ResourceUsageObserverConfiguration, config).every_evaluations
            ),
        ),
        _registration(
            ComponentKind.TERMINATION,
            "evaluation_budget",
            TerminationConfiguration,
            lambda config: EvaluationBudget(
                max_evaluations=cast(TerminationConfiguration, config).max_evaluations
            ),
            continuation_fingerprint_exclusions=frozenset({"max_evaluations"}),
        ),
        _registration(
            ComponentKind.FINAL_SELECTOR,
            "containment_marginal_quality",
            ContainmentMarginalQualityConfiguration,
            lambda config: ContainmentMarginalQualitySelector(
                max_objective_degradation=cast(
                    ContainmentMarginalQualityConfiguration, config
                ).max_objective_degradation,
                max_degradation_per_log_area_gain=cast(
                    ContainmentMarginalQualityConfiguration, config
                ).max_degradation_per_log_area_gain,
                objective_names=cast(
                    ContainmentMarginalQualityConfiguration, config
                ).objective_names,
            ),
        ),
        _registration(
            ComponentKind.FINAL_SELECTOR,
            "adaptive_residual_evidence_cover",
            AdaptiveResidualEvidenceCoverConfiguration,
            _adaptive_residual_selector,
        ),
        _registration(
            ComponentKind.SEARCH_ENGINE,
            "serial_mome",
            SerialMomeConfiguration,
            lambda config: SerialMomeSearchEngine(
                initial_population_size=cast(
                    SerialMomeConfiguration, config
                ).initial_population_size,
                configured_batch_size=cast(SerialMomeConfiguration, config).batch_size,
            ),
            default_for_search_family=True,
        ),
        _registration(
            ComponentKind.SEARCH_ENGINE,
            "pymoo_nsga2",
            PymooNsga2Configuration,
            lambda config: PymooNsga2SearchEngine(
                population_size=cast(PymooNsga2Configuration, config).population_size,
                eliminate_duplicates=cast(PymooNsga2Configuration, config).eliminate_duplicates,
            ),
            compatibility_notes=(
                "Uses the pymoo runtime bundled with the SALVI distribution.",
                "Accepts any registered SALVI crossover operator.",
                "Mutation operators must support pre-evaluation variation through pymoo.",
                "Archives, parent selection, mate selection, emitters, and schedulers "
                "are forbidden because NSGA-II does not consume them.",
                "Checkpoint resumption is not supported.",
            ),
            maturity=ComponentMaturity.EXPERIMENTAL,
            default_for_search_family=True,
        ),
    )


__all__ = ["default_runtime_registrations"]
