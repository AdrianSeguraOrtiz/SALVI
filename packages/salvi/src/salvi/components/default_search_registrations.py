"""Composition of the built-in component registry."""

from __future__ import annotations

from typing import cast

from salvi.components.advanced_emitters import (
    AlternatingPatternLocalSearchEmitter,
    AlternatingPatternLocalSearchEmitterConfiguration,
)
from salvi.components.catalog import ComponentMaturity
from salvi.components.configuration import EmptyConfiguration
from salvi.components.constraints import (
    BalancedBiclusterSizeRange,
    BalancedBiclusterSizeRangeConfiguration,
    MaximumInternalCoherence,
    MaximumInternalCoherenceConfiguration,
)
from salvi.components.default_registration import (
    _membership_registration,
    _registration,
)
from salvi.components.descriptors import ColumnCardinality, RowCardinality
from salvi.components.mate_selection import (
    CellFirstEvidenceCompatibleMateSelection,
    CellFirstEvidenceCompatibleMateSelectionConfiguration,
    RepertoireRandomMateSelection,
)
from salvi.components.membership_emitters import (
    CellCoverageRestartEmitter,
    CellCoverageRestartEmitterConfiguration,
    RandomMoveEmitter,
    RestartEmitter,
    RestartEmitterConfiguration,
    ShapeMoveEmitter,
    ShapeMoveEmitterConfiguration,
)
from salvi.components.objectives import (
    BalancedBiclusterSize,
    Contrast,
    ContrastConfiguration,
    InternalCoherence,
)
from salvi.components.operators import (
    BitFlipMembershipMutation,
    BitFlipMutationConfiguration,
    EvidenceWeightedRecombinationCrossover,
    HalfUniformMembershipCrossover,
    MembershipRecombinationConfiguration,
    MembershipRecombinationCrossover,
)
from salvi.components.parent_selection import (
    CellUniformQualityParentSelection,
    RepertoireUniformParentSelection,
)
from salvi.components.protocols import ComponentKind
from salvi.components.registry import ComponentRegistration
from salvi.components.schedulers import (
    AdaptiveCreditScheduler,
    AdaptiveCreditSchedulerConfiguration,
    CellBalancedAdaptiveCreditScheduler,
    CellBalancedAdaptiveCreditSchedulerConfiguration,
    FirstEmitterScheduler,
    FixedProportionScheduler,
    FixedProportionSchedulerConfiguration,
)
from salvi.components.variation_emitters import (
    CrossoverEmitter,
    CrossoverEmitterConfiguration,
    MutationEmitter,
    MutationEmitterConfiguration,
)
from salvi.domain.enums import PatternKind
from salvi.engine.archive import (
    DeepGridMomeArchive,
    DeepGridMomeConfiguration,
)


def default_search_registrations() -> tuple[ComponentRegistration, ...]:
    return (
        _registration(
            ComponentKind.OBJECTIVE,
            "internal_coherence",
            EmptyConfiguration,
            lambda _: InternalCoherence(),
            compatibility_notes=(
                "Requires a numeric transformation that provides robust-numeric-data.",
            ),
        ),
        _registration(
            ComponentKind.OBJECTIVE,
            "contrast",
            ContrastConfiguration,
            lambda config: Contrast(
                min_background_ratio=cast(ContrastConfiguration, config).min_background_ratio
            ),
            compatibility_notes=(
                "Requires a numeric transformation that provides robust-numeric-data.",
            ),
        ),
        _registration(
            ComponentKind.OBJECTIVE,
            "balanced_bicluster_size",
            EmptyConfiguration,
            lambda _: BalancedBiclusterSize(),
        ),
        _registration(
            ComponentKind.CONSTRAINT,
            "balanced_bicluster_size_range",
            BalancedBiclusterSizeRangeConfiguration,
            lambda config: BalancedBiclusterSizeRange(
                minimum=cast(BalancedBiclusterSizeRangeConfiguration, config).minimum,
                maximum=cast(BalancedBiclusterSizeRangeConfiguration, config).maximum,
            ),
        ),
        _registration(
            ComponentKind.CONSTRAINT,
            "maximum_internal_coherence",
            MaximumInternalCoherenceConfiguration,
            lambda config: MaximumInternalCoherence(
                maximum_error=cast(MaximumInternalCoherenceConfiguration, config).maximum_error,
            ),
            compatibility_notes=(
                "Requires a numeric transformation that provides robust-numeric-data.",
            ),
        ),
        _registration(
            ComponentKind.DESCRIPTOR,
            "row_cardinality",
            EmptyConfiguration,
            lambda _: RowCardinality(),
        ),
        _registration(
            ComponentKind.DESCRIPTOR,
            "column_cardinality",
            EmptyConfiguration,
            lambda _: ColumnCardinality(),
        ),
        _registration(
            ComponentKind.ARCHIVE,
            "deep_grid_mome",
            DeepGridMomeConfiguration,
            lambda config: DeepGridMomeArchive(
                axes=cast(DeepGridMomeConfiguration, config).axes,
                cell_capacity=cast(DeepGridMomeConfiguration, config).cell_capacity,
            ),
        ),
        _registration(
            ComponentKind.PARENT_SELECTION_POLICY,
            "repertoire_uniform",
            EmptyConfiguration,
            lambda _: RepertoireUniformParentSelection(),
        ),
        _registration(
            ComponentKind.PARENT_SELECTION_POLICY,
            "cell_uniform_quality",
            EmptyConfiguration,
            lambda _: CellUniformQualityParentSelection(),
        ),
        _registration(
            ComponentKind.MATE_SELECTION_POLICY,
            "repertoire_random",
            EmptyConfiguration,
            lambda _: RepertoireRandomMateSelection(),
        ),
        _registration(
            ComponentKind.MATE_SELECTION_POLICY,
            "cell_first_evidence_compatible",
            CellFirstEvidenceCompatibleMateSelectionConfiguration,
            lambda config: CellFirstEvidenceCompatibleMateSelection(
                parent_pool_size=cast(
                    CellFirstEvidenceCompatibleMateSelectionConfiguration, config
                ).parent_pool_size,
                mate_pool_size=cast(
                    CellFirstEvidenceCompatibleMateSelectionConfiguration, config
                ).mate_pool_size,
                minimum_row_jaccard=cast(
                    CellFirstEvidenceCompatibleMateSelectionConfiguration, config
                ).minimum_row_jaccard,
                minimum_column_jaccard=cast(
                    CellFirstEvidenceCompatibleMateSelectionConfiguration, config
                ).minimum_column_jaccard,
                cell_neighborhood_radius=cast(
                    CellFirstEvidenceCompatibleMateSelectionConfiguration, config
                ).cell_neighborhood_radius,
            ),
            compatibility_notes=(
                "Chooses source cells before local quality and structural compatibility.",
                "Both row and column Jaccard thresholds must be met.",
            ),
        ),
        _registration(
            ComponentKind.CROSSOVER_OPERATOR,
            "membership_recombination",
            MembershipRecombinationConfiguration,
            lambda config: MembershipRecombinationCrossover(
                application_probability=cast(
                    MembershipRecombinationConfiguration, config
                ).application_probability,
                row_exchange_probability=cast(
                    MembershipRecombinationConfiguration, config
                ).row_exchange_probability,
                column_exchange_probability=cast(
                    MembershipRecombinationConfiguration, config
                ).column_exchange_probability,
            ),
        ),
        _registration(
            ComponentKind.CROSSOVER_OPERATOR,
            "evidence_weighted_recombination",
            MembershipRecombinationConfiguration,
            lambda config: EvidenceWeightedRecombinationCrossover(
                application_probability=cast(
                    MembershipRecombinationConfiguration, config
                ).application_probability,
                row_exchange_probability=cast(
                    MembershipRecombinationConfiguration, config
                ).row_exchange_probability,
                column_exchange_probability=cast(
                    MembershipRecombinationConfiguration, config
                ).column_exchange_probability,
            ),
        ),
        _registration(
            ComponentKind.CROSSOVER_OPERATOR,
            "half_uniform_membership",
            MembershipRecombinationConfiguration,
            lambda config: HalfUniformMembershipCrossover(
                application_probability=cast(
                    MembershipRecombinationConfiguration, config
                ).application_probability,
                row_exchange_probability=cast(
                    MembershipRecombinationConfiguration, config
                ).row_exchange_probability,
                column_exchange_probability=cast(
                    MembershipRecombinationConfiguration, config
                ).column_exchange_probability,
            ),
        ),
        _registration(
            ComponentKind.MUTATION_OPERATOR,
            "bit_flip_membership",
            BitFlipMutationConfiguration,
            lambda config: BitFlipMembershipMutation(
                application_probability=cast(
                    BitFlipMutationConfiguration, config
                ).application_probability,
                bit_probability=cast(BitFlipMutationConfiguration, config).bit_probability,
            ),
        ),
        _registration(
            ComponentKind.EMITTER,
            "random_move",
            EmptyConfiguration,
            lambda _: RandomMoveEmitter(),
        ),
        _membership_registration("rows", "add"),
        _membership_registration("rows", "remove"),
        _membership_registration("rows", "swap"),
        _membership_registration("columns", "add"),
        _membership_registration("columns", "remove"),
        _membership_registration("columns", "swap"),
        _registration(
            ComponentKind.EMITTER,
            "shape_move",
            ShapeMoveEmitterConfiguration,
            lambda config: ShapeMoveEmitter(
                guided=cast(ShapeMoveEmitterConfiguration, config).guided,
                parent_pool_size=cast(ShapeMoveEmitterConfiguration, config).parent_pool_size,
                candidate_pool_size=cast(ShapeMoveEmitterConfiguration, config).candidate_pool_size,
            ),
        ),
        _registration(
            ComponentKind.EMITTER,
            "crossover",
            CrossoverEmitterConfiguration,
            lambda config: CrossoverEmitter(
                max_attempts=cast(CrossoverEmitterConfiguration, config).max_attempts,
            ),
        ),
        _registration(
            ComponentKind.EMITTER,
            "mutation",
            MutationEmitterConfiguration,
            lambda config: MutationEmitter(
                guided_parent_selection=cast(
                    MutationEmitterConfiguration, config
                ).guided_parent_selection,
                parent_pool_size=cast(MutationEmitterConfiguration, config).parent_pool_size,
                max_attempts=cast(MutationEmitterConfiguration, config).max_attempts,
            ),
        ),
        _registration(
            ComponentKind.EMITTER,
            "restart",
            RestartEmitterConfiguration,
            lambda config: RestartEmitter(
                strategy=cast(RestartEmitterConfiguration, config).strategy,
                cardinality_levels=cast(RestartEmitterConfiguration, config).cardinality_levels,
                joint_column_candidate_pool_size=cast(
                    RestartEmitterConfiguration, config
                ).joint_column_candidate_pool_size,
            ),
            parameter_patterns=(
                (
                    "joint_column_candidate_pool_size",
                    frozenset({PatternKind.ADDITIVE, PatternKind.MULTIPLICATIVE}),
                ),
            ),
        ),
        _registration(
            ComponentKind.EMITTER,
            "cell_coverage_restart",
            CellCoverageRestartEmitterConfiguration,
            lambda config: CellCoverageRestartEmitter(
                joint_column_candidate_pool_size=cast(
                    CellCoverageRestartEmitterConfiguration, config
                ).joint_column_candidate_pool_size,
            ),
            parameter_patterns=(
                (
                    "joint_column_candidate_pool_size",
                    frozenset({PatternKind.ADDITIVE, PatternKind.MULTIPLICATIVE}),
                ),
            ),
            compatibility_notes=("Requires row- and column-cardinality archive axes.",),
        ),
        _registration(
            ComponentKind.EMITTER,
            "alternating_pattern_local_search",
            AlternatingPatternLocalSearchEmitterConfiguration,
            lambda config: AlternatingPatternLocalSearchEmitter(
                parent_pool_size=cast(
                    AlternatingPatternLocalSearchEmitterConfiguration, config
                ).parent_pool_size,
                candidate_pool_size=cast(
                    AlternatingPatternLocalSearchEmitterConfiguration, config
                ).candidate_pool_size,
                cardinality_change_probability=cast(
                    AlternatingPatternLocalSearchEmitterConfiguration, config
                ).cardinality_change_probability,
                quality_parent_probability=cast(
                    AlternatingPatternLocalSearchEmitterConfiguration, config
                ).quality_parent_probability,
            ),
            maturity=ComponentMaturity.EXPERIMENTAL,
        ),
        _registration(
            ComponentKind.SCHEDULER,
            "first",
            EmptyConfiguration,
            lambda _: FirstEmitterScheduler(),
        ),
        _registration(
            ComponentKind.SCHEDULER,
            "fixed_proportion",
            FixedProportionSchedulerConfiguration,
            lambda config: FixedProportionScheduler(
                shares=dict(cast(FixedProportionSchedulerConfiguration, config).shares)
            ),
            prototype_component=FixedProportionScheduler(shares={"random_move": 1.0}),
        ),
        _registration(
            ComponentKind.SCHEDULER,
            "adaptive_credit",
            AdaptiveCreditSchedulerConfiguration,
            lambda config: AdaptiveCreditScheduler(
                exploration_weight=cast(
                    AdaptiveCreditSchedulerConfiguration, config
                ).exploration_weight,
                new_cell_reward=cast(AdaptiveCreditSchedulerConfiguration, config).new_cell_reward,
                insertion_reward=cast(
                    AdaptiveCreditSchedulerConfiguration, config
                ).insertion_reward,
            ),
        ),
        _registration(
            ComponentKind.SCHEDULER,
            "cell_balanced_adaptive_credit",
            CellBalancedAdaptiveCreditSchedulerConfiguration,
            lambda config: CellBalancedAdaptiveCreditScheduler(
                exploration_weight=cast(
                    CellBalancedAdaptiveCreditSchedulerConfiguration, config
                ).exploration_weight,
                new_cell_reward=cast(
                    CellBalancedAdaptiveCreditSchedulerConfiguration, config
                ).new_cell_reward,
                insertion_reward=cast(
                    CellBalancedAdaptiveCreditSchedulerConfiguration, config
                ).insertion_reward,
                underexplored_cell_weight=cast(
                    CellBalancedAdaptiveCreditSchedulerConfiguration, config
                ).underexplored_cell_weight,
            ),
        ),
    )


__all__ = ["default_search_registrations"]
