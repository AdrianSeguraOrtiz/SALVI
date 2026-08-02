"""Composition of the built-in component registry."""

from __future__ import annotations

from typing import cast

from salvi.components.candidate_initialization import (
    CellCoveragePatternAwareInitializer,
    CellCoveragePatternAwareInitializerConfiguration,
    PatternAwareInitializer,
    PatternAwareInitializerConfiguration,
    StratifiedInitializer,
    StratifiedInitializerConfiguration,
    UniformRandomInitializer,
)
from salvi.components.configuration import EmptyConfiguration
from salvi.components.default_registration import (
    _registration,
)
from salvi.components.evaluation_policies import (
    MinimumCardinality,
    MinimumCardinalityConfiguration,
    MinimumObservedSupport,
    ObservedSupportConfiguration,
)
from salvi.components.preprocessing import (
    DropAllMissingColumns,
    MedianModeImputation,
    MissingnessIndicators,
    MissingnessIndicatorsConfiguration,
    PreserveMissingValues,
    RejectMissingValues,
    RobustNumericScaling,
)
from salvi.components.protocols import ComponentKind
from salvi.components.registry import ComponentRegistration
from salvi.domain.enums import PatternKind


def default_data_registrations() -> tuple[ComponentRegistration, ...]:
    return (
        _registration(
            ComponentKind.MISSING_VALUES_POLICY,
            "preserve",
            EmptyConfiguration,
            lambda _: PreserveMissingValues(),
        ),
        _registration(
            ComponentKind.MISSING_VALUES_POLICY,
            "reject",
            EmptyConfiguration,
            lambda _: RejectMissingValues(),
        ),
        _registration(
            ComponentKind.MISSING_VALUES_POLICY,
            "median_mode_imputation",
            EmptyConfiguration,
            lambda _: MedianModeImputation(),
        ),
        _registration(
            ComponentKind.COLUMN_AUGMENTATION,
            "missingness_indicators",
            MissingnessIndicatorsConfiguration,
            lambda config: MissingnessIndicators(
                min_missing_ratio=cast(
                    MissingnessIndicatorsConfiguration, config
                ).min_missing_ratio,
                max_missing_ratio=cast(
                    MissingnessIndicatorsConfiguration, config
                ).max_missing_ratio,
            ),
        ),
        _registration(
            ComponentKind.SOURCE_COLUMN_FILTER,
            "drop_all_missing_columns",
            EmptyConfiguration,
            lambda _: DropAllMissingColumns(),
        ),
        _registration(
            ComponentKind.NUMERIC_TRANSFORMATION,
            "robust_numeric_scaling",
            EmptyConfiguration,
            lambda _: RobustNumericScaling(),
            compatibility_notes=(
                "Required by the current scientific objectives and pattern-aware "
                "candidate generation.",
            ),
        ),
        _registration(
            ComponentKind.CANDIDATE_VALIDITY_POLICY,
            "minimum_cardinality",
            MinimumCardinalityConfiguration,
            lambda config: MinimumCardinality(
                min_rows=cast(MinimumCardinalityConfiguration, config).min_rows,
                min_columns=cast(MinimumCardinalityConfiguration, config).min_columns,
            ),
        ),
        _registration(
            ComponentKind.EVALUATION_SUPPORT_POLICY,
            "minimum_observed_support",
            ObservedSupportConfiguration,
            lambda config: MinimumObservedSupport(
                min_observed_count=cast(ObservedSupportConfiguration, config).min_observed_count,
                min_observed_ratio=cast(ObservedSupportConfiguration, config).min_observed_ratio,
            ),
        ),
        _registration(
            ComponentKind.INITIALIZER,
            "uniform_random",
            EmptyConfiguration,
            lambda _: UniformRandomInitializer(),
        ),
        _registration(
            ComponentKind.INITIALIZER,
            "stratified",
            StratifiedInitializerConfiguration,
            lambda config: StratifiedInitializer(
                cardinality_levels=cast(
                    StratifiedInitializerConfiguration, config
                ).cardinality_levels,
            ),
        ),
        _registration(
            ComponentKind.INITIALIZER,
            "pattern_aware",
            PatternAwareInitializerConfiguration,
            lambda config: PatternAwareInitializer(
                cardinality_levels=cast(
                    PatternAwareInitializerConfiguration, config
                ).cardinality_levels,
                joint_column_candidate_pool_size=cast(
                    PatternAwareInitializerConfiguration, config
                ).joint_column_candidate_pool_size,
            ),
            parameter_patterns=(
                (
                    "joint_column_candidate_pool_size",
                    frozenset({PatternKind.ADDITIVE, PatternKind.MULTIPLICATIVE}),
                ),
            ),
            compatibility_notes=(
                "Joint additive and multiplicative anchors require at least two "
                "eligible numeric columns.",
            ),
        ),
        _registration(
            ComponentKind.INITIALIZER,
            "cell_coverage_pattern_aware",
            CellCoveragePatternAwareInitializerConfiguration,
            lambda config: CellCoveragePatternAwareInitializer(
                seeds_per_cell=cast(
                    CellCoveragePatternAwareInitializerConfiguration, config
                ).seeds_per_cell,
                max_attempts_per_cell=cast(
                    CellCoveragePatternAwareInitializerConfiguration, config
                ).max_attempts_per_cell,
                joint_column_candidate_pool_size=cast(
                    CellCoveragePatternAwareInitializerConfiguration, config
                ).joint_column_candidate_pool_size,
            ),
            parameter_patterns=(
                (
                    "joint_column_candidate_pool_size",
                    frozenset({PatternKind.ADDITIVE, PatternKind.MULTIPLICATIVE}),
                ),
            ),
            compatibility_notes=(
                "Requires row- and column-cardinality archive axes.",
                "Seeds each reachable archive cell independently for every allowed pattern.",
            ),
        ),
    )


__all__ = ["default_data_registrations"]
