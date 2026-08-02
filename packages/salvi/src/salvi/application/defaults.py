"""Public defaults for a complete, reusable scientific pipeline."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from salvi.domain.enums import SearchFamily


def default_scientific_configuration() -> dict[str, Any]:
    """Return the editable default pipeline without dataset or run bindings."""

    return {
        "schema_version": 1,
        "patterns": {
            "allowed": ["CONSTANT"],
            "min_improvement": 0.10,
            "max_iterations": 25,
            "convergence_tolerance": 1e-6,
        },
        "preprocessing": {
            "source_column_filters": [],
            "missing_values": {"name": "preserve", "parameters": {}},
            "column_augmentations": [],
            "numeric_transformations": [{"name": "robust_numeric_scaling", "parameters": {}}],
        },
        "evaluation": {
            "candidate_validity": {
                "name": "minimum_cardinality",
                "parameters": {"min_rows": 10, "min_columns": 10},
            },
            "observed_support": {
                "name": "minimum_observed_support",
                "parameters": {
                    "min_observed_count": 2,
                    "min_observed_ratio": 0.5,
                },
            },
        },
        "search": {
            "engine": {
                "name": "serial_mome",
                "parameters": {"initial_population_size": 64, "batch_size": 16},
            },
            "objectives": [
                {"name": "internal_coherence", "parameters": {}},
                {"name": "contrast", "parameters": {"min_background_ratio": 0.1}},
            ],
            "constraints": [],
            "descriptors": [
                {"name": "row_cardinality", "parameters": {}},
                {"name": "column_cardinality", "parameters": {}},
            ],
            "archive": {
                "name": "deep_grid_mome",
                "parameters": {
                    "axes": [
                        {
                            "descriptor": "row_cardinality",
                            "binning": "GEOMETRIC",
                            "bins": 8,
                        },
                        {
                            "descriptor": "column_cardinality",
                            "binning": "GEOMETRIC",
                            "bins": 8,
                        },
                    ],
                    "cell_capacity": 8,
                },
            },
            "parent_selection": {
                "name": "cell_uniform_quality",
                "parameters": {},
            },
            "mate_selection": {
                "name": "cell_first_evidence_compatible",
                "parameters": {
                    "parent_pool_size": 32,
                    "mate_pool_size": 4,
                    "minimum_row_jaccard": 0.1,
                    "minimum_column_jaccard": 0.1,
                    "cell_neighborhood_radius": 1,
                },
            },
            "crossover": {
                "name": "evidence_weighted_recombination",
                "parameters": {
                    "application_probability": 1.0,
                    "row_exchange_probability": 0.5,
                    "column_exchange_probability": 0.5,
                },
            },
            "initialization": {
                "name": "cell_coverage_pattern_aware",
                "parameters": {
                    "seeds_per_cell": 4,
                    "max_attempts_per_cell": 12,
                    "joint_column_candidate_pool_size": 32,
                },
            },
            "emitters": [
                {
                    "name": "add_row",
                    "parameters": {
                        "guided": True,
                        "parent_pool_size": 16,
                        "candidate_pool_size": 64,
                    },
                },
                {
                    "name": "remove_row",
                    "parameters": {
                        "guided": True,
                        "parent_pool_size": 16,
                        "candidate_pool_size": 64,
                    },
                },
                {
                    "name": "swap_row",
                    "parameters": {
                        "guided": True,
                        "parent_pool_size": 16,
                        "candidate_pool_size": 64,
                    },
                },
                {
                    "name": "add_column",
                    "parameters": {
                        "guided": True,
                        "parent_pool_size": 16,
                        "candidate_pool_size": 64,
                    },
                },
                {
                    "name": "remove_column",
                    "parameters": {
                        "guided": True,
                        "parent_pool_size": 16,
                        "candidate_pool_size": 64,
                    },
                },
                {
                    "name": "swap_column",
                    "parameters": {
                        "guided": True,
                        "parent_pool_size": 16,
                        "candidate_pool_size": 64,
                    },
                },
                {
                    "name": "shape_move",
                    "parameters": {
                        "guided": True,
                        "parent_pool_size": 16,
                        "candidate_pool_size": 64,
                    },
                },
                {"name": "crossover", "parameters": {"max_attempts": 8}},
                {
                    "name": "cell_coverage_restart",
                    "parameters": {
                        "joint_column_candidate_pool_size": 32,
                    },
                },
            ],
            "scheduler": {
                "name": "cell_balanced_adaptive_credit",
                "parameters": {
                    "exploration_weight": 0.5,
                    "new_cell_reward": 1.0,
                    "insertion_reward": 0.25,
                    "underexplored_cell_weight": 1.0,
                },
            },
            "termination": {
                "name": "evaluation_budget",
                "parameters": {"max_evaluations": 50_000},
            },
        },
        "execution": {
            "executor": {
                "name": "process_pool",
                "parameters": {
                    "integration_mode": "DETERMINISTIC",
                    "max_in_flight": 8,
                },
            },
            "workers": 4,
            "cancellation_grace_seconds": 5.0,
        },
        "monitoring": {
            "queue_capacity": 1024,
            "checkpoint_interval_evaluations": 5_000,
            "observers": [
                {"name": "search_progress", "parameters": {}},
                {"name": "archive_coverage", "parameters": {}},
                {
                    "name": "candidate_outcomes",
                    "parameters": {"every_evaluations": 1_000},
                },
                {
                    "name": "emitter_credit",
                    "parameters": {"every_evaluations": 1_000},
                },
                {
                    "name": "component_timing",
                    "parameters": {"every_evaluations": 1000},
                },
                {
                    "name": "evaluation_issues",
                    "parameters": {"every_evaluations": 1_000},
                },
                {
                    "name": "qd_archive_diagnostics",
                    "parameters": {
                        "every_evaluations": 1_000,
                        "include_cell_metrics": False,
                    },
                },
                {
                    "name": "objective_distribution",
                    "parameters": {
                        "every_evaluations": 1_000,
                    },
                },
                {
                    "name": "descriptor_distribution",
                    "parameters": {"every_evaluations": 1_000},
                },
                {
                    "name": "archive_descriptor_distribution",
                    "parameters": {"every_evaluations": 1_000},
                },
                {
                    "name": "candidate_diversity",
                    "parameters": {
                        "window_size": 512,
                        "distance_sample_size": 128,
                        "row_weight": 0.5,
                        "every_evaluations": 1_000,
                    },
                },
                {"name": "runtime_throughput", "parameters": {}},
                {
                    "name": "resource_usage",
                    "parameters": {"every_evaluations": 1_000},
                },
            ],
        },
        "final_selection": {
            "name": "adaptive_residual_evidence_cover",
            "parameters": {
                "objective_names": ["internal_coherence", "contrast"],
                "quality_scale": "unit_interval",
                "overlap_penalty": 0.50,
                "low_quality_penalty": 0.50,
                "complexity_penalty": 0.25,
                "minimum_marginal_evidence": 1.0,
                "maximum_dense_cells": 10_000_000,
                "minimum_quality_floor": 0.50,
                "maximum_quality_floor": 0.85,
                "minimum_candidates_for_knee": 8,
                "minimum_knee_prominence": 0.05,
                "fallback_quality_quantile": 0.50,
            },
        },
    }


def default_configuration_for_search_family(family: SearchFamily) -> dict[str, Any]:
    """Return the complete built-in architecture for one search family."""

    configuration = default_scientific_configuration()
    if family is SearchFamily.QUALITY_DIVERSITY:
        return configuration

    if family is not SearchFamily.CONVENTIONAL_MULTI_OBJECTIVE:
        raise ValueError(f"unsupported search family: {family.value}")

    search = configuration["search"]
    search.update(
        {
            "engine": {
                "name": "pymoo_nsga2",
                "parameters": {
                    "population_size": 64,
                    "eliminate_duplicates": True,
                },
            },
            "descriptors": [],
            "archive": None,
            "parent_selection": None,
            "mate_selection": None,
            "crossover": {
                "name": "half_uniform_membership",
                "parameters": {
                    "application_probability": 0.9,
                    "row_exchange_probability": 0.5,
                    "column_exchange_probability": 0.5,
                },
            },
            "mutation": {
                "name": "bit_flip_membership",
                "parameters": {
                    "application_probability": 1.0,
                    "bit_probability": None,
                },
            },
            "initialization": {
                "name": "pattern_aware",
                "parameters": {
                    "cardinality_levels": 8,
                    "joint_column_candidate_pool_size": 32,
                },
            },
            "emitters": [],
            "scheduler": None,
        }
    )
    monitoring = configuration["monitoring"]
    monitoring["checkpoint_interval_evaluations"] = None
    monitoring["observers"] = [
        {"name": "search_progress", "parameters": {}},
        {
            "name": "component_timing",
            "parameters": {"every_evaluations": 1_000},
        },
        {
            "name": "evaluation_issues",
            "parameters": {"every_evaluations": 1_000},
        },
        {
            "name": "objective_distribution",
            "parameters": {"every_evaluations": 1_000},
        },
        {
            "name": "candidate_diversity",
            "parameters": {
                "window_size": 512,
                "distance_sample_size": 128,
                "row_weight": 0.5,
                "every_evaluations": 1_000,
            },
        },
        {"name": "runtime_throughput", "parameters": {}},
        {
            "name": "resource_usage",
            "parameters": {"every_evaluations": 1_000},
        },
    ]
    return deepcopy(configuration)


__all__ = [
    "default_configuration_for_search_family",
    "default_scientific_configuration",
]
