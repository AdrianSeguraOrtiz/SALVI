"""Static descriptions and workflow metadata for built-in components."""

from __future__ import annotations

from salvi.components.catalog_models import (
    RolePresentation,
    WorkflowConnection,
    WorkflowConnectionKind,
    WorkflowStage,
    WorkflowStagePresentation,
)
from salvi.components.observer_catalog import _OBSERVER_PRESENTATIONS as _OBSERVER_PRESENTATIONS
from salvi.components.protocols import ComponentKind

_WORKFLOW_STAGES: tuple[WorkflowStagePresentation, ...] = (
    WorkflowStagePresentation(
        stage=WorkflowStage.INPUT,
        title="Input",
        description="Choose or import a dataset.",
        order=0,
        icon="database",
        theme="input",
    ),
    WorkflowStagePresentation(
        stage=WorkflowStage.PREPARATION,
        title="Preparation",
        description="Prepare source observations.",
        order=10,
        icon="wand",
        theme="preparation",
    ),
    WorkflowStagePresentation(
        stage=WorkflowStage.EVALUATION,
        title="Evaluation",
        description="Define validity, quality, and behavior.",
        order=20,
        icon="microscope",
        theme="evaluation",
        preferred_columns=2,
    ),
    WorkflowStagePresentation(
        stage=WorkflowStage.SEARCH,
        title="Search",
        description="Generate, evaluate, and retain candidates.",
        order=30,
        icon="radar",
        theme="search",
        preferred_columns=2,
    ),
    WorkflowStagePresentation(
        stage=WorkflowStage.OUTPUT,
        title="Output",
        description="Consolidate the terminal repertoire.",
        order=40,
        icon="archive",
        theme="output",
    ),
    WorkflowStagePresentation(
        stage=WorkflowStage.ANALYSIS,
        title="Analysis",
        description="Assess completed results independently.",
        order=50,
        icon="chart",
        theme="analysis",
    ),
)
_WORKFLOW_STAGE_ORDER = {item.stage: item.order for item in _WORKFLOW_STAGES}

_ROLE_ICONS: dict[ComponentKind, str] = {
    ComponentKind.SOURCE_COLUMN_FILTER: "filter",
    ComponentKind.MISSING_VALUES_POLICY: "alert",
    ComponentKind.COLUMN_AUGMENTATION: "table",
    ComponentKind.NUMERIC_TRANSFORMATION: "sliders",
    ComponentKind.CANDIDATE_VALIDITY_POLICY: "shield-check",
    ComponentKind.EVALUATION_SUPPORT_POLICY: "gauge",
    ComponentKind.OBJECTIVE: "crosshair",
    ComponentKind.CONSTRAINT: "shield-alert",
    ComponentKind.DESCRIPTOR: "scan",
    ComponentKind.INITIALIZER: "sparkles",
    ComponentKind.SEARCH_ENGINE: "cpu",
    ComponentKind.ARCHIVE: "boxes",
    ComponentKind.PARENT_SELECTION_POLICY: "user",
    ComponentKind.MATE_SELECTION_POLICY: "users",
    ComponentKind.CROSSOVER_OPERATOR: "shuffle",
    ComponentKind.MUTATION_OPERATOR: "dna",
    ComponentKind.EMITTER: "radio",
    ComponentKind.SCHEDULER: "fork",
    ComponentKind.EVALUATION_EXECUTOR: "zap",
    ComponentKind.TERMINATION: "timer",
    ComponentKind.OBSERVER: "chart-combined",
    ComponentKind.FINAL_SELECTOR: "list-filter",
}

_COMPONENT_DESCRIPTIONS: dict[tuple[ComponentKind, str], str] = {
    (ComponentKind.ARCHIVE, "deep_grid_mome"): (
        "Stores a bounded local Pareto front in each occupied descriptor cell."
    ),
    (ComponentKind.CANDIDATE_VALIDITY_POLICY, "minimum_cardinality"): (
        "Rejects biclusters below configured row or column cardinalities."
    ),
    (ComponentKind.DESCRIPTOR, "row_cardinality"): (
        "Uses the number of selected rows as a QD behavior descriptor."
    ),
    (ComponentKind.DESCRIPTOR, "column_cardinality"): (
        "Uses the number of selected columns as a QD behavior descriptor."
    ),
    (ComponentKind.EMITTER, "random_move"): (
        "Applies one uniformly selected valid add, remove, or swap move."
    ),
    (ComponentKind.EMITTER, "add_row"): "Adds one row to an archived bicluster.",
    (ComponentKind.EMITTER, "remove_row"): "Removes one row from an archived bicluster.",
    (ComponentKind.EMITTER, "swap_row"): "Exchanges one selected and one external row.",
    (ComponentKind.EMITTER, "add_column"): "Adds one column to an archived bicluster.",
    (ComponentKind.EMITTER, "remove_column"): ("Removes one column from an archived bicluster."),
    (ComponentKind.EMITTER, "swap_column"): ("Exchanges one selected and one external column."),
    (ComponentKind.EMITTER, "shape_move"): (
        "Changes row and column membership together while preserving a valid shape."
    ),
    (ComponentKind.EMITTER, "crossover"): (
        "Uses the configured mate-selection policy and crossover operator to produce "
        "offspring from two archived parents."
    ),
    (ComponentKind.EMITTER, "mutation"): (
        "Uses the configured parent-selection policy and mutation operator to alter "
        "one archived parent."
    ),
    (ComponentKind.EMITTER, "restart"): (
        "Generates a fresh candidate when the current archive needs renewed exploration."
    ),
    (ComponentKind.EMITTER, "cell_coverage_restart"): (
        "Generates pattern-aware restarts in the least represented reachable descriptor cells."
    ),
    (ComponentKind.EMITTER, "alternating_pattern_local_search"): (
        "Refines evaluated biclusters with one guided membership move, alternating rows "
        "and columns across accepted local-search steps."
    ),
    (ComponentKind.EVALUATION_SUPPORT_POLICY, "minimum_observed_support"): (
        "Requires enough originally observed values before scientific calculations are valid."
    ),
    (ComponentKind.MATE_SELECTION_POLICY, "repertoire_random"): (
        "Samples two different parents uniformly from the complete repertoire."
    ),
    (ComponentKind.MATE_SELECTION_POLICY, "cell_first_evidence_compatible"): (
        "Samples a descriptor cell first, then selects locally strong parents that overlap "
        "in rows and columns within a configurable cell neighborhood."
    ),
    (ComponentKind.CROSSOVER_OPERATOR, "membership_recombination"): (
        "Recombines row and column membership without using objective evidence."
    ),
    (ComponentKind.CROSSOVER_OPERATOR, "evidence_weighted_recombination"): (
        "Recombines membership while favoring columns supported by persisted objective evidence."
    ),
    (ComponentKind.CROSSOVER_OPERATOR, "half_uniform_membership"): (
        "Exchanges differing row and column memberships independently between two parents."
    ),
    (ComponentKind.MUTATION_OPERATOR, "bit_flip_membership"): (
        "Flips row and column membership bits while repairing minimum cardinalities."
    ),
    (ComponentKind.FINAL_SELECTOR, "containment_marginal_quality"): (
        "Traverses exact containment chains and retains the largest nested bicluster before "
        "its normalized objective degradation becomes material."
    ),
    (ComponentKind.FINAL_SELECTOR, "adaptive_residual_evidence_cover"): (
        "Greedily selects compact, complementary biclusters by their unexplained "
        "quality-weighted matrix evidence and adapts its quality floor to the repertoire."
    ),
    (ComponentKind.INITIALIZER, "uniform_random"): (
        "Samples initial row and column memberships uniformly over valid cardinalities."
    ),
    (ComponentKind.INITIALIZER, "stratified"): (
        "Distributes initial candidates across several row and column cardinality levels."
    ),
    (ComponentKind.INITIALIZER, "pattern_aware"): (
        "Combines cardinality strata with anchors supported by the allowed pattern catalog."
    ),
    (ComponentKind.INITIALIZER, "cell_coverage_pattern_aware"): (
        "Attempts several pattern-aware seeds in every reachable row/column descriptor cell."
    ),
    (ComponentKind.OBJECTIVE, "internal_coherence"): (
        "Minimizes the RMS pattern-fitting error across selected columns."
    ),
    (ComponentKind.OBJECTIVE, "contrast"): (
        "Maximizes separation between the fitted bicluster pattern and its background."
    ),
    (ComponentKind.OBJECTIVE, "balanced_bicluster_size"): (
        "Maximizes the harmonic mean of selected-row and selected-column coverage."
    ),
    (ComponentKind.CONSTRAINT, "balanced_bicluster_size_range"): (
        "Restricts balanced row/column coverage to an inclusive interval."
    ),
    (ComponentKind.CONSTRAINT, "maximum_internal_coherence"): (
        "Restricts the RMS inferred-pattern error to a configured maximum."
    ),
    (ComponentKind.PARENT_SELECTION_POLICY, "repertoire_uniform"): (
        "Samples eligible parents uniformly from the full terminal repertoire."
    ),
    (ComponentKind.PARENT_SELECTION_POLICY, "cell_uniform_quality"): (
        "Samples an occupied QD cell uniformly before choosing a local parent, preventing "
        "large cells from dominating reproduction."
    ),
    (ComponentKind.OBSERVER, "search_progress"): ("Persists only run-level evaluation progress."),
    (ComponentKind.OBSERVER, "archive_coverage"): (
        "Persists the current occupied-cell and retained-repertoire state."
    ),
    (ComponentKind.OBSERVER, "candidate_outcomes"): (
        "Persists mutually exclusive archive-retention outcomes over bounded windows."
    ),
    (ComponentKind.OBSERVER, "descriptor_distribution"): (
        "Persists descriptor summaries for recently evaluated candidates."
    ),
    (ComponentKind.OBSERVER, "archive_descriptor_distribution"): (
        "Persists descriptor and per-cell occupancy summaries for the current repertoire."
    ),
    (ComponentKind.OBSERVER, "objective_distribution"): (
        "Persists objective distribution summaries at a configurable cadence."
    ),
    (ComponentKind.OBSERVER, "emitter_credit"): (
        "Persists scheduler credit and allocation metrics for each emitter."
    ),
    (ComponentKind.OBSERVER, "candidate_diversity"): (
        "Measures exact uniqueness and row/column structural diversity of candidates."
    ),
    (ComponentKind.OBSERVER, "evaluation_issues"): (
        "Persists cumulative scientific invalidity counts grouped by issue code."
    ),
    (ComponentKind.OBSERVER, "component_timing"): (
        "Attributes wall-clock cost to setup, search phases, scientific components, "
        "individual evaluations, output, and observers."
    ),
    (ComponentKind.OBSERVER, "qd_archive_diagnostics"): (
        "Tracks QD-cell visits, acceptance, turnover, and stagnation as bounded temporal "
        "summaries or an optional exact two-dimensional cell map."
    ),
    (ComponentKind.SCHEDULER, "first"): (
        "Allocates every emission request to the first configured emitter."
    ),
    (ComponentKind.SCHEDULER, "fixed_proportion"): (
        "Allocates emitters by deterministic cumulative shares, enabling controlled "
        "operator mixtures without adaptive-credit feedback."
    ),
    (ComponentKind.SCHEDULER, "adaptive_credit"): (
        "Uses deterministic UCB allocation over cumulative emitter rewards."
    ),
    (ComponentKind.SCHEDULER, "cell_balanced_adaptive_credit"): (
        "Uses UCB allocation while rewarding useful work in less evaluated cells."
    ),
    (ComponentKind.SEARCH_ENGINE, "serial_mome"): (
        "Runs deterministic batched MOME through the standard ask/evaluate/tell lifecycle."
    ),
    (ComponentKind.SEARCH_ENGINE, "pymoo_nsga2"): (
        "Runs binary NSGA-II with explicit SALVI crossover and mutation operators and "
        "returns its terminal non-dominated population."
    ),
    (ComponentKind.TERMINATION, "evaluation_budget"): (
        "Stops search after an exact maximum number of candidate evaluations."
    ),
}

_PARAMETER_DESCRIPTIONS: dict[str, str] = {
    "axes": "Descriptor-specific archive discretizations and optional domain bounds.",
    "cell_capacity": "Maximum number of candidates retained per descriptor cell.",
    "exploration_weight": ("Strength of uncertainty-driven allocation by adaptive schedulers."),
    "min_rows": "Minimum number of rows allowed in a candidate bicluster.",
    "min_columns": "Minimum number of columns allowed in a candidate bicluster.",
    "minimum": "Inclusive lower bound accepted by the component.",
    "maximum": "Inclusive upper bound accepted by the component.",
    "maximum_error": "Largest internal-coherence error considered feasible.",
    "min_missing_ratio": "Minimum source-column missing fraction required to add an indicator.",
    "guided": "Whether persisted evaluation evidence guides the proposed membership move.",
    "parent_pool_size": "Number of strong archive members considered when choosing a parent.",
    "candidate_pool_size": (
        "Maximum number of membership alternatives scored on each changed side of a guided move."
    ),
    "row_exchange_probability": "Independent probability of inheriting each row from the donor.",
    "column_exchange_probability": (
        "Independent probability of inheriting each column from the donor."
    ),
    "mate_pool_size": (
        "Bound on the closest or highest-ranked compatible mating alternatives "
        "eligible for sampling."
    ),
    "max_attempts": "Maximum recombination attempts made before an explicit restart fallback.",
    "cardinality_levels": "Number of row and column size strata sampled during generation.",
    "seeds_per_cell": (
        "Number of accepted, pattern-balanced seeds requested in each reachable cell."
    ),
    "max_attempts_per_cell": (
        "Maximum evaluated seed attempts allowed to satisfy one cell's bootstrap target."
    ),
    "strategy": "Candidate-generation strategy used for a restart.",
    "joint_column_candidate_pool_size": (
        "Maximum number of numeric columns inspected when constructing additive or "
        "multiplicative joint-pattern anchors. Constant anchors do not use this parameter."
    ),
    "cardinality_change_probability": (
        "Probability that local refinement adds or removes membership instead of swapping."
    ),
    "quality_parent_probability": (
        "Probability of selecting the strongest sampled parent instead of an exploratory one."
    ),
    "shares": (
        "Emitter-name to long-run allocation share mapping. Every configured emitter must "
        "appear exactly once and the values must sum to one."
    ),
    "max_in_flight": "Optional bound on submitted evaluations that have not yet completed.",
    "min_observed_count": "Minimum count of originally observed values required for a fit.",
    "min_observed_ratio": "Minimum fraction of originally observed opportunities required.",
    "structural_similarity_threshold": (
        "Similarity at or above which nearby biclusters are consolidated."
    ),
    "row_weight": "Relative contribution of row membership to structural similarity.",
    "minimum_marginal_gain": "Smallest quality-novelty gain accepted into the final result.",
    "objective_names": (
        "Optional ordered subset of configured objectives used to rank final candidates. "
        "Null uses every objective."
    ),
    "max_per_cell": "Optional maximum number of reported biclusters from each occupied cell.",
    "min_background_ratio": (
        "Minimum external-row fraction required for a valid contrast calculation."
    ),
    "every_evaluations": "Evaluation cadence between successive metric samples.",
    "window_size": "Number of recent evaluated candidates used by windowed diversity metrics.",
    "distance_sample_size": (
        "Maximum deterministic sample from the diversity window used for pairwise distances. "
        "Uniqueness counts remain exact."
    ),
    "new_cell_reward": "Credit assigned when an emitter occupies a previously empty cell.",
    "insertion_reward": "Credit assigned when an emitter improves an occupied cell.",
    "underexplored_cell_weight": (
        "Strength of the scheduler reward multiplier for useful work in less evaluated cells."
    ),
    "initial_population_size": (
        "Number of candidates requested from ordinary initializers. Cell-coverage initializers "
        "derive their bootstrap budget from reachable cells instead."
    ),
    "population_size": "Number of candidates retained by the evolutionary population.",
    "batch_size": "Maximum candidates requested in each ask/evaluate/tell iteration.",
    "eliminate_duplicates": (
        "Whether the evolutionary backend rejects duplicate decision vectors during generation."
    ),
    "max_evaluations": "Exact total candidate-evaluation budget for the run.",
    "minimum_row_jaccard": (
        "Minimum row-membership Jaccard similarity required between recombination parents."
    ),
    "minimum_column_jaccard": (
        "Minimum column-membership Jaccard similarity required between recombination parents."
    ),
    "cell_neighborhood_radius": (
        "Maximum per-axis descriptor-cell distance allowed between recombination parents."
    ),
    "max_objective_degradation": (
        "Largest normalized loss accepted in any configured objective by the component."
    ),
    "max_degradation_per_log_area_gain": (
        "Largest normalized objective loss per logarithmic bicluster-area gain accepted along "
        "a containment chain."
    ),
    "quality_scale": (
        "Interpret selected objectives directly on the unit interval or normalize them "
        "empirically within the terminal repertoire."
    ),
    "overlap_penalty": "Penalty applied to evidence already explained by selected biclusters.",
    "low_quality_penalty": (
        "Penalty applied to observed cells whose column quality falls below the adaptive floor."
    ),
    "complexity_penalty": (
        "Weight of the combinatorial row-and-column membership complexity penalty."
    ),
    "minimum_marginal_evidence": (
        "Minimum positive residual-evidence gain required to retain another bicluster."
    ),
    "maximum_dense_cells": (
        "Largest prepared matrix stored as a dense coverage map before using sparse storage."
    ),
    "minimum_quality_floor": "Lower bound for the run-adaptive objective-quality floor.",
    "maximum_quality_floor": "Upper bound for the run-adaptive objective-quality floor.",
    "minimum_candidates_for_knee": (
        "Minimum unique feasible structures required before estimating a quality knee."
    ),
    "minimum_knee_prominence": (
        "Minimum two-sided prominence required to accept a detected quality transition."
    ),
    "fallback_quality_quantile": (
        "Quality quantile used when the repertoire has no sufficiently prominent knee."
    ),
}


def _incoming(
    source: ComponentKind,
    kind: WorkflowConnectionKind = WorkflowConnectionKind.SUPPORT,
) -> WorkflowConnection:
    return WorkflowConnection(source=source, kind=kind)


_ROLE_PRESENTATIONS: dict[ComponentKind, RolePresentation] = {
    ComponentKind.SOURCE_COLUMN_FILTER: RolePresentation(
        kind=ComponentKind.SOURCE_COLUMN_FILTER,
        title="Source filters",
        description="Remove source columns before scientific preprocessing.",
        stage=WorkflowStage.PREPARATION,
        order=10,
        repeatable=True,
        accepts_pipeline_input=True,
    ),
    ComponentKind.MISSING_VALUES_POLICY: RolePresentation(
        kind=ComponentKind.MISSING_VALUES_POLICY,
        title="Missing values",
        description="Choose how unavailable observations are represented.",
        stage=WorkflowStage.PREPARATION,
        order=20,
        incoming=(_incoming(ComponentKind.SOURCE_COLUMN_FILTER, WorkflowConnectionKind.PRIMARY),),
    ),
    ComponentKind.COLUMN_AUGMENTATION: RolePresentation(
        kind=ComponentKind.COLUMN_AUGMENTATION,
        title="Column augmentation",
        description="Derive additional columns from source observations.",
        stage=WorkflowStage.PREPARATION,
        order=30,
        repeatable=True,
        incoming=(_incoming(ComponentKind.MISSING_VALUES_POLICY, WorkflowConnectionKind.PRIMARY),),
    ),
    ComponentKind.NUMERIC_TRANSFORMATION: RolePresentation(
        kind=ComponentKind.NUMERIC_TRANSFORMATION,
        title="Numeric transformations",
        description="Transform numeric columns before pattern evaluation.",
        stage=WorkflowStage.PREPARATION,
        order=40,
        repeatable=True,
        incoming=(_incoming(ComponentKind.COLUMN_AUGMENTATION, WorkflowConnectionKind.PRIMARY),),
    ),
    ComponentKind.CANDIDATE_VALIDITY_POLICY: RolePresentation(
        kind=ComponentKind.CANDIDATE_VALIDITY_POLICY,
        title="Candidate validity",
        description="Define structural requirements for candidate biclusters.",
        stage=WorkflowStage.EVALUATION,
        order=10,
        incoming=(_incoming(ComponentKind.NUMERIC_TRANSFORMATION, WorkflowConnectionKind.PRIMARY),),
    ),
    ComponentKind.EVALUATION_SUPPORT_POLICY: RolePresentation(
        kind=ComponentKind.EVALUATION_SUPPORT_POLICY,
        title="Observed support",
        description="Define minimum observed evidence for scientific calculations.",
        stage=WorkflowStage.EVALUATION,
        order=20,
        incoming=(_incoming(ComponentKind.NUMERIC_TRANSFORMATION, WorkflowConnectionKind.PRIMARY),),
    ),
    ComponentKind.OBJECTIVE: RolePresentation(
        kind=ComponentKind.OBJECTIVE,
        title="Objectives",
        description="Score the scientific quality of candidate biclusters.",
        stage=WorkflowStage.EVALUATION,
        order=30,
        repeatable=True,
        incoming=(
            _incoming(ComponentKind.CANDIDATE_VALIDITY_POLICY),
            _incoming(ComponentKind.EVALUATION_SUPPORT_POLICY),
        ),
    ),
    ComponentKind.CONSTRAINT: RolePresentation(
        kind=ComponentKind.CONSTRAINT,
        title="Constraints",
        description="Define feasibility independently from objective quality.",
        stage=WorkflowStage.EVALUATION,
        order=40,
        repeatable=True,
        incoming=(
            _incoming(ComponentKind.CANDIDATE_VALIDITY_POLICY),
            _incoming(ComponentKind.EVALUATION_SUPPORT_POLICY),
        ),
    ),
    ComponentKind.DESCRIPTOR: RolePresentation(
        kind=ComponentKind.DESCRIPTOR,
        title="Descriptors",
        description="Locate candidates in quality-diversity behavior space.",
        stage=WorkflowStage.EVALUATION,
        order=50,
        repeatable=True,
        incoming=(_incoming(ComponentKind.CANDIDATE_VALIDITY_POLICY),),
    ),
    ComponentKind.INITIALIZER: RolePresentation(
        kind=ComponentKind.INITIALIZER,
        title="Initialization",
        description="Create the first candidate biclusters.",
        stage=WorkflowStage.SEARCH,
        order=30,
        incoming=(_incoming(ComponentKind.NUMERIC_TRANSFORMATION),),
    ),
    ComponentKind.SEARCH_ENGINE: RolePresentation(
        kind=ComponentKind.SEARCH_ENGINE,
        title="Search engine",
        description="Own the ask, evaluate, tell search lifecycle.",
        stage=WorkflowStage.SEARCH,
        order=40,
        incoming=(
            _incoming(ComponentKind.INITIALIZER, WorkflowConnectionKind.PRIMARY),
            _incoming(ComponentKind.OBJECTIVE),
            _incoming(ComponentKind.CONSTRAINT),
            _incoming(ComponentKind.DESCRIPTOR),
            _incoming(ComponentKind.EVALUATION_EXECUTOR, WorkflowConnectionKind.FEEDBACK),
        ),
    ),
    ComponentKind.ARCHIVE: RolePresentation(
        kind=ComponentKind.ARCHIVE,
        title="Archive",
        description="Retain evaluated candidates during quality-diversity search.",
        stage=WorkflowStage.SEARCH,
        order=30,
        incoming=(_incoming(ComponentKind.SEARCH_ENGINE, WorkflowConnectionKind.FEEDBACK),),
    ),
    ComponentKind.PARENT_SELECTION_POLICY: RolePresentation(
        kind=ComponentKind.PARENT_SELECTION_POLICY,
        title="Parent selection",
        description="Choose one parent for mutation-based generation.",
        stage=WorkflowStage.SEARCH,
        order=40,
        incoming=(_incoming(ComponentKind.ARCHIVE),),
    ),
    ComponentKind.MATE_SELECTION_POLICY: RolePresentation(
        kind=ComponentKind.MATE_SELECTION_POLICY,
        title="Mate selection",
        description="Choose compatible parents for recombination.",
        stage=WorkflowStage.SEARCH,
        order=50,
        incoming=(_incoming(ComponentKind.ARCHIVE),),
    ),
    ComponentKind.CROSSOVER_OPERATOR: RolePresentation(
        kind=ComponentKind.CROSSOVER_OPERATOR,
        title="Crossover",
        description="Recombine two parent biclusters.",
        stage=WorkflowStage.SEARCH,
        order=60,
        incoming=(
            _incoming(ComponentKind.MATE_SELECTION_POLICY),
            _incoming(ComponentKind.SEARCH_ENGINE),
        ),
    ),
    ComponentKind.MUTATION_OPERATOR: RolePresentation(
        kind=ComponentKind.MUTATION_OPERATOR,
        title="Mutation",
        description="Alter one parent bicluster.",
        stage=WorkflowStage.SEARCH,
        order=70,
        incoming=(
            _incoming(ComponentKind.PARENT_SELECTION_POLICY),
            _incoming(ComponentKind.SEARCH_ENGINE),
        ),
    ),
    ComponentKind.EMITTER: RolePresentation(
        kind=ComponentKind.EMITTER,
        title="Emitters",
        description="Generate candidate proposals for quality-diversity search.",
        stage=WorkflowStage.SEARCH,
        order=80,
        repeatable=True,
        incoming=(
            _incoming(ComponentKind.SCHEDULER, WorkflowConnectionKind.CONTROL),
            _incoming(ComponentKind.CROSSOVER_OPERATOR),
            _incoming(ComponentKind.MUTATION_OPERATOR),
        ),
    ),
    ComponentKind.SCHEDULER: RolePresentation(
        kind=ComponentKind.SCHEDULER,
        title="Scheduler",
        description="Allocate proposal requests among configured emitters.",
        stage=WorkflowStage.SEARCH,
        order=90,
        incoming=(_incoming(ComponentKind.SEARCH_ENGINE, WorkflowConnectionKind.CONTROL),),
    ),
    ComponentKind.EVALUATION_EXECUTOR: RolePresentation(
        kind=ComponentKind.EVALUATION_EXECUTOR,
        title="Evaluation executor",
        description="Evaluate candidate batches serially or in parallel.",
        stage=WorkflowStage.SEARCH,
        order=100,
        incoming=(
            _incoming(ComponentKind.EMITTER, WorkflowConnectionKind.PRIMARY),
            _incoming(ComponentKind.CROSSOVER_OPERATOR),
            _incoming(ComponentKind.MUTATION_OPERATOR),
        ),
    ),
    ComponentKind.TERMINATION: RolePresentation(
        kind=ComponentKind.TERMINATION,
        title="Termination",
        description="Decide when the search has consumed its budget.",
        stage=WorkflowStage.SEARCH,
        order=120,
        incoming=(_incoming(ComponentKind.SEARCH_ENGINE, WorkflowConnectionKind.CONTROL),),
    ),
    ComponentKind.OBSERVER: RolePresentation(
        kind=ComponentKind.OBSERVER,
        title="Observers",
        description="Persist selected views of search progress and diagnostics.",
        stage=WorkflowStage.SEARCH,
        order=130,
        repeatable=True,
        incoming=(_incoming(ComponentKind.SEARCH_ENGINE, WorkflowConnectionKind.CONTROL),),
    ),
    ComponentKind.FINAL_SELECTOR: RolePresentation(
        kind=ComponentKind.FINAL_SELECTOR,
        title="Final selection",
        description="Consolidate the terminal repertoire for reporting.",
        stage=WorkflowStage.OUTPUT,
        order=10,
        incoming=(
            _incoming(ComponentKind.SEARCH_ENGINE, WorkflowConnectionKind.PRIMARY),
            _incoming(ComponentKind.ARCHIVE),
        ),
        emits_pipeline_output=True,
    ),
}


_PARAMETER_UNITS = {
    "cancellation_grace_seconds": "seconds",
    "wall_clock_budget_seconds": "seconds",
    "convergence_tolerance": "error",
    "min_observed_ratio": "ratio",
    "min_background_ratio": "ratio",
    "row_weight": "ratio",
    "exploration_weight": "weight",
    "minimum_row_jaccard": "ratio",
    "minimum_column_jaccard": "ratio",
}

_ADVANCED_PARAMETERS = {
    "max_in_flight",
    "candidate_pool_size",
    "joint_column_candidate_pool_size",
    "convergence_tolerance",
}

_CONFIGURATION_PATHS: dict[ComponentKind, tuple[str, ...]] = {
    ComponentKind.SOURCE_COLUMN_FILTER: ("preprocessing", "source_column_filters"),
    ComponentKind.MISSING_VALUES_POLICY: ("preprocessing", "missing_values"),
    ComponentKind.COLUMN_AUGMENTATION: ("preprocessing", "column_augmentations"),
    ComponentKind.NUMERIC_TRANSFORMATION: ("preprocessing", "numeric_transformations"),
    ComponentKind.CANDIDATE_VALIDITY_POLICY: ("evaluation", "candidate_validity"),
    ComponentKind.EVALUATION_SUPPORT_POLICY: ("evaluation", "observed_support"),
    ComponentKind.SEARCH_ENGINE: ("search", "engine"),
    ComponentKind.OBJECTIVE: ("search", "objectives"),
    ComponentKind.CONSTRAINT: ("search", "constraints"),
    ComponentKind.DESCRIPTOR: ("search", "descriptors"),
    ComponentKind.ARCHIVE: ("search", "archive"),
    ComponentKind.PARENT_SELECTION_POLICY: ("search", "parent_selection"),
    ComponentKind.MATE_SELECTION_POLICY: ("search", "mate_selection"),
    ComponentKind.CROSSOVER_OPERATOR: ("search", "crossover"),
    ComponentKind.MUTATION_OPERATOR: ("search", "mutation"),
    ComponentKind.INITIALIZER: ("search", "initialization"),
    ComponentKind.EMITTER: ("search", "emitters"),
    ComponentKind.SCHEDULER: ("search", "scheduler"),
    ComponentKind.TERMINATION: ("search", "termination"),
    ComponentKind.EVALUATION_EXECUTOR: ("execution", "executor"),
    ComponentKind.OBSERVER: ("monitoring", "observers"),
    ComponentKind.FINAL_SELECTOR: ("final_selection",),
}
