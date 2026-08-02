# Configuration

SALVI accepts one reusable, versioned pipeline YAML document. It does not support
profiles, includes, inheritance, merge keys, aliases, environment interpolation,
or scientific CLI overrides. The pipeline deliberately does not contain a dataset
path, run identifier, random seed, checkpoint, or output directory.

Those values bind a pipeline to one concrete execution through `salvi run`.
SALVI writes the resulting complete configuration to the output directory as
`effective-configuration.yaml`, so reproducibility is retained without coupling a
saved algorithm design to a particular dataset.

Unknown and duplicate keys are errors. Component parameters are validated by the
configuration model registered for that component.

Source YAML may omit optional roles, empty collections, empty `parameters` maps,
and schema-level defaults. Component parameters explicitly supplied by the user
remain visible; omitted component parameters are resolved by their registered
configuration model. `salvi config format pipeline.yaml` validates and writes the
compact canonical form; add `--expanded` when auditing the full pipeline document.
Executed runs always persist the expanded, validated bound form.

The web builder edits this same document. Its component choices and parameter
forms are generated from the public registry metadata and Pydantic schemas; there
is no GUI-only configuration model or hidden profile.

Each public catalog entry declares its description, maturity, supported pattern
families, provided and required capabilities, explicit conflicts, compatibility
notes, and parameter schema. Parameters may additionally state the pattern
families to which they apply. Pipeline construction validates these declarations
centrally before instantiating components, and the capability pass then verifies
execution order. For example, the scientific objectives require
`robust-numeric-data`; omitting `robust_numeric_scaling` therefore fails
validation instead of silently changing their formulas. Runtime parallelism is
selected through the evaluation executor; no single-option kernel selector is
exposed.

## Pipeline schema

```yaml
schema_version: 1

patterns:
  allowed: [CONSTANT]
  min_improvement: 0.10
  max_iterations: 25
  convergence_tolerance: 1.0e-6

preprocessing:
  source_column_filters: []
  missing_values:
    name: preserve
    parameters: {}
  column_augmentations: []
  numeric_transformations:
    - name: robust_numeric_scaling
      parameters: {}

evaluation:
  candidate_validity:
    name: minimum_cardinality
    parameters:
      min_rows: 10
      min_columns: 10
  observed_support:
    name: minimum_observed_support
    parameters:
      min_observed_count: 2
      min_observed_ratio: 0.8

search:
  engine:
    name: serial_mome
    parameters:
      initial_population_size: 64
      batch_size: 16
  objectives:
    - name: internal_coherence
      parameters: {}
    - name: contrast
      parameters:
        min_background_ratio: 0.10
  descriptors:
    - name: row_cardinality
      parameters: {}
    - name: column_cardinality
      parameters: {}
  archive:
    name: deep_grid_mome
    parameters:
      axes:
        - descriptor: row_cardinality
          binning: GEOMETRIC
          bins: 8
        - descriptor: column_cardinality
          binning: LINEAR
          bins: 6
      cell_capacity: 8
  parent_selection:
    name: repertoire_uniform
    parameters: {}
  initialization:
    name: uniform_random
    parameters: {}
  emitters:
    - name: random_move
      parameters: {}
  scheduler:
    name: first
    parameters: {}
  termination:
    name: evaluation_budget
    parameters:
      max_evaluations: 10000

execution:
  executor:
    name: process_pool
    parameters:
      integration_mode: DETERMINISTIC
      max_in_flight: 16
  workers: 4
  cancellation_grace_seconds: 5.0

monitoring:
  queue_capacity: 1024
  checkpoint_interval_evaluations: 1000
  observers:
    - name: search_progress
      parameters: {}

final_selection:
  name: containment_marginal_quality
  parameters:
    max_objective_degradation: 0.15
    max_degradation_per_log_area_gain: 0.20

```

## Run binding

Supply the concrete data and runtime identity alongside the pipeline:

```bash
salvi validate pipeline.yaml \
  --dataset ./dataset-bundle \
  --seed 42

salvi run pipeline.yaml \
  --dataset ./dataset-bundle \
  --output ./run-output \
  --identifier example-run \
  --seed 42 \
  --overwrite
```

Final selection can also be reapplied to a persisted search repertoire without rerunning
the search:

```bash
salvi select pipeline.yaml \
  --dataset ./dataset-bundle \
  --repertoire ./run-output/artifacts/search-repertoire \
  --output ./selected-repertoire \
  --overwrite
```

This command prepares and validates the same dataset pipeline, verifies the
repertoire's dataset and prepared-column contract, and writes another canonical
`BiclusterSet` with preserved run and checkpoint provenance. The search
fingerprint deliberately excludes `final_selection`, so selector-only changes
do not invalidate a compatible archive.

`--resume-from-checkpoint` is also a launch-time setting. It is valid only when
the bound dataset and the scientific pipeline fingerprint match the checkpoint.
Validation never creates or modifies a run directory. Relative CLI binding paths
are resolved from the current working directory; paths in pipeline YAML component
parameters remain relative to the pipeline file itself.

## Final repertoire extraction

Omitting `final_selection` reports the search engine's repertoire directly.
There is no pass-through selector component.

`containment_marginal_quality` is the default final-selector strategy for
cardinality QD searches. It builds exact row-and-column containment chains and
tries to replace smaller nested candidates by progressively larger candidates.
The replacement stops when either the worst normalized objective loss exceeds
`max_objective_degradation` or that loss divided by logarithmic area gain
exceeds `max_degradation_per_log_area_gain`. Unrelated containment branches are
never compared, so a globally excellent tiny bicluster cannot suppress a
structurally unrelated larger detection.

Parent selection is independently configurable under `search`:

```yaml
search:
  parent_selection:
    name: cell_uniform_quality
    parameters: {}
```

`cell_uniform_quality` is the default QD policy. It samples an occupied cell
uniformly first and then selects locally, preventing deeper cells from receiving
more reproductive opportunities solely because they contain more candidates.
`repertoire_uniform` reproduces the original global-repertoire baseline. Both
policies preserve every emitter's own eligibility and guidance rules.

## Pattern modes

`allowed` accepts any non-empty, duplicate-free subset of the registered pattern
families. The current catalog contains `CONSTANT`, `ADDITIVE`, and
`MULTIPLICATIVE`. Examples include:

```yaml
patterns:
  allowed: [CONSTANT]
```

```yaml
patterns:
  allowed: [ADDITIVE]
```

```yaml
patterns:
  allowed: [MULTIPLICATIVE]
```

```yaml
patterns:
  allowed: [CONSTANT, ADDITIVE, MULTIPLICATIVE]
```

The defaults are `0.10`, `25`, and `1e-6` for minimum improvement, maximum
alternating-median iterations, and convergence tolerance respectively. The
improvement threshold compares each joint-pattern column against the registered
constant reference model. Boolean and categorical columns currently support only
`CONSTANT`; `ADDITIVE` and `MULTIPLICATIVE` operate on numeric subsets.

## Preprocessing components

Preprocessing is an explicit component hierarchy. `missing_values` selects one
policy. The remaining lists correspond directly to the
`SourceColumnFilteringStage`, `ColumnAugmentationStage`, and
`NumericTransformationStage` protocols. Each list is ordered and optional. The
families execute in the order shown below, and SALVI never inserts a component
that is absent from the YAML:

```text
SourceColumnFilteringStage(s)
  -> MissingValuesPolicy
  -> ColumnAugmentationStage(s)
  -> NumericTransformationStage(s)
```

Available missing-value policies:

- `preserve`: keep Arrow nulls and their support masks unchanged;
- `reject`: stop preparation if any value is missing;
- `median_mode_imputation`: fill numeric nulls with the observed median and
  Boolean or categorical nulls with a deterministic observed mode. It rejects
  all-missing columns. SALVI retains a separate original-observation mask, so
  imputed values never inflate scientific support.

Available column augmentations:

- `missingness_indicators`: append a Boolean `<column>__is_missing` column for
  every source column whose missing ratio is nonzero and lies within the closed
  interval from `min_missing_ratio` to `max_missing_ratio`. The maximum defaults
  to `1.0` for backward compatibility. Indicators are appended after source
  columns and retain their source index.

Available source-column filters:

- `drop_all_missing_columns`: remove columns without any originally observed
  value while retaining the original source-index mapping. Source filters run
  before the missing-value policy, allowing empty columns to be removed before
  median/mode imputation requires an observed prototype.

Available numeric transformations:

- `robust_numeric_scaling`: compute each numeric column's global median, 5th and
  95th percentiles, robust range, and standardized read-only view. Statistics
  use only originally observed values even when imputation is active.

An absent list is the identity operation for that family; no no-op component is
needed. There is still no ordinal categorical type. Components cannot reopen the
configured bundle.

## Evaluation policies

Evaluation-wide rules are components, but they are not data transformations.
They therefore live in the top-level `evaluation` block rather than under
`preprocessing`.

Available candidate-validity policies:

- `minimum_cardinality`: requires at least `min_rows` rows and `min_columns`
  prepared columns and rejects indices outside the prepared dataset. Defaults are
  two rows and two columns.

Available observed-support policies:

- `minimum_observed_support`: requires both `min_observed_count` and
  `ceil(min_observed_ratio * opportunity_count)` usable source observations,
  where the opportunity count is the relevant rows or columns for that
  calculation. The larger requirement wins. Defaults are two observations and
  ratio `0.8`.

The support policy deliberately counts original observations, not values made
available by imputation. Pattern fitters and objectives consume this one shared
policy instead of declaring independent local-support thresholds.

## Scientific objectives

The following objective components are registered:

- `internal_coherence`: no parameters; minimizes the RMS error of the inferred
  per-column patterns;
- `contrast`: maximizes separation from rows outside the bicluster and accepts
  `min_background_ratio` in `[0, 1)`, defaulting to `0.10`;
- `balanced_bicluster_size`: maximizes the harmonic mean of selected-row and
  selected-column coverage. Its per-column explanation is the objective loss
  caused by removing that equally weighted selected column.

For example:

```yaml
search:
  objectives:
    - name: internal_coherence
      parameters: {}
    - name: contrast
      parameters:
        min_background_ratio: 0.10
```

Every scientific objective emits one contribution and diagnostics per selected
column in addition to its aggregate value. Internal coherence and contrast require
the `robust_numeric_scaling` capability so numeric and mixed datasets use the
reviewed global robust scale; balanced size only reads candidate cardinalities.
Objective direction is intrinsic to the component and is persisted in canonical
results; it is not configured by the user.

## Constraints

Constraints are optional `search.constraints` components. They use the signed
convention `value <= 0` for feasibility and persist both the signed value and
derived positive violation. Search engines apply validity first, feasibility and
aggregate violation second, and objective Pareto dominance last.

Available constraints:

- `balanced_bicluster_size_range`: inclusive `minimum` and `maximum` in `[0, 1]`;
- `maximum_internal_coherence`: `maximum_error` in `[0, 1)`;

```yaml
search:
  constraints:
    - name: balanced_bicluster_size_range
      parameters:
        minimum: 0.10
        maximum: 0.40
```

An omitted or empty list means unconstrained search. Structural invalidity from
`minimum_cardinality` remains a hard evaluation error and is not converted into a
configurable constraint.

## QD search and execution

Search-engine composition is validated before any dataset work begins:

| Role | `serial_mome` | `pymoo_nsga2` |
| --- | --- | --- |
| Objectives | one or more | two or more |
| Constraints | zero or more | zero or more |
| Descriptors | one or more | optional |
| Archive | exactly one | forbidden |
| Parent selection | only when consumed by an emitter | forbidden |
| Mate selection | only with `crossover` emitter | forbidden |
| Crossover operator | optional, only when consumed | exactly one |
| Mutation operator | optional, only when consumed | exactly one |
| Emitters | one or more | forbidden |
| Scheduler | exactly one | forbidden |
| Final selector | optional | optional |

Configuring a forbidden role, omitting a required role, or configuring an
optional strategy that no active component consumes is an error. SALVI never
silently ignores a configured component.

`serial_mome` implements the stateful `initialize -> ask -> evaluate -> tell`
search loop. The name refers to its single ordered archive-integration controller,
not to the evaluation executor. `initial_population_size` controls ordinary
initializers and `batch_size` bounds one ask/evaluate/tell batch. A
cell-coverage initializer instead derives its bootstrap demand from reachable
archive cells, `seeds_per_cell`, and `max_attempts_per_cell`; it deliberately
does not interpret one candidate as a population member assigned to each cell.

`pymoo_nsga2` is an experimental conventional multiobjective engine provided by
the optional `evolution` installation extra. It uses one binary variable per row
and prepared column, repairs offspring to the configured minimum cardinalities,
and delegates NSGA-II selection, variation and survival to pymoo. Its engine
parameters are:

- `population_size`, default `64`;
- `eliminate_duplicates`, default `true`.

The current catalog offers `half_uniform_membership` under `search.crossover` and
`bit_flip_membership` under `search.mutation`. Their parameters hold crossover
application and exchange probabilities, and mutation application and per-bit
probabilities. Operators declare their pymoo factory lazily, allowing additional
pymoo wrappers to be registered without changing the engine. SALVI evaluates the
resulting candidates and preserves all pattern
and per-column diagnostics. The engine returns pymoo's feasible non-dominated
front, or its least-infeasible front if no feasible candidate exists. Archive,
parent selection, mate selection, emitters and scheduler are invalid in an
NSGA-II pipeline and fail validation. The population must not exceed the
evaluation budget. `resume_from_checkpoint` and periodic checkpoints are
rejected because this engine does not support exact resumption.

Candidate initialization is selected through `search.initialization`:

- `uniform_random` samples cardinalities and memberships uniformly inside the
  configured candidate-validity bounds. It is retained as a neutral baseline.
- `stratified` distributes attempts over geometrically spaced row and column
  cardinalities before sampling memberships. `cardinality_levels` controls the
  number of levels per dimension. Shapes are drawn from the Cartesian product of
  row and column levels, so off-diagonal combinations are covered rather than
  pairing only equally positioned levels.
- `pattern_aware` uses the same cardinality stratification and creates anchors for
  every allowed pattern family in deterministic round-robin order. Constant
  anchors group rows around a mixed-type reference profile. Additive anchors use
  similar numeric row differences, and multiplicative anchors use similar numeric
  row ratios. `joint_column_candidate_pool_size` bounds the columns inspected by
  one joint anchor. Ground truth is never consulted.
- `cell_coverage_pattern_aware` obtains every reachable representative
  row/column cardinality from the configured archive and requests
  `seeds_per_cell` accepted pattern-balanced seeds in each one. Failed,
  duplicate, or dominated attempts are retried up to `max_attempts_per_cell`.
  It uses the same pattern-specific anchor builders and never consults ground
  truth.
Every generated candidate stores immutable provenance: producer, operation,
generation sequence, parent identifiers, and an optional initialization pattern
hint. The provenance survives checkpoints and BiclusterSet output.

Available variation components are deliberately split by responsibility.
Mate-selection policies are:

- `repertoire_random`, which samples two different repertoire members;
- `cell_first_evidence_compatible`, which samples a source cell before local
  quality, then permits mates only within `cell_neighborhood_radius` that meet
  both overlap thresholds.

Crossover operators are:

- `membership_recombination`, a pattern-agnostic membership recombination;
- `evidence_weighted_recombination`, which favors columns supported by persisted
  coherence and contrast evidence;
- `half_uniform_membership`, the HUX-compatible operator required by
  `pymoo_nsga2`.

`bit_flip_membership` is the current mutation operator. It mutates row and column
membership and repairs configured minimum cardinalities.

Available QD emitters are:

- `add_row`, `remove_row`, `swap_row`, `add_column`, `remove_column`, and
  `swap_column`, each applying exactly the named membership operation;
- `shape_move`, which expands one dimension while contracting the other;
- `crossover`, which invokes the configured mate-selection policy and crossover
  operator and bounds clone-avoidance attempts before a restart fallback;
- `mutation`, which invokes the configured parent-selection policy and mutation
  operator;
- `restart`, which injects either `stratified` or `pattern_aware` candidates;
- `cell_coverage_restart`, which cycles through allowed pattern families and
  injects a pattern-aware candidate into the least represented reachable cell;
- `alternating_pattern_local_search`, which chooses an evaluated parent and
  proposes one guided add, remove, or swap move. If an accepted child from this
  emitter is selected again, the next proposal changes the opposite dimension.
  `cardinality_change_probability` controls how often add/remove is preferred
  over a cardinality-preserving swap. `quality_parent_probability` balances
  exploitation of strong sampled parents against broader parent exploration;
- `random_move`, retained as the neutral uninformed baseline.

Membership and shape emitters accept `guided`, `parent_pool_size`, and
`candidate_pool_size`. Guided
operation only reads the archived parent's persisted pattern fit and per-column
objective contributions. For column additions and swaps it scores a bounded pool
of alternatives against the receiving bicluster instead of choosing one
uniformly. The same bound applies to row and column additions, removals and
swaps. It never invokes objectives or pattern inference.
Candidate validity, including minimum row and column cardinality, always comes
from the configured validity-policy component.

`alternating_pattern_local_search` is not enabled implicitly. It reads the
persisted fit of its selected parent and proposes exactly one move before the
normal `ask -> evaluate -> tell` cycle. It does not run objectives or hidden
evaluations inside candidate generation.

For example, it can be added explicitly as:

```yaml
search:
  emitters:
    - name: alternating_pattern_local_search
      parameters:
        parent_pool_size: 16
        candidate_pool_size: 64
        cardinality_change_probability: 0.25
        quality_parent_probability: 0.25
```

`first` allocates every request to the first emitter and remains a deterministic
baseline. `adaptive_credit` initially samples every emitter and then applies a
deterministic upper-confidence allocation. Its credit is derived exclusively from
archive outcomes:

```text
credit =
  new_cell_reward * newly occupied cells
  + insertion_reward * accepted insertions into occupied cells
```

`exploration_weight` controls continued sampling of less-used emitters. Allocation
counts, feedback and scheduler state are checkpointed, so an interrupted run
resumes exactly.

`cell_balanced_adaptive_credit` is the default QD scheduler. It uses the same
UCB machinery and archive rewards, but multiplies useful work by an
`underexplored_cell_weight` bonus based on the evaluations previously mapped to
that cell. Rejections still count as search effort but do not become scientific
reward.

`fixed_proportion` is the controlled-ablation scheduler. Its `shares` mapping must
name every configured emitter exactly once and sum to one. It follows those
cumulative proportions deterministically, records normal feedback for diagnosis,
and never changes allocation in response to archive outcomes.

`row_cardinality` and `column_cardinality` declare integer semantic domains based
on the prepared dataset and `minimum_cardinality` policy. A descriptor does not
hard-code archive resolution. Instead, `deep_grid_mome.axes` binds each descriptor
to an independent discretization:

- `LINEAR`: requires `bins` and divides the numeric range uniformly;
- `GEOMETRIC`: requires `bins` and allocates more resolution near the positive
  lower bound;
- `EXACT`: creates one possible bin per integer value without preallocating cells;
- `CUSTOM`: requires a strictly increasing `boundaries` list of internal cuts.

Every axis may also specify optional `minimum` and `maximum` values inside the
descriptor domain. Candidates outside an explicitly narrowed axis are not
archived. Axis order defines coordinate order, while descriptor names bind values
unambiguously. All configured descriptors must appear exactly once.

The archive is sparse: a cell is created only after a candidate reaches it.
Unvisited cardinality regions consume no cell storage and are not interpreted as
coverage obligations. Within each occupied cell, objective directions define a
local Pareto front. Exact bicluster duplicates and dominated candidates are
rejected. `cell_capacity` bounds cell depth; deterministic crowding truncation
preserves objective extremes when the local front is full.

`evaluation_budget.max_evaluations` is exact even when it is not divisible by the
engine batch size.

`execution.executor` selects evaluation:

- `serial` requires `workers: 1` and integrates in submission order.
- `thread_pool` shares the immutable prepared dataset and one batch-scoped
  evaluation workspace across a bounded worker pool. `max_in_flight` limits
  submitted but unfinished candidates; when omitted it equals `workers`. It is
  appropriate for lightweight work or kernels that release the Python GIL, but
  can be slower for the current CPU-bound scientific evaluation.
- `process_pool` is the recommended CPU-oriented executor. It starts persistent
  workers using portable `spawn` semantics and transfers one immutable prepared
  runtime snapshot to each worker. Scientific work is then submitted as bounded
  candidate-only tasks. User scripts invoking it must follow Python's standard
  multiprocessing entry-point rule (`if __name__ == "__main__":`); the SALVI CLI
  and GUI already do so.

Both parallel modes accept `max_in_flight`. With
`integration_mode: DETERMINISTIC`, evaluations are returned in submission order
regardless of completion order. `THROUGHPUT` returns completion order and can
consequently change archive insertion, emitter credit and the final repertoire.
Parallel executors process finite engine batches, apply backpressure, and
propagate cooperative cancellation at task boundaries. A pool lives for the
whole run rather than being recreated per batch.

`monitoring.checkpoint_interval_evaluations` optionally writes versioned
checkpoints for engines that declare the `checkpoint-resume` capability. A
checkpoint may contain a generated but unevaluated batch, including
its emitter attribution and post-generation random/scheduler state. Worker failure
or cooperative cancellation writes a recovery checkpoint whenever work is
pending. To resume, pass `--resume-from-checkpoint` at launch time and use a new
output directory.
SALVI verifies the run, dataset, engine, and scientific configuration fingerprint
before restoring. Pending candidates are replayed without generation or
rescheduling. The evaluation budget may be extended when resuming.
`pymoo_nsga2` does not declare this capability and rejects both periodic
checkpoints and resumption.

## Monitoring observers

Observers are passive consumers of durable events. Their metrics are written to
the `metrics` table in `run.sqlite`; they do not modify archive retention,
scheduling or candidate generation.

- `search_progress`: cumulative evaluation-budget consumption only.
- `archive_coverage`: current occupied-cell and retained-repertoire counts.
- `candidate_outcomes`: mutually exclusive retained, invalid, duplicate,
  dominated, capacity and out-of-bounds rates over configurable evaluation
  windows. A rejected candidate was not retained; it is not necessarily
  scientifically invalid.
- `descriptor_distribution`: minimum, quartiles, median, mean and maximum
  descriptor values across every evaluated candidate in each complete sampling
  window.
- `archive_descriptor_distribution`: the same descriptor summaries over
  candidates currently retained in the repertoire, plus the distribution of
  retained members per occupied cell. It separates search effort from archive
  contents.
- `objective_distribution`: the same complete-window summaries for objective
  values.
- `emitter_credit`: current scheduler credit and allocation share together with
  windowed archive-retention and new-cell rates per emitter.
- `candidate_diversity`: exact cumulative uniqueness and windowed duplicate
  counts, plus nearest-neighbour Jaccard distances over a deterministic sample.
  `window_size` controls the exact rolling counts; `distance_sample_size`,
  `row_weight` and `every_evaluations` bound the quadratic distance cost.
- `evaluation_issues`: windowed valid/invalid rates and the fraction of evaluated
  candidates affected by each scientific issue. Issue categories may overlap.
- `component_timing`: wall-clock attribution for loading, preprocessing,
  initialization, candidate generation, individual and batched scientific
  evaluation, objectives, descriptors, constraints, search updates, observers,
  final selection and artifact serialization. `every_evaluations` bounds
  windowed timing output.
- `qd_archive_diagnostics`: visits, acceptance, turnover and stagnation globally
  and, optionally, for every visited QD cell.
- `runtime_throughput`: evaluations per second, active worker capacity and peak
  in-flight work reported by the executor. Detailed duration attribution belongs
  exclusively to `component_timing`.
- `resource_usage`: process CPU time, interval CPU percentage, active thread count
  and resident memory when the platform exposes it. `every_evaluations` controls
  sampling frequency.
Observers that expose `every_evaluations` sample when an evaluation batch crosses
the next configured threshold. They therefore retain the requested cadence even
when the batch size does not divide it exactly. Event delivery uses a bounded
best-effort observer queue; durable search events are never dropped when a
monitor is slow.

Every observer metric is described in the component catalog with its unit,
value kind, temporal scope and observed population. Visual clients must not
place metrics with incompatible semantics on one axis. In particular,
cumulative counters, current gauges, windowed rates and batch distributions are
separate concepts even when all happen to contain numeric values.

## Commands

```bash
salvi validate pipeline.yaml --dataset dataset-bundle
salvi inspect pipeline.yaml --dataset dataset-bundle
salvi run pipeline.yaml --dataset dataset-bundle --output run-output
salvi run pipeline.yaml --dataset dataset-bundle --output run-output \
  --progress always --monitor-interval 1.0
salvi select pipeline.yaml --dataset dataset-bundle \
  --repertoire run-output/artifacts/search-repertoire \
  --output selected-repertoire
salvi profile pipeline.yaml profile-output --dataset dataset-bundle \
  --output profiled-run-output --repetitions 3 --overwrite --run-overwrite
salvi components --kind search_engine
salvi config format pipeline.yaml
salvi schemas
salvi gui
salvi gui --port 6087 --no-open
salvi-exp convert gbic SOURCE DESTINATION
```

`validate` verifies the YAML, DatasetBundle, registered components, requirements,
capabilities, descriptor domains, and archive axes. `inspect` performs the same
scientific composition and preprocessing, then reports dimensions, component
roles, descriptor domains, reachable archive cells, and termination semantics
without starting the search or writing run artifacts. `run` executes the
configured scientific search and always writes `artifacts/repertoire`. When a
final selector is configured, it also writes `artifacts/search-repertoire` with
the unfiltered engine result; otherwise `repertoire` is that raw result directly.
During `run`, the CLI reads the same `run.sqlite` event store used by the GUI and
prints concise progress to `stderr`. The default `--progress auto` enables this
only for interactive terminals, `--progress always` also prints in captured
sessions, and `--quiet` disables it. The final JSON summary remains on `stdout`
for scripts.
`salvi-exp convert gbic` converts either one paired G-Bic dataset or a directory tree into
canonical DatasetBundles. It preserves constant, additive, multiplicative, and
mixed planted
patterns and accepts `--overwrite` only when replacement is intentional.

## Experiment manifests

Experiment manifests live in the separate `salvi-experiments` package. Each
dataset or benchmark case references a reusable pipeline YAML and a DatasetBundle.
Manifests may group pipelines, but they cannot patch scientific components or
inject implicit variables. Dataset, output, identifier, and seed are explicit
job bindings; every execution persists its fully bound effective configuration.
