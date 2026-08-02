# Architecture

## Design goals

SALVI separates scientific policy from search orchestration and infrastructure.
Components can be replaced independently, while the search engine retains the
intrinsic `initialize -> ask -> evaluate -> tell -> archive -> terminate` lifecycle.
Neither the CLI nor the GUI owns execution logic; both call `RunService`.

The constant, additive, multiplicative, and mixed scientific evaluation layer is
operational. A deterministic serial MOME engine generates candidates and retains
them in a sparse bounded archive. Read-only pipeline inspection composes the real
scientific components and reports dataset-dependent domains without starting a
search or creating run artifacts.

## Package boundaries

`salvi` contains domain models, component contracts, composition, execution,
canonical artifacts, CLI, and the optional GUI. `salvi-experiments` is a sibling
package that may depend on the public `salvi` API. It owns scientific experiment
protocols and external interoperability adapters such as GBIC and HBIC. The
reverse dependency is forbidden.

The core is arranged around these boundaries:

- `domain`: immutable scientific and run models.
- `components`: protocols and explicit registrations.
- `api`: fluent programmatic composition.
- `application`: configuration, composition, and `RunService`.
- `patterns` and `evaluation`: reusable scientific contracts.
- `engine`: lifecycle implementations.
- `infrastructure`: Parquet, YAML, JSON, SQLite, and filesystem adapters.
- `cli` and `web`: replaceable presentation adapters.

`SALVI-old` is not imported, linked, or executed. Scientific parity is protected by
versioned fixtures whose expected values are reviewed and stored in this
repository.

## Composition

`SalviRun.builder(dataset)` accepts unique components through `with_*` methods and
collections through `add_*` methods. `build()` validates required cardinalities and
component capabilities, then returns an immutable `RunSpecification`. Optional
builder arguments carry the run identifier, seed, allowed-pattern configuration,
and worker count into the same specification received by the engine; YAML values
are therefore never hidden in presentation-layer state.

Configuration-driven construction uses an explicit `ComponentRegistry`. A
registration declares:

- component kind and public name;
- strict Pydantic configuration model;
- factory;
- provided capabilities;
- required capabilities.

There is deliberately no implicit import scanning or entry-point discovery in the
foundation release. This keeps composition deterministic and makes unsupported
configuration fail before execution.

Component families are represented by concrete protocols and registry kinds, not
by naming conventions. The hierarchy is intentionally shallow:

```text
Component protocol
  -> concrete registered implementation
     -> private strategy object only when the implementation has a genuinely
        interchangeable internal algorithm
```

For example, preprocessing exposes `MissingValuesPolicy`,
`SourceColumnFilteringStage`, `ColumnAugmentationStage`, and
`NumericTransformationStage` directly. A generic "preprocessor" is not a public
registry category. This keeps YAML classification, protocol ownership, and code
location aligned without turning every helper into a configurable component.

## Execution and observation

`RunService` is the only execution gateway. It owns configuration and artifact
I/O, applies the missing-values policy and ordered preprocessing pipeline exactly
once, drives the engine protocol, persists events, and coordinates cancellation.
Execution follows:

```text
configuration
  -> canonical dataset validation
  -> registered component composition
  -> one canonical Parquet load
  -> missing-values policy / ordered preprocessing components
  -> immutable RunContext construction
  -> initialize / ask / evaluate / tell / archive / result
  -> event and metadata persistence
```

The `SearchEngine` owns algorithmic state and the semantics of `ask`, `tell`,
archive updates, emitter scheduling, and termination. It never reloads data,
reapplies preprocessing, writes artifacts, or calls presentation code. An
`EvaluationExecutor` evaluates one submitted batch. Every batch receives a fresh
`EvaluationWorkspace`; durable fit information is copied into each `Evaluation`
instead of retaining an unbounded run-wide cache.

The engine runs outside the GUI process. Events are immutable, typed, and written
by one SQLite writer using WAL mode. Monitors use a replayable event source and
never access engine internals. The same source exposes incremental typed metrics
and the current artifact index. Observer callbacks are isolated behind a bounded
queue; durable events are never dropped when a monitor is slow. This supports
real-time charts and bicluster inspection without coupling scientific code to a
particular user-interface toolkit.

The local FastAPI application exposes registry metadata, partial composition,
uploads, execution, events and artifacts through versioned endpoints. Its React
client builds the same reusable YAML accepted by the CLI. Scientific execution
occurs in a spawned child process. Live charts and event history consume only the
SQLite event-source contract, while final inspection uses a paged canonical
BiclusterSet reader and bounded source-matrix fragments. Optional input adapters
and post-run analyses are discovered from installed providers, preserving the
one-way dependency from `salvi-experiments` to `salvi`.

## Scientific lifecycle

The search engine owns lifecycle and scheduling. Components own scientific
choices:

- objectives decide quality;
- constraints define feasibility without becoming optimization dimensions;
- descriptors map candidates into QD behavior space;
- archives decide cell retention;
- emitters generate candidates;
- schedulers allocate evaluation credit;
- final selectors extract the reported bicluster set.

Objective direction is explicit. Every objective declares `MINIMIZE` or
`MAXIMIZE`, and every persisted `ObjectiveValue` carries that direction. Archives
must therefore compare raw scientific values without relying on undocumented
sign changes or objective ordering.

The recommended QD formulation uses row and column cardinality as descriptors.
The catalog also exposes balanced bicluster size as an objective and as a bounded
constraint for conventional controls and explicit ablations; neither is inserted
implicitly.
Pattern inference is shared through a per-evaluation workspace so objectives and
future emitters or observers reuse one exact fit for a candidate. The immutable
fit and every per-column objective contribution are copied into `Evaluation`, so
output never depends on a live cache or repeats scientific calculations.
Concurrent requests for the same signature share one in-flight result, while
different signatures are never serialized by a global inference lock. Objective
results are also cached by candidate signature and objective name, allowing
constraints, operators and output writers to reuse completed scientific work.

Descriptors own behavioral meaning and declare their value kind, valid domain,
supported discretization strategies, and a recommended default. The archive owns
the actual discretization used by a run. Each configured archive axis binds one
descriptor to its own strategy and resolution, so row cardinality may use
geometric bins while column cardinality uses linear, exact, or manual boundaries.
This keeps semantic validation close to the descriptor without freezing an
experiment-specific resolution inside it.

`DeepGridMomeArchive` stores only occupied coordinates. Each cell contains a
bounded local constrained Pareto front evaluated with the directions persisted by
the objectives. Feasible candidates dominate infeasible candidates; among
infeasible candidates lower aggregate violation is preferred. Exact structural
duplicates, invalid evaluations, dominated
candidates, and values outside explicitly narrowed axes are rejected through
typed insertion outcomes. Empty regions are neither allocated nor treated as
required coverage. When a cell exceeds its configured depth, deterministic
crowding preserves objective extremes and removes one interior member.

Final extraction is a separate component and cannot alter search state. The
default `AdaptiveResidualEvidenceCoverSelector` removes exact duplicates and
greedily covers still-unexplained observed matrix cells with quality-weighted
evidence, while penalizing overlap, weak columns, and membership complexity. Its
quality floor is inferred from the terminal repertoire without ground truth.
`ContainmentMarginalQualitySelector` remains an alternative that consolidates
exact row-and-column containment chains under objective-loss limits. Each
reported evaluation retains its archive coordinate and a normalized selection
record linking consolidated source candidates back to the terminal checkpoint.
Selectors consume the terminal search repertoire and return a canonical
`Repertoire`; they cannot evaluate new candidates or mutate search state.

`SerialMomeSearchEngine` owns initial and pending candidates, evaluation counters,
archive updates, emitter scheduling, exact budget handling, and checkpoint state.
`RunService` drives evaluation and I/O. Checkpoints include the archive, remaining
initialization, an optional generated batch with exact emitter attribution,
deterministic random-stream and scheduler states, counters, integration mode, and
a scientific configuration fingerprint. Resuming a pending checkpoint replays
the candidates without invoking initialization, emitters or scheduling again.

`PymooNsga2SearchEngine` is an optional conventional multiobjective control behind
the same `SearchEngine` protocol. It translates SALVI biclusters to binary
row/column membership vectors and delegates NSGA-II selection, variation and
survival to pymoo. SALVI still owns initialization,
scientific evaluation, parallel execution, events, explanations, artifacts and
final selection. The currently registered `half_uniform_membership` crossover and
`bit_flip_membership` mutation expose lazy pymoo factory specifications. The engine
depends on that generic provider contract rather than concrete operator classes,
so future pymoo-backed catalog entries do not require NSGA-II changes. The adapter
returns NSGA-II's constrained non-dominated front through the
canonical `Repertoire` contract. Archive, parent selection, mate selection,
emitters and scheduler are forbidden for this engine rather than silently
ignored. This adapter deliberately does not support checkpoint resumption.

Each search-engine registration owns an explicit composition contract. Roles not
listed by that contract are forbidden, mandatory roles must be present, and
optional variation strategies are rejected unless an active engine or emitter
declares that it consumes them. MOME requires objectives, descriptors, one
archive, one or more emitters and a scheduler. NSGA-II requires at least two
objectives plus one crossover and one mutation operator, and explicitly forbids
descriptors because it has no QD behavior space. Both engines accept zero or more
constraints. Every engine also declares a search family. The catalog registers
one default engine per family, allowing clients to request an architecture
transition without encoding engine names or compatibility rules. A missing optional
stage, including final selection, is represented by its absence rather than by a
null or pass-through component.

Candidate generation is provenance-first. The engine assigns one monotonic
generation sequence, initializers and emitters declare their producer, operation,
parents and optional pattern hint, and the resulting immutable record travels with
the candidate through evaluation, checkpoints and canonical output. Emitters are
stateless scientific transformations: optional guidance reads only persisted
`PatternFit` and per-column objective contributions from archived evaluations.
They never call objectives or pattern inference.

Parent selection is a component shared by single-parent emitters. The baseline
samples uniformly from every eligible repertoire member. A QD-aware policy may
instead sample an occupied cell uniformly before applying the emitter's local
random or quality-guided preference. Pairwise variation is decomposed into a
mate-selection policy, a reusable crossover operator and the generic QD
`crossover` emitter. Mutation is similarly decomposed into a reusable mutation
operator and the generic `mutation` emitter. This lets conventional engines and
QD emitters share the same public operator catalog without making emitters part
of NSGA-II.

Schedulers allocate explicit evaluation counts. After `tell`, the engine
attributes typed archive insertion outcomes to the emitter that produced each
candidate and updates scheduler credit. The adaptive scheduler rewards newly
occupied cells more than insertions into existing cells while retaining
deterministic exploration and tie handling. Its complete statistics are part of
the checkpoint.

Observers derive metrics from immutable events on the bounded
observer queue. The SQLite event is persisted before any callback runs, and
observer-derived metrics reference that event sequence. Observers are passive:
coverage, objective/descriptor distributions, emitter credit and candidate
diversity cannot affect the engine.

Evaluation executors are independent of the search engine. The serial
executor is the reference path. The thread-pool executor shares the read-only
prepared dataset and thread-safe batch workspace, bounds unfinished work, and
keeps its workers alive across batches. The process-pool executor uses portable
spawned workers, initializes each with one immutable runtime snapshot, and sends
only bounded candidate tasks thereafter; it is the CPU-oriented path for the
current Python evaluators. Deterministic integration restores submission order
after arbitrary completion; throughput integration deliberately preserves
completion order. Runtime, throughput and resource observers consume only durable
events and therefore cannot influence either policy.

Pattern implementations are registered in a deterministic catalog. Each one
couples its fitter and contrast strategy with compatible column kinds and a
column-local or joint-subset scope. Exactly one column-scoped implementation is
declared as the scientific reference model; inference does not hard-code its
identity. The current generic assignment strategy compares local alternatives,
iteratively prunes joint proposals, resolves overlaps deterministically, retains
all stable competing alternatives, and stores explicit group membership.
Inference and contrast can therefore support future pattern families without a
constant-versus-additive decision tree in either objective.

## Prepared data boundary

`Dataset` is canonical source identity and metadata. `PreparedDataset` is the sole
in-memory scientific representation. `RunService` loads the Parquet matrix once,
constructs read-only Arrow and NumPy views, including precomputed integer codes and
global frequencies for Boolean and categorical columns, applies one missing-values policy and
the configured ordered preprocessing stages, and then creates one immutable
`RunContext` shared by the engine and all downstream scientific components.

The base context carries prepared data, pattern settings, deterministic named
random streams, candidate validity, and observed-support policy. It deliberately
carries no ground truth, GUI, event store, experiment object, archive, or
variation strategy. Search families enrich this boundary with their own private
runtime state; for example, MOME creates a `QdRunContext` for parent selection,
variation, and archive-cell targets. Components outside that family never depend
on QD state. Component factories continue to receive configuration only; runtime
scientific dependencies are passed explicitly. No component independently opens
a DatasetBundle.

`PreparedDataset` has separate read-only masks for values observed in the source
and values available after preprocessing. This distinction is permanent:
imputation may make a value available, but it cannot increase original support.
Runtime column metadata also records each prepared column's source index and
derivation, allowing augmentation and filtering without losing canonical
provenance. The two masks share one immutable NumPy allocation until a configured
policy actually changes availability.

Preprocessing follows the same composition model as the rest of SALVI. A run
selects ordered source-column filters, exactly one `MissingValuesPolicy`, then
ordered lists of column augmentations and numeric transformations. Source filters
run first so unusable columns can be removed before a policy needs to estimate a
replacement. Each component consumes and returns `PreparedDataset`; there are no
hidden transformations. The built-ins currently comprise:

- `preserve`, which keeps nulls unavailable and visible to support-aware code;
- `reject`, which fails when any null is present;
- `median_mode_imputation`, which fills values but preserves source support;
- `missingness_indicators`, which appends provenance-aware Boolean columns;
- `drop_all_missing_columns`, which removes unusable prepared columns;
- `robust_numeric_scaling`, which adds global median and `P95 - P05` statistics
  plus a read-only standardized numeric view.

An omitted preprocessing family performs no work, so no identity component is
registered. Robust statistics always use source observations; available imputed
values are transformed using those statistics. All-missing numeric columns remain
missing unless an explicit filter removes them, and imputation rejects them
  because no defensible prototype exists. Numeric columns with robust range at or
  below `1e-12` map values at the median to zero and rare finite deviations to
  `-1` or `1`; unavailable values remain unavailable.

Candidate validity and evaluation support are separate component families. The
former owns structural requirements such as minimum row and column cardinality;
the latter owns the shared original-observation threshold used by future fitters
and objectives. Both are placed in `RunContext` because they are runtime
scientific policy, whereas `RunContext` itself remains an internal immutable
dependency container rather than a user-selectable component.

### Preprocessing performance baseline

Every run records loading time, each preprocessing component's time,
and approximate prepared-memory ownership in `run-metadata.json`; these values are
the portable baseline for the machine executing SALVI. As an illustrative
development measurement, a heterogeneous GBIC matrix with 1000 rows, 500 columns
(252 numeric and 248 categorical) took a median 0.272 seconds to verify and load
and 0.041 seconds to construct and robustly scale over five warm runs on Python
3.13.5 and an Intel Xeon Platinum 8358. The Arrow table occupied 3.54 MB, the base
prepared representation 6.06 MB, and the scaled representation 8.08 MB. These
figures are not acceptance thresholds; CI tests the storage accounting and
deterministic values rather than unstable wall-clock limits.

Archive, descriptor, termination, checkpoint, progress, candidate provenance,
emitter feedback, scheduler credit, and bounded parallel submission have explicit
ownership contracts. Component registrations are checked against their concrete
protocol, and capability requirements are validated in lifecycle order so
circular or backward dependencies fail at build time.

## Acceleration boundary

SALVI does not expose a kernel-backend component while Python is the only real
implementation. The evaluation executor is the public runtime choice because it
changes execution semantics and resource use. Native acceleration will become a
component only after profiling identifies a stable kernel, a second implementation
exists, and parity tests can define a meaningful user choice.
