# Scientific contract

This document fixes the scientific behavior implemented by SALVI's evaluation
layer. The QD search that will generate candidates is introduced in later phases;
the fitters and objectives described here are already executable components.

## Data semantics and missing values

SALVI accepts numeric, Boolean, and nominal categorical columns. Ordinal
categorical variables are outside the current contract. Original column names and
category labels are preserved.

Missing-value handling is explicit and never inferred from the dataset. The
recommended `preserve` policy keeps nulls unavailable and ignores them where a
statistic can be estimated with sufficient original support. `reject` accepts only
complete datasets. `median_mode_imputation` is available for controlled
ablations: it uses the observed median for numeric columns and a deterministic
observed mode for Boolean and categorical columns, and rejects columns with no
observed prototype.

SALVI always retains two masks. The source-observation mask records which values
were genuinely present; the availability mask records which runtime values can be
used after the selected policy. Imputation changes only availability. Consequently
it cannot make a candidate satisfy an observed-support requirement that the
source data did not satisfy.

The optional `missingness_indicators` augmentation creates explicit Boolean
features from the source-observation mask. It is independent of the selected
missing-value policy, so indicators remain correct after imputation. The optional
`drop_all_missing_columns` filter removes columns for which no source prototype can
be estimated. Every derived or retained prepared column preserves its canonical
source index.

Numeric columns are standardized once using the global median and robust range
between the 5th and 95th percentiles. Scales at or below `1e-12` receive the
documented zero-scale treatment rather than division by an unstable value.
This calculation is implemented as the explicit `robust_numeric_scaling`
preprocessing stage; scientific objectives declare the resulting
`robust-numeric-data` capability.

Candidate structure and observed support are independent policies. The default
candidate-validity component requires at least two rows and two columns. The
default support component requires at least two original observations and an
original-observation ratio of `0.8`. This prevents a one-value column from
receiving perfect constant coherence. Fitters may impose stronger
pattern-specific mathematical requirements, such as two observations for an
additive row effect, but may not weaken the configured global policy.

## Allowed patterns

A run allows any non-empty combination of the registered constant, additive, and
multiplicative patterns. Pattern assignment is inferred during candidate
evaluation and is not part of the genotype.

Constant patterns apply to every supported column kind:

- numeric columns use a robust local prototype;
- Boolean and categorical columns use the local mode and original label.

Additive patterns apply only to numeric columns. A bicluster contains at most one
joint additive subgroup. The model is fitted on original values:

```text
x_ij = alpha_i + beta_j + residual_ij
```

Alternating medians estimate the row and column effects, recenter row effects to
median zero, ignore missing observations, and stop on convergence or iteration
limit. The global robust range of each column normalizes its residual only after
model fitting. Columns are not independently centered or scaled before fitting:
doing so would turn every positive affine or proportional relation into an
apparently perfect additive pattern.

Multiplicative patterns also apply only to numeric columns and currently form at
most one joint subgroup. They model proportional evolution without taking
logarithms:

```text
x_ij / scale_j = alpha_i * beta_j + residual_ij
```

`scale_j` is the global `P95 - P05` range. A zero-range column uses
`max(abs(global_median), 1)` as a numerically safe scale, but must still improve
over the constant reference to enter a mixed group. Initialization uses the
available column with greatest robust dispersion as a deterministic proportional
anchor and is linear in the fitted matrix size. Alternating medians of ratios then
estimate row and column effects. Effects whose denominators are numerically zero
do not supply information to that update.
Identifiability is fixed by making the median absolute nonzero row effect equal to
one and choosing a deterministic sign. This supports zero and negative data and
avoids a log transform that would change the accepted data domain.

Pattern implementations are registered in an explicit catalog. Each entry couples
its fitter and contrast strategy, and may provide a bounded group-candidate
generator based on the pattern's own invariants. Its definition declares supported
column kinds, whether it fits one `COLUMN` independently or a joint `SUBSET`, its
minimum column count, its maximum number of groups, and whether it is the reference model.
Exactly one reference model is required. This metadata lets inference and contrast
dispatch decide which implementations may compete without assuming that every
future pattern is simply "not constant". Adding another joint pattern requires a
fitter, a contrast strategy, and one catalog entry, not binary rewrites of mixed
inference or contrast.

Pattern-aware initialization remains separate from this scientific catalog. Its
current anchor builder has explicit behavior for every registered family and
rejects an unsupported family rather than silently treating it as another
pattern. Anchor generation only proposes structures; it never calls objectives or
inspects ground truth.

In the current mixed mode, nominal and Boolean columns are constant. Numeric
columns begin as candidates for every compatible implementation. Column-scoped
fits establish independent references. Each built-in joint model proposes a
bounded set of column neighborhoods from its invariant profile: raw centered
profiles for additive shifts and robust-scaled proportional profiles for
multiplicative effects. The assignment engine fits those proposals, enforces each
model's minimum cardinality, partitions overlaps by normalized improvement, and
refits the selected disjoint groups. A future joint pattern supplies its own
candidate generator through the same catalog contract; inference contains no
pattern-specific branch.

Winning and losing alternatives are retained for explanation. If two models are
observationally equivalent within numerical tolerance, one deterministic
assignment is used operationally and the column diagnostics explicitly report the
ambiguity, equivalent families, and error margin. Additive and multiplicative
subgroups need at least two informative columns and each row effect needs at least
two observed members. Remaining numeric columns use the constant model when it is
allowed. The current additive and multiplicative definitions each permit at most
one group per bicluster; this is catalog metadata rather than an inference-engine
assumption.

## Shared evaluation

One immutable `PatternFit` is computed for each exact row/column signature. It
contains every tested alternative and, per column, the final assignment, error,
parameter and scale, original and available support, optional joint-group identity,
and diagnostics. Joint groups store their columns, row parameters, iteration count,
and convergence state. A batch-scoped `EvaluationWorkspace` shares that fit and a
derived contrast result while a candidate is evaluated. The resulting `Evaluation`
owns the durable fit used later by archives, emitters, observers, and output.

Single-pattern workspaces construct only their requested implementation and the
lightweight reference model needed for evidence comparison. A constant-only
workspace constructs no joint fitter and skips joint assignment entirely.

## Internal coherence

Each selected column contributes the error of its inferred pattern. Constant
numeric columns use absolute deviations from the local median divided by the global
`P95 - P05` range and clipped to one. Constant categorical and Boolean columns use
mode disagreement normalized by the number of distinct values observed globally,
with deterministic semantic-order tie handling. Additive columns use raw-model
residuals divided by their global robust range. Multiplicative columns use
residuals from their robust-scaled rank-one model. Both therefore share the same
dimensionless scale and are clipped to one.

The objective aggregates column errors with a root-mean-square operation. This
makes one badly fitted column matter without allowing the number of selected
columns to change the scale. Zero is perfect and one is the invalid or worst fit;
the objective direction is `MINIMIZE`.

## Balanced bicluster size and constraints

The optional `balanced_bicluster_size` objective is the harmonic mean of selected
row coverage and selected prepared-column coverage. It reaches one only for the
complete matrix and rewards balanced growth instead of allowing one dimension to
compensate linearly for the other. Its column explanation is the marginal loss
caused by removing each equally weighted selected column.

The recommended MOME formulation still uses both cardinalities as descriptors;
this objective exists for conventional controls and explicit ablations. The same
scalar is reused by `balanced_bicluster_size_range`, which makes values outside an
inclusive interval infeasible without scanning matrix values.
`maximum_internal_coherence` similarly constrains the existing RMS fit error and
reuses the candidate's cached `PatternFit`.

## Contrast

Contrast compares the fitted local pattern with rows outside the bicluster. Its
scale is common across column kinds: `0` is inverse separation, `0.5` is neutral,
and `1` is correct maximal separation.

- Constant numeric contrast compares local and background residual distributions.
- Categorical and Boolean contrast compares bilateral local/background prototype
  frequencies without treating category codes as ordered numbers.
- Additive contrast tests whether the fitted joint profile is reproduced by
  external rows after estimating their row effects.
- Multiplicative contrast estimates an external proportional row effect from the
  selected subgroup and compares local and background rank-one residuals.

Insufficient observed support, an invalid fit, or too little usable background is
penalized explicitly. Contrast is not multiplied by the fraction of rows left in
the background; candidate size is represented separately by QD descriptors.
The objective direction is `MAXIMIZE`.

## Column-level explanations

Every scientific objective returns both its aggregate value and one value for each
selected column in canonical column order. Internal coherence publishes the fitted
pattern error. Contrast publishes the separation score produced by that same
assignment. Each contribution carries a validity flag and calculation diagnostics;
invalid columns are not silently replaced by neutral values.

Canonical BiclusterSet output stores these contributions in
`column-objectives.parquet`. `column-patterns.parquet` stores assignments,
alternatives, model parameters, supports, and group identities;
`pattern-groups.parquet` and `pattern-row-parameters.parquet` store generic joint
model information. Output reuses the immutable evaluation and never refits a
bicluster, so its explanation is exactly the one used to score it.

## QD descriptors

Row cardinality and column cardinality are independent behavior descriptors in the
recommended QD formulation. SALVI does not add the optional balanced-size objective
or constraint unless they are explicitly configured.
The future archive will bound and group descriptor space so evaluations are not
wasted on every exact cardinality pair.

Ground truth is reserved for evaluation protocols and must never influence pattern
inference, archive boundaries, emitters, selection, or termination.
