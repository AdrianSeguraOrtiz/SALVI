# Scientific experiments

SALVI separates scientific search from evaluation protocols. The core package
produces and consumes canonical `DatasetBundle` and `BiclusterSet` artifacts.
`salvi-experiments` measures objective alignment and biclustering accuracy without
changing a reusable pipeline or using ground truth during search.

## Installation

From the repository root:

```bash
uv sync --all-packages --all-extras --dev
uv run salvi-exp --help
```

All experiment YAML paths are relative to the YAML file itself. Unknown and
duplicate fields are rejected.

The CLI reports concise progress to `stderr` during long-running protocols and
prints only the final result directory to `stdout`. Use `salvi-exp --quiet ...`
when a pipeline should suppress progress messages entirely.

## Task scope

Every protocol declares the ground-truth patterns included in its scientific task:

```yaml
task:
  included_patterns: [CONSTANT, ADDITIVE, MULTIPLICATIVE]
```

This is the default and includes mixed biclusters. A bicluster is retained only
when every declared column pattern belongs to the task. Restricting the list is
therefore an explicit change of scientific question, not an invisible filtering
option. The effective scope is stored in every report and manifest.

## Objective alignment

Dataset-level objective alignment evaluates known ground-truth biclusters using
the exact preprocessing, pattern inference, support policies, objectives, and
executor declared by a reusable SALVI pipeline YAML, explicitly bound to the
experiment's DatasetBundle. It does not initialize or run a search engine.

For every ground-truth bicluster it also generates:

- random controls with matching row and column cardinality;
- controls formed by removing rows or columns;
- controls formed by adding rows or columns when possible.

The exact candidate values are written to `candidates.parquet` and
`candidates.csv`. `objective-alignment.parquet` summarizes, per objective and
ground-truth bicluster, the fraction of controls that are no better than the
ground truth and the direction-aware objective improvement. PNG and SVG figures
show distributions and favorable fractions.

```bash
uv run salvi-exp dataset objective-alignment \
  examples/experiments/objective-alignment.yaml
```

Use the benchmark command to repeat the same protocol over reusable pipelines
and explicit DatasetBundles:

```bash
uv run salvi-exp benchmark objective-alignment \
  examples/experiments/objective-alignment-benchmark.yaml
```

The benchmark never overrides a referenced pipeline. Each case supplies its own
DatasetBundle, so the same component design can be evaluated on multiple data
sets without embedding paths in the pipeline YAML.

## Accuracy

Accuracy compares any canonical `BiclusterSet` with the optional ground truth in
its corresponding `DatasetBundle`.

- **REL** (relevance) asks how well each reported bicluster matches its best real
  bicluster. Extra weak detections reduce REL.
- **REC** (recovery) asks how well each real bicluster is represented by its best
  reported bicluster. Extra detections do not reduce REC.
- **BE** (biclustering error complement) uses a one-to-one optimal matching of
  bicluster cell memberships and accounts for overlap multiplicity.

Row and column similarities use Jaccard overlap and are combined geometrically.
All three main scores range from zero to one and are better when larger.
Target-coverage values report the fraction of real biclusters whose best
structural match reaches each configured threshold. REL and REC intervals
resample the biclusters defining their respective averages; BE intervals
independently resample reported and ground-truth biclusters before repeating the
optimal assignment.

The protocol writes exact metrics, confidence intervals, every best match, PNG
and SVG figures, a JSON report, and a checksummed manifest:

```bash
uv run salvi-exp dataset accuracy examples/experiments/accuracy.yaml
```

Algorithm metadata explicitly records whether the real target count was known,
the evaluation and wall-clock budgets, observed time and memory when available,
and post-processing and final-selection policies. Missing resource measurements
remain null; they are never inferred from standard output.

## Clinical datasets

`salvi-exp convert uci` imports an official UCI resource through a strict,
checksum-pinned curation recipe:

```bash
uv run salvi-exp convert uci uci-import.yaml clinical-dataset
```

The recipe declares UCI ID, missing tokens, explicit mappings, column roles and
derived annotations. It does not contain a SALVI pipeline. The resulting
`ClinicalDatasetBundle` keeps only `SEARCH` columns in its nested
`DatasetBundle`; `OUTCOME`, `COVARIATE` and `SUPPLEMENTARY` values remain
external to search.

A completed raw or final `BiclusterSet` can then be characterized independently:

```bash
uv run salvi-exp dataset clinical-validation clinical-validation.yaml
```

The strict configuration binds one `ClinicalDatasetBundle`, one
`BiclusterSet`, one output directory and testing thresholds. The protocol
reports cardinalities, pattern composition, missingness-indicator use, Fisher
tests with odds ratios and risk differences, chi-squared tests with Cramér's V,
Mann-Whitney tests with rank effects, and survival log-rank tests with
univariate hazard ratios. Benjamini-Hochberg correction is applied to the
evaluable associations. Biclusters below configured member, nonmember or event
counts are retained as explicitly non-evaluable rather than silently removed.

`calculate_repertoire_stability` is the reusable API for comparing independent
results or subsamples. It performs one-to-one structural matching with:

```text
sqrt(row_jaccard * column_jaccard) * pattern_concordance
```

and reports coverage at configurable thresholds. Benchmark-level clinical
orchestration and dataset-specific curation recipes deliberately live outside
the reusable package.

## Benchmarks and comparison

Benchmark protocols run cases sequentially by default. They can orchestrate
independent cases in parallel with:

```yaml
execution:
  workers: 4
  allow_nested_parallelism: false
  allow_cpu_oversubscription: false
```

Parallel benchmark execution is available for `benchmark objective-alignment`
and `benchmark accuracy`. Dataset-level commands do not add an extra orchestration
layer; `dataset objective-alignment` uses the executor declared in the referenced
SALVI pipeline YAML. `benchmark compare` only aggregates existing tables and remains
sequential.

When benchmark workers are greater than one, `salvi-experiments` checks the
referenced SALVI pipeline configurations. It rejects nested parallelism or CPU
oversubscription independently. `allow_nested_parallelism: true` permits running
multiple benchmark cases while each referenced SALVI pipeline also uses more than one
worker. `allow_cpu_oversubscription: true` separately permits configurations
whose estimated active workers exceed the detected CPU count.

An accuracy benchmark runs the dataset protocol for one algorithm over multiple
cases:

```bash
uv run salvi-exp benchmark accuracy \
  examples/experiments/accuracy-benchmark.yaml
```

It aggregates REL, REC, BE, target coverage, detected cardinality, and available
time and memory fields with descriptive statistics and bootstrap confidence
intervals.

Algorithm comparison consumes existing dataset- or benchmark-accuracy result
directories:

```bash
uv run salvi-exp benchmark compare examples/experiments/comparison.yaml
```

All algorithms must cover exactly the same dataset identifiers and task scope.
The first algorithm is the declared baseline for paired REL, REC, and BE deltas.
The comparison writes machine-readable Parquet and CSV tables plus PNG and SVG
figures.

## SALVI ablations

`benchmark ablation` compares two or more complete SALVI pipeline YAMLs over the
same selected DatasetBundles:

```bash
uv run salvi-exp benchmark ablation examples/experiments/ablation.yaml
```

The experiment never patches component parameters. Population size, batch size,
evaluation budget, archive capacity, component instances, and every other
scientific choice remain part of each referenced pipeline. This permits fair
comparisons of either one component or a complete design while preserving the
exact effective configurations.

Dataset and stochastic-run selection are independent:

```yaml
benchmark_root: /path/to/benchmark
datasets:
  replicates: [101, 103]
  identifiers: [EXP-1_NL20_101, EXP-4_PA_101]
run_seeds: [7, 19, 31]
```

`datasets.replicates` filters dataset identifiers ending in `_101`, `_102`, and
so on. `datasets.identifiers` optionally restricts the result further to exact
identifiers. `run_seeds` repeats SALVI independently on every resulting
pipeline/dataset pair. Use `replicates: ALL` and an empty identifier list to
include the complete benchmark.

Pattern handling must be explicit:

- `pattern_binding: PIPELINE` preserves `patterns.allowed` from each YAML.
- `pattern_binding: GROUND_TRUTH` binds the pattern families declared by each
  synthetic DatasetBundle. It does not expose bicluster count, cardinalities, or
  membership to the search.

The protocol measures both `SEARCH` and `FINAL` by default. This separates the
quality and diversity generated by QD search from the effect of final selection.
Optional `selectors` apply several final-selector components offline to every
completed search repertoire. A raw variant has no component name:

```yaml
selectors:
  - identifier: raw
    name: null
    parameters: {}
  - identifier: containment
    name: containment_marginal_quality
    parameters:
      max_objective_degradation: 0.15
      max_degradation_per_log_area_gain: 0.20
```

Selector variants do not repeat candidate search. Their effective pipelines,
status, timing, and outputs are cached independently below each search case.
When this list is non-empty, pipeline-level final selectors are deliberately
removed from the search process so every variant receives the same raw archive.
When `selectors` is omitted, each pipeline's own `final_selection` component is
preserved and measured normally; it is never silently discarded.
The CLI reports completed search cases, selector applications, and measured
outputs as separate progress stages.
For every completed run it records:

- REL, REC, BE, target coverage, and detected bicluster count;
- wall time, throughput, candidate-generation, evaluation, and archive-update
  time;
- accepted, invalid, duplicate, and unique evaluated candidates;
- visited and created QD cells plus per-emitter final credit statistics;
- exact repertoire uniqueness and sampled nearest-neighbour structural
  diversity.

Case-level confidence intervals default to zero bootstrap repetitions because
they are unnecessarily expensive in a large ablation. Aggregate intervals
retain 2,000 repetitions by default. Both are configurable independently.

Outputs include `case-status`, optional `selection-status`, `run-metrics`, `emitter-metrics`,
`repertoire-metrics`, aggregate summaries, paired deltas against the first
pipeline/selector combination, configuration and selector differences, and
accuracy/runtime/diversity figures in CSV, Parquet, PNG, SVG, and JSON forms.

Paired tables retain the raw `delta = compared - baseline`, declare whether
`HIGHER` or `LOWER` is preferred for each metric, and also expose
`favorable_delta`, where a positive value always means that the compared
pipeline improved over the baseline. When offline selectors are configured,
`comparison_scope` separates search comparisons made under the same selector
from selector comparisons made over the same terminal archive.

With `execution.resume: true`, completed cases are reused only when the dataset,
validated effective pipeline, pattern binding, run seed, and installed SALVI
source fingerprint still match. Internal implementation changes therefore
invalidate stale search results even when the YAML text is unchanged.

## G-Bic and HBIC

G-Bic datasets are converted at the experiment interoperability boundary. A source may
be one dataset triplet or a complete directory tree:

```bash
uv run salvi-exp convert gbic /path/to/GBIC-data /path/to/DatasetBundles
```

The converter discovers matching `_data.tsv`, `_bics.txt`, and `_bics.json`
artifacts and preserves constant, additive, multiplicative, and mixed ground
truth in canonical bundles.

`py-hbic` returns each bicluster as a pair containing a row Boolean mask and a
column Boolean mask. The direct adapter accepts that result without an
intermediate format:

```python
from pathlib import Path

from hbic import Hbic
from salvi_experiments.interop import HbicConverter

result = Hbic(n_clusters=5).fit_predict(frame)
HbicConverter(
    dataset_bundle=Path("datasets/example"),
).convert_result(result, Path("results/hbic-example"))
```

For cross-process workflows, `salvi-exp convert hbic` accepts the versioned JSON
document described in [Artifact contracts](artifact-contracts.md). Imported HBIC
results deliberately contain no fabricated objectives or pattern fits. They can
be passed directly to the accuracy protocols.

When comparing algorithms, record whether `n_clusters` or any equivalent real
target count was supplied. Knowing that count changes the information available
to the algorithm and must not be hidden in an aggregate table.
