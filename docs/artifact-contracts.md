# Artifact contracts

All computational tables use Parquet. YAML describes human-edited configuration
and datasets; JSON describes machine-produced manifests and metadata. Every
contract is versioned independently.

## DatasetBundle version 1

```text
DatasetBundle/
├── dataset.yaml
├── data.parquet
├── row-identifiers.parquet # optional
└── ground-truth.json       # optional
```

`dataset.yaml` contains the dataset identifier, dimensions, ordered column
metadata, file names, and SHA-256 checksums. Column indices are contiguous and
zero-based. Supported semantic kinds are `NUMERIC`, `BOOLEAN`, and `CATEGORICAL`;
categorical metadata contains unique original labels.

The Parquet column order and names must exactly match the manifest. Observed
numeric values must be finite; missing values use Arrow nulls rather than `NaN`
or infinity. `row-identifiers.parquet` stores unique, non-null original string
identifiers. When absent, original zero-based positions become identifiers.

Canonical `ground-truth.json` stores the dataset identity and dimensions plus
sorted row/column coordinates and a pattern assignment for every selected column.
It supports constant, additive, multiplicative, and mixed biclusters. Ground truth
is optional,
validated and retained for future experiments, but `DatasetBundleReader.load`
does not parse it and `RunContext` has no field through which it could guide a
run. Experiment adapters must request it explicitly through
`read_ground_truth`.

## ClinicalDatasetBundle version 1

The `salvi_experiments` namespace wraps a search-only `DatasetBundle` with clinical
annotations that must never reach `RunContext`:

```text
ClinicalDatasetBundle/
├── clinical-dataset.yaml
├── dataset/
├── annotations.parquet
├── effective-import-recipe.yaml
└── source/
    ├── metadata.json
    └── data.csv
```

The manifest records the UCI identity, source checksum, row count, typed
annotation roles, units, ordinal levels, derived fields and checksums. Only
columns explicitly assigned `SEARCH` are present below `dataset/`. Outcomes,
covariates and supplementary annotations share the original row identifiers in
`annotations.parquet` and are consumed only by post-run analyses. The reader
validates row alignment and rejects any non-search annotation that leaks into
the nested `DatasetBundle`.

## BiclusterSet version 7

```text
BiclusterSet/
├── manifest.json
├── columns.parquet
├── biclusters.parquet
├── column-patterns.parquet          # optional
├── column-objectives.parquet        # optional
├── pattern-groups.parquet           # optional
├── pattern-row-parameters.parquet   # optional
└── final-selection.parquet          # optional
```

`columns.parquet` defines the result column coordinate system after preprocessing.
For every prepared column it records its contiguous result index, name, semantic
kind, categories, original source-column index, and optional derivation. This is
required because filtering may renumber source columns and augmentation may add
derived columns that do not have an independent source coordinate.

`biclusters.parquet` records stable identifiers, generation, immutable candidate
provenance, sorted row and prepared-column index lists, ordered objective
summaries, signed constraints, aggregate feasibility, ordered descriptors, typed
issues, and scientific validity. Candidate
provenance includes producer, operation, generation sequence, parent identifiers
and an optional pattern hint. The table also retains the archive-cell coordinate
of each reported evaluation. Every objective value carries its explicit
`MINIMIZE` or `MAXIMIZE` direction. Row indices remain original dataset
coordinates. Column indices refer to `columns.parquet`; its
`source_column_index` field provides the explicit mapping back to the original
dataset. All index lists must be unique, sorted, non-negative, and in range.
The reader also accepts version 5 sets, treating their absent constraint list as
empty and deriving feasibility from validity.

Scientific explanations remain normalized in separate tables:

- `column-patterns.parquet` contains one row per selected column with its assigned
  pattern, joint-group identity, fit error, typed parameter and scale, original and
  available support, prototype support, all tested pattern alternatives, and
  diagnostics;
- `column-objectives.parquet` contains one row per objective and selected column
  with direction, contribution, validity, and diagnostics;
- `pattern-groups.parquet` describes every joint fitted group without assuming a
  particular pattern kind;
- `pattern-row-parameters.parquet` contains row parameters keyed by group.
- `final-selection.parquet` records selector identity, deterministic selection
  rank, quality, novelty and marginal-gain scores, plus every consolidated source
  candidate and archive coordinate.

These files are optional only when the repertoire contains no corresponding
records. A scientific evaluation writes a complete row for every selected column,
including invalid or unassigned columns, so explanations cannot become
misaligned with aggregate fitness. Readers reconstruct the immutable `Evaluation`
and validate group membership, signatures, objective order, and persisted validity.
Writers never recompute scientific values during serialization.

`PagedBiclusterSetReader` validates the same manifest and checksums, indexes only
identifier columns, and materializes bounded Parquet pages plus the scientific
detail of those biclusters. GUI and other interactive consumers can therefore
inspect large repertoires without constructing every `Evaluation` in memory.

The manifest records schema version, dataset identifier, original and prepared
dimensions, run provenance, creation time, file names, and SHA-256 checksums.
Scientific runs additionally identify the final search-state checkpoint, its
evaluation count, and its checksum. Candidate identifiers in final-selection
provenance can therefore be traced to complete evaluations, emitter ancestry, and
archive cells in that checkpoint.
Writers stage directories atomically and refuse to overwrite by default.

## Run directory

```text
run-output/
├── effective-configuration.yaml
├── run-metadata.json
├── run.sqlite
├── logs/
├── checkpoints/
└── artifacts/
    ├── search-repertoire/     # present when final selection is configured
    └── repertoire/            # selected output, or raw search result
```

`run-metadata.json` records package version, run identity, seed, effective
configuration, timestamps, terminal state, executor/runtime details, bundle
loading time, preprocessing component timings, prepared-memory accounting, search
and selection time, and the configured termination progress.
`run.sqlite` contains
append-only events plus metric and artifact indexes. It uses WAL mode so a monitor
can read while the engine writes.

Scientific checkpoint version 4 is compact durable JSON written through `fsync` and
atomic replacement. It contains evaluation and insertion counters, remaining
initial candidates, the retained repertoire, deterministic named random-stream
and scheduler states, integration mode, and a fingerprint of every configuration
field that can affect scientific continuation. It may additionally contain a
generated but unevaluated candidate batch and its emitter attribution. Such work
is replayed exactly on resume.

Every completed resumable scientific run writes a terminal checkpoint even when
periodic checkpointing is disabled. BiclusterSet manifests reference that exact
state. When final selection is configured, `search-repertoire` contains the
engine result before selection and `repertoire` contains the selected output.
Without a selector, only `repertoire` is written and it contains the engine
result directly. Periodic checkpoints remain optional and serve continuation
rather than result provenance. A search engine without `checkpoint-resume`, such
as `pymoo_nsga2`, writes no misleading checkpoint and uses null checkpoint
provenance in its manifest.

Output, monitoring, resume location, and the evaluation budget are excluded from
the fingerprint; the budget may therefore be extended without changing the
resumed trajectory. A checkpoint from a different dataset, run, engine,
integration mode, or scientific configuration is rejected. Recovery checkpoints
are emitted on cooperative cancellation or worker failure whenever candidates
are pending.

`SQLiteRunEventSource` is the replayable monitor boundary. Consumers can poll from
the last event or metric sequence, page event history, list and read bounded
metric series, and inspect the current artifact index. They do not query engine
objects or depend on GUI classes.

## Presentation exports

`salvi-exp export csv` writes normalized CSV tables and an
`export-manifest.json`. This export preserves membership, objectives,
descriptors, patterns, per-column explanations, row parameters, and final
selection provenance. It is not accepted as a runtime artifact and never replaces
the canonical checksummed Parquet representation.

`salvi-exp convert hbic` writes a canonical `BiclusterSet` after validating an
external result against a specified `DatasetBundle`. Imported biclusters retain
HBIC provenance and row/column descriptors. Objectives and pattern explanations
remain absent because HBIC does not provide SALVI's scientific evaluations.

`salvi-exp convert uci RECIPE.yaml OUTPUT` downloads an official UCI resource,
checks its pinned SHA-256, applies explicit roles, mappings, missing tokens and
derived annotations, then writes a `ClinicalDatasetBundle`. A modified upstream
resource fails rather than silently changing an experiment.

## Versioning rules

- Readers reject unknown schema versions.
- Checksums cover every declared data file exactly.
- New optional files require a schema-compatible manifest field.
- Breaking field or semantic changes increment the corresponding schema version.
- CSV is an import/export convenience and is never a canonical runtime contract.

The current writer and reader ranges, migration rules and release procedure are
defined in [Versioning and release policy](versioning-and-release.md). The
installed machine-readable registry is available through `salvi schemas`.
