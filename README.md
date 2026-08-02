# SALVI

SALVI is a Python 3.11+ framework for quality-diversity biclustering of
heterogeneous data. It is organized around explicit components for data
preparation, pattern inference, objective evaluation, candidate generation,
archives, observers, final selection, interoperability and experimentation.

The current implementation is executable end to end. It loads canonical
DatasetBundles, preprocesses heterogeneous columns without imputing missing
values by default, evaluates constant, additive, multiplicative and mixed
column-pattern hypotheses, and stores per-column explanations for every reported
bicluster. Internal coherence, contrast and balanced bicluster size are available
as objectives. Optional constraints can bound balanced size or internal-coherence
error. Pattern inference is shared across objectives and constraints so each
candidate is fit once per evaluation.

The search layer provides a bounded MOME-style quality-diversity engine. Row and
column cardinalities are behavioral descriptors, archive axes own their own
discretization, and each occupied cell keeps a bounded local Pareto front.
Initializers, emitters, schedulers, termination criteria, executors and observers
are selected through the same component registry used by the CLI and GUI.
An optional `pymoo` backend also exposes conventional NSGA-II using the same
SALVI objectives, constraints, initialization and evaluation contracts.

Runs can execute serially, with a thread pool, or with a persistent process pool.
Checkpoints capture archives, pending work, scheduler state and random streams.
Passive SQLite observers record progress, coverage,
diversity, throughput and resource usage. Omitting final selection reports the search
repertoire directly. When a selector is configured, SALVI persists both the
unfiltered `search-repertoire` and the selected `repertoire`, so search coverage
and reporting quality can be audited independently.

The sibling `salvi-experiments` package contains scientific protocols that use
only public SALVI artifacts: objective alignment against ground truth, accuracy
assessment, clinical association and stability analysis, benchmark aggregation,
and comparison reports. GBIC, HBIC and UCI adapters convert external data and
results into SALVI's canonical formats.

## Development setup

SALVI requires Python 3.11 or newer and uses a versioned `uv` workspace.

```bash
uv sync --all-packages --all-extras --dev
uv run pytest
uv run ruff check .
uv run mypy packages/salvi/src packages/salvi-experiments/src
```

For a local command-line installation from the repository root:

```bash
python -m pip install "packages/salvi[gui,evolution]"
python -m pip install packages/salvi-experiments
salvi --help
salvi-exp --help
```

Install only the core and the optional conventional evolutionary backend with:

```bash
python -m pip install "packages/salvi[evolution]"
```

The repository root is an orchestration-only uv workspace and deliberately does
not build a third distribution. `salvi` and `salvi-experiments` remain independently
installable packages. Workspace development should use
`uv sync --all-packages --all-extras --dev`.

Create a small canonical dataset, inspect the concrete pipeline, and run a short
scientific smoke search:

```bash
uv run python examples/create-example-dataset.py
uv run salvi validate examples/smoke-configuration.yaml \
  --dataset examples/example-dataset
uv run salvi inspect examples/smoke-configuration.yaml \
  --dataset examples/example-dataset
uv run salvi run examples/smoke-configuration.yaml \
  --dataset examples/example-dataset --output examples/smoke-output \
  --identifier example-run --seed 42
uv run salvi components --kind search_engine
uv run salvi config format examples/smoke-configuration.yaml
uv run salvi schemas
uv run salvi-exp schemas
```

`salvi run` reports concise live progress from `run.sqlite` to `stderr` when
executed in an interactive terminal. The final machine-readable JSON summary is
kept on `stdout`. Use `--progress always`, `--progress never`, `--quiet`, or
`--monitor-interval 1.0` to control console monitoring.

`salvi select` applies the pipeline's final-selector component to an existing
search repertoire. This makes selector ablations independent from the
expensive candidate search while preserving a canonical `BiclusterSet` and its
source-run/checkpoint provenance.

The optional local web application is installed through the `gui` extra and
launched with `uv run salvi gui`. It provides a catalog-driven workflow builder,
SQLite-backed live monitoring and on-demand result inspection at
`http://127.0.0.1:8765`. It needs neither Qt nor a graphical session; on a remote
machine, forward port `8765` through VS Code or SSH. See
[docs/gui.md](docs/gui.md).

External HBIC results and human-readable CSV exports use explicit adapters:

```bash
uv run salvi-exp convert gbic /path/to/gbic /path/to/dataset-bundles
uv run salvi-exp convert uci uci-import.yaml clinical-dataset
uv run salvi-exp convert hbic hbic-results.json hbic-biclusters \
  --dataset-bundle example-dataset
uv run salvi-exp export csv hbic-biclusters hbic-csv
```

Scientific protocols are provided by the sibling package:

```bash
uv run salvi-exp dataset objective-alignment \
  examples/experiments/objective-alignment.yaml
uv run salvi-exp dataset accuracy examples/experiments/accuracy.yaml
uv run salvi-exp dataset clinical-validation clinical-validation.yaml
uv run salvi-exp benchmark accuracy \
  examples/experiments/accuracy-benchmark.yaml
uv run salvi-exp benchmark compare examples/experiments/comparison.yaml
uv run salvi-exp benchmark ablation examples/experiments/ablation.yaml
```

## Documentation

- [Architecture](docs/architecture.md)
- [Configuration](docs/configuration.md)
- [Artifact contracts](docs/artifact-contracts.md)
- [Local web application](docs/gui.md)
- [Scientific experiments](docs/experiments.md)
- [Scientific contract](docs/scientific-contract.md)
- [Performance profiling](docs/performance.md)
- [Versioning and release](docs/versioning-and-release.md)

SALVI is licensed under the MIT License.
