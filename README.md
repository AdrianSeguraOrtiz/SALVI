# SALVI

SALVI is a component-oriented framework for biclustering heterogeneous data with
quality-diversity optimization. It searches for coherent and contrasting
submatrices while using row and column cardinality as behavioral descriptors,
so biclusters of different shapes can be explored without competing in one
global Pareto population.

The same component catalog drives the Python API, CLI, and local web interface.
SALVI supports numerical, categorical, and boolean columns; constant, additive,
and multiplicative patterns; missing values without mandatory imputation;
parallel evaluation; durable SQLite monitoring; canonical Parquet artifacts;
and per-column explanations for every reported bicluster. The installation also
includes the `salvi_experiments` Python namespace and the `salvi-exp` command for
objective-alignment, accuracy, benchmark, clinical, and interoperability tasks.

SALVI requires Python 3.11 or newer. Until a PyPI release is available, install
the complete distribution directly from GitHub:

```bash
python -m pip install \
  "git+https://github.com/AdrianSeguraOrtiz/SALVI.git@<commit-sha>"
```

Using a commit SHA makes the installation reproducible. Replace
`<commit-sha>` with `main` to track the latest development revision. A source
checkout can be installed for development with:

```bash
git clone https://github.com/AdrianSeguraOrtiz/SALVI.git
cd SALVI
python -m pip install -e ".[dev]"
```

## Programmatic API

`SalviRun.builder(...)` composes concrete component instances. Unique roles use
`with_*`; repeatable roles use `add_*`. `build()` validates role cardinalities,
capabilities, scientific requirements, and incompatibilities before search
starts.

The following compact example builds and executes a QD pipeline in memory:

```python
from pathlib import Path

from salvi import (
    DatasetBundleReader,
    PatternConfiguration,
    PatternKind,
    SalviRun,
    execute_in_memory,
)
from salvi.components.candidate_initialization import UniformRandomInitializer
from salvi.components.descriptors import ColumnCardinality, RowCardinality
from salvi.components.evaluation_policies import MinimumCardinality, MinimumObservedSupport
from salvi.components.execution import SerialEvaluationExecutor
from salvi.components.membership_emitters import RandomMoveEmitter
from salvi.components.objectives import Contrast, InternalCoherence
from salvi.components.parent_selection import RepertoireUniformParentSelection
from salvi.components.preprocessing import PreserveMissingValues, RobustNumericScaling
from salvi.components.schedulers import FirstEmitterScheduler
from salvi.components.termination import EvaluationBudget
from salvi.engine.archive import DeepGridMomeArchive, DeepGridMomeConfiguration
from salvi.engine.mome import SerialMomeSearchEngine

dataset = DatasetBundleReader().read(Path("dataset-bundle"))
archive = DeepGridMomeConfiguration()

specification = (
    SalviRun.builder(
        dataset,
        run_identifier="example",
        seed=42,
        patterns=PatternConfiguration(allowed=(PatternKind.CONSTANT,)),
    )
    .with_missing_values_policy(PreserveMissingValues())
    .add_numeric_transformation(RobustNumericScaling())
    .with_candidate_validity_policy(MinimumCardinality(min_rows=2, min_columns=2))
    .with_evaluation_support_policy(MinimumObservedSupport())
    .with_search_engine(SerialMomeSearchEngine())
    .add_objective(InternalCoherence())
    .add_objective(Contrast())
    .add_descriptor(RowCardinality())
    .add_descriptor(ColumnCardinality())
    .with_archive(DeepGridMomeArchive(axes=archive.axes, cell_capacity=archive.cell_capacity))
    .with_parent_selection_policy(RepertoireUniformParentSelection())
    .with_initializer(UniformRandomInitializer())
    .add_emitter(RandomMoveEmitter())
    .with_scheduler(FirstEmitterScheduler())
    .with_executor(SerialEvaluationExecutor())
    .with_termination(EvaluationBudget(max_evaluations=5_000))
    .build()
)

result = execute_in_memory(specification)
print(result.evaluations, len(result.repertoire.evaluations))
```

`execute_in_memory` is intended for direct scripting and returns both the raw
search repertoire and the final repertoire. It does not create SQLite events,
checkpoints, or files. Use `RunService` or the CLI for durable, observable runs.

## CLI

SALVI pipeline YAML is reusable: it describes scientific and search components,
while the dataset, output directory, run identifier, and seed are supplied when
launching a run. A minimal configuration is available at
[`examples/smoke-configuration.yaml`](https://github.com/AdrianSeguraOrtiz/SALVI/blob/main/examples/smoke-configuration.yaml), and
the current full QD pipeline is at
[`examples/scientific-configuration.yaml`](https://github.com/AdrianSeguraOrtiz/SALVI/blob/main/examples/scientific-configuration.yaml).

```yaml
schema_version: 1
patterns:
  allowed: [CONSTANT]
preprocessing:
  missing_values: {name: preserve}
  numeric_transformations:
    - {name: robust_numeric_scaling}
evaluation:
  candidate_validity:
    name: minimum_cardinality
    parameters: {min_rows: 2, min_columns: 2}
  observed_support: {name: minimum_observed_support}
search:
  engine: {name: serial_mome}
  objectives:
    - {name: internal_coherence}
    - {name: contrast}
  descriptors:
    - {name: row_cardinality}
    - {name: column_cardinality}
  archive: {name: deep_grid_mome}
  parent_selection: {name: repertoire_uniform}
  initialization: {name: uniform_random}
  emitters:
    - {name: random_move}
  scheduler: {name: first}
  termination:
    name: evaluation_budget
    parameters: {max_evaluations: 5000}
execution:
  executor: {name: serial}
monitoring:
  observers:
    - {name: search_progress}
```

Validate, inspect, and execute it against a canonical `DatasetBundle`:

```bash
salvi validate pipeline.yaml --dataset dataset-bundle
salvi inspect pipeline.yaml --dataset dataset-bundle
salvi run pipeline.yaml \
  --dataset dataset-bundle \
  --output run-output \
  --identifier experiment-01 \
  --seed 42
```

The main CLI also provides:

```text
salvi components [--kind KIND] [--format text|json|markdown]
salvi config format PIPELINE.yaml [--output NORMALIZED.yaml]
salvi select PIPELINE.yaml --dataset DATASET --repertoire INPUT --output OUTPUT
salvi profile PIPELINE.yaml REPORT --dataset DATASET --output RUN_OUTPUT
salvi schemas
salvi gui
```

`salvi run` writes the effective configuration, metadata, checkpoints, canonical
bicluster artifacts, and `run.sqlite` under the output directory. It reports
concise live progress on `stderr`; `--progress always|never|auto`, `--quiet`, and
`--monitor-interval` control that view.

The same installation exposes the complete experiment and interoperability CLI:

```bash
salvi-exp schemas
salvi-exp convert gbic /path/to/GBIC-data /path/to/DatasetBundles
salvi-exp convert uci uci-import.yaml clinical-dataset
salvi-exp convert hbic hbic-result.json bicluster-set --dataset-bundle dataset-bundle
salvi-exp export csv bicluster-set csv-output
salvi-exp dataset objective-alignment objective-alignment.yaml
salvi-exp dataset accuracy accuracy.yaml
salvi-exp dataset clinical-validation clinical-validation.yaml
salvi-exp benchmark objective-alignment objective-alignment-benchmark.yaml
salvi-exp benchmark accuracy accuracy-benchmark.yaml
salvi-exp benchmark ablation ablation.yaml
salvi-exp benchmark compare comparison.yaml
```

Run `salvi --help`, `salvi <command> --help`, or `salvi-exp --help` for the full
interface.

## GUI

Launch the local web application with the same installation:

```bash
salvi gui
```

SALVI opens `http://127.0.0.1:8765`. Use `salvi gui --no-open` on a headless or
remote machine, then forward port `8765` through VS Code Remote SSH or SSH. The
server is loopback-only because this single-user version has no authentication.

The **Build** view constructs a reusable YAML pipeline directly from catalog
metadata and compatibility rules:

![SALVI Build view](https://raw.githubusercontent.com/AdrianSeguraOrtiz/SALVI/main/docs/images/salvi-build.png)

The **Monitor** view reads durable SQLite events and renders only the observers
selected in the pipeline:

![SALVI Monitor view](https://raw.githubusercontent.com/AdrianSeguraOrtiz/SALVI/main/docs/images/salvi-monitor.png)

The **Results** view compares the raw repertoire with final selection and exposes
matrix values, structure, objectives, patterns, provenance, and per-column
contributions:

![SALVI Results view](https://raw.githubusercontent.com/AdrianSeguraOrtiz/SALVI/main/docs/images/salvi-results.png)

## Documentation

- [Architecture](https://github.com/AdrianSeguraOrtiz/SALVI/blob/main/docs/architecture.md)
- [Configuration reference](https://github.com/AdrianSeguraOrtiz/SALVI/blob/main/docs/configuration.md)
- [Artifact contracts](https://github.com/AdrianSeguraOrtiz/SALVI/blob/main/docs/artifact-contracts.md)
- [Local web application](https://github.com/AdrianSeguraOrtiz/SALVI/blob/main/docs/gui.md)
- [Scientific experiments](https://github.com/AdrianSeguraOrtiz/SALVI/blob/main/docs/experiments.md)
- [Scientific contract](https://github.com/AdrianSeguraOrtiz/SALVI/blob/main/docs/scientific-contract.md)
- [Performance](https://github.com/AdrianSeguraOrtiz/SALVI/blob/main/docs/performance.md)
- [Versioning and release](https://github.com/AdrianSeguraOrtiz/SALVI/blob/main/docs/versioning-and-release.md)

SALVI is licensed under the MIT License.
