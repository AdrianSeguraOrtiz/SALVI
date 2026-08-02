# salvi

`salvi` is the component-oriented core of the SALVI quality-diversity
biclustering framework. It provides canonical heterogeneous datasets, constant,
additive and multiplicative pattern inference, per-column objective
explanations, serial and bounded parallel MOME search, checkpoints, final
repertoire extraction and an optional local web application.

The optional `evolution` extra adds pymoo-backed conventional search engines.
`pymoo_nsga2` is currently available as an experimental non-QD control while
retaining SALVI evaluation and canonical artifacts. QD-only archive, emitter and
scheduler roles are rejected for this engine:

```bash
python -m pip install "salvi[evolution]"
```

Install the core command and inspect its public contracts:

```bash
python -m pip install salvi
salvi --version
salvi schemas
salvi validate configuration.yaml --dataset dataset-bundle
salvi inspect configuration.yaml --dataset dataset-bundle
salvi run configuration.yaml --dataset dataset-bundle --output run-output
salvi components --kind search_engine
```

`salvi run` keeps its final JSON summary on `stdout` and writes optional live
progress to `stderr` by polling the run SQLite event store. Use `--progress` and
`--monitor-interval` to control that console monitor.

Install and launch the optional local web application with:

```bash
python -m pip install "salvi[gui]"
salvi gui
```

It serves the catalog-driven Build, Monitor and Results views on
`http://127.0.0.1:8765` without requiring Node.js at runtime. The `evolution`
and `gui` extras are optional. Complete configuration, artifact, scientific
and performance documentation is maintained in the repository `docs/`
directory. External GBIC/HBIC adapters and CSV exports are owned by the
separate `salvi-experiments` package.
