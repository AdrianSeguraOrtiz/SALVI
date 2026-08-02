# salvi-experiments

`salvi-experiments` contains reproducible scientific protocols built exclusively
on the public SALVI API and canonical artifacts.

Available commands:

```text
salvi-exp dataset objective-alignment CONFIG.yaml
salvi-exp dataset accuracy CONFIG.yaml
salvi-exp benchmark objective-alignment CONFIG.yaml
salvi-exp benchmark accuracy CONFIG.yaml
salvi-exp benchmark compare CONFIG.yaml
salvi-exp benchmark ablation CONFIG.yaml
salvi-exp convert gbic SOURCE DESTINATION
salvi-exp convert uci RECIPE.yaml DESTINATION
salvi-exp convert hbic SOURCE DESTINATION --dataset-bundle DATASET_BUNDLE
salvi-exp dataset clinical-validation CONFIG.yaml
salvi-exp export csv BICLUSTER_SET DESTINATION
salvi-exp schemas
```

Every command receives one strict, self-contained YAML document. Paths are
resolved relative to that document. Mixed ground-truth biclusters are included by
default; a reduced pattern scope must be declared explicitly and is recorded in
the report and manifest.

Commands write concise progress messages to `stderr` and keep the final result
directory on `stdout`. Use `salvi-exp --quiet ...` to suppress progress output.
Benchmark YAMLs may set `execution.workers` to run independent cases in
parallel. Nested SALVI workers are rejected unless the YAML explicitly enables
`execution.allow_nested_parallelism`; CPU oversubscription is controlled
separately with `execution.allow_cpu_oversubscription`.

See [Scientific experiments](../../docs/experiments.md) for configuration fields,
metric definitions, output artifacts, and end-to-end examples.
