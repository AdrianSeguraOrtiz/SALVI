# Versioning and release policy

SALVI publishes one Python distribution, `salvi`, with one semantic version. It
contains both the `salvi` core namespace and the `salvi_experiments` scientific
protocol namespace, ensuring that objective, artifact, and experiment semantics
are installed as one reproducible unit. Development status, package version, and
user-visible changes are recorded in `CHANGELOG.md`.

## Public schemas

Artifact schemas evolve independently from package versions. The installed
versions can be inspected without loading a dataset:

```bash
salvi --version
salvi schemas
salvi-exp --version
salvi-exp schemas
```

The current contracts are:

| Contract | Writer version | Oldest readable version |
| --- | ---: | ---: |
| Run configuration | 1 | 1 |
| DatasetBundle | 1 | 1 |
| Ground truth | 1 | 1 |
| BiclusterSet | 7 | 5 |
| Search checkpoint | 4 | 4 |
| Run metadata | 2 | 2 |
| Profile report | 1 | 1 |
| Experiment configuration (`salvi-exp`) | 1 | 1 |
| Experiment report (`salvi-exp`) | 1 | 1 |

The core authority is `salvi.versioning.SCHEMA_VERSIONS`; experiment-owned
contracts live in `salvi_experiments.versioning.EXPERIMENT_SCHEMA_VERSIONS`.
Tests verify that these values agree with the runtime Pydantic models.

## Compatibility and migration

- Adding an optional field with unchanged semantics may retain a schema version.
- A required field, changed coordinate system, changed scientific meaning or
  removed field increments the affected schema.
- Readers reject unsupported versions explicitly. They never guess, silently
  reinterpret or modify an input artifact in place.
- A future migration writes a new artifact atomically, retains provenance and
  checksums for the source, and validates the result through the normal reader.
- Checkpoints are intentionally stricter than result artifacts. A checkpoint is
  resumable only when its schema and scientific configuration fingerprint match.
- Experimental CSV files are presentation exports, not migration inputs.
- External formats enter through `salvi_experiments` adapters such as `gbic` and
  `hbic`; converters do not manufacture missing objective values or pattern
  diagnostics.

No migration command is shipped in version 0.1.0 because every current reader has
one supported version. The first schema bump must add an explicit migrator and
fixture before its release.

## Release procedure

1. Update the root package version and the changelog.
2. Run `python tools/check_release_versions.py`.
3. Run the complete quality suite and all clean-install smokes.
4. Run `python tools/verify_reproducible_builds.py`.
5. Regenerate the performance baseline described in `docs/performance.md`.
6. Tag the exact commit as `v<version>`.

The tag workflow validates that the tag matches the package, rebuilds and checks
the wheel and source distribution, and stores them as a GitHub Actions
artifact. It deliberately does not publish to PyPI automatically. Publication
requires a separately authorized action after the artifacts have been inspected.
