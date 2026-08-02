# Changelog

All notable user-visible changes are documented here. SALVI follows semantic
versioning for its Python distributions and independent integer versions for
artifact schemas.

## 0.1.0 - Unreleased

- Introduced component-oriented preprocessing, evaluation, QD search, candidate
  generation, execution, monitoring and final repertoire extraction.
- Added constant, additive and multiplicative pattern inference for heterogeneous
  data, including mixed assignment and per-column scientific explanations.
- Added deterministic serial and bounded parallel execution with checkpoints and
  resumability.
- Separated reusable pipeline YAML from dataset/run bindings, added compact
  canonical formatting, real pipeline inspection and registry catalog commands,
  and materialized registered defaults in reproducibility artifacts.
- Added canonical DatasetBundle and BiclusterSet artifacts, GBIC/HBIC conversion,
  UCI clinical import, CSV export, CLI, a catalog-driven local FastAPI/React
  application and scientific experiment protocols.
- Added versioned runtime profiling, reproducible distribution checks and
  clean-wheel installation smokes for Python 3.11-3.13 on Linux, Windows and
  macOS.
