# Changelog

All notable user-visible changes are documented here. SALVI follows semantic
versioning for its Python distribution and independent integer versions for
artifact schemas.

## 0.1.0 - Unreleased

- Introduced component-oriented preprocessing, evaluation, QD search, candidate
  generation, execution, monitoring and final repertoire extraction.
- Added constant, additive and multiplicative pattern inference for heterogeneous
  data, including mixed assignment and per-column scientific explanations.
- Kept additive and multiplicative model identities distinct by fitting strict
  raw additive shifts and normalizing only residuals; mixed inference now uses
  bounded pattern-specific neighborhoods and reports observational ambiguity.
- Made pattern-aware QD initialization and restarts respect exact cell ranges,
  observed-support feasibility and pattern-specific reachability.
- Added lightweight structural BiclusterSet reads and vectorized clinical
  repertoire matching to avoid rebuilding full evaluations during stability analysis.
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
- Consolidated the core, GUI, conventional evolutionary backend, interoperability,
  and experiment protocols into one installable `salvi` distribution, and added
  direct in-memory execution for fluent programmatic compositions.
