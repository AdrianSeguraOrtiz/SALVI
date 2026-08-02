# Performance profiling

SALVI profiles a reusable pipeline bound to one explicit dataset and run output,
rather than isolated toy functions. The profiler records wall and CPU time, process and child-process
resource use, Python allocation peaks, Linux process I/O when available, input
and output sizes, preprocessing timings and evaluations per second. It also
writes a standard `pstats` profile for every repetition.

```bash
salvi profile PIPELINE.yaml PROFILE_DIRECTORY \
  --dataset DATASET_BUNDLE --output RUN_OUTPUT \
  --repetitions 3 --overwrite --run-overwrite
```

`profile-report.json` is schema version 1. RSS is a process-lifetime high-water
mark on platforms that expose it. Traced memory covers Python allocations;
NumPy and Arrow may own buffers outside that counter. The CPU profile observes
the coordinator process, so serial execution must be used when investigating
scientific kernel call stacks. Parallel profiles still report end-to-end wall
time and child CPU/RSS.

For executor comparisons and release baselines, disable call-stack and allocation
instrumentation so serial code and worker processes receive equivalent treatment:

```bash
salvi profile PIPELINE.yaml PROFILE_DIRECTORY \
  --dataset DATASET_BUNDLE --output RUN_OUTPUT \
  --repetitions 3 --lightweight --overwrite --run-overwrite
```

## Release workload

The repository contains a deterministic 240-row, 12-column heterogeneous fixture
generator and four reusable pipelines:

- constant patterns with the serial executor;
- additive patterns with the serial executor;
- constant, additive and multiplicative inference with the serial executor;
- the same mixed workload with four deterministic process workers.

Regenerate and profile them with:

```bash
python tools/generate_release_fixture.py --overwrite
python tools/run_release_benchmarks.py --repetitions 3
```

The harness uses lightweight mode. The compact result is stored in
`benchmarks/performance-baseline-v0.1.0.json`. Full `pstats`, run directories and
generated data remain under ignored `benchmarks/generated/`.

These figures are regression references, not promises for other hardware.
Comparisons are meaningful only for the same fixture, configuration checksum,
Python version and machine class. A release should explain changes above 15% in
median throughput or peak RSS.

The version 0.1.0 development baseline used CPython 3.13.5 on an eight-logical-CPU
x86-64 Linux machine:

| Workload | Median wall time | Throughput | Coordinator peak RSS |
| --- | ---: | ---: | ---: |
| Constant, serial | 0.826 s | 77.52 eval/s | 131.1 MiB |
| Additive, serial | 2.906 s | 22.02 eval/s | 127.7 MiB |
| Mixed, serial | 9.506 s | 6.73 eval/s | 134.4 MiB |
| Mixed, four processes | 5.476 s | 11.69 eval/s | 138.2 MiB |

The process-pool row additionally observed a 136.9 MiB child-process high-water
mark. The exact configuration hashes and unrounded values remain in the JSON
baseline.

## Current optimizations

Representative profiling identified repeated array preparation, scalar pattern
fitting, guided-emitter scoring and per-event persistence as the main avoidable
costs. The optimized runtime now:

- precomputes immutable support masks and exposes shared numeric and discrete
  matrix views to trusted kernels;
- batches constant fitting and vectorizes additive and multiplicative row and
  column effects without changing the reviewed formulas;
- evaluates guided row and column candidate pools with shared matrix scans;
- caches direction-aware per-column losses on each immutable evaluation and the
  materialized repertoire while its archive state is unchanged;
- microbatches deterministic process evaluation while enforcing the configured
  candidate-level `max_in_flight` bound;
- commits related run events and observer metrics in bounded SQLite
  transactions, and does not start an observer dispatcher when no observers are
  configured;
- derives detailed event payloads from observer declarations and skips
  per-candidate objective, descriptor, constraint and validity timing unless
  `component_timing` is configured;
- keeps cumulative and rolling structural uniqueness exact while bounding
  nearest-neighbour calculations with the configurable deterministic
  `distance_sample_size`.

On the same development machine and fixture as the versioned 0.1.0 baseline, a
three-repetition verification after these changes produced:

| Workload | Median wall time | Throughput | Baseline speedup |
| --- | ---: | ---: | ---: |
| Constant, serial | 0.289 s | 221.61 eval/s | 2.86x |
| Additive, serial | 0.829 s | 77.16 eval/s | 3.50x |
| Mixed, serial | 3.353 s | 19.09 eval/s | 2.84x |
| Mixed, four processes | 2.492 s | 25.68 eval/s | 2.20x |

The versioned baseline remains unchanged so it continues to represent the
pre-optimization reference. The figures above are a verification snapshot, not
a portable performance guarantee.

No Rust extension is included in 0.1.0. Current measured improvements are in
Python orchestration and monitoring, while the scientific kernels remain readable
Python reference implementations. Native code should be added only after a stable
kernel dominates representative serial profiles and a Python parity suite exists;
until then there is no artificial backend selector in the public pipeline.
