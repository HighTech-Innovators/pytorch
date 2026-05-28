# ADR: PyTorch Benchmarking Infrastructure

**Status**: ACTIVE  
**Last Updated**: 2026-05-28

## Architectural Role

The benchmarks subsystem provides reproducible performance measurement infrastructure for PyTorch. It consists of specialized benchmark suites that measure performance characteristics across multiple dimensions: operator-level dispatch overhead, autograd recording cost, distributed training scaling, and full model training throughput. Benchmarks enable regression detection and performance-driven development by establishing baseline timings and comparing performance across commits.

Benchmarks are classified as **Performance Sensitive** (per Chapter 10) — they directly measure and validate the performance characteristics that are critical to PyTorch's production viability. The subsystem owns no runtime functionality; it is purely a measurement and validation tool.

## Responsibilities

### What benchmarks Own
- **Benchmark suites**: Specialized measurement harnesses for different performance domains (operator microbenchmarks, functional autograd, FastRNN, Dynamo, distributed scaling, inference optimization)
- **Measurement infrastructure**: Timing harness, result aggregation, baseline comparison, reporting formats (JSON, CSV, Markdown)
- **Model definition**: Reference models used for end-to-end performance testing (GPT-fast, vision models, RNNs, transformers)
- **Result comparison**: Utilities to detect performance regressions by comparing baseline runs against current runs
- **Performance data**: Generated timing results and baseline measurements for regression detection

### What benchmarks Does Not Own
- **Production kernels**: Operators themselves are owned by ATen (aten/src/ATen/native/)
- **Testing framework**: Test execution (pytest) is owned by torch/test
- **Runtime profiling**: Profiler infrastructure is owned by torch/csrc/autograd/profiler
- **Neural network definitions**: torch.nn modules are owned by torch/nn
- **Distributed primitives**: Collectives and process groups are owned by torch/csrc/distributed/

## Dependencies

### Internal Dependencies (benchmarks → other modules)
- **torch**: Direct import for model definitions, tensor operations, training utilities
- **torch.nn**: Reference models inherit from nn.Module
- **torch.distributed**: Distributed benchmark suites use collectives and process groups
- **torch.profiler**: Some benchmarks integrate profiler for detailed timeline analysis
- **aten**: Operators executed during benchmark runs
- **c10**: Device and allocator abstractions used by operators

### External Dependencies (other modules → benchmarks)
- **Performance validation**: Other modules may depend on benchmark results for performance assertions (e.g., "dispatch overhead must be < 50 cycles per operation")
- **Regression detection**: CI/CD pipelines depend on benchmarks to detect performance degradation before merging

**Fan-in**: Moderate — benchmarks are referenced by CI systems and development workflows, but not by runtime code.

## Trade-offs and Design Decisions

### 1. Separate Benchmark Harnesses vs. Integrated Performance Tests
**Decision**: Maintain specialized benchmark suites (`benchmarks/operator_benchmark/`, `benchmarks/dynamo/`, etc.) rather than embedding performance tests within functional test suites.

**Rationale**:
- Specialized harnesses allow fine-grained measurement of specific performance domains (dispatch overhead, memory allocation, communication costs)
- Functional tests verify correctness; performance tests verify cost. Separating them clarifies test intent.
- Benchmark result aggregation (JSON export, baseline comparison) requires structured output that differs from functional test assertions

**Trade-off**: Maintaining two test hierarchies (functional in `torch/test`, performance in `benchmarks/`) requires coordination. However, clear ownership and purpose boundary prevents confusion about test intent.

**Evidence**: The `benchmarks/` structure shows specialized infrastructure:
- `benchmarks/compare-fastrnn-results.py` (line 5-50) implements result aggregation with JSON input and Markdown/CSV output
- `benchmarks/operator_benchmark/` contains operator microbenchmark harness (not functional tests)
- `benchmarks/dynamo/` contains TorchDynamo compilation timing, not correctness tests

### 2. Baseline Comparison vs. Absolute Thresholds
**Decision**: Compare against baseline measurements and flag percent-change regressions rather than enforcing absolute performance thresholds.

**Rationale**:
- Absolute thresholds are brittle: they depend on hardware, may break on legitimate optimizations, and require maintenance
- Baseline comparison detects regressions (5%+ slowdown) while tolerating platform variation
- Supports regression-driven development: ensure new changes don't slow down existing performance

**Trade-off**: Baseline-relative comparisons require maintaining baseline measurements across commits. However, this is acceptable because baselines can be updated on performance improvements.

**Evidence**: The `compare-fastrnn-results.py` script (line 30-40) loads two JSON files and computes percent differences, indicating baseline-relative comparison.

### 3. Microbenchmarks (operator-level) vs. Macrobenchmarks (end-to-end model training)
**Decision**: Include both microbenchmarks (measure operator dispatch, memory allocation cost) and macrobenchmarks (measure full model training throughput).

**Rationale**:
- Microbenchmarks isolate individual performance-critical paths (dispatch: 20-40 cycles, autograd recording: 150-300 cycles per Chapter 10)
- Macrobenchmarks validate that microbenchmark optimizations translate to real model performance
- Different regression detection strategies: microbenchmarks catch low-level regressions; macrobenchmarks catch high-level performance cliffs

**Trade-off**: Maintaining both types requires effort. However, microbenchmarks are too fine-grained to catch all regressions, and macrobenchmarks are too noisy to catch all optimizations.

**Evidence**: 
- `benchmarks/operator_benchmark/` contains operator microbenchmarks (from Chapter 10: "Operator microbenchmarks: Dispatch overhead, memory allocation cost")
- `benchmarks/gpt_fast/` contains end-to-end model training (from Chapter 10: "Model training: ResNet, BERT, Transformer throughput")

### 4. Python Benchmark Scripts vs. C++ Native Benchmarks
**Decision**: Implement most benchmarks in Python, but allow native C++ benchmarks for performance-critical paths.

**Rationale**:
- Python benchmarks are easier to maintain and update (faster iteration)
- Python directly uses public APIs (e.g., torch.nn, torch.optim), validating the API surface
- C++ benchmarks can measure lower-level costs (dispatch, allocation) that Python layers add overhead to

**Trade-off**: Python benchmarks may not measure absolute kernel performance (GIL, Python overhead), but this is acceptable because the goal is to detect regressions, not measure bare metal performance.

**Evidence**: `benchmarks/` is primarily Python scripts (from listing: `gpt_fast/*.py`, `functional_autograd_benchmark/*.py`), with native code invoked indirectly through torch API calls.

## Extension Boundaries

### Adding a New Benchmark Suite
1. Create a new folder under `benchmarks/` (e.g., `benchmarks/my_feature_benchmark/`)
2. Implement timing harness following patterns in `benchmarks/operator_benchmark/` or `benchmarks/dynamo/`
3. Export results in JSON format (see `compare-fastrnn-results.py` line 30-35 for JSON schema)
4. Add entry to `benchmarks/README.md` linking to suite-specific documentation

### Integrating Benchmarks into CI
1. Benchmark results must be JSON (for aggregation via `compare-fastrnn-results.py`)
2. Store baseline measurements in a versioned artifact or repository
3. Run benchmarks on reference hardware to avoid platform variation
4. Compare current run against baseline, flag regressions > 5%

### Measuring New Performance Domains
1. If measuring operator cost: use `benchmarks/operator_benchmark/` pattern (timing kernel dispatch, allocation)
2. If measuring model training: add to `benchmarks/gpt_fast/` or create new suite for that model type
3. If measuring distributed performance: extend `benchmarks/distributed/` with new collective or scaling scenario

## Runtime Implications

### Lifecycle
- Benchmarks are not part of the runtime; they are developer/CI tools
- Benchmark code does not ship with PyTorch distributions
- Benchmark scripts are run on-demand during development or in CI pipelines

### Concurrency
- Benchmarks are single-threaded by default (to avoid confounding factors like lock contention)
- Some benchmarks (e.g., distributed benchmark suites) intentionally multi-threaded or multi-process to measure scaling
- Benchmarks do not share state with runtime; each run is independent

### Failure Behavior
- Benchmark failure (e.g., timing exceeds threshold) does not crash runtime
- Failure is recorded and compared against baseline for regression detection
- If regression is detected in CI, the commit may be blocked (depends on CI policy)

## Performance Implications

### Measurement Overhead
- Benchmarks add measurement overhead (timing calls, result aggregation) that is separate from operator cost
- Overhead is typically < 5% of total time for macrobenchmarks (e.g., training throughput)
- Microbenchmarks repeat operations to amortize measurement cost, typically 1000+ iterations per measurement

### Hotspots
- Critical path for benchmark execution: operator invocation (inherited from runtime)
- Critical path for result comparison: JSON aggregation and percent-change computation (see `compare-fastrnn-results.py` lines 30-55)
- No allocations in inner loop of most benchmarks; memory use is O(1) for timing measurement

### Cache Behavior
- Benchmark results depend on L1/L2 cache state; macrobenchmarks may show variance if run on loaded system
- Microbenchmarks are typically run on idle hardware to reduce variance
- Baseline measurements should be taken on same hardware as regression-detection runs

## Ownership Boundaries

### Benchmarks Own
- Timing infrastructure: measurement harnesses, result formatting, baseline comparison
- Benchmark models: GPT-fast, vision model definitions used for performance testing
- Performance baselines: measured timings that serve as reference for regression detection

### Benchmarks Borrow
- Operators: Execute ATen kernels, but do not own them
- Models: Import torch.nn modules, but do not define them
- Device abstractions: Use c10 Device, Allocator for timing different backends

### Adjacent Ownership
- **Testing (torch/test)**: Owns functional tests; benchmarks own performance tests
- **Profiling (torch.profiler)**: Owns runtime profiler; benchmarks own aggregate timing measurements
- **Performance development**: Developers optimize based on benchmark results; benchmarks measure impact

## Key Files and References

| File/Folder | Purpose |
|---|---|
| `benchmarks/README.md` | Entry point; lists all benchmark suites |
| `benchmarks/compare-fastrnn-results.py` | Utility for comparing baseline vs. current run (JSON → Markdown/CSV) |
| `benchmarks/operator_benchmark/` | Operator microbenchmark suite (dispatch overhead, allocation cost) |
| `benchmarks/dynamo/` | TorchDynamo compilation and execution timing |
| `benchmarks/gpt_fast/` | End-to-end GPT model training performance |
| `benchmarks/functional_autograd_benchmark/` | Functional autograd recording cost |
| `benchmarks/distributed/` | Distributed training scaling benchmarks |
| `benchmarks/data/` | Benchmark data and fixtures |

## Related Architecture
- **Chapter 10 (Performance and Scalability)**: Defines performance domains measured by benchmarks (dispatch overhead, autograd cost, communication, memory allocation)
- **Chapter 03 (ATen Dispatch)**: Dispatch overhead is primary target for `benchmarks/operator_benchmark/`
- **Chapter 04 (Autograd Engine)**: Autograd recording cost is measured by `benchmarks/functional_autograd_benchmark/`
- **Chapter 08 (Distributed Training)**: Collective performance and gradient synchronization is measured by `benchmarks/distributed/`
