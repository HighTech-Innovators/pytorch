# `benchmarks`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`benchmarks` owns the repository's reproducible performance measurement suites and comparison scripts. It packages operator, framework, model, and compiler benchmarks as runnable Python modules and shell entry points.

## Key Files

| File | Purpose |
|---|---|
| `README.md` | Lists the supported benchmark suites and the expected setup flow for running them |
| `operator_benchmark/benchmark_core.py` | Implements benchmark registration, test-case construction, warmup, measurement, and result export |
| `operator_benchmark/README.md` | Documents the operator microbenchmark workflow, CLI, output format, and CI integration |
| `compare.sh` | Runs the FastRNN benchmark twice with different fusers and compares the resulting JSON reports |
| `compare-fastrnn-results.py` | Formats paired FastRNN benchmark outputs for side-by-side review |

## Public Interface

`operator_benchmark/benchmark_core.py` exposes `TestConfig`, `BENCHMARK_TESTER`, `_register_test()`, `_build_test()`, and `BenchmarkRunner`. The directory also exposes runnable module entry points such as `python -m pt.add_test`, `python -m benchmark_all_test`, and the shell wrapper `compare.sh`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch](torch/ADR.md) | depends-on | Benchmark code imports `torch`, uses backend availability checks, and executes real operators under test |
| [torch/utils](torch/utils/ADR.md) | depends-on | `operator_benchmark/benchmark_core.py` uses `torch.utils.benchmark.Timer` for synchronized GPU timing |
| [functorch](functorch/ADR.md) | depended-on-by | The top-level tree includes `functional_autograd_benchmark/` and other suites that measure transform performance against the functorch surface |

## Runtime Behaviour

Importing operator benchmark modules records benchmark metadata in `BENCHMARK_TESTER` through `_register_test()`, and `_build_test()` turns each config into instantiated operator cases after filtering unsupported `cuda`, `mps`, and `xpu` inputs. `BenchmarkRunner.run()` warms each test, fixes NumPy's seed from `full_test_id`, selects `run_forward`, `run_backward`, `run_jit_forward`, or `run_compile_forward`, and emits terminal, CSV, or JSON output from measured latencies and memory data.

The directory also contains suite-specific orchestration. `compare.sh` runs `python -m fastrnns.bench --fuser=old` and `--fuser=te`, stores `old.json` and `te.json`, and then feeds both files to `compare-fastrnn-results.py`.

## Performance Profile

- **Allocation sites** - `_build_test()` deep-copies operator objects for backward-input variants, materializes input dictionaries in `op.init(...)`, and creates per-record dataclass payloads in `_output_json()`.
- **Synchronization costs** - `BenchmarkRunner._launch_forward()` switches to `torch.utils.benchmark.Timer` for `cuda` and `xpu` cases so timing includes the backend synchronization needed for correct GPU measurements.
- **Data movement** - Benchmarks move measured results into CSV and JSON files, and `compare.sh` writes full benchmark result sets to `old.json` and `te.json` before comparison.
- **Redundant or repeated work** - `BenchmarkRunner.run()` always performs warmup iterations before real measurements, and multi-run benchmarks intentionally repeat the same operator case to reduce variance and report stable numbers.

## Design Rationale

PyTorch keeps benchmark code in a dedicated top-level component so performance measurement stays separate from correctness tests and production kernels. The registration-based design in `operator_benchmark/benchmark_core.py` lets new suites add cases declaratively while reusing one runner for warmup, filtering, timing, and export.
