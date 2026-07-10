# ADR Scope

| Directory | Source files present | Status | Reason (if EXCLUDED) |
|---|---|---|---|
| `android` | yes (Java/C++) | EXCLUDED | Android platform bindings; out of CPU-only Linux deployment scope |
| `aten` | yes (C++) | COVERED | Covered by `aten/src/ATen`, `aten/src/ATen/native`, `aten/src/ATen/native/cpu` ADRs |
| `benchmarks` | yes (Python) | EXCLUDED | Benchmark scripts, not production architecture |
| `binaries` | yes (`.cc`) | EXCLUDED | Standalone benchmark/tool entry-point executables |
| `c10` | yes (C++) | COVERED | Parent ADR + `c10/core`, `c10/util`, `c10/cuda` |
| `caffe2` | yes (C++/Python) | EXCLUDED | Legacy Caffe2 subsystem, deprecated |
| `cmake` | no (build config) | EXCLUDED | Build configuration only |
| `docs` | minimal | EXCLUDED | Documentation |
| `functorch` | yes (Python/C++) | COVERED | Function-transform frontend ADR |
| `mypy_plugins` | yes (Python) | EXCLUDED | Type-checker plugins (tooling) |
| `scripts` | yes (Python/shell) | EXCLUDED | CI/tooling scripts |
| `test` | yes (Python) | EXCLUDED | Test suite |
| `third_party` | yes (vendored) | EXCLUDED | Vendored third-party code |
| `tools` | yes (Python) | COVERED | Architecturally significant `tools/autograd` covered; remainder is build tooling (EXCLUDED) |
| `torch` | yes (Python/C++) | COVERED | Parent ADR + subsystem ADRs |
| `torchgen` | yes (Python) | COVERED | Code-generation subsystem ADR |
| `c10/core` | yes (C++) | PENDING | |
| `c10/util` | yes (C++) | PENDING | |
| `c10/cuda` | yes (C++) | PENDING | CUDA source present; documented but CPU-only at deployment |
| `aten/src/ATen` | yes (C++) | PENDING | |
| `aten/src/ATen/native` | yes (C++) | PENDING | |
| `aten/src/ATen/native/cpu` | yes (C++) | PENDING | |
| `torch/csrc` | yes (C++) | PENDING | |
| `torch/csrc/autograd` | yes (C++) | PENDING | |
| `torch/csrc/jit` | yes (C++) | PENDING | |
| `torch/csrc/api` | yes (C++) | PENDING | |
| `torch/autograd` | yes (Python) | PENDING | |
| `torch/nn` | yes (Python) | PENDING | |
| `torch/nn/parallel` | yes (Python) | PENDING | |
| `torch/distributed` | yes (Python) | PENDING | |
| `torch/_dynamo` | yes (Python) | PENDING | |
| `torch/_inductor` | yes (Python) | PENDING | |
| `torch/fx` | yes (Python) | PENDING | |
| `torch/profiler` | yes (Python) | PENDING | |
| `torch/jit` | yes (Python) | PENDING | |
| `tools/autograd` | yes (Python) | PENDING | |
