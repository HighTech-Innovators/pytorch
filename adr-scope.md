# ADR Scope

| Directory | Source files present | Status | Reason (if EXCLUDED) |
|---|---|---|---|
| `android` | yes (Java/C++) | EXCLUDED | Build/config only |
| `aten` | yes (C++) | COVERED | Parent ADR + `aten/src/ATen`, `aten/src/ATen/native`, `aten/src/ATen/native/cpu` sub-ADRs |
| `benchmarks` | yes (Python) | EXCLUDED | Build/config only |
| `binaries` | yes (`.cc`) | EXCLUDED | Build/config only |
| `c10` | yes (C++) | COVERED | Parent ADR + `c10/core`, `c10/util`, `c10/cuda` |
| `adr` | yes (Python tooling) | EXCLUDED | Build/config only |
| `caffe2` | yes (C++) | COVERED | Legacy Caffe2 serialization container, SIMD embedding kernels, and shared utilities |
| `cmake` | no (build config) | EXCLUDED | Build/config only |
| `docs` | minimal | EXCLUDED | Build/config only |
| `functorch` | yes (Python/C++) | COVERED | Function-transform compatibility shim ADR |
| `mypy_plugins` | yes (Python) | EXCLUDED | Build/config only |
| `scripts` | yes (Python/shell) | EXCLUDED | Build/config only |
| `test` | yes (Python) | EXCLUDED | Test suite |
| `third_party` | yes (vendored) | EXCLUDED | Vendored/third-party |
| `tools` | yes (Python) | COVERED | Parent ADR + architecturally significant `tools/autograd` sub-ADR; remainder is build tooling |
| `torch` | yes (Python/C++) | COVERED | Parent ADR + subsystem ADRs |
| `torchgen` | yes (Python) | COVERED | Code-generation subsystem ADR |
| `c10/core` | yes (C++) | COVERED | |
| `c10/util` | yes (C++) | COVERED | |
| `c10/cuda` | yes (C++) | COVERED | CUDA source present; documented but CPU-only at deployment |
| `aten/src/ATen` | yes (C++) | COVERED | |
| `aten/src/ATen/native` | yes (C++) | COVERED | |
| `aten/src/ATen/native/cpu` | yes (C++) | COVERED | |
| `torch/csrc` | yes (C++) | COVERED | |
| `torch/csrc/autograd` | yes (C++) | COVERED | |
| `torch/csrc/jit` | yes (C++) | COVERED | |
| `torch/csrc/api` | yes (C++) | COVERED | |
| `torch/autograd` | yes (Python) | COVERED | |
| `torch/nn` | yes (Python) | COVERED | |
| `torch/nn/parallel` | yes (Python) | COVERED | |
| `torch/distributed` | yes (Python) | COVERED | |
| `torch/_dynamo` | yes (Python) | COVERED | |
| `torch/_inductor` | yes (Python) | COVERED | |
| `torch/fx` | yes (Python) | COVERED | |
| `torch/profiler` | yes (Python) | COVERED | |
| `torch/jit` | yes (Python) | COVERED | |
| `tools/autograd` | yes (Python) | COVERED | |
