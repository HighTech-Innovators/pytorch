# ADR Coverage Complete

Gate passed: 2026-07-10
Validator: work/2-validate-adrs.md

## Coverage Table

| Directory | Status | ADR path | Exclusion reason |
|---|---|---|---|
| `android` | EXCLUDED | — | Build/config only |
| `aten` | COVERED | `aten/ADR.md` | |
| `aten/src/ATen` | COVERED | `aten/src/ATen/ADR.md` | |
| `aten/src/ATen/native` | COVERED | `aten/src/ATen/native/ADR.md` | |
| `aten/src/ATen/native/cpu` | COVERED | `aten/src/ATen/native/cpu/ADR.md` | |
| `adr` | EXCLUDED | — | Build/config only |
| `benchmarks` | EXCLUDED | — | Build/config only |
| `binaries` | EXCLUDED | — | Build/config only |
| `c10` | COVERED | `c10/ADR.md` | |
| `c10/core` | COVERED | `c10/core/ADR.md` | |
| `c10/cuda` | COVERED | `c10/cuda/ADR.md` | |
| `c10/util` | COVERED | `c10/util/ADR.md` | |
| `caffe2` | COVERED | `caffe2/ADR.md` | |
| `cmake` | EXCLUDED | — | Build/config only |
| `docs` | EXCLUDED | — | Build/config only |
| `functorch` | COVERED | `functorch/ADR.md` | |
| `mypy_plugins` | EXCLUDED | — | Build/config only |
| `scripts` | EXCLUDED | — | Build/config only |
| `test` | EXCLUDED | — | Test suite |
| `third_party` | EXCLUDED | — | Vendored/third-party |
| `tools` | COVERED | `tools/ADR.md` | |
| `tools/autograd` | COVERED | `tools/autograd/ADR.md` | |
| `torch` | COVERED | `torch/ADR.md` | |
| `torch/_dynamo` | COVERED | `torch/_dynamo/ADR.md` | |
| `torch/_inductor` | COVERED | `torch/_inductor/ADR.md` | |
| `torch/autograd` | COVERED | `torch/autograd/ADR.md` | |
| `torch/csrc` | COVERED | `torch/csrc/ADR.md` | |
| `torch/csrc/api` | COVERED | `torch/csrc/api/ADR.md` | |
| `torch/csrc/autograd` | COVERED | `torch/csrc/autograd/ADR.md` | |
| `torch/csrc/jit` | COVERED | `torch/csrc/jit/ADR.md` | |
| `torch/distributed` | COVERED | `torch/distributed/ADR.md` | |
| `torch/fx` | COVERED | `torch/fx/ADR.md` | |
| `torch/jit` | COVERED | `torch/jit/ADR.md` | |
| `torch/nn` | COVERED | `torch/nn/ADR.md` | |
| `torch/nn/parallel` | COVERED | `torch/nn/parallel/ADR.md` | |
| `torch/profiler` | COVERED | `torch/profiler/ADR.md` | |
| `torchgen` | COVERED | `torchgen/ADR.md` | |

## Book Subsystem Cross-reference

| Subsystem (from book architecture-map) | Directory | Status |
|---|---|---|
| c10 core library | `c10/` | COVERED |
| c10 core types | `c10/core/` | COVERED |
| c10 CUDA types | `c10/cuda/` | COVERED |
| c10 utilities | `c10/util/` | COVERED |
| ATen operator layer | `aten/src/ATen/` | COVERED |
| ATen native operators | `aten/src/ATen/native/` | COVERED |
| ATen vectorized CPU kernels | `aten/src/ATen/native/cpu/` | COVERED |
| TorchGen code generation | `torchgen/` | COVERED |
| Python-C++ bridge | `torch/csrc/` | COVERED |
| C++ autograd engine | `torch/csrc/autograd/` | COVERED |
| TorchScript JIT | `torch/csrc/jit/` | COVERED |
| C++ frontend (LibTorch) | `torch/csrc/api/` | COVERED |
| Python public API | `torch/` | COVERED |
| Python autograd | `torch/autograd/` | COVERED |
| nn.Module system | `torch/nn/` | COVERED |
| DDP / DataParallel | `torch/nn/parallel/` | COVERED |
| Distributed training | `torch/distributed/` | COVERED |
| TorchDynamo compiler | `torch/_dynamo/` | COVERED |
| TorchInductor codegen | `torch/_inductor/` | COVERED |
| FX graph IR | `torch/fx/` | COVERED |
| Profiler | `torch/profiler/` | COVERED |
| TorchScript Python API | `torch/jit/` | COVERED |
| Serialization (`torch/serialization.py`) | `torch/` | COVERED (file within covered `torch/` directory) |
| Gradient formula tooling | `tools/autograd/` | COVERED |
| Test suite | `test/` | EXCLUDED — Test suite; CI-only, not an architectural unit |

## Known Partial Coverage

None.
