# ADR Coverage Complete

Gate passed: 2026-07-09
Validator: work/2-validate-adrs.md

## Coverage Table

| Directory | Status | ADR path | Exclusion reason |
|---|---|---|---|
| ./adr | EXCLUDED | — | Build/config only — ADR generation and validation tooling; not a PyTorch architectural component |
| ./android | EXCLUDED | — | Build/config only — Android platform deployment; no core architectural logic |
| ./aten | EXCLUDED | — | Build/config only — pass-through root with no source files; architectural unit is aten/src/ATen (COVERED via book cross-reference) |
| ./benchmarks | EXCLUDED | — | Test suite — benchmark harness code |
| ./binaries | EXCLUDED | — | Build/config only — compiled binary entry points |
| ./c10 | COVERED | ./c10/ADR.md | |
| ./caffe2 | EXCLUDED | — | Vendored/third-party — bundled legacy Caffe2 runtime; not owned architecture |
| ./cmake | EXCLUDED | — | Build/config only — CMake configuration modules |
| ./docs | EXCLUDED | — | Build/config only — documentation build infrastructure; no architectural logic |
| ./functorch | COVERED | ./functorch/ADR.md | |
| ./mypy_plugins | EXCLUDED | — | Build/config only — mypy type-checking plugin |
| ./scripts | EXCLUDED | — | Build/config only — CI and developer scripts |
| ./test | EXCLUDED | — | Test suite — 48 subdirectories of comprehensive tests |
| ./third_party | EXCLUDED | — | Vendored/third-party — external dependencies (protobuf, NCCL, etc.) |
| ./tools | COVERED | ./tools/ADR.md | |
| ./torch | COVERED | ./torch/ADR.md | |
| ./torchgen | COVERED | ./torchgen/ADR.md | |
| ./c10/benchmark | EXCLUDED | — | Test suite — micro-benchmarks for c10 internals |
| ./c10/core | COVERED | ./c10/core/ADR.md | |
| ./c10/cuda | COVERED | ./c10/cuda/ADR.md | |
| ./c10/hip | EXCLUDED | — | Leaf with no architectural boundary — ROCm/HIP mirror of c10/cuda; covered by c10/cuda ADR |
| ./c10/macros | EXCLUDED | — | Build/config only — preprocessor macro definitions |
| ./c10/metal | COVERED | ./c10/metal/ADR.md | |
| ./c10/mobile | COVERED | ./c10/mobile/ADR.md | |
| ./c10/test | EXCLUDED | — | Test suite — c10 unit tests |
| ./c10/util | COVERED | ./c10/util/ADR.md | |
| ./c10/xpu | COVERED | ./c10/xpu/ADR.md | |
| ./aten/src | EXCLUDED | — | Build/config only — pass-through directory with no direct source files |
| ./aten/src/ATen | COVERED | ./aten/src/ATen/ADR.md | |
| ./torch/accelerator | COVERED | ./torch/accelerator/ADR.md | |
| ./torch/amp | COVERED | ./torch/amp/ADR.md | |
| ./torch/ao | EXCLUDED | — | Leaf with no architectural boundary — alpha operator support; covered by torch ADR |
| ./torch/autograd | COVERED | ./torch/autograd/ADR.md | |
| ./torch/_awaits | EXCLUDED | — | Leaf with no architectural boundary — async future primitives; covered by distributed ADR |
| ./torch/backends | EXCLUDED | — | Leaf with no architectural boundary — backend configuration flags |
| ./torch/_C | EXCLUDED | — | Build/config only — compiled extension stub; implementation is in torch/csrc |
| ./torch/compiler | COVERED | ./torch/compiler/ADR.md | |
| ./torch/contrib | EXCLUDED | — | Leaf with no architectural boundary — 152-line experimental contrib module with no independent architectural role |
| ./torch/cpu | COVERED | ./torch/cpu/ADR.md | |
| ./torch/csrc | COVERED | ./torch/csrc/ADR.md | |
| ./torch/cuda | COVERED | ./torch/cuda/ADR.md | |
| ./torch/_custom_op | COVERED | ./torch/_custom_op/ADR.md | |
| ./torch/_decomp | COVERED | ./torch/_decomp/ADR.md | |
| ./torch/_dispatch | COVERED | ./torch/_dispatch/ADR.md | |
| ./torch/distributed | COVERED | ./torch/distributed/ADR.md | |
| ./torch/distributions | COVERED | ./torch/distributions/ADR.md | |
| ./torch/_dynamo | COVERED | ./torch/_dynamo/ADR.md | |
| ./torch/export | COVERED | ./torch/export/ADR.md | |
| ./torch/_export | COVERED | ./torch/_export/ADR.md | |
| ./torch/fft | COVERED | ./torch/fft/ADR.md | |
| ./torch/func | COVERED | ./torch/func/ADR.md | |
| ./torch/_functorch | COVERED | ./torch/_functorch/ADR.md | |
| ./torch/futures | COVERED | ./torch/futures/ADR.md | |
| ./torch/fx | COVERED | ./torch/fx/ADR.md | |
| ./torch/_higher_order_ops | COVERED | ./torch/_higher_order_ops/ADR.md | |
| ./torch/_inductor | COVERED | ./torch/_inductor/ADR.md | |
| ./torch/jit | COVERED | ./torch/jit/ADR.md | |
| ./torch/linalg | COVERED | ./torch/linalg/ADR.md | |
| ./torch/multiprocessing | COVERED | ./torch/multiprocessing/ADR.md | |
| ./torch/nested | COVERED | ./torch/nested/ADR.md | |
| ./torch/nn | COVERED | ./torch/nn/ADR.md | |
| ./torch/optim | COVERED | ./torch/optim/ADR.md | |
| ./torch/profiler | COVERED | ./torch/profiler/ADR.md | |
| ./torch/_prims | COVERED | ./torch/_prims/ADR.md | |
| ./torch/quantization | COVERED | ./torch/quantization/ADR.md | |
| ./torch/sparse | COVERED | ./torch/sparse/ADR.md | |
| ./torch/utils | COVERED | ./torch/utils/ADR.md | |
| ./torchgen/api | COVERED | ./torchgen/api/ADR.md | |
| ./torchgen/dest | COVERED | ./torchgen/dest/ADR.md | |
| ./torchgen/aoti | COVERED | ./torchgen/aoti/ADR.md | |
| ./functorch/compile | EXCLUDED | — | Leaf with no architectural boundary — compile helpers; covered by functorch ADR |
| ./functorch/_src | EXCLUDED | — | Leaf with no architectural boundary — internal functorch source; covered by functorch ADR |

## Book Subsystem Cross-reference

| Subsystem (from book) | Directory | Status |
|---|---|---|
| Core Python API | ./torch | COVERED |
| Autograd system | ./torch/autograd | COVERED |
| Neural network module system | ./torch/nn | COVERED |
| Module implementations | ./torch/nn/modules | COVERED (via torch/nn ancestor) |
| Optimisers | ./torch/optim | COVERED |
| Distributed training | ./torch/distributed | COVERED |
| FSDP parameter sharding | ./torch/distributed/fsdp | COVERED (via torch/distributed ancestor) |
| RPC framework | ./torch/distributed/rpc | COVERED (via torch/distributed ancestor) |
| FX graph system | ./torch/fx | COVERED |
| JIT compilation | ./torch/jit | COVERED |
| TorchDynamo | ./torch/_dynamo | COVERED |
| TorchInductor | ./torch/_inductor | COVERED |
| Profiler | ./torch/profiler | COVERED |
| CUDA support | ./torch/cuda | COVERED |
| Mixed precision | ./torch/amp | COVERED |
| Utilities | ./torch/utils | COVERED |
| Quantisation | ./torch/quantization | COVERED |
| Export pipeline | ./torch/_export | COVERED |
| Functional transforms | ./torch/_functorch | COVERED |
| Core abstractions | ./c10/core | COVERED |
| C++ utilities | ./c10/util | COVERED |
| CUDA abstractions | ./c10/cuda | COVERED |
| Mobile support | ./c10/mobile | COVERED |
| ATen tensor library | ./aten/src/ATen | COVERED |
| ATen core dispatch | ./aten/src/ATen/core | COVERED (via aten/src/ATen ancestor) |
| Dispatcher | ./aten/src/ATen/core/dispatch | COVERED (via aten/src/ATen ancestor) |
| Native kernels | ./aten/src/ATen/native | COVERED (via aten/src/ATen ancestor) |
| CPU backend | ./aten/src/ATen/cpu | COVERED (via aten/src/ATen ancestor) |
| CUDA backend | ./aten/src/ATen/cuda | COVERED (via aten/src/ATen ancestor) |
| Code generation | ./torchgen | COVERED |
| API translation | ./torchgen/api | COVERED |
| Code generation targets | ./torchgen/dest | COVERED |
| C++ binding bridge | ./torch/csrc | COVERED |
| C++ autograd | ./torch/csrc/autograd | COVERED (via torch/csrc ancestor) |
| JIT backend | ./torch/csrc/jit | COVERED (via torch/csrc ancestor) |
| JIT serialisation | ./torch/csrc/jit/serialization | COVERED (via torch/csrc ancestor) |
| Functional transforms bridge | ./functorch | COVERED |
| AOT Autograd | ./functorch/_src/aot_autograd | COVERED (via functorch ancestor) |
| Build tools | ./tools | COVERED |

## Known Partial Coverage

None.
