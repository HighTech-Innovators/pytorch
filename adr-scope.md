# ADR Scope

| Directory | Source files present | Status | Reason (if EXCLUDED) |
|---|---|---|---|
| ./android | yes | EXCLUDED | Build/config only — Android platform deployment; no core architectural logic |
| ./aten | no | EXCLUDED | Build/config only — pass-through root with no source files; architectural unit is aten/src/ATen (COVERED via book cross-reference) |
| ./benchmarks | yes | EXCLUDED | Test suite — benchmark harness code |
| ./binaries | yes | EXCLUDED | Build/config only — compiled binary entry points |
| ./c10 | yes | COVERED | Core C++ library; depth-2 units covered below |
| ./caffe2 | yes | EXCLUDED | Vendored/third-party — bundled legacy Caffe2 runtime; not owned architecture |
| ./cmake | no | EXCLUDED | Build/config only — CMake configuration modules |
| ./docs | yes | EXCLUDED | Build/config only — documentation build infrastructure; no architectural logic |
| ./functorch | yes | COVERED | Functional transforms bridge package |
| ./mypy_plugins | yes | EXCLUDED | Build/config only — mypy type-checking plugin |
| ./scripts | yes | EXCLUDED | Build/config only — CI and developer scripts |
| ./test | yes | EXCLUDED | Test suite — 48 subdirectories of comprehensive tests |
| ./third_party | yes | EXCLUDED | Vendored/third-party — external dependencies (protobuf, NCCL, etc.) |
| ./tools | yes | EXCLUDED | Build/config only — build utilities and code-generation helper scripts |
| ./torch | yes | COVERED | Python API; depth-2 architectural units covered below |
| ./torchgen | yes | COVERED | Code generation system |
| ./c10/benchmark | yes | EXCLUDED | Test suite — micro-benchmarks for c10 internals |
| ./c10/core | yes | COVERED | Core abstractions — TensorImpl, Storage, Allocator, DispatchKey, Device |
| ./c10/cuda | yes | COVERED | CUDA device abstractions |
| ./c10/hip | yes | EXCLUDED | Leaf with no architectural boundary — ROCm/HIP mirror of c10/cuda; covered by c10/cuda ADR |
| ./c10/macros | no | EXCLUDED | Build/config only — preprocessor macro definitions |
| ./c10/metal | yes | EXCLUDED | Leaf with no architectural boundary — Metal/Apple GPU mirror; covered by parent |
| ./c10/mobile | yes | COVERED | Mobile portability layer |
| ./c10/test | yes | EXCLUDED | Test suite — c10 unit tests |
| ./c10/util | yes | COVERED | C++ utility library |
| ./c10/xpu | yes | EXCLUDED | Leaf with no architectural boundary — Intel XPU device mirror; covered by parent |
| ./aten/src | no | EXCLUDED | Build/config only — pass-through directory with no direct source files |
| ./aten/src/ATen | yes | COVERED | ATen tensor library — operator dispatch, native kernels, kernel implementations (added via book cross-reference) |
| ./torch/accelerator | yes | EXCLUDED | Leaf with no architectural boundary — thin device-agnostic accelerator API; covered by torch ADR |
| ./torch/amp | yes | EXCLUDED | Leaf with no architectural boundary — automatic mixed precision; covered by torch ADR |
| ./torch/ao | yes | EXCLUDED | Leaf with no architectural boundary — alpha operator support; covered by torch ADR |
| ./torch/autograd | yes | COVERED | Autograd engine and automatic differentiation |
| ./torch/_awaits | yes | EXCLUDED | Leaf with no architectural boundary — async future primitives; covered by distributed ADR |
| ./torch/backends | yes | EXCLUDED | Leaf with no architectural boundary — backend configuration flags |
| ./torch/_C | no | EXCLUDED | Build/config only — compiled extension stub; implementation is in torch/csrc |
| ./torch/compiler | yes | EXCLUDED | Leaf with no architectural boundary — public torch.compile API shim |
| ./torch/contrib | yes | EXCLUDED | Leaf with no architectural boundary — 152-line experimental contrib module with no independent architectural role |
| ./torch/cpu | yes | EXCLUDED | Leaf with no architectural boundary — CPU thread management shim |
| ./torch/csrc | yes | COVERED | C++ binding bridge — pybind11 bindings, autograd C++ engine |
| ./torch/cuda | yes | EXCLUDED | Leaf with no architectural boundary — Python CUDA device API; covered by c10/cuda ADR |
| ./torch/_custom_op | yes | EXCLUDED | Leaf with no architectural boundary — custom operator registration helper |
| ./torch/_decomp | yes | EXCLUDED | Leaf with no architectural boundary — operator decompositions; covered by torchgen ADR |
| ./torch/_dispatch | yes | EXCLUDED | Leaf with no architectural boundary — Python-level dispatch helpers |
| ./torch/distributed | yes | COVERED | Distributed training — c10d, DDP, FSDP, RPC, DeviceMesh |
| ./torch/distributions | yes | EXCLUDED | Leaf with no architectural boundary — probability distributions; covered by torch ADR |
| ./torch/_dynamo | yes | COVERED | TorchDynamo — bytecode analysis and FX graph extraction |
| ./torch/export | yes | EXCLUDED | Leaf with no architectural boundary — public export API shim; covered by torch/_export ADR |
| ./torch/_export | yes | COVERED | Export pipeline — model export to portable formats |
| ./torch/fft | yes | EXCLUDED | Leaf with no architectural boundary — FFT operation wrappers |
| ./torch/func | yes | EXCLUDED | Leaf with no architectural boundary — public func API shim; covered by functorch ADR |
| ./torch/_functorch | yes | COVERED | Functional transforms implementation — vmap, grad, AOT autograd |
| ./torch/futures | yes | EXCLUDED | Leaf with no architectural boundary — async future primitives |
| ./torch/fx | yes | COVERED | FX graph system — IR, tracing, transformation |
| ./torch/_higher_order_ops | yes | EXCLUDED | Leaf with no architectural boundary — higher-order operator support |
| ./torch/_inductor | yes | COVERED | TorchInductor — compilation backend and code generation |
| ./torch/jit | yes | COVERED | JIT and TorchScript — scripting, tracing, serialisation |
| ./torch/linalg | yes | EXCLUDED | Leaf with no architectural boundary — linear algebra operation wrappers |
| ./torch/multiprocessing | yes | EXCLUDED | Leaf with no architectural boundary — multiprocessing utilities |
| ./torch/nested | yes | EXCLUDED | Leaf with no architectural boundary — nested tensor support |
| ./torch/nn | yes | COVERED | Neural network module system |
| ./torch/optim | yes | COVERED | Optimizers — parameter update algorithms |
| ./torch/profiler | yes | COVERED | Profiler — performance measurement and observability |
| ./torch/_prims | yes | EXCLUDED | Leaf with no architectural boundary — primitive operations; covered by torch/_functorch |
| ./torch/quantization | yes | EXCLUDED | Leaf with no architectural boundary — quantisation API; lower priority |
| ./torch/sparse | yes | EXCLUDED | Leaf with no architectural boundary — sparse tensor support |
| ./torch/utils | yes | EXCLUDED | Leaf with no architectural boundary — utilities; no single architectural boundary |
| ./torchgen/api | yes | EXCLUDED | Leaf with no architectural boundary — API translation helpers; covered by torchgen ADR |
| ./torchgen/dest | yes | EXCLUDED | Leaf with no architectural boundary — codegen output targets; covered by torchgen ADR |
| ./torchgen/aoti | yes | EXCLUDED | Leaf with no architectural boundary — AOT Inductor codegen; covered by torchgen ADR |
| ./functorch/compile | yes | EXCLUDED | Leaf with no architectural boundary — compile helpers; covered by functorch ADR |
| ./functorch/_src | yes | EXCLUDED | Leaf with no architectural boundary — internal functorch source; covered by functorch ADR |
