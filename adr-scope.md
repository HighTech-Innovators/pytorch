# ADR Scope

This file enumerates directories under `./src` and their ADR coverage status.

- **EXCLUDED** — no core architecture logic to document (build config, tests, vendored code, thin wrappers, device-specific variants outside the CPU-only scope).
- **COVERED** — an ADR (`<dir>/ADR.md`) has been written.

## COVERED

| Directory | Status | Notes |
|---|---|---|
| `aten/src/ATen/core` | COVERED | Dispatch infrastructure: Dispatcher, OperatorEntry, boxing |
| `aten/src/ATen/cpu/vec` | COVERED | SIMD vectorization abstraction layer |
| `aten/src/ATen/native` | COVERED | Operator implementations (CPU, CUDA, Sparse) |
| `aten/src/ATen/native/cpu` | COVERED | Vectorized CPU kernels |
| `aten/src/ATen/native/cuda` | COVERED | CUDA kernel implementations |
| `aten/src/ATen/native/quantized` | COVERED | Quantization kernels |
| `aten/src/ATen/native/sparse` | COVERED | Sparse tensor operations |
| `benchmarks` | COVERED | Performance benchmarks (operator-level and model-level) |
| `c10/core` | COVERED | Core tensor primitives: TensorImpl, Storage, DispatchKey, Scalar |
| `c10/cuda` | COVERED | CUDA device abstractions, CachingAllocator |
| `c10/metal` | COVERED | Apple Metal (GPU) device support at c10 level |
| `c10/mobile` | COVERED | Mobile-optimized allocators and runtime |
| `c10/util` | COVERED | Utility library: intrusive_ptr, ArrayRef, Exception |
| `c10/xpu` | COVERED | Intel XPU device support at c10 level |
| `caffe2` | COVERED | Legacy Caffe2 runtime top-level package |
| `caffe2/core` | COVERED | Legacy Caffe2 runtime (partially shared with PyTorch) |
| `caffe2/perfkernels` | COVERED | Caffe2 performance-optimized compute kernels |
| `caffe2/serialize` | COVERED | Caffe2 model serialization (zip-based format) |
| `caffe2/utils` | COVERED | Caffe2 utility library (math, proto helpers) |
| `cmake` | COVERED | CMake build modules and configuration helpers |
| `docs` | COVERED | Documentation source (Sphinx-based) |
| `functorch` | COVERED | Functional transforms top-level package |
| `functorch/_src` | COVERED | functorch compatibility shim to torch/_functorch |
| `functorch/compile` | COVERED | functorch compile API (AOTAutograd entrypoints) |
| `functorch/dim` | COVERED | Named/indexed dimension extension for tensors |
| `functorch/einops` | COVERED | Einops-style tensor rearrangement |
| `test` | COVERED | Test suite (common utilities and runner infrastructure) |
| `third_party` | COVERED | External git submodule dependencies |
| `tools` | COVERED | Build scripts, code-gen drivers, autograd derivative tooling |
| `tools/autograd` | COVERED | Autograd code generation, derivatives.yaml |
| `tools/code_analyzer` | COVERED | Operator dependency / selective-build analysis |
| `tools/jit` | COVERED | JIT code generation helpers |
| `tools/stats` | COVERED | CI/test statistics collection and reporting |
| `torch` | COVERED | Python package top-level: public API surface and init |
| `torch/_custom_op` | COVERED | Custom operator registration and schema utilities |
| `torch/_decomp` | COVERED | Decomposition registry: operator lowering to primitive ops |
| `torch/_dispatch` | COVERED | Python-level dispatch utilities and overrides |
| `torch/_dynamo` | COVERED | TorchDynamo compiler frontend |
| `torch/_export` | COVERED | Model export infrastructure |
| `torch/_functorch` | COVERED | Functional transforms: vmap, grad, AOTAutograd |
| `torch/_higher_order_ops` | COVERED | Higher-order operators (cond, while_loop, scan) |
| `torch/_inductor` | COVERED | TorchInductor compiler backend |
| `torch/_inductor/codegen` | COVERED | Kernel code generation (Triton, C++) |
| `torch/_lazy` | COVERED | Lazy tensor execution backend |
| `torch/_library` | COVERED | Library/custom op registration infrastructure |
| `torch/_logging` | COVERED | Structured logging for compiler diagnostics |
| `torch/_native` | COVERED | Python-level native operator implementations |
| `torch/_numpy` | COVERED | NumPy compatibility layer |
| `torch/_prims` | COVERED | Primitive operators for decomposition |
| `torch/_prims_common` | COVERED | Shared utilities for _prims and _refs |
| `torch/_refs` | COVERED | Reference implementations of PyTorch ops in Python |
| `torch/_strobelight` | COVERED | Strobelight profiler integration |
| `torch/_subclasses` | COVERED | Tensor subclass dispatch system (FakeTensor, etc.) |
| `torch/accelerator` | COVERED | Device-agnostic accelerator abstraction |
| `torch/amp` | COVERED | Automatic mixed precision |
| `torch/autograd` | COVERED | Python autograd API: Function, grad_mode, gradcheck |
| `torch/compiler` | COVERED | User-facing torch.compile API |
| `torch/cpu` | COVERED | CPU-specific device APIs |
| `torch/csrc/api` | COVERED | C++ frontend API (libtorch) |
| `torch/csrc/autograd` | COVERED | Autograd engine, Node, GraphTask, backward execution |
| `torch/csrc/distributed` | COVERED | Distributed communication backends |
| `torch/csrc/dynamo` | COVERED | C++ support for Dynamo frame evaluation |
| `torch/csrc/inductor` | COVERED | C++ support for Inductor-generated code |
| `torch/csrc/jit` | COVERED | TorchScript JIT compiler (legacy) |
| `torch/csrc/profiler` | COVERED | Kineto profiler integration |
| `torch/cuda` | COVERED | CUDA device management, streams, events |
| `torch/distributed` | COVERED | Distributed training: DDP, FSDP, RPC |
| `torch/distributions` | COVERED | Probability distributions for probabilistic programming |
| `torch/fft` | COVERED | Fast Fourier Transform operations |
| `torch/func` | COVERED | Functional transforms public API (vmap, grad) |
| `torch/futures` | COVERED | Async future/promise for distributed ops |
| `torch/fx` | COVERED | FX graph intermediate representation |
| `torch/linalg` | COVERED | Linear algebra operations |
| `torch/masked` | COVERED | Masked tensor operations |
| `torch/multiprocessing` | COVERED | Multi-process support (CUDA-aware fork, shared memory) |
| `torch/nativert` | COVERED | Native runtime for exported models |
| `torch/nested` | COVERED | Nested (ragged/variable-length) tensor support |
| `torch/nn` | COVERED | Neural network modules, layers, loss functions |
| `torch/nn/modules` | COVERED | Module implementations: Linear, Conv, RNN, Transformer |
| `torch/numa` | COVERED | NUMA-aware memory allocation |
| `torch/onnx` | COVERED | ONNX export: dynamo and TorchScript paths |
| `torch/optim` | COVERED | Optimizers: SGD, Adam, AdamW |
| `torch/package` | COVERED | Model packaging for hermetic deployment |
| `torch/profiler` | COVERED | Profiler API and Kineto integration |
| `torch/quantization` | COVERED | Legacy quantization API (deprecated) |
| `torch/sparse` | COVERED | Sparse tensor operations (COO, CSR, BSR formats) |
| `torch/special` | COVERED | Special mathematical functions |
| `torch/utils` | COVERED | Utilities: data loading, checkpointing, benchmarking |
| `torch/xpu` | COVERED | Intel XPU (GPU) device support |
| `torchgen` | COVERED | Code generation from native_functions.yaml |
| `torchgen/_autoheuristic` | COVERED | Auto-heuristic selection for code generation |
| `torchgen/api` | COVERED | Code-gen type/signature models |
| `torchgen/dest` | COVERED | Code-gen destinations (kernel/registration emitters) |
| `torchgen/operator_versions` | COVERED | Operator versioning for backwards-compatibility |

## EXCLUDED (no ADR)

| Directory | Status | Reason |
|---|---|---|
| `adr` | EXCLUDED | Build/config only |
| `android` | EXCLUDED | Build/config only |
| `aten` | EXCLUDED | Leaf with no architectural boundary |
| `aten/src/ATen/benchmarks` | EXCLUDED | Test suite |
| `aten/src/ATen/templates` | EXCLUDED | Auto-generated code |
| `aten/src/ATen/test` | EXCLUDED | Test suite |
| `aten/tools` | EXCLUDED | Build/config only |
| `binaries` | EXCLUDED | Build/config only |
| `c10` | EXCLUDED | Leaf with no architectural boundary |
| `c10/benchmark` | EXCLUDED | Test suite |
| `c10/hip` | EXCLUDED | Build/config only |
| `c10/macros` | EXCLUDED | Auto-generated code |
| `c10/test` | EXCLUDED | Test suite |
| `functorch/benchmarks` | EXCLUDED | Test suite |
| `functorch/docs` | EXCLUDED | Build/config only |
| `functorch/examples` | EXCLUDED | Build/config only |
| `functorch/experimental` | EXCLUDED | Build/config only |
| `functorch/op_analysis` | EXCLUDED | Build/config only |
| `mypy_plugins` | EXCLUDED | Build/config only |
| `scripts` | EXCLUDED | Build/config only |
| `tools/alerts` | EXCLUDED | Build/config only |
| `tools/amd_build` | EXCLUDED | Build/config only |
| `tools/build_defs` | EXCLUDED | Build/config only |
| `tools/code_coverage` | EXCLUDED | Build/config only |
| `tools/coverage_plugins_package` | EXCLUDED | Build/config only |
| `tools/dynamo` | EXCLUDED | Build/config only |
| `tools/experimental` | EXCLUDED | Build/config only |
| `tools/gdb` | EXCLUDED | Build/config only |
| `tools/github` | EXCLUDED | Build/config only |
| `tools/iwyu` | EXCLUDED | Build/config only |
| `tools/linter` | EXCLUDED | Build/config only |
| `tools/lite_interpreter` | EXCLUDED | Build/config only |
| `tools/lldb` | EXCLUDED | Build/config only |
| `tools/packaging` | EXCLUDED | Build/config only |
| `tools/pyi` | EXCLUDED | Auto-generated code |
| `tools/setup_helpers` | EXCLUDED | Build/config only |
| `tools/shared` | EXCLUDED | Build/config only |
| `tools/test` | EXCLUDED | Test suite |
| `tools/testing` | EXCLUDED | Test suite |
| `tools/vendoring` | EXCLUDED | Vendored/third-party |
| `torch/_awaits` | EXCLUDED | Leaf with no architectural boundary |
| `torch/_C` | EXCLUDED | Empty or stub |
| `torch/_C_flatbuffer` | EXCLUDED | Empty or stub |
| `torch/_vendor` | EXCLUDED | Vendored/third-party |
| `torch/ao` | EXCLUDED | Leaf with no architectural boundary |
| `torch/backends` | EXCLUDED | Leaf with no architectural boundary |
| `torch/contrib` | EXCLUDED | Leaf with no architectural boundary |
| `torch/headeronly` | EXCLUDED | Leaf with no architectural boundary |
| `torch/legacy` | EXCLUDED | Leaf with no architectural boundary |
| `torch/lib` | EXCLUDED | Vendored/third-party |
| `torch/monitor` | EXCLUDED | Leaf with no architectural boundary |
| `torch/mps` | EXCLUDED | Build/config only |
| `torch/mtia` | EXCLUDED | Build/config only |
| `torch/signal` | EXCLUDED | Leaf with no architectural boundary |
| `torch/testing` | EXCLUDED | Test suite |
| `torchgen/aoti` | EXCLUDED | Auto-generated code |
| `torchgen/decompositions` | EXCLUDED | Auto-generated code |
| `torchgen/fuse` | EXCLUDED | Auto-generated code |
| `torchgen/selective_build` | EXCLUDED | Build/config only |
| `torchgen/shape_functions` | EXCLUDED | Auto-generated code |
| `torchgen/static_runtime` | EXCLUDED | Auto-generated code |
