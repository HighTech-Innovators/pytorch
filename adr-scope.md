# ADR Scope

This file enumerates directories under `./src` and their ADR coverage status.

- **EXCLUDED** — no core architecture logic to document (build config, tests, vendored code, thin wrappers, device-specific variants outside the CPU-only scope).
- **COVERED** — an ADR (`<dir>/ADR.md`) has been written.

## COVERED

| Directory | Status | Notes |
|---|---|---|
| `aten/src/ATen/core` | COVERED | Dispatch infrastructure: Dispatcher, OperatorEntry, boxing |
| `aten/src/ATen/native` | COVERED | Operator implementations (CPU, CUDA, Sparse) |
| `aten/src/ATen/native/cpu` | COVERED | Vectorized CPU kernels |
| `aten/src/ATen/native/cuda` | COVERED | CUDA kernel implementations |
| `aten/src/ATen/native/quantized` | COVERED | Quantization kernels |
| `aten/src/ATen/native/sparse` | COVERED | Sparse tensor operations |
| `c10/core` | COVERED | Core tensor primitives: TensorImpl, Storage, DispatchKey, Scalar |
| `c10/cuda` | COVERED | CUDA device abstractions, CachingAllocator |
| `c10/mobile` | COVERED | Mobile-optimized allocators and runtime |
| `c10/util` | COVERED | Utility library: intrusive_ptr, ArrayRef, Exception |
| `caffe2/core` | COVERED | Legacy Caffe2 runtime (partially shared with PyTorch) |
| `functorch/_src` | COVERED | functorch compatibility shim to torch/_functorch |
| `functorch/compile` | COVERED | functorch compile API (AOTAutograd entrypoints) |
| `tools/autograd` | COVERED | Autograd code generation, derivatives.yaml |
| `tools/code_analyzer` | COVERED | Operator dependency / selective-build analysis |
| `tools/jit` | COVERED | JIT code generation helpers |
| `torch/_decomp` | COVERED | Decomposition registry: operator lowering to primitive ops |
| `torch/_dynamo` | COVERED | TorchDynamo compiler frontend |
| `torch/_export` | COVERED | Model export infrastructure |
| `torch/_functorch` | COVERED | Functional transforms: vmap, grad, AOTAutograd |
| `torch/_inductor` | COVERED | TorchInductor compiler backend |
| `torch/_inductor/codegen` | COVERED | Kernel code generation (Triton, C++) |
| `torch/amp` | COVERED | Automatic mixed precision |
| `torch/autograd` | COVERED | Python autograd API: Function, grad_mode, gradcheck |
| `torch/csrc/api` | COVERED | C++ frontend API (libtorch) |
| `torch/csrc/autograd` | COVERED | Autograd engine, Node, GraphTask, backward execution |
| `torch/csrc/distributed` | COVERED | Distributed communication backends |
| `torch/csrc/dynamo` | COVERED | C++ support for Dynamo frame evaluation |
| `torch/csrc/inductor` | COVERED | C++ support for Inductor-generated code |
| `torch/csrc/jit` | COVERED | TorchScript JIT compiler (legacy) |
| `torch/csrc/profiler` | COVERED | Kineto profiler integration |
| `torch/cuda` | COVERED | CUDA device management, streams, events |
| `torch/distributed` | COVERED | Distributed training: DDP, FSDP, RPC |
| `torch/fx` | COVERED | FX graph intermediate representation |
| `torch/nn` | COVERED | Neural network modules, layers, loss functions |
| `torch/nn/modules` | COVERED | Module implementations: Linear, Conv, RNN, Transformer |
| `torch/onnx` | COVERED | ONNX export: dynamo and TorchScript paths |
| `torch/optim` | COVERED | Optimizers: SGD, Adam, AdamW |
| `torch/profiler` | COVERED | Profiler API and Kineto integration |
| `torch/utils` | COVERED | Utilities: data loading, checkpointing, benchmarking |
| `torchgen` | COVERED | Code generation from native_functions.yaml |
| `torchgen/api` | COVERED | Code-gen type/signature models |
| `torchgen/dest` | COVERED | Code-gen destinations (kernel/registration emitters) |

## EXCLUDED (no ADR)

| Directory | Status | Reason |
|---|---|---|
| `android` | EXCLUDED | Build/config only |
| `aten` | EXCLUDED | Leaf with no architectural boundary |
| `aten/src/ATen/benchmarks` | EXCLUDED | Test suite |
| `aten/src/ATen/templates` | EXCLUDED | Auto-generated code |
| `aten/src/ATen/test` | EXCLUDED | Test suite |
| `aten/tools` | EXCLUDED | Build/config only |
| `benchmarks` | EXCLUDED | Test suite |
| `binaries` | EXCLUDED | Build/config only |
| `c10` | EXCLUDED | Leaf with no architectural boundary |
| `c10/benchmark` | EXCLUDED | Test suite |
| `c10/hip` | EXCLUDED | Build/config only |
| `c10/macros` | EXCLUDED | Auto-generated code |
| `c10/metal` | EXCLUDED | Build/config only |
| `c10/test` | EXCLUDED | Test suite |
| `c10/xpu` | EXCLUDED | Build/config only |
| `caffe2` | EXCLUDED | Leaf with no architectural boundary |
| `caffe2/perfkernels` | EXCLUDED | Leaf with no architectural boundary |
| `caffe2/serialize` | EXCLUDED | Leaf with no architectural boundary |
| `caffe2/utils` | EXCLUDED | Leaf with no architectural boundary |
| `cmake` | EXCLUDED | Build/config only |
| `docs` | EXCLUDED | Build/config only |
| `functorch` | EXCLUDED | Leaf with no architectural boundary |
| `functorch/benchmarks` | EXCLUDED | Test suite |
| `functorch/dim` | EXCLUDED | Leaf with no architectural boundary |
| `functorch/docs` | EXCLUDED | Build/config only |
| `functorch/einops` | EXCLUDED | Leaf with no architectural boundary |
| `functorch/examples` | EXCLUDED | Build/config only |
| `functorch/experimental` | EXCLUDED | Build/config only |
| `functorch/op_analysis` | EXCLUDED | Build/config only |
| `mypy_plugins` | EXCLUDED | Build/config only |
| `scripts` | EXCLUDED | Build/config only |
| `test` | EXCLUDED | Test suite |
| `third_party` | EXCLUDED | Vendored/third-party |
| `tools` | EXCLUDED | Leaf with no architectural boundary |
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
| `tools/stats` | EXCLUDED | Build/config only |
| `tools/test` | EXCLUDED | Test suite |
| `tools/testing` | EXCLUDED | Test suite |
| `tools/vendoring` | EXCLUDED | Vendored/third-party |
| `torch` | EXCLUDED | Leaf with no architectural boundary |
| `torch/_awaits` | EXCLUDED | Leaf with no architectural boundary |
| `torch/_C` | EXCLUDED | Empty or stub |
| `torch/_C_flatbuffer` | EXCLUDED | Empty or stub |
| `torch/_custom_op` | EXCLUDED | Leaf with no architectural boundary |
| `torch/_dispatch` | EXCLUDED | Leaf with no architectural boundary |
| `torch/_higher_order_ops` | EXCLUDED | Leaf with no architectural boundary |
| `torch/_lazy` | EXCLUDED | Leaf with no architectural boundary |
| `torch/_library` | EXCLUDED | Leaf with no architectural boundary |
| `torch/_logging` | EXCLUDED | Leaf with no architectural boundary |
| `torch/_native` | EXCLUDED | Leaf with no architectural boundary |
| `torch/_numpy` | EXCLUDED | Leaf with no architectural boundary |
| `torch/_prims` | EXCLUDED | Leaf with no architectural boundary |
| `torch/_prims_common` | EXCLUDED | Leaf with no architectural boundary |
| `torch/_refs` | EXCLUDED | Leaf with no architectural boundary |
| `torch/_strobelight` | EXCLUDED | Leaf with no architectural boundary |
| `torch/_subclasses` | EXCLUDED | Leaf with no architectural boundary |
| `torch/_vendor` | EXCLUDED | Vendored/third-party |
| `torch/accelerator` | EXCLUDED | Leaf with no architectural boundary |
| `torch/ao` | EXCLUDED | Leaf with no architectural boundary |
| `torch/backends` | EXCLUDED | Leaf with no architectural boundary |
| `torch/compiler` | EXCLUDED | Leaf with no architectural boundary |
| `torch/contrib` | EXCLUDED | Leaf with no architectural boundary |
| `torch/cpu` | EXCLUDED | Leaf with no architectural boundary |
| `torch/distributions` | EXCLUDED | Leaf with no architectural boundary |
| `torch/fft` | EXCLUDED | Leaf with no architectural boundary |
| `torch/func` | EXCLUDED | Leaf with no architectural boundary |
| `torch/futures` | EXCLUDED | Leaf with no architectural boundary |
| `torch/headeronly` | EXCLUDED | Leaf with no architectural boundary |
| `torch/legacy` | EXCLUDED | Leaf with no architectural boundary |
| `torch/lib` | EXCLUDED | Vendored/third-party |
| `torch/linalg` | EXCLUDED | Leaf with no architectural boundary |
| `torch/masked` | EXCLUDED | Leaf with no architectural boundary |
| `torch/monitor` | EXCLUDED | Leaf with no architectural boundary |
| `torch/mps` | EXCLUDED | Build/config only |
| `torch/mtia` | EXCLUDED | Build/config only |
| `torch/multiprocessing` | EXCLUDED | Leaf with no architectural boundary |
| `torch/nativert` | EXCLUDED | Leaf with no architectural boundary |
| `torch/nested` | EXCLUDED | Leaf with no architectural boundary |
| `torch/numa` | EXCLUDED | Leaf with no architectural boundary |
| `torch/package` | EXCLUDED | Leaf with no architectural boundary |
| `torch/quantization` | EXCLUDED | Leaf with no architectural boundary |
| `torch/signal` | EXCLUDED | Leaf with no architectural boundary |
| `torch/sparse` | EXCLUDED | Leaf with no architectural boundary |
| `torch/special` | EXCLUDED | Leaf with no architectural boundary |
| `torch/testing` | EXCLUDED | Test suite |
| `torch/xpu` | EXCLUDED | Build/config only |
| `torchgen/aoti` | EXCLUDED | Auto-generated code |
| `torchgen/_autoheuristic` | EXCLUDED | Leaf with no architectural boundary |
| `torchgen/decompositions` | EXCLUDED | Auto-generated code |
| `torchgen/fuse` | EXCLUDED | Auto-generated code |
| `torchgen/operator_versions` | EXCLUDED | Leaf with no architectural boundary |
| `torchgen/selective_build` | EXCLUDED | Build/config only |
| `torchgen/shape_functions` | EXCLUDED | Auto-generated code |
| `torchgen/static_runtime` | EXCLUDED | Auto-generated code |
