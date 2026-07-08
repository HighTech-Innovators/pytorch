# ADR Scope

This file enumerates directories under `./src` and their ADR coverage status.

- **EXCLUDED** — no core architecture logic to document (build config, tests, vendored code, thin wrappers, device-specific variants outside the CPU-only scope).
- **PENDING** — an ADR is required; not yet written.
- **COVERED** — an ADR (`<dir>/ADR.md`) has been written.

## PENDING (ADRs required)

| Directory | Status | Notes |
|---|---|---|
| `c10/core` | PENDING | Core tensor primitives: TensorImpl, Storage, DispatchKey, Scalar |
| `c10/util` | PENDING | Utility library: intrusive_ptr, ArrayRef, Exception |
| `c10/cuda` | PENDING | CUDA device abstractions, CachingAllocator (documented; CPU-only build) |
| `c10/mobile` | PENDING | Mobile-optimized allocators and runtime |
| `aten/src/ATen/core` | PENDING | Dispatch infrastructure: Dispatcher, OperatorEntry, boxing |
| `aten/src/ATen/native` | PENDING | Operator implementations (CPU, CUDA, Sparse) |
| `aten/src/ATen/native/cpu` | PENDING | Vectorized CPU kernels |
| `aten/src/ATen/native/cuda` | PENDING | CUDA kernel implementations |
| `aten/src/ATen/native/sparse` | PENDING | Sparse tensor operations |
| `aten/src/ATen/native/quantized` | PENDING | Quantization kernels |
| `torch/csrc/autograd` | PENDING | Autograd engine, Node, GraphTask, backward execution |
| `torch/csrc/jit` | PENDING | TorchScript JIT compiler (legacy) |
| `torch/csrc/api` | PENDING | C++ frontend API (libtorch) |
| `torch/csrc/distributed` | PENDING | Distributed communication backends |
| `torch/csrc/inductor` | PENDING | C++ support for Inductor-generated code |
| `torch/csrc/dynamo` | PENDING | C++ support for Dynamo frame evaluation |
| `torch/csrc/profiler` | PENDING | Kineto profiler integration |
| `torch/autograd` | PENDING | Python autograd API: Function, grad_mode, gradcheck |
| `torch/nn` | PENDING | Neural network modules, layers, loss functions |
| `torch/nn/modules` | PENDING | Module implementations: Linear, Conv, RNN, Transformer |
| `torch/optim` | PENDING | Optimizers: SGD, Adam, AdamW |
| `torch/_dynamo` | PENDING | TorchDynamo compiler frontend |
| `torch/_inductor` | PENDING | TorchInductor compiler backend |
| `torch/_inductor/codegen` | PENDING | Kernel code generation (Triton, C++) |
| `torch/fx` | PENDING | FX graph intermediate representation |
| `torch/_functorch` | PENDING | Functional transforms: vmap, grad, AOTAutograd |
| `torch/_export` | PENDING | Model export infrastructure |
| `torch/distributed` | PENDING | Distributed training: DDP, FSDP, RPC |
| `torch/cuda` | PENDING | CUDA device management, streams, events |
| `torch/amp` | PENDING | Automatic mixed precision |
| `torch/profiler` | PENDING | Profiler API and Kineto integration |
| `torch/utils` | PENDING | Utilities: data loading, checkpointing, benchmarking |
| `torchgen` | PENDING | Code generation from native_functions.yaml |
| `torchgen/api` | PENDING | Code-gen type/signature models |
| `torchgen/dest` | PENDING | Code-gen destinations (kernel/registration emitters) |
| `tools/autograd` | PENDING | Autograd code generation, derivatives.yaml |
| `tools/code_analyzer` | PENDING | Operator dependency / selective-build analysis |
| `tools/jit` | PENDING | JIT code generation helpers |
| `caffe2/core` | PENDING | Legacy Caffe2 runtime (partially shared with PyTorch) |
| `functorch/compile` | PENDING | functorch compile API (AOTAutograd entrypoints) |
| `functorch/_src` | PENDING | functorch compatibility shim to torch/_functorch |

## EXCLUDED (no ADR)

| Directory | Status | Reason |
|---|---|---|
| `benchmarks` | EXCLUDED | Benchmark scripts, no core logic |
| `binaries` | EXCLUDED | Binary entrypoints |
| `cmake` | EXCLUDED | Build configuration |
| `docs` | EXCLUDED | Documentation |
| `android` | EXCLUDED | Mobile/device-specific, non-core |
| `scripts` | EXCLUDED | Scripts |
| `test` | EXCLUDED | Test suite |
| `third_party` | EXCLUDED | Vendored code |
| `mypy_plugins` | EXCLUDED | Type-checking plugins |
| `c10/benchmark` | EXCLUDED | Benchmarks |
| `c10/hip` | EXCLUDED | ROCm variant |
| `c10/macros` | EXCLUDED | Header macros only |
| `c10/metal` | EXCLUDED | Metal GPU variant |
| `c10/test` | EXCLUDED | Tests |
| `c10/xpu` | EXCLUDED | XPU variant |
| `aten/src/ATen/templates` | EXCLUDED | Code-gen templates |
| `aten/src/ATen/test` | EXCLUDED | Tests |
| `aten/src/ATen/benchmarks` | EXCLUDED | Benchmarks |
| `aten/tools` | EXCLUDED | ATen build tooling |
| `tools/alerts` | EXCLUDED | CI alerting |
| `tools/amd_build` | EXCLUDED | ROCm build |
| `tools/build_defs` | EXCLUDED | Build definitions |
| `tools/code_coverage` | EXCLUDED | Coverage tooling |
| `tools/coverage_plugins_package` | EXCLUDED | Coverage plugins |
| `tools/dynamo` | EXCLUDED | Dynamo dev tooling |
| `tools/experimental` | EXCLUDED | Experimental tooling |
| `tools/gdb` | EXCLUDED | Debugger scripts |
| `tools/github` | EXCLUDED | CI/GitHub tooling |
| `tools/iwyu` | EXCLUDED | include-what-you-use config |
| `tools/linter` | EXCLUDED | Linters |
| `tools/lite_interpreter` | EXCLUDED | Lite interpreter tooling |
| `tools/lldb` | EXCLUDED | Debugger scripts |
| `tools/packaging` | EXCLUDED | Packaging |
| `tools/pyi` | EXCLUDED | Stub generation |
| `tools/setup_helpers` | EXCLUDED | Setup helpers |
| `tools/shared` | EXCLUDED | Shared build helpers |
| `tools/stats` | EXCLUDED | CI stats |
| `tools/test` | EXCLUDED | Tests |
| `tools/testing` | EXCLUDED | Test infra |
| `tools/vendoring` | EXCLUDED | Vendoring |
| `functorch/benchmarks` | EXCLUDED | Benchmarks |
| `functorch/docs` | EXCLUDED | Documentation |
| `functorch/examples` | EXCLUDED | Examples |
| `functorch/experimental` | EXCLUDED | Experimental |
| `functorch/op_analysis` | EXCLUDED | Analysis scripts |
| `functorch/dim` | EXCLUDED | Named-dim prototype |
| `functorch/einops` | EXCLUDED | einops shim |
| `caffe2/perfkernels` | EXCLUDED | Legacy perf kernels |
| `caffe2/serialize` | EXCLUDED | Legacy serialization |
| `caffe2/utils` | EXCLUDED | Legacy utils |
| `torch/_awaits` | EXCLUDED | Thin JIT await wrapper |
| `torch/backends` | EXCLUDED | Backend config flags |
| `torch/_C` | EXCLUDED | Native extension stub |
| `torch/_C_flatbuffer` | EXCLUDED | Native stub |
| `torch/compiler` | EXCLUDED | Thin compiler facade |
| `torch/contrib` | EXCLUDED | Contrib extras |
| `torch/cpu` | EXCLUDED | Thin CPU facade |
| `torch/_custom_op` | EXCLUDED | Legacy custom-op shim |
| `torch/_decomp` | EXCLUDED | Decomposition tables |
| `torch/_dispatch` | EXCLUDED | Python dispatch helpers |
| `torch/fft` | EXCLUDED | Thin functional wrapper |
| `torch/func` | EXCLUDED | Thin wrapper over _functorch |
| `torch/futures` | EXCLUDED | Thin futures wrapper |
| `torch/headeronly` | EXCLUDED | Header-only shims |
| `torch/_higher_order_ops` | EXCLUDED | HOP definitions |
| `torch/_lazy` | EXCLUDED | Lazy tensor backend |
| `torch/legacy` | EXCLUDED | Legacy code |
| `torch/lib` | EXCLUDED | Prebuilt libs |
| `torch/_library` | EXCLUDED | Library registration helpers |
| `torch/linalg` | EXCLUDED | Thin functional wrapper |
| `torch/_logging` | EXCLUDED | Logging config |
| `torch/masked` | EXCLUDED | Masked tensor prototype |
| `torch/monitor` | EXCLUDED | Monitoring hooks |
| `torch/mps` | EXCLUDED | MPS device variant |
| `torch/mtia` | EXCLUDED | MTIA device variant |
| `torch/multiprocessing` | EXCLUDED | MP wrappers |
| `torch/_native` | EXCLUDED | Native shim |
| `torch/nativert` | EXCLUDED | Native runtime prototype |
| `torch/nested` | EXCLUDED | Nested tensor wrapper |
| `torch/numa` | EXCLUDED | NUMA bindings |
| `torch/_numpy` | EXCLUDED | NumPy compat layer |
| `torch/onnx` | EXCLUDED | ONNX export (self-contained) |
| `torch/package` | EXCLUDED | Packaging |
| `torch/_prims` | EXCLUDED | Primitive op refs |
| `torch/_prims_common` | EXCLUDED | Prim helpers |
| `torch/quantization` | EXCLUDED | Legacy quant shim (see torch/ao) |
| `torch/_refs` | EXCLUDED | Reference decompositions |
| `torch/signal` | EXCLUDED | Signal windows |
| `torch/sparse` | EXCLUDED | Thin sparse wrapper |
| `torch/special` | EXCLUDED | Thin functional wrapper |
| `torch/_strobelight` | EXCLUDED | Profiling integration |
| `torch/_subclasses` | EXCLUDED | Fake/functional tensor subclasses |
| `torch/testing` | EXCLUDED | Test utilities |
| `torch/_vendor` | EXCLUDED | Vendored code |
| `torch/xpu` | EXCLUDED | XPU device variant |
| `torch/accelerator` | EXCLUDED | Accelerator facade |
| `torch/ao` | EXCLUDED | Quant/sparsity (large, self-contained) |
| `torch/distributions` | EXCLUDED | Probability distributions |
| `torchgen/aoti` | EXCLUDED | AOTInductor codegen |
| `torchgen/_autoheuristic` | EXCLUDED | Autoheuristic |
| `torchgen/decompositions` | EXCLUDED | Decomp codegen |
| `torchgen/fuse` | EXCLUDED | Fusion codegen |
| `torchgen/operator_versions` | EXCLUDED | Operator versioning |
| `torchgen/selective_build` | EXCLUDED | Selective build |
| `torchgen/shape_functions` | EXCLUDED | Shape function codegen |
| `torchgen/static_runtime` | EXCLUDED | Static runtime codegen |
