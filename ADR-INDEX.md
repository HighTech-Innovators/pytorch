# ADR Index

94 Architecture Decision Records · generated 2026-07-08

Each `ADR.md` describes the role, key files, public interface, dependencies,
design rationale, constraints, and observability of one directory.

---

| # | Path | Role |
|---:|---|---|
| 1 | [aten/src/ATen/core/ADR.md](aten/src/ATen/core/ADR.md) | `aten/src/ATen/core` contains the C++ operator-dispatch core that turns schemas and tensor arguments into kernel calls. |
| 2 | [aten/src/ATen/cpu/vec/ADR.md](aten/src/ATen/cpu/vec/ADR.md) | `aten/src/ATen/cpu/vec` owns ATen's CPU SIMD abstraction layer. |
| 3 | [aten/src/ATen/native/ADR.md](aten/src/ATen/native/ADR.md) | `aten/src/ATen/native` is the main ATen operator implementation layer. |
| 4 | [aten/src/ATen/native/cpu/ADR.md](aten/src/ATen/native/cpu/ADR.md) | `aten/src/ATen/native/cpu` contains CPU backend kernels for ATen native operators. |
| 5 | [aten/src/ATen/native/cuda/ADR.md](aten/src/ATen/native/cuda/ADR.md) | `aten/src/ATen/native/cuda` contains CUDA backend kernels and CUDA library integrations for ATen native operators. |
| 6 | [aten/src/ATen/native/quantized/ADR.md](aten/src/ATen/native/quantized/ADR.md) | `aten/src/ATen/native/quantized` implements quantized tensor creation, quantizer-aware tensor operations, quantized operator schemas, and dispatch plumbing for low-precision inference kernels. |
| 7 | [aten/src/ATen/native/sparse/ADR.md](aten/src/ATen/native/sparse/ADR.md) | `aten/src/ATen/native/sparse` implements ATen operators for sparse COO and compressed sparse layouts. |
| 8 | [benchmarks/ADR.md](benchmarks/ADR.md) | `benchmarks` owns the repository's reproducible performance measurement suites and comparison scripts. |
| 9 | [c10/core/ADR.md](c10/core/ADR.md) | `c10/core` defines the universal tensor representation and the primitives every other subsystem builds on. |
| 10 | [c10/cuda/ADR.md](c10/cuda/ADR.md) | `c10/cuda` defines the CUDA runtime substrate below ATen kernels: device guards, stream objects, allocator interfaces, event helpers, error checks, and peer-access utilities. |
| 11 | [c10/metal/ADR.md](c10/metal/ADR.md) | `c10/metal` provides the shared Metal-side utility layer that ATen MPS host code and `.metal` kernels include directly. |
| 12 | [c10/mobile/ADR.md](c10/mobile/ADR.md) | `c10/mobile` provides mobile-specific CPU allocation strategies for inference workloads. |
| 13 | [c10/util/ADR.md](c10/util/ADR.md) | `c10/util` supplies the low-level C++ utility layer that `c10/core`, ATen, and dispatcher code include on hot paths. |
| 14 | [c10/xpu/ADR.md](c10/xpu/ADR.md) | `c10/xpu` is the c10 runtime substrate for Intel XPU devices. |
| 15 | [caffe2/ADR.md](caffe2/ADR.md) | `caffe2` owns the remaining Caffe2 compatibility layer that PyTorch still builds and reuses. |
| 16 | [caffe2/core/ADR.md](caffe2/core/ADR.md) | `caffe2/core` is the remaining minimal Caffe2 core shim that PyTorch still builds for legacy Caffe2 compatibility and shared configuration reporting. |
| 17 | [caffe2/perfkernels/ADR.md](caffe2/perfkernels/ADR.md) | `caffe2/perfkernels` owns CPU-only fast paths for a small set of legacy Caffe2 kernels that still back ATen operators. |
| 18 | [caffe2/serialize/ADR.md](caffe2/serialize/ADR.md) | `caffe2/serialize` owns the C++ ZIP-container reader and writer that TorchScript, libtorch, and related loaders use for archive persistence. |
| 19 | [caffe2/utils/ADR.md](caffe2/utils/ADR.md) | `caffe2/utils` collects small Caffe2 utility surfaces that PyTorch still uses outside the main tensor core. |
| 20 | [cmake/ADR.md](cmake/ADR.md) | `cmake` owns the reusable CMake modules that configure, generate, and assemble the PyTorch build. |
| 21 | [docs/ADR.md](docs/ADR.md) | `docs` owns the Sphinx documentation source tree and the build entry points for published PyTorch reference documentation. |
| 22 | [functorch/_src/ADR.md](functorch/_src/ADR.md) | `functorch/_src` is a legacy private compatibility shim for functorch internals, mapped to book chapters 07 and 13 through its forwarding to `torch/_functorch`. |
| 23 | [functorch/ADR.md](functorch/ADR.md) | `functorch` owns the legacy top-level Python package for composable function transforms. |
| 24 | [functorch/compile/ADR.md](functorch/compile/ADR.md) | `functorch/compile` is the legacy public compile namespace for functorch's AOTAutograd and graph-partitioning APIs, mapped to book chapters 07 and 13. |
| 25 | [functorch/dim/ADR.md](functorch/dim/ADR.md) | `functorch/dim` implements first-class dimensions for tensors in Python. |
| 26 | [functorch/einops/ADR.md](functorch/einops/ADR.md) | `functorch/einops` provides an einops-style rearrangement frontend on top of first-class dimensions. |
| 27 | [test/ADR.md](test/ADR.md) | `test` owns the top-level Python and C++ validation suites for PyTorch. |
| 28 | [third_party/ADR.md](third_party/ADR.md) | `third_party` owns the vendored dependency source trees that PyTorch builds against. |
| 29 | [tools/ADR.md](tools/ADR.md) | `tools` owns the repository's shared Python build, code generation, maintenance, and developer-automation scripts. |
| 30 | [tools/autograd/ADR.md](tools/autograd/ADR.md) | `tools/autograd` generates PyTorch's differentiable operator layer from `native_functions.yaml` and `tools/autograd/derivatives.yaml`. |
| 31 | [tools/code_analyzer/ADR.md](tools/code_analyzer/ADR.md) | `tools/code_analyzer` generates selective-build operator lists for mobile and custom builds. |
| 32 | [tools/jit/ADR.md](tools/jit/ADR.md) | `tools/jit` contains JIT-oriented code generation helpers for ATen operator integration. |
| 33 | [tools/stats/ADR.md](tools/stats/ADR.md) | `tools/stats` turns CI artifacts and live host telemetry into normalized JSON records for dashboards and backend storage. |
| 34 | [torch/_custom_op/ADR.md](torch/_custom_op/ADR.md) | `torch/_custom_op` owns the deprecated Python custom-operator API that predates `torch.library.custom_op`. |
| 35 | [torch/_decomp/ADR.md](torch/_decomp/ADR.md) | `torch/_decomp` owns the operator decomposition registry: a set of tables mapping `torch.ops.aten.*` operator overloads to Python functions that express each op in terms of simpler primitives. |
| 36 | [torch/_dispatch/ADR.md](torch/_dispatch/ADR.md) | `torch/_dispatch` owns small Python-side helpers for manipulating dispatcher modes and validating functionalization behavior. |
| 37 | [torch/_dynamo/ADR.md](torch/_dynamo/ADR.md) | `torch/_dynamo` is the Python compiler frontend behind `torch.compile()`, as mapped in book chapter 07. |
| 38 | [torch/_export/ADR.md](torch/_export/ADR.md) | `torch/_export` contains private implementation pieces for PyTorch export, serialization, verifier, pass infrastructure, trace wrappers, and legacy TorchScript-to-ExportedProgram conversion, as mapped in book chapter 11. |
| 39 | [torch/_functorch/ADR.md](torch/_functorch/ADR.md) | `torch/_functorch` implements PyTorch's functional transforms and AOTAutograd compiler frontend, as mapped across book chapters 07 and 13. |
| 40 | [torch/_higher_order_ops/ADR.md](torch/_higher_order_ops/ADR.md) | `torch/_higher_order_ops` owns Python higher-order operators that keep control flow, subgraphs, and effectful calls explicit across `torch.compile()` and `torch.export()`. |
| 41 | [torch/_inductor/ADR.md](torch/_inductor/ADR.md) | `torch/_inductor` is PyTorch's default compiler backend for FX graphs, as mapped in book chapter 08. |
| 42 | [torch/_inductor/codegen/ADR.md](torch/_inductor/codegen/ADR.md) | `torch/_inductor/codegen` emits executable source and wrapper code from scheduled Inductor IR, as mapped in book chapter 08. |
| 43 | [torch/_lazy/ADR.md](torch/_lazy/ADR.md) | `torch/_lazy` is the Python control layer for lazy tensor execution backends. |
| 44 | [torch/_library/ADR.md](torch/_library/ADR.md) | `torch/_library` owns the Python infrastructure behind `torch.library`, including custom operator definition, fake implementation registration, autograd bridging, and small side registries that do not live in dispatcher tables. |
| 45 | [torch/_logging/ADR.md](torch/_logging/ADR.md) | `torch/_logging` owns PyTorch's structured and component-scoped logging system. |
| 46 | [torch/_native/ADR.md](torch/_native/ADR.md) | `torch/_native` owns Python-level native DSL operator overrides for ATen. |
| 47 | [torch/_numpy/ADR.md](torch/_numpy/ADR.md) | `torch/_numpy` implements a NumPy-like Python API on top of `torch.Tensor`. |
| 48 | [torch/_prims_common/ADR.md](torch/_prims_common/ADR.md) | `torch/_prims_common` centralizes shared type aliases, metadata rules, stride logic, and wrapper decorators used by PrimTorch and Python reference implementations. |
| 49 | [torch/_prims/ADR.md](torch/_prims/ADR.md) | `torch/_prims` defines PrimTorch primitive operators and the small execution utilities that trace ordinary `torch.*` programs into those primitives. |
| 50 | [torch/_refs/ADR.md](torch/_refs/ADR.md) | `torch/_refs` provides Python reference implementations and decompositions for existing PyTorch operators. |
| 51 | [torch/_strobelight/ADR.md](torch/_strobelight/ADR.md) | `torch/_strobelight` integrates PyTorch with Meta's Strobelight profiling tools. |
| 52 | [torch/_subclasses/ADR.md](torch/_subclasses/ADR.md) | `torch/_subclasses` owns tensor-subclass infrastructure used by PyTorch compilers, most notably FakeTensor and FunctionalTensor. |
| 53 | [torch/accelerator/ADR.md](torch/accelerator/ADR.md) | `torch/accelerator` owns the device-agnostic accelerator facade for Python. |
| 54 | [torch/ADR.md](torch/ADR.md) | `torch` owns the public Python package for PyTorch. |
| 55 | [torch/amp/ADR.md](torch/amp/ADR.md) | `torch/amp` provides automatic mixed precision for training and inference, the performance optimization identified in book Chapter 13 as the `torch.amp` path for FP16/BF16 throughput. |
| 56 | [torch/autograd/ADR.md](torch/autograd/ADR.md) | `torch/autograd` is the Python-facing automatic differentiation API layered over the C++ autograd engine described in book Chapter 05. |
| 57 | [torch/compiler/ADR.md](torch/compiler/ADR.md) | `torch/compiler` owns the public Python compiler facade. |
| 58 | [torch/cpu/ADR.md](torch/cpu/ADR.md) | `torch/cpu` owns the CPU-side compatibility surface used by device-agnostic Python code. |
| 59 | [torch/csrc/api/ADR.md](torch/csrc/api/ADR.md) | `torch/csrc/api` implements the C++ frontend, also known as libtorch. |
| 60 | [torch/csrc/autograd/ADR.md](torch/csrc/autograd/ADR.md) | `torch/csrc/autograd` implements PyTorch's C++ reverse-mode automatic differentiation runtime described in book Chapter 05, "Autograd Engine". |
| 61 | [torch/csrc/distributed/ADR.md](torch/csrc/distributed/ADR.md) | `torch/csrc/distributed` implements PyTorch's C++ distributed runtime: c10d process groups, stores, collective work handles, DistributedDataParallel gradient reduction, RPC agents, RRefs, and distributed autograd. |
| 62 | [torch/csrc/dynamo/ADR.md](torch/csrc/dynamo/ADR.md) | `torch/csrc/dynamo` implements the native CPython frame-evaluation hook, guard fast path, cache-entry storage, frame-local mapping, and compiled-autograd bridge used by TorchDynamo. |
| 63 | [torch/csrc/inductor/ADR.md](torch/csrc/inductor/ADR.md) | `torch/csrc/inductor` provides the C++ runtime support for TorchInductor-generated code and AOTInductor packages. |
| 64 | [torch/csrc/jit/ADR.md](torch/csrc/jit/ADR.md) | `torch/csrc/jit` implements the legacy TorchScript compiler, IR, optimizer, serializer, mobile runtime, and C++ execution API. |
| 65 | [torch/csrc/profiler/ADR.md](torch/csrc/profiler/ADR.md) | `torch/csrc/profiler` implements the C++ profiler backend for PyTorch observability. |
| 66 | [torch/cuda/ADR.md](torch/cuda/ADR.md) | `torch/cuda` is the Python CUDA device-management layer. |
| 67 | [torch/distributed/ADR.md](torch/distributed/ADR.md) | `torch/distributed` is the Python distributed-training layer. |
| 68 | [torch/distributions/ADR.md](torch/distributions/ADR.md) | `torch/distributions` owns PyTorch's probability-distribution framework. |
| 69 | [torch/fft/ADR.md](torch/fft/ADR.md) | `torch/fft` owns the Python namespace for Fourier-transform operators. |
| 70 | [torch/func/ADR.md](torch/func/ADR.md) | `torch/func` owns the public functional-transform API. |
| 71 | [torch/futures/ADR.md](torch/futures/ADR.md) | `torch/futures` owns the Python future abstraction used for asynchronous PyTorch work. |
| 72 | [torch/fx/ADR.md](torch/fx/ADR.md) | `torch/fx` provides PyTorch's Python-level graph IR, symbolic tracer, generated-code module wrapper, and graph interpreter, as mapped in book chapter 09. |
| 73 | [torch/linalg/ADR.md](torch/linalg/ADR.md) | `torch/linalg` owns the Python namespace for linear algebra operators. |
| 74 | [torch/masked/ADR.md](torch/masked/ADR.md) | `torch/masked` owns PyTorch's masked-value API. |
| 75 | [torch/multiprocessing/ADR.md](torch/multiprocessing/ADR.md) | `torch/multiprocessing` owns PyTorch's multiprocessing compatibility layer. |
| 76 | [torch/nativert/ADR.md](torch/nativert/ADR.md) | `torch/nativert` owns the NativeRT runtime for exported models. |
| 77 | [torch/nested/ADR.md](torch/nested/ADR.md) | `torch/nested` owns the public nested-tensor API for ragged and variable-length data. |
| 78 | [torch/nn/ADR.md](torch/nn/ADR.md) | `torch/nn` is the public Python neural-network package described in book Chapter 10, "Neural Network Modules". |
| 79 | [torch/nn/modules/ADR.md](torch/nn/modules/ADR.md) | `torch/nn/modules` implements the stateful neural-network module system and standard layer catalog described in book Chapter 10. |
| 80 | [torch/numa/ADR.md](torch/numa/ADR.md) | `torch/numa` owns NUMA-aware CPU-affinity utilities for multi-device jobs. |
| 81 | [torch/onnx/ADR.md](torch/onnx/ADR.md) | `torch/onnx` exports PyTorch models to the ONNX (Open Neural Network Exchange) interchange format. |
| 82 | [torch/optim/ADR.md](torch/optim/ADR.md) | `torch/optim` implements parameter update algorithms for tensors produced by `torch.nn.Module.parameters()`, as covered by book Chapter 10's training loop architecture. |
| 83 | [torch/package/ADR.md](torch/package/ADR.md) | `torch/package` owns hermetic packaging for Python modules, pickled objects, and resources. |
| 84 | [torch/profiler/ADR.md](torch/profiler/ADR.md) | `torch/profiler` is the Python profiler API over the native Kineto and `RecordFunction` backend. |
| 85 | [torch/quantization/ADR.md](torch/quantization/ADR.md) | `torch/quantization` owns the legacy quantization namespace kept for backward compatibility. |
| 86 | [torch/sparse/ADR.md](torch/sparse/ADR.md) | `torch/sparse` owns the Python sparse-tensor namespace. |
| 87 | [torch/special/ADR.md](torch/special/ADR.md) | `torch/special` owns the Python namespace for special mathematical functions. |
| 88 | [torch/utils/ADR.md](torch/utils/ADR.md) | `torch/utils` collects the framework-level Python utilities that do not belong to a single tensor operator namespace. |
| 89 | [torch/xpu/ADR.md](torch/xpu/ADR.md) | `torch/xpu` owns the Intel XPU backend's Python runtime. |
| 90 | [torchgen/_autoheuristic/ADR.md](torchgen/_autoheuristic/ADR.md) | `torchgen/_autoheuristic` trains and emits learned heuristics for compiler-time choice selection. |
| 91 | [torchgen/ADR.md](torchgen/ADR.md) | `torchgen` is the ATen operator code-generation control plane. |
| 92 | [torchgen/api/ADR.md](torchgen/api/ADR.md) | `torchgen/api` defines the signature and type models that turn a JIT-style `FunctionSchema` into each concrete C++ interface PyTorch emits. |
| 93 | [torchgen/dest/ADR.md](torchgen/dest/ADR.md) | `torchgen/dest` contains the destination renderers that turn parsed operator models and API signatures into concrete generated C++ fragments. |
| 94 | [torchgen/operator_versions/ADR.md](torchgen/operator_versions/ADR.md) | `torchgen/operator_versions` generates the mobile operator-upgrader translation unit that preserves bytecode backward compatibility. |
