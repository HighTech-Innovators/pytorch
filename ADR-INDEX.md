# ADR Index

48 Architecture Decision Records · generated 2026-07-09

Each `ADR.md` describes the role, key files, public interface, dependencies,
design rationale, constraints, and observability of one directory.

---

| # | Path | Role |
|---:|---|---|
| 1 | [aten/src/ATen/ADR.md](aten/src/ATen/ADR.md) | `aten/src/ATen` (A TENsor library) implements all PyTorch tensor operations: the multi-backend operator dispatcher, native C++ kernel implementations, and the code-generated operator registration glue. |
| 2 | [c10/ADR.md](c10/ADR.md) | `c10` is the minimal-dependency C++ foundation library for PyTorch. |
| 3 | [c10/core/ADR.md](c10/core/ADR.md) | `c10/core` owns PyTorch's foundational C++ abstractions: tensor metadata, memory ownership, device representation, and the dispatch-key type system. |
| 4 | [c10/cuda/ADR.md](c10/cuda/ADR.md) | `c10/cuda` provides the CUDA device abstraction layer: stream management, the CUDA caching allocator, device guards, and CUDA-specific exception handling. |
| 5 | [c10/metal/ADR.md](c10/metal/ADR.md) | `c10/metal` provides shared Metal shader utilities for PyTorch's MPS and sparse MPS kernels. |
| 6 | [c10/mobile/ADR.md](c10/mobile/ADR.md) | `c10/mobile` provides CPU memory allocators optimised for mobile and embedded inference deployments, where the system allocator's aggressive memory reclamation behaviour causes page faults that harm latency. |
| 7 | [c10/util/ADR.md](c10/util/ADR.md) | `c10/util` provides the C++ utility library that `c10/core` and ATen depend on: intrusive reference counting, array views, exception macros, numeric types (`BFloat16`, `Half`), thread-local state helpers, and diagnostic infrastructure. |
| 8 | [c10/xpu/ADR.md](c10/xpu/ADR.md) | `c10/xpu` owns the low-level C++ runtime integration for PyTorch's XPU backend. |
| 9 | [functorch/ADR.md](functorch/ADR.md) | `functorch` is the public backward-compatibility package for PyTorch's functional transforms. |
| 10 | [tools/ADR.md](tools/ADR.md) | `tools` contains repository-level automation for building, release maintenance, source checkout management, generated assets, and developer diagnostics. |
| 11 | [torch/_custom_op/ADR.md](torch/_custom_op/ADR.md) | `torch/_custom_op` contains the deprecated Python custom-operator facade that predates the production `torch.library` API. |
| 12 | [torch/_decomp/ADR.md](torch/_decomp/ADR.md) | `torch/_decomp` owns Python decompositions that rewrite selected `torch.ops.aten` operators into simpler ATen, reference, primitive, or functional RNG operations. |
| 13 | [torch/_dispatch/ADR.md](torch/_dispatch/ADR.md) | `torch/_dispatch` provides Python-level helpers around PyTorch's dispatcher controls. |
| 14 | [torch/_dynamo/ADR.md](torch/_dynamo/ADR.md) | `torch/_dynamo` is TorchDynamo: a Python-level JIT compiler that hooks into CPython's frame evaluation API (PEP 523) to intercept bytecode execution, extract contiguous PyTorch operation sequences as FX graphs, and dispatch them to a configurable compilation backend (default: TorchInductor). |
| 15 | [torch/_export/ADR.md](torch/_export/ADR.md) | `torch/_export` implements the model export pipeline: it captures a `torch.export.ExportedProgram` — a portable, serialisable representation of a PyTorch model with verified input/output signatures — that can be lowered to ONNX, AOT Inductor, or flatbuffer formats for deployment without a Python runtime. |
| 16 | [torch/_functorch/ADR.md](torch/_functorch/ADR.md) | `torch/_functorch` implements PyTorch's composable function transforms: `vmap` (vectorised map), `grad`, `vjp`, `jvp`, `jacrev`, `jacfwd`, and AOT autograd. |
| 17 | [torch/_higher_order_ops/ADR.md](torch/_higher_order_ops/ADR.md) | `torch/_higher_order_ops` defines compiler-aware higher-order operators that carry Python callables, FX graph regions, structured control flow, Triton kernels, and custom execution regions through PyTorch dispatch. |
| 18 | [torch/_inductor/ADR.md](torch/_inductor/ADR.md) | `torch/_inductor` is TorchInductor: the default compilation backend for `torch.compile`. |
| 19 | [torch/_prims/ADR.md](torch/_prims/ADR.md) | `torch/_prims` defines PrimTorch primitive operations, their metadata functions, ATen implementations, and tracing helpers. |
| 20 | [torch/accelerator/ADR.md](torch/accelerator/ADR.md) | `torch/accelerator` provides a device-generic Python facade for the currently compiled accelerator backend. |
| 21 | [torch/ADR.md](torch/ADR.md) | `torch` is the Python-facing API surface for PyTorch. |
| 22 | [torch/amp/ADR.md](torch/amp/ADR.md) | `torch/amp` provides the device-generic automatic mixed precision interface for PyTorch training and inference. |
| 23 | [torch/autograd/ADR.md](torch/autograd/ADR.md) | `torch/autograd` is the Python API surface for automatic differentiation. |
| 24 | [torch/compiler/ADR.md](torch/compiler/ADR.md) | `torch/compiler` provides the public Python namespace for compiler-facing APIs around `torch.compile`, Dynamo tracing controls, guard policies, compile stances, cache artifact hot loading, and nested compile regions. |
| 25 | [torch/cpu/ADR.md](torch/cpu/ADR.md) | `torch/cpu` provides the Python-facing CPU device facade that mirrors selected `torch.cuda` stream, event, synchronization, and device-query APIs for device-agnostic code. |
| 26 | [torch/csrc/ADR.md](torch/csrc/ADR.md) | `torch/csrc` is the C++ binding bridge: it provides pybind11 bindings that expose ATen tensor operations, the C++ autograd engine, JIT IR, distributed collectives, and profiler internals to Python as the `torch._C` extension module. |
| 27 | [torch/cuda/ADR.md](torch/cuda/ADR.md) | `torch/cuda` provides the Python-facing CUDA runtime surface for PyTorch tensors, streams, events, graphs, allocator controls, and device queries. |
| 28 | [torch/distributed/ADR.md](torch/distributed/ADR.md) | `torch/distributed` provides distributed training infrastructure: the collective communication library (c10d), `DistributedDataParallel` (DDP), `FullyShardedDataParallel` (FSDP), `DeviceMesh`, RPC, and the functional collectives API. |
| 29 | [torch/distributions/ADR.md](torch/distributions/ADR.md) | `torch/distributions` implements parameterized probability distributions, constraints, transforms, and KL divergence utilities on top of PyTorch tensors. |
| 30 | [torch/export/ADR.md](torch/export/ADR.md) | `torch/export` captures `torch.nn.Module` programs into a normalized ahead-of-time graph representation with explicit inputs, outputs, state, constants, and dynamic-shape constraints. |
| 31 | [torch/fft/ADR.md](torch/fft/ADR.md) | `torch/fft` owns the Python namespace for PyTorch spectral transforms. |
| 32 | [torch/func/ADR.md](torch/func/ADR.md) | `torch/func` exposes PyTorch's function-transform API as a public namespace over `torch._functorch` and related functorch functionality. |
| 33 | [torch/futures/ADR.md](torch/futures/ADR.md) | `torch/futures` exposes the Python future abstraction used by asynchronous PyTorch APIs. |
| 34 | [torch/fx/ADR.md](torch/fx/ADR.md) | `torch/fx` provides the FX intermediate representation (IR): a Python-level graph of operations over tensors, a tracer that captures module calls symbolically, and an interpreter/transformer framework for graph analysis and rewriting. |
| 35 | [torch/jit/ADR.md](torch/jit/ADR.md) | `torch/jit` provides TorchScript: a statically-typed subset of Python that compiles `nn.Module` subclasses and standalone functions to a portable IR. |
| 36 | [torch/linalg/ADR.md](torch/linalg/ADR.md) | `torch/linalg` defines the Python `torch.linalg` namespace for linear algebra operations. |
| 37 | [torch/multiprocessing/ADR.md](torch/multiprocessing/ADR.md) | `torch/multiprocessing` wraps Python's `multiprocessing` module with tensor-aware serialization. |
| 38 | [torch/nested/ADR.md](torch/nested/ADR.md) | `torch/nested` provides PyTorch's public nested tensor API for representing batches with ragged dimensions. |
| 39 | [torch/nn/ADR.md](torch/nn/ADR.md) | `torch/nn` defines the neural network module system: the `Module` base class, all standard layer implementations, the functional API, and weight-initialization utilities. |
| 40 | [torch/optim/ADR.md](torch/optim/ADR.md) | `torch/optim` implements gradient-based parameter update algorithms — SGD, Adam, AdamW, Adagrad, RMSprop, and others — together with the `Optimizer` base class that owns parameter groups and per-parameter state, and learning-rate schedulers in `lr_scheduler.py`. |
| 41 | [torch/profiler/ADR.md](torch/profiler/ADR.md) | `torch/profiler` provides PyTorch's primary performance observability interface: the `profile` context manager for collecting CPU and GPU operation traces, a `schedule`-based step-triggered recording model, memory profiling, and export to Chrome trace format and TensorBoard. |
| 42 | [torch/quantization/ADR.md](torch/quantization/ADR.md) | `torch/quantization` preserves the legacy quantization import surface while the implementation lives under `torch.ao.quantization`. |
| 43 | [torch/sparse/ADR.md](torch/sparse/ADR.md) | `torch/sparse` provides the Python-facing sparse tensor namespace for sparse matrix operations, sparse reductions, invariant checking, sparse gradcheck support, semi-structured sparsity, and Triton-backed BSR helpers. |
| 44 | [torch/utils/ADR.md](torch/utils/ADR.md) | `torch/utils` provides Python utilities that support extension compilation, tree-structured argument handling, activation checkpointing, private backend registration, environment inspection, FLOP accounting, and smaller helper APIs. |
| 45 | [torchgen/ADR.md](torchgen/ADR.md) | `torchgen` is PyTorch's code generation system. |
| 46 | [torchgen/aoti/ADR.md](torchgen/aoti/ADR.md) | `torchgen/aoti` owns the operator allowlists and ABI metadata that drive AOTInductor C shim generation. |
| 47 | [torchgen/api/ADR.md](torchgen/api/ADR.md) | `torchgen/api` translates parsed operator schemas into typed API models used by PyTorch's generated C++, Python binding, dispatcher, autograd, lazy, functionalization, and ufunc code. |
| 48 | [torchgen/dest/ADR.md](torchgen/dest/ADR.md) | `torchgen/dest` contains destination-specific emitters that turn `torchgen` schema and API models into C++ source fragments. |
