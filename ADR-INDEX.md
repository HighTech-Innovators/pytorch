# ADR Index

27 Architecture Decision Records · generated 2026-07-10

Each `ADR.md` describes the role, key files, public interface, dependencies,
design rationale, constraints, and observability of one directory.

---

| # | Path | Role |
|---:|---|---|
| 1 | [aten/ADR.md](aten/ADR.md) | `aten` is the ATen tensor library root. |
| 2 | [aten/src/ATen/ADR.md](aten/src/ATen/ADR.md) | `aten/src/ATen` ("A TENsor library") is PyTorch's C++ operator library. |
| 3 | [aten/src/ATen/native/ADR.md](aten/src/ATen/native/ADR.md) | `aten/src/ATen/native` holds the concrete implementations of PyTorch's ~2000 operators — the actual kernel bodies for arithmetic, reductions, matrix ops, convolutions, normalization, activations, indexing, and shape operations. |
| 4 | [aten/src/ATen/native/cpu/ADR.md](aten/src/ATen/native/cpu/ADR.md) | `aten/src/ATen/native/cpu` holds the vectorized CPU kernel bodies for element-wise and reduction operators. |
| 5 | [c10/ADR.md](c10/ADR.md) | `c10` ("Caffe2 + ATen") is PyTorch's foundational C++ library. |
| 6 | [c10/core/ADR.md](c10/core/ADR.md) | `c10/core` owns the core tensor runtime state (`TensorImpl`), the storage abstraction (`Storage`/`StorageImpl`), the memory-allocation interface (`Allocator`/`DataPtr`), the device model (`Device`/`DeviceType`), the scalar-type registry (`ScalarType`), and the dispatch-key registry (`DispatchKey`/`DispatchKeySet`) that selects which kernel runs for every operator. |
| 7 | [c10/cuda/ADR.md](c10/cuda/ADR.md) | `c10/cuda` provides the CUDA-specific counterparts of the core `c10` abstractions: the CUDA caching allocator, CUDA stream and event wrappers, device guards, and CUDA error handling. |
| 8 | [c10/util/ADR.md](c10/util/ADR.md) | `c10/util` provides the low-level C++ utility layer that the rest of PyTorch builds on: intrusive reference counting, the exception/assertion system, thread-local storage, small-buffer-optimized containers, half-precision types, logging, and a type-registry. |
| 9 | [caffe2/ADR.md](caffe2/ADR.md) | `caffe2` is the legacy Caffe2 layer retained within the PyTorch repository. |
| 10 | [functorch/ADR.md](functorch/ADR.md) | `functorch` is a compatibility shim that re-exports the function-transform APIs (`vmap`, `grad`, `jvp`, `vjp`, `jacfwd`, `jacrev`) from their canonical implementation in `torch._functorch`. |
| 11 | [tools/ADR.md](tools/ADR.md) | `tools` is the build-time tooling root. |
| 12 | [tools/autograd/ADR.md](tools/autograd/ADR.md) | `tools/autograd` is a build-time code-generation subsystem. |
| 13 | [torch/_dynamo/ADR.md](torch/_dynamo/ADR.md) | `torch/_dynamo` is TorchDynamo: a Python-level JIT that captures PyTorch programs into FX graphs by symbolically executing Python bytecode via CPython's PEP 523 frame evaluation hook. |
| 14 | [torch/_inductor/ADR.md](torch/_inductor/ADR.md) | `torch/_inductor` is TorchInductor: the default backend for `torch.compile`. |
| 15 | [torch/ADR.md](torch/ADR.md) | `torch` is the top-level Python package and public API surface. |
| 16 | [torch/autograd/ADR.md](torch/autograd/ADR.md) | `torch/autograd` is the Python surface of automatic differentiation. |
| 17 | [torch/csrc/ADR.md](torch/csrc/ADR.md) | `torch/csrc` is the Python↔C++ bridge — the C++ source of the `torch._C` extension module (`libtorch_python`). |
| 18 | [torch/csrc/api/ADR.md](torch/csrc/api/ADR.md) | `torch/csrc/api` is the Python-free C++ frontend (LibTorch). |
| 19 | [torch/csrc/autograd/ADR.md](torch/csrc/autograd/ADR.md) | `torch/csrc/autograd` implements PyTorch's reverse-mode automatic-differentiation engine in C++. |
| 20 | [torch/csrc/jit/ADR.md](torch/csrc/jit/ADR.md) | `torch/csrc/jit` is the C++ implementation of TorchScript and the legacy JIT compiler. |
| 21 | [torch/distributed/ADR.md](torch/distributed/ADR.md) | `torch/distributed` is the Python surface for multi-process collective communication. |
| 22 | [torch/fx/ADR.md](torch/fx/ADR.md) | `torch/fx` is PyTorch's graph intermediate representation and transformation framework. |
| 23 | [torch/jit/ADR.md](torch/jit/ADR.md) | `torch/jit` is the Python surface of TorchScript: the system that compiles Python/PyTorch code into a portable, serializable IR executable without the Python runtime. |
| 24 | [torch/nn/ADR.md](torch/nn/ADR.md) | `torch/nn` is PyTorch's neural-network module system. |
| 25 | [torch/nn/parallel/ADR.md](torch/nn/parallel/ADR.md) | `torch/nn/parallel` provides data-parallel training wrappers: `DistributedDataParallel` (DDP), which replicates a module across processes and synchronizes gradients via collective all-reduce, and the legacy single-process, multi-GPU `DataParallel`. |
| 26 | [torch/profiler/ADR.md](torch/profiler/ADR.md) | `torch/profiler` is the Python profiling API. |
| 27 | [torchgen/ADR.md](torchgen/ADR.md) | `torchgen` is the YAML-driven code-generation engine for PyTorch operators. |
