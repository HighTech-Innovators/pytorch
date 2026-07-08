# `torch/csrc/inductor`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/csrc/inductor` provides the C++ runtime support for TorchInductor-generated code and AOTInductor packages. Book Chapter 08 (`book/08-torchinductor.md`) describes Python-side graph lowering, scheduling, fusion, and source generation; this directory supplies the native operators, stable C ABI shims, model containers, runners, and device wrappers that generated C++/CUDA/MPS/XPU code calls at runtime. It bridges compiled model shared libraries back to ATen tensors and PyTorch runtime services without re-entering the Python compiler stack.

## Key Files

| File | Purpose |
|---|---|
| `inductor_ops.cpp` | Registers Inductor-specific ATen operators such as `_alloc_from_pool`, `_reinterpret_tensor`, `_mm_plus_mm`, and `accumulate_grad_` |
| `inductor_ops.h` | Declares native Inductor helper operators used by generated or compiled graphs |
| `aoti_runtime/model_container.h` | Header-only AOTInductor model pool, constant buffer state machine, constant folding, and thread-safe run coordination |
| `aoti_runtime/interface.h` | Stable C ABI types and function declarations shared between generated model libraries and runners |
| `aoti_runner/model_container_runner.h` | C++ runner that loads a model shared object, resolves C ABI function pointers, converts tensor handles, and runs models |
| `aoti_runner/model_container_runner_cuda.h` | CUDA-specific AOTI runner specialization with device stream handling |
| `aoti_torch/c/shim.h` | C shim surface for generated code to call PyTorch tensor and operator services |
| `aoti_torch/shim_cpu.cpp` | CPU shim implementations for generated calls into ATen native and oneDNN paths |
| `aoti_torch/tensor_converter.h` | Conversion helpers between `AtenTensorHandle` and `at::Tensor` |
| `cpp_wrapper/lazy_triton_compile.h` | Runtime support for generated wrappers that lazily compile Triton kernels |

## Public Interface

The public interface has two layers. The dispatcher layer registers `inductor` and `inductor_prims` operators through `TORCH_LIBRARY_FRAGMENT`, so generated graphs can call helpers with normal ATen dispatch semantics. The AOTI layer exposes C ABI functions and opaque handles such as `AOTInductorModelContainerHandle`, `AtenTensorHandle`, and `AOTIProxyExecutorHandle`; `AOTIModelContainerRunner` loads a compiled model library, calls create/run/delete symbols, updates constants, and returns `std::vector<at::Tensor>` results.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [c10/core](c10/core/ADR.md) | depends-on | Uses tensor metadata, storage, scalar types, devices, dispatch keys, and error handling in native helpers and shims |
| [aten/src/ATen/core](aten/src/ATen/core/ADR.md) | depends-on | Uses ATen tensor handles, `at::Tensor`, dispatcher registration, and generated operator calls |
| [aten/src/ATen/native](aten/src/ATen/native/ADR.md) | depends-on | CPU shim calls native oneDNN, quantized, and other native kernels directly where generated code needs stable wrappers |
| [torch/csrc/autograd](torch/csrc/autograd/ADR.md) | depends-on | `accumulate_grad_` uses autograd accumulation semantics for compiled autograd graphs |
| [torch/_inductor](torch/_inductor/ADR.md) | depended-on-by | Python Inductor code generation emits calls into these runtime headers, C shims, and model runners |

## Runtime Behaviour

Generated Inductor code can allocate views from pooled storage through `_alloc_from_pool`, reinterpret tensor sizes and strides without view tracking through `_reinterpret_tensor`, and use `accumulate_grad_` to model compiled autograd side effects as fresh outputs. AOTI runners load a generated shared object, resolve the C ABI entry points, convert input `at::Tensor` values into stolen `AtenTensorHandle` arrays, call the model container on the requested stream, and convert returned handles back to tensors. `AOTInductorModelContainer` owns a pool of model instances, loads constants once, lazily runs constant folding when the first execution sees initialized constants, and supports inactive-buffer updates followed by swaps. Device-specific runner and shim files route the same model-container protocol to CPU, CUDA, MPS, and XPU implementations.

## Performance Profile

The Python compiler performs fusion and scheduling before this runtime executes, so the C++ code focuses on low-overhead tensor handle conversion, shared-library calls, stream propagation, and constant reuse. `aoti_runtime` headers deliberately avoid ATen and c10 includes except for the stable C ABI, which keeps generated `model.so` binaries smaller and isolates them from C++ ABI churn. The model container preallocates multiple model instances and tracks available/pending models under locks, allowing concurrent inference while protecting constant-buffer updates. Pool allocation and reinterpret helpers avoid extra tensor data copies by constructing new `TensorImpl` metadata over existing storage.

## Design Rationale

Inductor needs a native runtime because generated kernels and AOT packages must run after Python graph lowering has finished. The stable C ABI lets generated model libraries call PyTorch services across binary boundaries without depending on C++ template layout. Model containers centralize constant management, stream passing, and proxy execution so codegen templates stay small and device-specific behavior remains behind runner and shim files. Dispatcher-registered helper ops keep eager, compiled, and functionalization paths consistent for operations that the generated graph cannot represent as plain ATen calls.
