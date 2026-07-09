# `torch/csrc`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/csrc` is the C++ binding bridge: it provides pybind11 bindings that expose ATen tensor operations, the C++ autograd engine, JIT IR, distributed collectives, and profiler internals to Python as the `torch._C` extension module. Every Python → C++ call in PyTorch passes through this directory.

## Key Files

| File | Purpose |
|---|---|
| `Module.cpp` | Root pybind11 module initialiser: `PyInit__C` that calls `init*` functions for all submodules; includes ATen, autograd, JIT, distributed, profiler, CUDA init |
| `autograd/engine.h` / `autograd/engine.cpp` | C++ backward engine: `Engine::execute` — BFS/topological backward traversal; `ReadyQueue`, `GraphTask`, `InputBuffer`; `MAX_DEPTH = 60` reentrant guard |
| `autograd/function.h` | `Node` — base class for all autograd graph nodes; `edge_list`, `apply()` contract |
| `jit/` | C++ JIT backend: `torch::jit::Graph`, `IValue`, `CompilationUnit`, `GraphExecutor`, IR passes |
| `DynamicTypes.cpp` / `DynamicTypes.h` | `py::object` ↔ `at::Tensor` conversion; `pyobj_to_tensor`, `tensor_to_pyobj` |
| `Device.cpp` / `Device.h` | Python ↔ `c10::Device` conversion |
| `Dtype.cpp` / `Dtype.h` | Python ↔ `c10::ScalarType` conversion |
| `Exceptions.h` / `Exceptions.cpp` | Exception translation: `c10::Error` → `RuntimeError`, `TypeError`, `IndexError` in Python |
| `Generator.cpp` / `Generator.h` | `THPGenerator` Python wrapper over `at::Generator` |
| `PyInterpreter.cpp` / `PyInterpreter.h` | `PyInterpreter` singleton: manages the single active Python interpreter hook for `TensorImpl.pyobj_slot_` |
| `DataLoader.cpp` | C++ multiprocessing worker for `torch.utils.data.DataLoader`; signal handling, shared-memory FD passing |

## Public Interface

| Symbol | Description |
|---|---|
| `torch._C` | The compiled extension module; loaded by `torch/__init__.py` via `from torch import _C as _C` |
| `torch._C._EngineBase.run_backward` | Calls `Engine::execute`; entry point for `torch.autograd.backward` |
| `torch._C._dynamo.eval_frame.set_eval_frame` | Installs the TorchDynamo frame evaluation callback |
| `torch._C._distributed_c10d.ProcessGroup` | C++ process group object; all collective calls dispatch through it |
| `torch._C._autograd._enable_profiler` | Activates Kineto profiling |
| `torch._C._jit_*` | JIT compilation and serialisation functions |
| `torch._C.TensorBase` | Base Python type for `at::Tensor`; `__add__`, `__matmul__`, `.grad`, `.requires_grad_` all defined here |
| `torch._C._set_grad_enabled(bool)` | Sets the thread-local `c10::GradMode` flag |

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [aten/src/ATen](aten/src/ATen/ADR.md) | depends-on | Wraps `at::Tensor` in `THPVariable`; all tensor operations call ATen through `Module.cpp` |
| [c10/core](c10/core/ADR.md) | depends-on | `c10::Device`, `c10::ScalarType`, `c10::TensorImpl`, `c10::GradMode` conversions |
| [c10/util](c10/util/ADR.md) | depends-on | `c10::Error` caught in `Exceptions.cpp` and translated to Python exception types |
| Python C API / pybind11 | depends-on | `PyObject*`, `py::class_<>`, `py::module_`, `PYBIND11_MODULE` |
| [torch/autograd](torch/autograd/ADR.md) | depended-on-by | `_engine_run_backward` in `graph.py` calls `torch._C._EngineBase.run_backward` |
| [torch/_dynamo](torch/_dynamo/ADR.md) | depended-on-by | Dynamo uses `torch._C._dynamo.eval_frame.set_eval_frame` to install frame hooks |
| [torch/distributed](torch/distributed/ADR.md) | depended-on-by | `torch._C._distributed_c10d` is the C++ collective backend |
| [torch/profiler](torch/profiler/ADR.md) | depended-on-by | `torch._C._autograd._enable_profiler`, `torch._C._profiler` bindings |

## Runtime Behaviour

`PyInit__C` is called once when `import torch._C` executes; it calls `initModule` in `Module.cpp`, which registers all pybind11 classes (`THPVariable`, `THPStorage`, `THPGenerator`, etc.) and calls `init*` functions for each submodule (autograd, JIT, distributed, profiler, CUDA). `THPVariable` (in `autograd/python_variable.cpp`) holds an `at::Tensor` by value; Python-level attribute access (`.grad`, `.shape`, `.device`) calls C++ getters on the embedded `TensorImpl`. `PyInterpreter` provides the Python GIL handoff needed when C++ code (running on non-Python threads, e.g., the autograd engine's worker threads) must call Python callbacks; it uses CPython's `Py_BEGIN_ALLOW_THREADS` / `Py_END_ALLOW_THREADS` pattern.

## Performance Profile

- **Allocation sites**: each new Python tensor creates a `THPVariable` Python object wrapping an `at::Tensor` (which itself wraps a `TensorImpl`). This is two heap allocations per tensor creation from Python. C++-to-C++ tensor creation (inside ATen kernels) does not go through `THPVariable`.
- **Synchronization costs**: the Python GIL is held during all Python → C++ calls. The autograd engine releases the GIL when running C++-only `Node::apply()` implementations via `Py_BEGIN_ALLOW_THREADS`. CUDA operations release the GIL after submitting the kernel.
- **Data movement**: no data movement occurs in `torch/csrc` itself; it is a binding layer. Type conversion in `DynamicTypes.cpp` is zero-copy for tensors.
- **Redundant or repeated work**: pybind11 argument parsing (`py::object` → C++ type) runs on every cross-language call; argument type-checking is the dominant cost for short operations. `Device.cpp` and `Dtype.cpp` cache Python ↔ C++ mappings in static arrays to avoid repeated string comparisons.

## Design Rationale

pybind11 was chosen over hand-written CPython API code because it generates correct reference-counting code and exception handling automatically, reducing the risk of memory leaks and segfaults in the binding layer. `THPVariable` stores `at::Tensor` by value (not pointer) so that Python garbage collection releases the `TensorImpl` reference count correctly when the Python object is deleted. The `PyInterpreter` singleton exists because `TensorImpl` in C++ must hold a weak back-reference to its Python wrapper for the `__torch_dispatch__` protocol, but cannot directly include Python headers — the singleton provides an indirection that satisfies both the no-Python-in-c10 constraint and the back-reference requirement.
