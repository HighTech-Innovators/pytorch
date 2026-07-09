# `torch`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch` is the Python-facing API surface for PyTorch. It orchestrates the import of the compiled C++ extension (`torch._C`), exposes the `Tensor` class, mathematical operations, and device management, and provides the package boundary policy that organises 66+ subpackages under a single coherent namespace.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | 3,598-line package entry point: loads `torch._C` via `_load_global_deps()` and `from torch._C import *`; calls `_C._initExtension()` to register backends; imports `Tensor`, `storage`, `autograd`, `amp`, `serialization`, and all public submodules; defines top-level functions (`set_default_dtype`, `set_deterministic_algorithms`, `set_float32_matmul_precision`, etc.) |
| `_tensor.py` | `Tensor` class — Python wrapper over `c10::TensorImpl`; defines `__torch_function__`, `backward()`, `grad` property, indexing operators, and `__repr__` delegation |
| `storage.py` | `Storage` and `TypedStorage` — Python wrappers over `c10::Storage`; exposes raw buffer access, `data_ptr()`, `nbytes()`, and serialisation hooks |
| `_utils.py` | `_import_dotted_name`, `classproperty`, `_functionalize_sync`; internal utilities used by the package initialisation sequence |
| `functional.py` | Pure-Python functional operator wrappers (`einsum`, `meshgrid`, `broadcast_tensors`, `norm`); re-exports via `from torch.functional import *` at line 2712 |
| `torch_version.py` | Exposes `__version__` sourced from `version.txt` (`2.14.0a0` at analysis time) |
| `_ops.py` | `torch.ops` namespace: the Python-accessible operator registry that wraps `torch._C._VariableFunctions` and custom-op registration |
| `_classes.py` | `torch.classes` namespace: Python-accessible TorchScript custom class registry |
| `serialization.py` | `torch.save` / `torch.load` — pickle-based tensor serialisation with `map_location` support |
| `random.py` | `torch.manual_seed`, `torch.seed`, `torch.get_rng_state`, `torch.set_rng_state` — RNG state management |
| `amp/__init__.py` | `autocast`, `GradScaler` — automatic mixed-precision context managers, imported at package level |

## Public Interface

| Symbol | Description |
|---|---|
| `torch.Tensor` | Primary user-facing tensor class; wraps `c10::TensorImpl` |
| `torch.tensor()` / `torch.empty()` / `torch.zeros()` / `torch.ones()` / `torch.randn()` | Tensor factory functions, routed through `torch._C._VariableFunctions` |
| `torch.save()` / `torch.load()` | Pickle-based serialisation for tensors and arbitrary objects |
| `torch.set_default_dtype()` / `torch.set_default_device()` | Global state setters backed by `torch._C._set_default_dtype` / `c10::AutogradState` |
| `torch.set_deterministic_algorithms()` | Determinism mode toggle backed by `torch._C._set_deterministic_algorithms` |
| `torch.set_float32_matmul_precision()` | TF32/BF16 matmul precision selector backed by `torch._C._set_float32_matmul_precision` |
| `torch.compile()` | Public entry point for `torch/_dynamo`; dispatches to `_dynamo.optimize()` |
| `torch.autocast` / `torch.GradScaler` | AMP context managers from `torch.amp` |
| `torch.ops` | Operator registry namespace — access to all registered ATen and custom operators |
| `torch.classes` | TorchScript custom class registry namespace |
| `torch.vmap()` | Vectorised map from `torch.func`, re-exported at line 3379 |
| `torch.cond()` / `torch.while_loop()` | Higher-order control-flow operators from `torch._higher_order_ops` |

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [c10/core](c10/core/ADR.md) | depends-on | `Tensor` wraps `c10::TensorImpl`; `set_default_dtype` / `set_deterministic_algorithms` call into `c10` C++ APIs |
| [torch/csrc](torch/csrc/ADR.md) | depends-on | `torch._C` compiled extension exposes all C++ operations to Python; loaded via `from torch import _C` at line 1502 and `from torch._C import *` at line 541 |
| [aten/src/ATen](aten/src/ATen/ADR.md) | depends-on | All tensor factory functions and math operations are implemented in ATen and exposed through `torch._C._VariableFunctions` |
| [torch/autograd](torch/autograd/ADR.md) | depends-on | `torch.__init__` imports `backward`, `grad`, `no_grad`, `Function` from `torch.autograd` at line 2748 |
| [torch/nn](torch/nn/ADR.md) | depended-on-by | `torch.nn` subpackage imports `torch.Tensor`, `torch.autograd`, and operator functions as its runtime foundation |
| [torch/_dynamo](torch/_dynamo/ADR.md) | depended-on-by | `torch.compile()` is a shim over `torch._dynamo.optimize()`; `_dynamo` imports `torch` at the package level |
| [torch/fx](torch/fx/ADR.md) | depended-on-by | FX tracing calls `torch.Tensor` operations through the proxy mechanism; imports `torch` as the traced namespace |
| [torch/_inductor](torch/_inductor/ADR.md) | depended-on-by | Inductor code generation references `torch` operator schemas and `torch.Tensor` methods for lowering |
| [torch/distributed](torch/distributed/ADR.md) | depended-on-by | Distributed collectives wrap `torch.Tensor` operations and depend on `torch` device and dtype APIs |
| [torch/jit](torch/jit/ADR.md) | depended-on-by | TorchScript scripting and tracing entry points (`torch.jit.script`, `torch.jit.trace`) are exposed through the `torch` namespace |
| [torch/optim](torch/optim/ADR.md) | depended-on-by | Optimiser parameter-update loops call `torch.Tensor` in-place operations |
| [torch/profiler](torch/profiler/ADR.md) | depended-on-by | Profiler instruments `torch` API calls and reads `torch.Tensor` metadata for reporting |

## Runtime Behaviour

`import torch` executes `torch/__init__.py` sequentially. Early in the sequence (lines 469–558), `_load_global_deps()` opens `libtorch_global_deps.so` with `RTLD_GLOBAL` when `USE_RTLD_GLOBAL_WITH_LIBTORCH` is set, ensuring CUDA and MKL symbols are globally visible before `torch._C` is loaded. The compiled extension is then imported via `from torch import _C` (line 1502) and `from torch._C import *` (line 541), which runs `PyInit__C` in `torch/csrc/Module.cpp` and registers all pybind11 bindings. `_C._initExtension(_manager_path())` (line 2646) performs late-binding initialisation: it registers ATen operator implementations, sets up the default memory format constants (`contiguous_format`, `preserve_format`, `channels_last`), and initialises backend hooks.

`torch` holds global mutable state via `torch._C`: default dtype, default device, determinism mode, float32 matmul precision, and warn-always flag. These are set via C++ calls (e.g., `_C._set_default_tensor_type`, `_C._set_deterministic_algorithms`) and are process-wide and thread-unsafe to mutate concurrently. Thread-local state (grad mode, inference mode) is owned by `c10::AutogradState` and accessed via `torch.autograd.grad_mode` context managers.

## Performance Profile

- **Allocation sites:** `torch.Tensor` construction always allocates a `c10::TensorImpl` (one intrusive-pointer-managed heap object) plus the data buffer from the registered device allocator. The factory functions (`torch.empty`, `torch.zeros`, `torch.randn`) in `torch._C._VariableFunctions` go through ATen's dispatcher before reaching the allocator, adding dispatcher overhead on every factory call.
- **Synchronization costs:** `torch` itself does not hold locks; synchronisation is delegated to `c10` allocators and the ATen dispatcher. The GIL is released by `torch._C` for compute-bound C++ operations (operator kernels), but is re-acquired for every Python-level callback, including `__torch_function__` dispatch and custom autograd `Function.apply`.
- **Data movement:** `torch.load()` in `serialization.py` deserialises tensors via `pickle` and reconstructs storage objects; if `map_location` is specified, data is moved across devices using `torch.Tensor.to()`. No implicit data movement occurs at the `torch` package level beyond what user code requests.
- **Redundant work:** The `torch/__init__.py` import chain is sequential and performs no lazy loading of subpackages at the top level; all imports listed in lines 2388–2875 execute on `import torch`. On large machines with many installed backends this increases cold-start import time. Subpackages that declare their own `__init__.py` lazy-import patterns (e.g., `torch._dynamo`) defer expensive initialisation until first use.

## Design Rationale

`torch/__init__.py` acts as the single integration point for a multi-layer system: it must load a compiled C++ extension, wire up Python types to C++ objects, and present a flat, ergonomic namespace to users. The separation between `torch` (Python API), `torch/csrc` (C++ bindings), `aten/src/ATen` (operator implementations), and `c10` (core types) reflects a deliberate layering policy visible in the `c10/CMakeLists.txt` comment about dependency cost. The `torch.ops` and `torch.classes` namespaces are late-bound proxy objects that forward attribute access into the operator and class registries, allowing custom operators to integrate without modifying `__init__.py`.
