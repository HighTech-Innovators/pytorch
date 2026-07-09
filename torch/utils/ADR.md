# `torch/utils`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/utils` provides Python utilities that support extension compilation, tree-structured argument handling, activation checkpointing, private backend registration, environment inspection, FLOP accounting, and smaller helper APIs. The package is intentionally broad, but the entry file `__init__.py` keeps the public surface focused on utilities such as `set_module`, `swap_tensors`, `ThroughputBenchmark`, `rename_privateuse1_backend`, and `generate_methods_for_privateuse1_backend`.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Imports common utility subpackages and defines `set_module()`, `swap_tensors()`, and `cmake_prefix_path` |
| `cpp_extension.py` | Builds C++, CUDA, and SYCL extensions through `BuildExtension`, `CppExtension`, `CUDAExtension`, `load()`, and `load_inline()` |
| `_pytree.py` | Implements Python pytree registration, flattening, unflattening, mapping, key paths, and serialization |
| `checkpoint.py` | Implements activation checkpointing, RNG-state preservation, selective checkpoint contexts, and reentrant/non-reentrant paths |
| `flop_counter.py` | Registers FLOP formulas and implements `FlopCounterMode` on top of `TorchDispatchMode` |
| `backend_registration.py` | Renames the PrivateUse1 backend and generates tensor, module, and storage methods for a custom backend |
| `collect_env.py` | Collects diagnostic information about the Python, PyTorch, compiler, CUDA, and system environment |

## Public Interface

The package-level interface includes `set_module(obj, mod)`, `swap_tensors(t1, t2)`, `cmake_prefix_path`, `ThroughputBenchmark`, `get_cpp_backtrace`, `rename_privateuse1_backend()`, and `generate_methods_for_privateuse1_backend()`. `cpp_extension.py` exports `BuildExtension`, `CppExtension`, `CUDAExtension`, `SyclExtension`, `include_paths()`, `library_paths()`, `load()`, `load_inline()`, `get_default_build_root()`, and compiler/ninja checks. `_pytree.py` exports `TreeSpec`, `LeafSpec`, `register_pytree_node()`, `tree_flatten()`, `tree_unflatten()`, `tree_map()`, `tree_leaves()`, `treespec_dumps()`, and `treespec_loads()`. `checkpoint.py` exports `checkpoint()`, `checkpoint_sequential()`, `CheckpointFunction`, `CheckpointError`, `DefaultDeviceType`, `CheckpointPolicy`, `SelectiveCheckpointContext`, and `create_selective_checkpoint_contexts()`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch](torch/ADR.md) | depends-on | Utilities call `torch._C`, tensor methods, accelerator modules, dispatcher state, and top-level torch operators |
| [torch/autograd](torch/autograd/ADR.md) | depends-on | `checkpoint.py` subclasses `torch.autograd.Function`, saves device RNG state, and interacts with gradient edges |
| [torch/fx](torch/fx/ADR.md) | depended-on-by | FX, export, and compiler paths use `_pytree.TreeSpec`, key paths, and pytree flatten/unflatten helpers |
| [c10/core](c10/core/ADR.md) | depends-on | `cpp_extension.py` links extensions against `c10`, `torch`, `torch_cpu`, and optionally `torch_python` |
| [torch/_dynamo](torch/_dynamo/ADR.md) | depended-on-by | Dynamo and compiler flows use pytree utilities, dispatch modes, checkpoint integration, and FLOP accounting helpers |

## Runtime Behaviour

`__init__.py` imports selected subpackages, exposes `cmake_prefix_path`, and implements `swap_tensors()` by checking weak references, swapping Python slots/dictionaries/classes, and finally calling `torch._C._swap_tensor_impl()`. `cpp_extension.py` lazily initializes `BuildExtension`, discovers CUDA/ROCm/SYCL paths, validates compiler versions, writes Ninja build files, and loads compiled extension modules with cache versioning. `_pytree.py` maintains registries protected by `_NODE_REGISTRY_LOCK` and recursively flattens containers into leaves plus `TreeSpec` metadata. `checkpoint.py` wraps forward functions so activations are recomputed during backward, preserves CPU and device RNG states with `get_device_states()` and `set_device_states()`, and supports selective caching through dispatch modes.

## Performance Profile

`_pytree.py` documents that the Python implementation has meaningful overhead, so hot compiler paths reduce repeated flattening and optionally integrate with `torch.utils._cxx_pytree` when optree support is available. `checkpoint.py` deliberately trades additional forward recomputation for lower activation memory, and its RNG-state preservation adds device-state reads and writes when checkpointed inputs include accelerator tensors. `cpp_extension.py` pays high one-time costs in compiler discovery, Ninja generation, and shared-library loading, then reuses build directories and versioning to avoid unnecessary rebuilds. `FlopCounterMode` runs under `TorchDispatchMode`, so it adds per-dispatch Python accounting overhead while avoiding real kernel instrumentation.

## Design Rationale

`torch/utils` groups functionality that is reusable across PyTorch subsystems but does not belong in the core tensor API. The design keeps high-level Python services such as pytrees, extension building, and checkpoint policies separate from ATen kernels and `torch/csrc` bindings while still allowing them to call into `torch._C` when identity swaps, backend renames, or C++ backtraces require it. This layout gives downstream users stable utility imports without forcing every utility into the root `torch` namespace.
