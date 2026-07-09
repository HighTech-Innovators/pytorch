# `torchgen/dest`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torchgen/dest` contains destination-specific emitters that turn `torchgen` schema and API models into C++ source fragments. It owns generated registration code, native-function declarations, ufunc kernels, lazy IR node classes, and lazy backend method bodies.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Re-exports destination entry points such as `RegisterDispatchKey`, `GenLazyIR`, `compute_native_function_declaration`, and ufunc generators |
| `register_dispatch_key.py` | Generates dispatch-key registrations, structured kernel wrappers, out/in-place wrappers, device checks, and per-backend helper code |
| `lazy_ir.py` | Generates lazy IR node class declarations, TS lazy lowering, lazy native function definitions, shape inference definitions, and non-native lazy nodes |
| `ufunc.py` | Generates CPU and CUDA ufunc dispatch stubs, functors, dtype switches, scalar specializations, and TensorIterator kernels |
| `native_functions.py` | Generates `NativeFunctions.h` declarations for structured and unstructured native kernels |
| `lazy_ts_lowering.py` | Builds TorchScript lazy lowering bodies for `LazyIrSchema` nodes |

## Public Interface

| Symbol | Description |
|---|---|
| `gen_registration_headers()` | Emits backend-specific include lists for generated registration files, including XPU `ATen/xpu/EmptyTensor.h` |
| `RegisterDispatchKey` | Main callable dataclass for registration generation across anonymous definitions, namespaced definitions, declarations, and registrations |
| `RegisterDispatchKey.gen_out_inplace_wrapper()` | Emits functional-to-out or functional-to-in-place wrappers using `_copy_from` or `_copy_from_and_resize` |
| `StructuredRegisterDispatchKey` | Emits structured kernel classes with `set_output_strided`, `set_output_raw_strided`, output proxies, and optional device guards |
| `GenLazyIR` / `GenTSLazyIR` | Generate lazy IR node classes with `ClassOpKind`, constructors, `ToString`, reuse checks, and optional lowering functions |
| `GenLazyNativeFuncDefinition` | Emits lazy tensor extraction, fallback, and backend native function definitions |
| `compute_ufunc_cuda()` / `compute_ufunc_cpu()` / `compute_ufunc_cpu_kernel()` | Emit TensorIterator ufunc wrappers, dispatch stubs, dtype switches, and CPU vectorized kernel bodies |
| `compute_native_function_declaration()` | Emits native kernel forward declarations for `NativeFunction` and `NativeFunctionsGroup` entries |

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torchgen](torchgen/ADR.md) | depends-on | Consumes `BackendIndex`, `DispatchKey`, `NativeFunction`, `NativeFunctionsGroup`, `SchemaKind`, `UfuncKey`, `Target`, and generation context decorators |
| [torchgen/api](torchgen/api/ADR.md) | depends-on | Calls `cpp`, `meta`, `structured`, `translate`, `DispatcherSignature`, `NativeSignature`, `kernel_signature`, and ufunc API helpers |
| [aten/src/ATen](aten/src/ATen/ADR.md) | depended-on-by | Emits ATen registration files, native declarations, `REGISTER_DISPATCH`, TensorIterator kernels, and backend wrappers compiled into ATen |
| [torch/csrc](torch/csrc/ADR.md) | depended-on-by | Generated dispatcher registrations and lazy lowering support the C++ extension paths that Python calls through `torch._C` |

## Runtime Behaviour

The generators execute during PyTorch code generation and return lists of C++ strings for selected operators. `RegisterDispatchKey.__call__()` accepts either a `NativeFunctionsGroup` or `NativeFunction`, routes structured groups to `gen_structured()`, and otherwise maps unstructured functions through `gen_unstructured()`. `StructuredRegisterDispatchKey` creates C++ structs that allocate or resize outputs in `set_output_strided` and `set_output_raw_strided`, install `OptionalDeviceGuard` fields for CUDA, MPS, XPU, MTIA, and composite paths, and expose `maybe_get_output()` for structured meta logic. `GenLazyIR.gen()` converts a `LazyIrSchema` into a C++ class with operands, scalar fields, optional-value bitfields, `ToString()`, shape construction, and backend-overridable `Create`, `CanBeReused`, and lowering hooks.

## Performance Profile

`register_dispatch_key.py` does most work with linear passes over native functions and string templates, and it uses `mapMaybe()` to skip functions without backend kernels. Generated structured wrappers favor direct code over runtime reflection: `gen_empty_impl_names()` selects backend-specific empty implementations once at generation time, and `translate()` resolves wrapper arguments before C++ is compiled. `ufunc.py` minimizes generated runtime overhead by emitting `AT_DISPATCH_SWITCH` dtype cases, direct CUDA `gpu_kernel` functor calls, and CPU `cpu_kernel_vec` paths when vector loops exist. Lazy generation stores scalar hashes in `torch::lazy::MHash` and emits `CanBeReused()` comparisons so lazy backends can reuse nodes rather than rebuilding equivalent IR.

## Design Rationale

The directory isolates destination formatting from the semantic API layer so backend registration, ufunc, native declaration, and lazy IR code can evolve independently. Structured kernel generation centralizes output allocation and resizing rules because functional, in-place, and out variants need consistent shape checks and device guards. Ufunc generation keeps CPU and CUDA emitters together because both share `torchgen.api.ufunc` signatures but require different TensorIterator and functor code. Lazy code generation exposes overridable methods in `GenLazyIR` so TS lazy and backend-specific lazy implementations reuse schema handling while customizing lowering and node creation.
