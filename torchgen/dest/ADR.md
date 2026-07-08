# `torchgen/dest`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torchgen/dest` contains the destination renderers that turn parsed operator models and API signatures into concrete generated C++ fragments. It owns the code for native kernel declarations, per-dispatch-key wrapper kernels, dispatcher `TORCH_LIBRARY_IMPL` registrations, structured kernel classes, CPU/CUDA ufunc loops, and lazy tensor IR nodes. If `torchgen/api` defines what a call should look like, this directory decides where that call appears and what boilerplate surrounds it.

## Key Files

| File | Purpose |
|---|---|
| `register_dispatch_key.py` | Generates `Register{DispatchKey}.cpp` fragments, wrapper kernels, device checks, device guards, structured classes, out/in-place helpers, and `m.impl()` registrations |
| `native_functions.py` | Generates `NativeFunctions.h` declarations for unstructured and structured kernels implemented under `aten/src/ATen/native` |
| `ufunc.py` | Generates CPU dispatch stubs, CUDA functors, dtype branches, scalar specializations, and structured ufunc kernel bodies |
| `lazy_ir.py` | Generates lazy tensor IR node classes, eager fallback calls, meta-tensor shape inference plumbing, and lazy native function bodies |
| `lazy_ts_lowering.py` | Generates TorchScript lazy lowering bodies for lazy IR nodes |
| `native_functions.py` | Computes structured kernel declarations that inherit from `at::meta::structured_*` classes |

## Public Interface

The main callable classes are `RegisterDispatchKey`, `StructuredRegisterDispatchKey`, `GenLazyIR`, `GenTSLazyIR`, `GenLazyNativeFuncDefinition`, and `GenLazyShapeInferenceDefinition`. Functional entry points include `compute_native_function_declaration()`, `gen_registration_headers()`, `gen_registration_helpers()`, `compute_ufunc_cuda()`, `compute_ufunc_cpu()`, and `generate_non_native_lazy_ir_nodes()`. `torchgen/gen.py` instantiates these objects with `BackendIndex`, `Target`, `SelectiveBuilder`, ROCm, SymInt, and static-dispatch options, then feeds them `NativeFunction` or `NativeFunctionsGroup` objects.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torchgen](torchgen/ADR.md) | depends-on | Consumes `NativeFunction`, `NativeFunctionsGroup`, `BackendIndex`, `DispatchKey`, `SchemaKind`, `Target`, and selector state |
| [torchgen/api](torchgen/api/ADR.md) | depends-on | Uses dispatcher, native, C++, structured, meta, ufunc, lazy, and translation APIs to compute signatures and call expressions |
| [aten/src/ATen/core](aten/src/ATen/core/ADR.md) | generates-for | Emits dispatcher registration code and `at::_ops` call/redispatch wrappers used by the core dispatch system |
| [aten/src/ATen/native](aten/src/ATen/native/ADR.md) | generates-for | Emits native declarations and wrapper code that call `at::native` CPU, CUDA, Meta, Composite, and structured kernels |
| [torch/csrc/jit](torch/csrc/jit/ADR.md) | generates-for | Lazy and TorchScript lowering destinations produce code consumed by JIT/lazy integrations |

## Runtime Behaviour

At generation time, `RegisterDispatchKey.__call__()` selects structured or unstructured generation, skips unselected operators through `SelectiveBuilder`, emits declarations, definitions, anonymous wrapper kernels, or registrations based on `Target`, and translates wrapper arguments into native kernel arguments. For backend wrappers, it inserts device checks, optional device guards, backend-specific `empty` helpers, meta/in-place/out resizing helpers, and finally calls `m.impl("aten::op", TORCH_FN(wrapper))`. Book chapter 03's dispatch-table registration path depends on these generated `m.impl()` calls to populate each operator's `OperatorEntry` table.

## Performance Profile

This directory shifts dispatch and structured-kernel boilerplate out of handwritten kernels and into generated C++ so runtime calls use direct unboxed wrappers with predictable code. Device checks and `OptionalDeviceGuard` creation appear only for backends whose `BackendIndex` requires them, and selector checks omit registrations for selective builds. Ufunc generation creates dtype-specific CPU and CUDA bodies plus scalar specializations, increasing generated source size but avoiding generic boxed argument handling in pointwise kernels. Structured generation centralizes output allocation and resizing, which keeps kernel implementations focused on computation while preserving out/in-place correctness.

## Design Rationale

Destination renderers exist because each generated file has a different contract even when it starts from the same schema. Dispatcher registrations need `m.impl()` payloads, native headers need declarations, structured kernels need meta inheritance and `set_output` mechanics, ufuncs need dtype loops, and lazy backends need IR node construction. Keeping this code in `torchgen/dest` prevents policy from leaking into `model.py` or `api` modules. The design matches book chapter 03: generated destination code fills the dispatcher tables, while generated wrappers implement redispatch-friendly middleware boundaries without forcing native kernels to know the dispatcher registration API.
