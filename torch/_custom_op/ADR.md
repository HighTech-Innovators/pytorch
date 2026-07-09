# `torch/_custom_op`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/_custom_op` contains the deprecated Python custom-operator facade that predates the production `torch.library` API. It still implements registration, schema validation, dispatcher integration, fake/meta implementations, and explicit autograd formulas for existing internal and compatibility users.

## Key Files

| File | Purpose |
|---|---|
| `torch/_custom_op/__init__.py` | Empty package marker; the implementation lives in `impl.py` and `autograd.py` |
| `torch/_custom_op/impl.py` | Defines `custom_op`, `CustomOp`, dispatcher registration, schema validation, device implementation registration, and error callbacks |
| `torch/_custom_op/autograd.py` | Builds autograd indirection kernels and generated `torch.autograd.Function` classes for `impl_backward` and `impl_save_for_backward` |

## Public Interface

| Symbol | Description |
|---|---|
| `custom_op(qualname, manual_schema=None)` | Deprecated decorator that validates a Python function, infers or parses a schema, defines a `torch.library.Library` fragment, and returns a `CustomOp` |
| `CustomOp` | Callable operator wrapper storing `_schema`, `_cpp_ns`, `_ophandle`, `_qualname`, registered implementations, and autograd state |
| `CustomOp.impl(device_types)` | Registers CPU or CUDA kernels through `torch.library.impl` after duplicate-dispatch checks |
| `CustomOp.impl_factory()` | Registers a factory implementation on `BackendSelect` |
| `CustomOp.impl_abstract()` | Registers a Meta implementation with fake context error handling |
| `CustomOp.impl_save_for_backward()` / `CustomOp.impl_backward()` | Register the two functions required to construct the generated autograd kernel |
| `get_op`, `_find_custom_op`, `get_abstract_impl` | Lookup helpers for dispatcher operators and the `global_registry` |

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch](torch/ADR.md) | depends-on | Uses `torch.library`, `torch.ops`, tensor predicates, `torch._library.fake_impl`, and Python dispatcher-facing APIs |
| [torch/csrc](torch/csrc/ADR.md) | depends-on | Calls `_dispatch_find_schema_or_throw`, `_dispatch_call_boxed`, `_dispatch_has_kernel_for_dispatch_key`, and `_dispatch_set_report_error_callback` |
| [torch/autograd](torch/autograd/ADR.md) | depends-on | Generates subclasses of `torch.autograd.Function` and uses `_AutoDispatchBelowAutograd` to avoid recursive autograd dispatch |
| [torchgen](torchgen/ADR.md) | depends-on | Parses and validates `FunctionSchema`, `OperatorName`, `SchemaKind`, `BaseType`, and `ListType` from `torchgen.model` |

## Runtime Behaviour

`custom_op` emits a deprecation warning, parses `ns::name`, rejects reserved namespaces, infers a functional schema with `infer_schema` unless a manual schema is supplied, defines a `torch.library.Library(ns, "FRAGMENT")`, and creates a `CustomOp` from the dispatcher handle. A `CustomOp` call bypasses `torch.ops.*` packet lookup and invokes `_C._dispatch_call_boxed` on the stored `_DispatchOperatorHandle`. Registering `impl_save_for_backward` and `impl_backward` installs an autograd indirection kernel, constructs a generated `torch.autograd.Function`, saves pytree state, validates gradient dictionaries against schema tensor-like arguments, and returns flattened gradients to autograd.

## Performance Profile

Registration performs schema parsing, Python signature inspection, dispatch-key duplicate checks, and error-callback installation once per custom operator. Runtime `CustomOp.__call__` uses boxed dispatcher invocation, which favors compatibility with dynamic schemas over the unboxed generated paths used by native ATen operators. The autograd path flattens and unflattens pytrees, caches namedtuple argument classes with `functools.lru_cache`, and stores non-tensor context separately from tensors saved through `ctx.save_for_backward`.

## Design Rationale

The package remains as a compatibility layer while steering new users to `torch.library.custom_op` through explicit deprecation warnings. It layers Python validation and helpful dispatcher error messages on top of the C++ dispatcher so invalid namespaces, non-functional schemas, missing Meta kernels, and missing CPU/CUDA kernels fail with operator-specific diagnostics. The separate autograd indirection exists because the dispatcher registration itself cannot be swapped after creation, while users can register `save_for_backward` and `backward` later.
