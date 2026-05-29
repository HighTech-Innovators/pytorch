# Architecture Decision Record: aten/src/ATen/core

## Architectural Role

`aten/src/ATen/core` defines the operator schemas, dispatch infrastructure foundation, and core abstractions for PyTorch's tensor operations. It provides the `FunctionSchema` definition system (which describes operator signatures), the dispatcher routing foundation, and the basic operator registration patterns. This layer bridges the c10 tensor abstractions with the actual kernel implementations, making it Runtime Critical for operator execution.

## Responsibilities

This module is responsible for:
- Defining `FunctionSchema` (name, overload, arguments, return types, tags) used to describe all PyTorch operators
- Implementing the dispatcher registration interfaces (`RegistrationHandleRAII`, schema registration)
- Defining the operator lookup table and operator entry management (but not the dispatch key routing logic itself—that's in aten/src/ATen/core/dispatch)
- Providing the `TORCH_LIBRARY` and `TORCH_LIBRARY_IMPL` macro infrastructure for static operator registration
- Defining type wrapping and boxing utilities that enable the dispatcher to invoke kernels with type-erased arguments
- Supporting operator tagging (e.g., `inplace_only`, `structured`) for special handling
- Implementing the operator cache for fast dispatch lookups

The module does **not** implement kernel code or dispatch key routing; those are the responsibilities of aten/src/ATen/native and aten/src/ATen/core/dispatch respectively.

## Dependencies

**Inbound** (what depends on aten/src/ATen/core):
- `aten/src/ATen/native` for concrete kernel implementations registered against schemas
- `aten/src/ATen/core/dispatch` for dispatch key routing
- `torch/csrc/autograd` for autograd operation registration
- All frontend bindings (Python layer, C++ bindings)

**Outbound** (what aten/src/ATen/core depends on):
- `c10/core` for TensorImpl, DispatchKeySet, Device
- `c10/util` for exception handling and intrusive pointers
- Standard library (STL containers for schema storage)

## Trade-offs

**Static registration at library load time**: Operators are registered via static initializers (TORCH_LIBRARY macros) that run when the shared library is loaded. This means the operator table is fully populated before any user code runs, enabling fast O(1) lookups. The trade-off is inflexibility—operators cannot be dynamically added or removed at runtime without modifying source code and rebuilding.

**Schema first, kernel registration second**: Operators must have a schema defined (via `TORCH_LIBRARY`) before kernels can be registered (via `TORCH_LIBRARY_IMPL`). This two-phase approach is safer than optional schemas but requires discipline in registration order.

**Type erasure via boxing**: Arguments are boxed (converted to a uniform type-erased representation) before dispatch and unboxed afterward. This enables uniform dispatcher code but adds small per-call overhead compared to direct function pointers.

## Extension Boundaries

- **Custom operator registration**: Third-party code can define new operators via `TORCH_LIBRARY` and register kernels via `TORCH_LIBRARY_IMPL(custom_namespace, ...)`, enabling extensibility without modifying ATen internals.
- **New dispatch keys**: New functionality keys (e.g., custom hardware backends) can be registered and new kernel implementations for those keys can be added.

## Runtime Implications

**Initialization order**: During library load, all `TORCH_LIBRARY` and `TORCH_LIBRARY_IMPL` static initializers run sequentially, populating the dispatcher with schemas and kernel registrations. By the time user code runs, the dispatcher is fully initialized.

**Lookup semantics**: `Dispatcher.findSchema(op_name)` returns the schema for an operator. `Dispatcher.lookup(op_name, dispatch_key)` returns the kernel function for that operator on that dispatch key.

**Fallback behavior**: If a specific dispatch key has no kernel registered, the dispatcher consults fallback kernels registered for that key (e.g., the Autograd fallback wraps all operators to record operations).

## Performance Implications

**Lookup cost**: Schema and kernel lookup are O(1) hash table operations (flat_hash_map). Lookups occur once per operator call, adding minimal overhead.

**Registration overhead**: Static initializers run sequentially at library load, which can take hundreds of milliseconds for large libraries. This is a one-time cost at startup.

**Type erasure cost**: Boxing and unboxing arguments adds a small per-call overhead (1-2 cycles for parameter setup), offset by the ability to reuse dispatcher infrastructure across diverse operators.

## Ownership Boundaries

- **FunctionSchema owns**: operator name, overload, argument types, return types, tags
- **Dispatcher owns**: schema table, operator entry table, fallback kernel table (but not individual kernels—kernels are owned by their implementations)
- **OperatorEntry owns**: per-dispatch-key kernel pointers (but kernels themselves are owned by their implementations)

## Verification Points

- `aten/src/ATen/core/TensorBody.h` — Operator schema interface
- `aten/src/ATen/core/boxing/BoxedKernel.h` — Type-erased kernel invocation
- `aten/src/ATen/native/NegateFallback.cpp:18-22` — Example of TORCH_LIBRARY_IMPL registration
- `torch/csrc/autograd/VariableTypeManual.cpp:350` — Autograd operation registration
