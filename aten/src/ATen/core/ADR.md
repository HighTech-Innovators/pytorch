# `aten/src/ATen/core`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`aten/src/ATen/core` contains the C++ operator-dispatch core that turns schemas and tensor arguments into kernel calls. Its `dispatch`, `boxing`, stack, schema, generator, list, dict, and IValue support code sits between `c10/core` tensor metadata and `aten/src/ATen/native` kernel implementations. Book Chapter 03 maps directly onto this directory: `Dispatcher`, `OperatorEntry`, `DispatchKeyExtractor`, and `KernelFunction` implement the operator dispatch flow.

## Key Files

| File | Purpose |
|---|---|
| `dispatch/Dispatcher.h` | Singleton operator registry, schema lookup, boxed calls, unboxed calls, redispatch, and registration internals |
| `dispatch/OperatorEntry.h` | Per-operator schema, dispatch table, kernel registrations, fallback updates, and error reporting |
| `dispatch/DispatchKeyExtractor.h` | Tensor-argument dispatch-key extraction for boxed and unboxed operator calls |
| `boxing/KernelFunction.h` | Wrapper that stores boxed and unboxed kernels and bridges between calling conventions |
| `function_schema.h` | Operator schema representation used by registration and validation |
| `stack.h` | Boxed dispatcher stack type used for generic operator invocation |
| `Tensor.h` | ATen tensor handle layer over `TensorImpl` |

## Public Interface

The main public interface is `c10::Dispatcher::singleton()`, `findSchemaOrThrow`, `call`, `redispatch`, `callBoxed`, and registration methods reached through the library registration API. `OperatorHandle` and `TypedOperatorHandle` identify registered operators and carry typed call signatures. `OperatorEntry` is internal, but its dispatch table and `DispatchKeyExtractor` define the observable behaviour of every operator call. `KernelFunction` accepts boxed kernels, unboxed function pointers, functors, and lambdas, then calls them through boxed `Stack*` or templated unboxed `call<Return, Args...>` entry points.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [c10/core](c10/core/ADR.md) | depends-on | `DispatchKeySet`, `DispatchKey`, `TensorImpl`, `SafePyObject`, `SymInt`, and device metadata |
| [c10/util](c10/util/ADR.md) | depends-on | `Exception`, `LeftRight`, type lists, `flat_hash_map`, bitsets, and intrusive ownership helpers |
| [aten/src/ATen/native](aten/src/ATen/native/ADR.md) | depended-on-by | Generated native wrappers register kernels and call through this dispatcher |
| [torch/csrc/autograd](torch/csrc/autograd/ADR.md) | depended-on-by | Autograd wrappers use dispatch keys and redispatch to wrap leaf kernels |
| [torchgen](torchgen/ADR.md) | depended-on-by | Code generation emits schemas, typed wrappers, and registrations against these APIs |

## Runtime Behaviour

`Dispatcher::singleton()` returns the process-wide dispatcher; on non-mobile builds the inline accessor caches a reference to `realSingleton`, while mobile builds call `realSingleton` directly to avoid duplicating static-guard code across operator stubs. `DispatchKeyExtractor` unions key sets from `Tensor`, optional `Tensor`, tensor lists, `ITensorListRef`, and defined `Generator` arguments, applies thread-local included and excluded keys, then masks out fallthrough keys for that operator. `OperatorEntry::lookup` converts the effective `DispatchKeySet` into a dispatch-table index, validates that a kernel exists, and returns the `KernelFunction`. `Dispatcher::call` uses the unboxed path for typed ATen calls, while `callBoxed` and `redispatchBoxed` run through a `Stack` for middleware and generic operators.

## Performance Profile

`OperatorEntry` stores a fixed `std::array<KernelFunction, c10::num_runtime_entries>`, so dispatch lookup is an array index rather than a hash-map lookup. `lookup` checks `isValidUnboxed()` first because generated ATen APIs usually call unboxed kernels and can avoid touching the boxed kernel in the common case. `DispatchKeyExtractor` precomputes boxed stack argument positions from the schema and uses `unsafeToTensorImpl` in boxed extraction to avoid a tensor refcount bump. `KernelFunction::makeFromUnboxedFunction` accepts a compile-time function pointer so wrapper code can inline more aggressively than a runtime function pointer path.

## Design Rationale

Chapter 03 describes dispatch as a central registry plus per-operator tables, and this directory implements that split. A singleton dispatcher supports dynamic registration, schema lookup, tracing, backend fallbacks, and global middleware without putting a virtual table on every tensor. Per-operator `OperatorEntry` tables keep the hot path small while preserving rich registration metadata for diagnostics. Boxed and unboxed calls coexist because leaf kernels need direct typed speed and middleware needs a signature-independent stack representation.

