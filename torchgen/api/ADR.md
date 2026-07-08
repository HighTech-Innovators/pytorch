# `torchgen/api`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torchgen/api` defines the signature and type models that turn a JIT-style `FunctionSchema` into each concrete C++ interface PyTorch emits. It distinguishes the public C++ API, dispatcher unboxed API, native kernel API, Python binding signatures, autograd saved-variable metadata, structured/meta APIs, ufunc signatures, lazy APIs, and generated unboxing glue. This directory sits between `torchgen/model.py` and output generators: it decides what arguments exist, what C++ types they use, how TensorOptions scatter or gather, and how one API calls another.

## Key Files

| File | Purpose |
|---|---|
| `cpp.py` | Maps schemas to the public C++ API, including TensorOptions packing, defaults, method/function naming, return types, and SymInt overloads |
| `dispatcher.py` | Maps schemas to the unboxed dispatcher calling convention used by `at::_ops` and dispatcher registrations |
| `native.py` | Maps schemas to `at::native` kernel declarations, keeping native and dispatcher signatures close to avoid wrapper overhead |
| `python.py` | Models Python parser signatures, `.pyi` signatures, Python-to-C++ binding expressions, and TensorOptions `requires_grad` handling |
| `translate.py` | Synthesizes expressions that convert one set of `NamedCType` bindings into another API's expected bindings |
| `autograd.py` | Defines differentiability metadata, saved attributes, backward formulas, forward derivatives, and matching helpers for `derivatives.yaml` |
| `ufunc.py` | Defines CPU and CUDA ufunc argument conventions, functor constructor/apply bindings, and dtype-specific computation types |
| `unboxing.py` | Emits IValue-to-C++ conversion snippets for generated JIT/mobile unboxing wrappers |

## Public Interface

Generator modules import functions such as `cpp.arguments()`, `cpp.returns_type()`, `dispatcher.arguments()`, `native.arguments()`, `structured.meta_arguments()`, and `translate.translate()` to render calls and declarations. `types/signatures.py` exposes `CppSignature`, `CppSignatureGroup`, `DispatcherSignature`, `NativeSignature`, `StructuredImplSignature`, and related classes with `decl()`, `defn()`, `arguments()`, `returns_type()`, and `ptr_type()` methods. `python.py` exposes Python signature models that feed `tools/autograd/gen_python_functions.py`. `autograd.py` exposes `DifferentiabilityInfo`, `Derivative`, `ForwardDerivative`, and `SavedAttribute`, which `tools/autograd/load_derivatives.py` fills from `derivatives.yaml`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torchgen](torchgen/ADR.md) | depends-on | Consumes `FunctionSchema`, `NativeFunction`, `Argument`, `Return`, `DispatchKey`, and type dataclasses from the core model |
| [torchgen/dest](torchgen/dest/ADR.md) | depended-on-by | Destination generators use API signatures and `translate()` to render wrapper bodies and registrations |
| [tools/autograd](tools/autograd/ADR.md) | depended-on-by | Autograd codegen uses `api.autograd`, `CppSignatureGroup`, and C++ type models to save inputs and emit backward nodes |
| [tools/jit](tools/jit/ADR.md) | depended-on-by | JIT unboxing generation uses `api.unboxing`, `CppSignatureGroup`, and `translate()` for stack-to-kernel wrappers |
| [aten/src/ATen/core](aten/src/ATen/core/ADR.md) | generates-for | Dispatcher signatures match the unboxed `Dispatcher::call`/`redispatch` entry points used by generated `at::_ops` |

## Runtime Behaviour

At generation time, API modules receive immutable schemas and return typed `Binding` lists, C++ type strings, default expressions, and expression lists. `translate.translate()` builds a context of available `NamedCType` expressions, performs forward inference for optional tensors, scalar opmath conversions, TensorOptions fields, and method `self`, then satisfies target bindings or raises `UnsatError` with the available context. Python binding generation separately models parser signatures and C++ dispatch lambdas so `torch._C._VariableFunctions` can parse PyObjects, release the GIL, call ATen, and wrap results as described in book chapter 03's Python-to-dispatch path.

## Performance Profile

This directory performs pure Python schema-to-string computation during builds, and its hot paths run once per generated operator and per API surface. The type model keeps semantic names in `NamedCType`, so generators avoid expensive and error-prone string matching when reordering, packing, or unpacking arguments. The generated code benefits directly: dispatcher APIs avoid TensorOptions packing, native APIs track dispatcher conventions to minimize wrapper translation, and unboxed signatures avoid the boxed `IValue` path except where JIT/mobile explicitly asks for generated unboxing.

## Design Rationale

PyTorch carries several legitimate C++ interfaces for the same operator because public C++, dispatcher, native kernels, Python bindings, autograd wrappers, and JIT stacks impose different constraints. `torchgen/api` isolates those conventions so output destinations do not duplicate type policy. `NamedCType` encodes both C++ type and semantic role, which prevents accidental substitution of unrelated booleans such as `pin_memory` and `requires_grad`. The split matches book chapter 03's distinction between Python bindings, typed dispatcher calls, and boxed fallbacks, and it gives book chapter 05's autograd generator the saved-variable and derivative metadata it needs without changing the core schema model.
