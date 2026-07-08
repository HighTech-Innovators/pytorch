# `torch/csrc/dynamo`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/csrc/dynamo` implements the native CPython frame-evaluation hook, guard fast path, cache-entry storage, frame-local mapping, and compiled-autograd bridge used by TorchDynamo. Book Chapter 07 (`book/07-torchdynamo.md`) describes Dynamo's bytecode capture and guard model; this directory supplies the C/C++ machinery that intercepts CPython frames, stores optimized code on code objects, and evaluates tensor guards before Python re-enters tracing. The code is tightly coupled to CPython internals and version-specific frame layouts.

## Key Files

| File | Purpose |
|---|---|
| `eval_frame.c` | CPython eval-frame shim, thread-local callback state, `_PyInterpreterFrame` Python wrapper, and frame interception entry points |
| `eval_frame.h` | C interface for installing and calling the custom frame evaluator |
| `cache_entry.h` | Defines `CacheEntry` nodes storing guard managers, optimized code objects, compile ids, backend references, and trace annotations |
| `extra_state.h` | Owns per-code-object extra state containing cache-entry lists and shared frame state |
| `guards.h` | Defines `LocalState`, `TensorCheck`, root guard-manager entry points, and dispatch-key masking semantics |
| `framelocals_mapping.h` | Maps CPython frame locals, cell variables, globals, and stack references into guard-checkable values |
| `compiled_autograd.h` | C++ abstraction for calling a Python autograd compiler from libtorch without directly depending on Python implementation code |
| `python_compiled_autograd.cpp` | Python-facing compiled-autograd bridge implementation |
| `stackref_bridge.c` | CPython stack-reference compatibility layer for newer Python frame internals |
| `init.cpp` | Registers the `_dynamo` native module surface exposed through `torch._C` |

## Public Interface

The native surface exposes frame-evaluation control, cache inspection and mutation helpers, guard hooks, frame wrappers, and compiled-autograd registration through `torch._C._dynamo` and related `torch._C` bindings. Python Dynamo installs a callback that receives frame objects, returns guarded bytecode, and stores cache entries. Guard managers call `run_root_guard_manager` against `FrameLocalsMapping`, while compiled autograd installs a `PyCompilerInterface` implementation through `PyCompilerGuard` so C++ autograd nodes can delegate graph capture back to Python.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [c10/core](c10/core/ADR.md) | depends-on | Uses dispatch key sets, grad mode, symbolic sizes, scalar types, and tensor metadata for guard checks |
| [aten/src/ATen/core](aten/src/ATen/core/ADR.md) | depends-on | Checks `at::Tensor` dtype, device, sizes, strides, dispatch keys, and autograd metadata |
| [torch/csrc/autograd](torch/csrc/autograd/ADR.md) | depends-on | Compiled autograd mirrors autograd nodes, saved variables, input metadata, and gradient accumulation callbacks |
| [torch/_dynamo](torch/_dynamo/ADR.md) | depended-on-by | Python Dynamo installs callbacks, builds guard managers, manages cache limits, and handles graph breaks around this native hook |
| [torch/_inductor](torch/_inductor/ADR.md) | related | Compiled Dynamo entries commonly call Inductor-generated backends after guards pass |

## Runtime Behaviour

When Dynamo is active, `eval_frame.c` replaces CPython's frame evaluator with `dynamo__custom_eval_frame_shim`, reads a thread-local Python callback, and wraps the live interpreter frame so Python Dynamo can inspect code, locals, globals, builtins, closures, and instruction offsets. A code object's extra state stores a linked list of `CacheEntry` objects; each entry contains a guard manager and optimized code object for one set of assumptions. On a function call, native guard code maps frame locals through `FrameLocalsMapping`, checks tensor type, dtype, device, dispatch keys, dynamic sizes, dynamic strides, and grad mode, then returns cached code when the guard passes. If no entry matches, the callback compiles or graph-breaks in Python and appends a new cache entry.

## Performance Profile

The frame-evaluation hook sits on every compiled function invocation, so the hot path keeps guard data in C++ objects and performs tensor checks without re-running Python bytecode analysis. `TensorCheck` stores compact dispatch-key, dtype, device index, requires-grad, dimension, size, and stride expectations; dynamic dimensions use optional `SymInt` values so stable dimensions stay cheap. Cache entries live on the code object's extra scratch space, which avoids a global dictionary lookup for the common case. The code uses CPython-version conditionals and stack-reference bridges because direct frame access is faster than public reflection APIs but changes across Python releases.

## Design Rationale

Dynamo captures ordinary Python by intercepting CPython frames instead of requiring users to author a restricted IR. Native eval-frame integration is necessary because Python has no public high-performance hook with enough access to locals, bytecode offsets, and frame state. Guard checks live in C++ because they run on every cache hit, while graph construction remains in Python where Dynamo can reuse FX, symbolic variables, and backend configuration. Compiled autograd uses an abstract C++ `PyCompilerInterface` so libtorch_cpu can participate in compiled backward graphs without directly depending on libtorch_python symbols.
