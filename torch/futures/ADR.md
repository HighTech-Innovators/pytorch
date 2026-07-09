# `torch/futures`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/futures` exposes the Python future abstraction used by asynchronous PyTorch APIs. The module wraps `torch._C.Future` with typed Python methods, callback helpers, result and exception setters, and collection helpers for groups of futures.

## Key Files

| File | Purpose |
|---|---|
| `torch/futures/__init__.py` | Defines `Future`, `collect_all`, and `wait_all` and exports them in `__all__`. |

## Public Interface

| Symbol | Description |
|---|---|
| `Future(devices=None)` | Subclass of `torch._C.Future` that accepts optional CUDA devices and converts them to `torch.device` before calling the C++ constructor. |
| `Future.done()` | Delegates to the C++ future and reports whether a result or exception is present. |
| `Future.wait()` | Blocks until completion and returns the value, with CUDA stream synchronization handled by the C++ future. |
| `Future.value()` | Returns the already-completed value without performing the extra synchronization documented for `wait`. |
| `Future.then(callback)` | Registers a callback and returns the child `Future` produced by the C++ `then` implementation. |
| `Future.add_done_callback(callback)` | Registers an inline completion callback without returning a chained future. |
| `Future.set_result(result)` | Completes the future and triggers callbacks. |
| `Future.set_exception(result)` | Installs an unwrap function that raises the supplied `Exception`, then completes the future with that exception object. |
| `collect_all(futures)` | Calls `torch._C._collect_all` and returns a future that completes with the input futures. |
| `wait_all(futures)` | Calls `_collect_all(...).wait()` and then waits each returned future, returning the list of values. |

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch](torch/ADR.md) | depends-on | Imports `torch`, constructs `torch.device` objects, subclasses `torch._C.Future`, and calls `torch._C._collect_all`. |
| [torch/csrc](torch/csrc/ADR.md) | depends-on | Relies on the C++ `Future` object for storage, completion state, callback execution, CUDA stream synchronization, and `_collect_all`. |
| [torch/distributed](torch/distributed/ADR.md) | depended-on-by | Distributed RPC APIs such as `torch.distributed.rpc.rpc_async` return and consume `torch.futures.Future` objects. |

## Runtime Behaviour

`Future.__init__` normalizes each entry in `devices` to a `torch.device` and passes the device list to `torch._C.Future`, which owns the actual completion state. `done`, `wait`, `value`, `then`, `add_done_callback`, and `set_result` call their C++ base methods directly, while `then` casts the returned base future to the Python generic type. `set_exception` requires an `Exception`, registers a small `raise_error` unwrap function through `_set_unwrap_func`, and completes the future by storing the exception object as the result. `collect_all` and `wait_all` batch futures through `torch._C._collect_all` so the C++ layer observes all completions.

## Performance Profile

The hot completion, waiting, callback registration, and CUDA stream synchronization logic lives in `torch._C.Future`, so Python overhead is limited to method dispatch, type casts, and device normalization at construction. `then` allocates a child future and carries callback scheduling cost, while `add_done_callback` avoids returning a chained future when no callback result needs synchronization. `collect_all` performs a single C++ aggregation call over the input list, but `wait_all` performs an additional Python list comprehension that calls `wait()` on each completed subfuture. Futures carrying CUDA tensors record and consume stream synchronization events in the C++ layer, which avoids a full host blocking synchronization when the current streams can be ordered instead.

## Design Rationale

The Python class gives users typed, documented methods while preserving the C++ future as the source of truth for concurrency and stream semantics. The API separates `then` from `add_done_callback` so callers can choose between callback chaining and a cheaper fire-and-forget notification. `collect_all` and `wait_all` provide bulk coordination without exposing the internal `_collect_all` binding as the public interface.
