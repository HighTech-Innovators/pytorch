# `torch/futures`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/futures` owns the Python future abstraction used for asynchronous PyTorch work. It wraps the native `torch._C.Future` with Python typing, callback chaining, and convenience combinators for waiting on many asynchronous results.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Defines the `Future` wrapper class and the `collect_all` and `wait_all` helpers |

## Public Interface

`Future`, `collect_all`, and `wait_all` are the package-level exports. Important methods on `Future` are `done`, `wait`, `value`, `then`, `add_done_callback`, `set_result`, and `set_exception`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/distributed](torch/distributed/ADR.md) | depended-on-by | RPC and distributed communication hooks use `torch.futures.Future` for asynchronous completion and callback chaining |
| [torch/csrc](torch/csrc/ADR.md) | depends-on | `Future` subclasses the native `torch._C.Future`, and `collect_all` delegates to `torch._C._collect_all` |

## Runtime Behaviour

`Future.__init__()` normalizes the optional `devices` list to `torch.device` objects and passes them to the C++ base class so GPU results can be synchronized correctly. `wait()` and `value()` simply delegate to the native future, but their docstrings define when GPU synchronization is inserted and when callers must stay on the same stream. `then()` returns a new future by registering a callback on the underlying C++ future, while `add_done_callback()` uses the same callback mechanism for side-effecting observers. `set_exception()` installs an unwrap function that re-raises the stored exception and then completes the future through `set_result()`.

## Performance Profile

The wrapper adds very little scheduling overhead because completion, callback registration, and collection all live in the native `torch._C.Future` implementation. Device-aware futures can avoid blocking host synchronization by recording the right stream dependencies and letting GPU work finish asynchronously after `wait()` returns. `collect_all()` batches completion tracking into one future instead of forcing callers to poll or join each child individually. `wait_all()` is eager and sequential after collection, so it is convenient but not the lowest-overhead path when a caller can continue composing futures instead of blocking.

## Design Rationale

PyTorch needs one asynchronous abstraction that works for RPC, distributed communication, and device-aware callbacks. Wrapping the C++ future in Python keeps the fast path native while exposing a typed, ergonomic API that matches Python users' expectations.
