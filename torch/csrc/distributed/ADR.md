# `torch/csrc/distributed`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/csrc/distributed` implements PyTorch's C++ distributed runtime: c10d process groups, stores, collective work handles, DistributedDataParallel gradient reduction, RPC agents, RRefs, and distributed autograd. It is the native extension behind the Python distributed APIs and follows the Python-to-C++ binding pattern described in book Chapter 06 (`book/06-python-bindings.md`). The directory turns Python calls such as `all_reduce`, DDP reducer hooks, RPC sends, and distributed backward into asynchronous C++ communication and autograd work.

## Key Files

| File | Purpose |
|---|---|
| `c10d/ProcessGroup.hpp` | Abstract collective and point-to-point communication API with backend type, rank/size, timeouts, and asynchronous `Work` objects |
| `c10d/Work.hpp` | Completion, wait, exception, future, and result abstraction returned by asynchronous communication calls |
| `c10d/Store.hpp` | Rendezvous key-value store base used by TCP, file, hash, and prefix stores |
| `c10d/ProcessGroupNCCL.hpp` | NCCL backend for GPU collectives and CUDA stream/event coordination |
| `c10d/ProcessGroupGloo.hpp` | Gloo backend for CPU and selected CUDA collective paths |
| `c10d/reducer.hpp` | DDP reducer that buckets gradients, installs autograd hooks, and runs communication hooks or allreduce |
| `rpc/rpc_agent.h` | Asynchronous RPC agent base with worker identity, retry options, message sending, callbacks, and futures |
| `rpc/rref_context.h` | RRef ownership and lifetime coordination for remote references |
| `autograd/autograd.h` | Public C++ distributed autograd `backward` entry point for a distributed context |
| `autograd/engine/dist_engine.h` | Distributed autograd engine that coordinates graph traversal and gradient propagation across workers |

## Public Interface

The c10d interface exposes `ProcessGroup`, backend-specific process groups, `Store`, `Work`, collective option structs, `GradBucket`, and communication hook registration. RPC exposes `RpcAgent`, `WorkerInfo`, messages, request callbacks, RRefs, and TensorPipe-backed implementations. Distributed autograd exposes `torch::distributed::autograd::backward(context_id, roots, retain_graph)`. Python bindings in `c10d/init.cpp`, `rpc/init.cpp`, and `autograd/init.cpp` publish these native classes and functions under `torch._C._distributed_c10d`, `torch._C._distributed_rpc`, and distributed autograd modules.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [c10/core](c10/core/ADR.md) | depends-on | Uses devices, scalar types, `IValue`, intrusive futures, and custom class holders for process groups and RPC objects |
| [aten/src/ATen/core](aten/src/ATen/core/ADR.md) | depends-on | Communicates `at::Tensor` payloads and registers functional collective operators through the dispatcher |
| [torch/csrc/autograd](torch/csrc/autograd/ADR.md) | depends-on | DDP reducer installs autograd hooks, and distributed autograd builds on autograd nodes and variable lists |
| [torch/csrc/profiler](torch/csrc/profiler/ADR.md) | depends-on | Records profiler and flight-recorder metadata for communication, RPC, and distributed debugging paths |
| [torch/distributed](torch/distributed/ADR.md) | depended-on-by | Python distributed APIs wrap these C++ classes and route user calls into c10d, RPC, and reducer implementations |

## Runtime Behaviour

A process group owns a fixed membership and delegates each collective to a backend such as Gloo, NCCL, UCC, MPI, XCCL, or a custom backend; calls return a `Work` object that represents asynchronous progress and exposes wait/future/error state. DDP constructs a `Reducer` with model parameters and bucket assignments, attaches autograd hooks to mark gradients ready, and launches allreduce or a registered communication hook when each bucket becomes complete. RPC agents send serialized `Message` objects to `WorkerInfo` destinations, complete `JitFuture` results on responses, and use request callbacks to execute Python, TorchScript, or builtin functions. Distributed autograd associates RPC traffic with a context id, discovers graph dependencies from roots, propagates gradients, and accumulates leaf gradients across workers.

## Performance Profile

Collectives run asynchronously so backward computation can overlap with network transfers; the reducer's default bucket caps of 1 MiB for the first bucket and 25 MiB for later buckets balance early overlap against bandwidth efficiency. NCCL and CUDA paths coordinate streams, events, sequence numbers, and flight-recorder metadata so GPU communication avoids unnecessary host synchronization. Communication hooks support compression, custom futures, and optimizer-in-backward paths, which reduce bandwidth or eliminate gradient copies when configured. RPC performance depends on worker threads, message serialization, timeout/retry settings, and TensorPipe device maps, so the base `RpcAgent` keeps sends nonblocking and returns futures immediately.

## Design Rationale

The architecture separates process-group collectives, RPC, and distributed autograd because they require different communication patterns but share tensors, futures, and Python bindings. `ProcessGroup` abstracts backend differences behind one collective API while still exposing backend ids for profiling and debugging. DDP reduction lives in C++ because it must react to autograd hooks on the training hot path and overlap communication with gradient production. RPC and RRef abstractions preserve Python object semantics across workers while keeping transport, retries, and futures in native code.
