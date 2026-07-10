# `torch/csrc/autograd`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/csrc/autograd` implements PyTorch's reverse-mode automatic-differentiation engine in C++. It owns the backward-graph node/edge model, the singleton `Engine` with per-device ready queues, gradient accumulation via `InputBuffer`, and the `PyNode`/`THPVariable` bridge that lets Python-defined `Function` subclasses participate in the C++ backward pass.

## Key Files

| File | Purpose |
|---|---|
| `torch/csrc/autograd/engine.cpp` | `Engine::execute`, `thread_main`, `evaluate_function`, `compute_dependencies`, `call_function`; fork safety |
| `torch/csrc/autograd/engine.h` | `Engine`, `ReadyQueue`, `NodeTask`, `MAX_DEPTH`, `ThreadPoolShared` |
| `torch/csrc/autograd/function.h` | `Node` base class, `Edge`, `collect_next_edges`, input metadata |
| `torch/csrc/autograd/graph_task.h` | `GraphTask`, `ExecInfo`, `outstanding_tasks_`, `captured_vars_` |
| `torch/csrc/autograd/input_buffer.cpp` | Accumulates multiple gradient contributions before a node runs |
| `torch/csrc/autograd/python_variable.cpp` | `THPVariable` tensor wrapper and autograd metadata access |
| `torch/csrc/autograd/python_function.cpp` | `PyNode`: bridges a Python `Function.backward` into the engine |
| `torch/csrc/autograd/custom_function.cpp` | C++ custom `Function` support |

## Public Interface

`Engine::get_default_engine()`, `Engine::execute()`, `Node`, `Edge`, `collect_next_edges()`, `GraphTask`, `ReadyQueue`, `InputBuffer`, `PyNode`, `THPVariable_Wrap()`, `THPVariable_Unpack()`, `AutogradMeta`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [c10/core](c10/core/ADR.md) | depends-on | `TensorImpl::autograd_meta_`, dispatch keys (`AutogradCPU`) |
| [aten/src/ATen](aten/src/ATen/ADR.md) | depends-on | Backward nodes call ATen ops to compute gradients |
| [tools/autograd](tools/autograd/ADR.md) | depends-on | `derivatives.yaml` generates the per-operator backward `Node` subclasses |
| [torch/autograd](torch/autograd/ADR.md) | depended-on-by | Python `backward()`/`grad()` drive this engine |
| CPython | mutual | `PyNode` calls Python `Function.backward`; requires GIL |

## Runtime Behaviour

`loss.backward()` reaches `Engine::execute()`, which validates outputs, builds a `GraphTask`, and calls `compute_dependencies()` to count how many gradient inputs each `Node` will receive. `execute_with_graph_task()` enqueues the root `NodeTask` and calls `thread_main()`, whose loop pops the highest-`sequence_nr_` task from a per-device `ReadyQueue`, invokes `evaluate_function()` → `call_function()` (which runs the node's `apply`), merges outputs into each target's `InputBuffer`, and decrements `outstanding_tasks_`; a node becomes ready only when its dependency count hits zero. Reentrant backward is bounded by `MAX_DEPTH = 60`, and a `pthread_atfork` handler poisons the engine after a fork so child-process backward calls raise.

## Performance Profile

On CPU the whole backward runs through the `cpu_ready_queue`, so queue management, `InputBuffer` gradient accumulation (tensor adds), and per-node dispatch can dominate when a graph has many tiny ops — the framework overhead exceeds the arithmetic. Priority ordering by `sequence_nr_` (a `std::priority_queue`) adds a log-factor per push/pop. Each `PyNode` invocation reacquires the GIL, so Python-defined autograd functions serialize the backward pass. Recording gradients unnecessarily allocates `AutogradMeta` on `TensorImpl` and builds no-op nodes — `torch.inference_mode()` removes this entire trace.

## Design Rationale

Reverse-mode with a tape of `Node`/`Edge` objects (rather than symbolic differentiation) matches scalar-loss training and keeps the backward graph a direct image of the forward computation via `collect_next_edges`. Per-device ready queues avoid cross-device synchronization; `outstanding_tasks_` as an atomic counter ensures exactly one thread enqueues a newly-ready node. `InputBuffer` implements the chain rule's sum-over-paths by accumulating gradients before a node executes. The `MAX_DEPTH` limit prevents stack exhaustion under reentrant backward. In this CPU-only build there are no CUDA streams to overlap, so the control flow (`Engine::execute` → ready queue → `evaluate_function`) is the clean, dominant path.
