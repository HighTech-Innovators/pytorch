# `torch/csrc/autograd`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/csrc/autograd` implements PyTorch's C++ reverse-mode automatic differentiation runtime described in book Chapter 05, "Autograd Engine". It owns graph vertices (`Node`), graph edges (`Edge`), tensor autograd metadata (`AutogradMeta` in `variable.h`), saved forward tensors (`SavedVariable`), and the `Engine`/`GraphTask` scheduler that executes backward graphs. It bridges generated derivative wrappers, Python autograd bindings, C++ API entry points, profiler/anomaly hooks, and dispatch-key state so forward operations record a dynamic graph and backward execution computes gradients.

## Key Files

| File | Purpose |
|---|---|
| `node.h` | Defines `Node`, `edge_list`, input metadata, sequence numbers, topological numbers, stream selection, and next-edge graph connectivity |
| `edge.h` | Defines `Edge` as the `(Node, input_nr)` pointer used to connect backward functions |
| `engine.h` | Declares `Engine`, `ReadyQueue`, `NodeTask`, dependency computation, and the main backward execution API |
| `engine.cpp` | Implements ready-queue scheduling, dependency counting, node evaluation, stream synchronization, graph-task completion, and reentrant backward handling |
| `graph_task.h` | Defines `GraphTask` state for one backward run: dependencies, not-ready input buffers, captured grads, thread-local state, stream bookkeeping, and completion future |
| `input_buffer.h` | Defines `InputBuffer`, the per-node gradient accumulator used when multiple downstream edges feed one node input |
| `variable.h` | Defines `AutogradMeta` fields and helper APIs for `grad_fn`, `grad_accumulator`, version counters, hooks, views, and `requires_grad` |
| `saved_variable.h` | Defines `SavedVariable`, the version-checked snapshot object used by backward formulas to recover forward tensors |
| `autograd.cpp` | Implements the pure C++ `backward` and `grad` entry points that gather roots and call `Engine::execute` |
| `python_engine.cpp` | Binds the C++ engine to Python's `_EngineBase.run_backward` path |

## Public Interface

The C++ interface exposes `torch::autograd::backward`, `torch::autograd::grad`, `Engine::get_default_engine().execute`, `Engine::execute_with_graph_task`, `Node::operator()`, `Node::next_edges`, `impl::gradient_edge`, `impl::set_gradient_edge`, `impl::grad_accumulator`, and the `SavedVariable::unpack`/hook surface. Python reaches this runtime through `python_engine.cpp`, `python_function.cpp`, and generated autograd wrappers; C++ frontend users reach it through `autograd.h` and `variable.h`. Custom C++ autograd functions derive from `Node` or related function classes and return `variable_list` gradients from `apply`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [c10/core](c10/core/ADR.md) | depends-on | Uses tensors, streams, intrusive pointers, dispatch-key state, and device metadata carried by `TensorImpl` and `Stream` |
| [c10/util](c10/util/ADR.md) | depends-on | Uses `intrusive_ptr`, `SmallVector`, `Exception`, hash helpers, and range utilities throughout graph and engine code |
| [aten/src/ATen/core](aten/src/ATen/core/ADR.md) | depends-on | Uses `Tensor`, `ThreadLocalState`, `IValue`, `SequenceNumber`, and operator dispatch interfaces during graph construction and execution |
| [torch/autograd](torch/autograd/ADR.md) | depended-on-by | Python APIs normalize user inputs, call `_engine_run_backward`, and expose hooks, grad mode, gradcheck, and custom Function support over this runtime |
| [torch/csrc/profiler](torch/csrc/profiler/ADR.md) | depended-on-by | Profiler events use `Node` sequence numbers and thread ids to correlate backward nodes with forward operations |
| [tools/autograd](tools/autograd/ADR.md) | depends-on | Generated VariableType wrappers and derivative classes use formulas and saved-variable requirements generated from autograd definitions |

## Runtime Behaviour

Forward operator wrappers create `Node` subclasses, save required inputs in `SavedVariable`, and attach output tensors to graph edges through `impl::set_gradient_edge`; leaf tensors instead receive gradient accumulators through `impl::grad_accumulator`. `Node` stores next edges and topological numbers, so the engine prunes unreachable paths and prioritizes later-created backward functions as Chapter 05 describes. `Engine::execute` validates roots and initial gradients, creates a `GraphTask`, computes dependency counts from the graph root, and then drives `thread_main` on the caller CPU queue and device ready queues. `Engine::evaluate_function` waits for accelerator events recorded in `InputBuffer`, calls the node, releases saved variables when `keep_graph_` is false, decrements downstream dependencies, accumulates not-ready inputs, and pushes ready `NodeTask`s to the appropriate queue.

## Performance Profile

The hot path avoids Python and virtual tensor metadata dispatch during backward scheduling: `Edge` is a compact intrusive pointer plus input index, `ReadyQueue` orders tasks by reentrant depth and `Node::sequence_nr`, and `GraphTask::dependencies_` lets each edge transition a node to ready with a decrement. `InputBuffer::add` moves the first gradient into place and only performs tensor addition when multiple gradient contributions target the same input slot. `compute_dependencies` uses each node's `topological_nr` to skip graph branches below the requested outputs, matching Chapter 05's pruning model. Accelerator execution records ready streams and leaf streams so synchronization occurs through events instead of device-wide barriers.

## Design Rationale

PyTorch keeps autograd as a dynamic C++ graph because eager Python programs contain arbitrary control flow while still needing compiled, thread-aware backward execution. `Node` and `Edge` represent the minimal graph abstraction: formulas live in node subclasses, connectivity lives in edge lists, and tensors carry only the metadata needed to find their gradient edge. `GraphTask` isolates one backward run so `autograd.grad`, `backward(inputs=...)`, reentrant backward, callbacks, error state, and stream synchronization coexist without global graph state. Saved tensors keep references and version counters instead of copies because activation memory dominates training cost and in-place modification checks preserve correctness.
