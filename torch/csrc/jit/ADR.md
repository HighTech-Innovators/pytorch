# `torch/csrc/jit`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/csrc/jit` is the C++ implementation of TorchScript and the legacy JIT compiler. It owns the TorchScript frontend (parser, IR emitter, tracer), the graph IR and its optimization passes, the bytecode interpreter / graph executor, and model serialization. It compiles Python-subset code and traced models into an optimizable graph that can run without the Python interpreter.

## Key Files

| File | Purpose |
|---|---|
| `torch/csrc/jit/ir/ir.cpp` | Core graph IR: `Graph`, `Node`, `Value`, `Block` |
| `torch/csrc/jit/ir/alias_analysis.cpp` | Alias analysis enabling safe mutation/optimization passes |
| `torch/csrc/jit/frontend/parser.cpp` | Parser for the TorchScript Python subset |
| `torch/csrc/jit/frontend/ir_emitter.cpp` | Lowers parsed AST to graph IR |
| `torch/csrc/jit/frontend/tracer.cpp` | Records ops executed during a traced run |
| `torch/csrc/jit/runtime/interpreter.cpp` | Bytecode interpreter that executes IR graphs |
| `torch/csrc/jit/runtime/graph_executor.cpp` | Compiles and dispatches graph execution |
| `torch/csrc/jit/runtime/profiling_graph_executor_impl.cpp` | Profile-guided specialization/optimization |
| `torch/csrc/jit/passes/` | Graph optimization passes (fusion, DCE, constant folding) |
| `torch/csrc/jit/serialization/` | Save/load of scripted modules (ZIP archive format) |

## Public Interface

`torch::jit::Graph`, `Node`, `Value`, `Block`, `torch::jit::compile()`, `GraphExecutor`, `InterpreterState`, `torch::jit::script::Module`, `torch::jit::trace()`, `torch::jit::load()`/`save()`, the passes in `torch::jit::` (e.g. `runOptimization`, `PeepholeOptimize`).

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [aten/src/ATen](aten/src/ATen/ADR.md) | depends-on | IR nodes call ATen operators; schemas define op signatures |
| [c10/core](c10/core/ADR.md) | depends-on | Tensor/IValue types, dispatch keys |
| [torch/csrc](torch/csrc/ADR.md) | depended-on-by | `initModule()` wires JIT Python bindings via `torch/csrc/jit/python/` |
| [torch/jit](torch/jit/ADR.md) | depended-on-by | Python `torch.jit.script`/`trace` front the C++ compiler |

## Runtime Behaviour

Scripting parses the Python-subset source (`frontend/parser.cpp`) and emits graph IR (`frontend/ir_emitter.cpp`); tracing instead records the ops a concrete run executes (`frontend/tracer.cpp`). The resulting `Graph` (`ir/ir.cpp`) is run through optimization passes, then executed either directly or through the `ProfilingGraphExecutor`, which first runs a profiling pass to record tensor shapes/types, then generates a specialized optimized graph guarded by those observations. `interpreter.cpp` executes the compiled bytecode. `alias_analysis.cpp` gates which passes are safe by tracking which values may alias.

## Performance Profile

The graph executor's value proposition is amortizing Python interpreter overhead: once compiled, a graph runs in the C++ interpreter without per-op Python dispatch or the GIL. Profile-guided specialization enables fusion of element-wise chains (via `tensorexpr/` and fusion passes), cutting intermediate-tensor allocation and memory round-trips. The cost is up-front compilation and profiling latency on the first runs, plus guard checks that trigger re-specialization when observed shapes change. Serialization/deserialization cost is paid at model load.

## Design Rationale

A dedicated IR with alias analysis lets PyTorch apply graph-level optimizations (fusion, DCE, constant folding) that eager execution cannot, and lets models deploy to Python-free C++ inference. Separating a profiling executor from a plain one allows shape-specialized optimization without requiring static shapes up front. TorchScript is officially deprecated in favor of `torch.compile`/`torch.export`, so this subsystem is maintained primarily for the installed base of deployed scripted models — a dual-path maintenance burden shared with `torch/_dynamo`. The CUDA-oriented fusion backends here are inert in this CPU-only deployment.
