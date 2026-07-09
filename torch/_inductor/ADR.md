# `torch/_inductor`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/_inductor` is TorchInductor: the default compilation backend for `torch.compile`. It receives an FX `GraphModule` from TorchDynamo, lowers it through a multi-level lowering pipeline (FX → Inductor IR → Triton kernels for GPU or C++/OpenMP for CPU), and returns an executable compiled artifact. It also implements Ahead-of-Time (AOT) compilation for deployment.

## Key Files

| File | Purpose |
|---|---|
| `compile_fx.py` | Primary entry point: `compile_fx(gm, example_inputs)` — orchestrates the full pipeline from FX graph to compiled callable; calls AOT autograd, then Inductor lowering |
| `codecache.py` | `FxGraphCache`, `CompiledFxGraph` — caches compiled kernels by graph hash to avoid recompilation; handles loading cached `.so` / `.py` artifacts |
| `config.py` | ~200 configuration flags: `max_autotune`, `triton.cudagraphs`, `cpp.threads`, `benchmark_kernel`, `fx_graph_cache`, `coordinate_descent_tuning` |
| `graph.py` | `GraphLowering` — lowers ATen FX graph to Inductor IR; owns `SchedulerNode` schedule |
| `scheduler.py` | `Scheduler` — fuses `SchedulerNode` objects, determines kernel boundaries, handles reductions |
| `codegen/triton.py` | Triton kernel code generator: emits `.py` files with `@triton.jit` kernels; handles tiling, masking, reductions |
| `codegen/cpp.py` | CPU kernel code generator: emits C++ with OpenMP parallelism |
| `lowering.py` | `lowering` registry: maps ATen operator schemas to Inductor IR node constructors |
| `ir.py` | Inductor IR: `Buffer`, `ComputedBuffer`, `Pointwise`, `Reduction`, `ExternKernel`; layout and stride inference |
| `async_compile.py` | `AsyncCompile` — parallel kernel compilation using a process pool; non-blocking during graph execution |
| `cache_key.py` | Graph hash computation for `FxGraphCache`; includes source hash, compiler config, and input shapes |
| `autotune_process.py` | Remote autotuning process: benchmarks candidate Triton tile configurations |
| `constant_folding.py` | Pre-compilation constant propagation on the FX graph |

## Public Interface

| Symbol | Description |
|---|---|
| `torch._inductor.compile_fx(gm, example_inputs)` | Main entry; returns a callable that accepts real tensors and runs the compiled kernels |
| `torch._inductor.config` | Module of configuration flags; `torch._inductor.config.max_autotune = True` enables autotuning |
| `torch.compile(fn, backend="inductor")` | User entry point; Dynamo calls `compile_fx` internally |
| `torch._inductor.aoti_compile_and_package(ep, ...)` | AOT compile an `ExportedProgram` to a packaged `.pt2` artifact |
| `torch._inductor.codecache.FxGraphCache` | Graph-level cache; `FxGraphCache.load(key)` / `FxGraphCache.save(key, compiled_graph)` |

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/_dynamo](torch/_dynamo/ADR.md) | depends-on | Receives `GraphModule` from Dynamo; `compile_fx` is registered as the `"inductor"` backend |
| [torch/_functorch](torch/_functorch/ADR.md) | depends-on | `compile_fx` calls `aot_autograd` from `torch._functorch` to separate forward from backward before lowering |
| [torch/fx](torch/fx/ADR.md) | depends-on | Traverses `Graph.nodes` to lower each FX node to Inductor IR |
| [aten/src/ATen](aten/src/ATen/ADR.md) | depends-on | `lowering.py` maps ATen ops to Inductor primitives; `ExternKernel` calls back into ATen for unsupported ops |
| Triton (external) | depends-on | `codegen/triton.py` generates `@triton.jit` Python code; compiled by `triton.compiler.compile` |
| `torch._C._inductor` | depends-on | C++ Inductor utilities: `aoti_*` functions for AOT compiled artifacts |

## Runtime Behaviour

`compile_fx(gm, example_inputs)` first calls `aot_export_module` from `torch._functorch.aot_autograd` to produce a joint forward-backward graph. Inductor's `GraphLowering.run()` traverses the FX graph and calls the corresponding lowering function from `lowering.py` for each node, producing Inductor IR `Buffer` and `ComputedBuffer` objects. `Scheduler.codegen()` fuses adjacent pointwise nodes and assigns each fused group to a Triton or C++ kernel. `AsyncCompile` submits kernel compilation jobs to a process pool so that multiple kernels compile in parallel while earlier kernels run. On first execution the compiled kernels are loaded from disk cache (if available) or compiled on-the-fly; subsequent calls skip all Python overhead and call directly into the compiled artifact.

## Performance Profile

- **Allocation sites**: Inductor pre-allocates output buffers during lowering and passes pre-allocated tensors to compiled kernels. At inference time with `torch.compile` and `mode="reduce-overhead"`, CUDA graph capture eliminates per-call allocation overhead.
- **Synchronization costs**: `AsyncCompile`'s process pool introduces inter-process communication during compilation. `CudaGraphs` mode captures an entire forward pass in a CUDA graph, eliminating CPU-kernel-launch latency at runtime.
- **Data movement**: Inductor fuses pointwise operations into single kernels to eliminate intermediate tensor allocations and data movement. Non-fusible operations (e.g., reductions with incompatible tiling) are kept as separate kernels.
- **Redundant or repeated work**: autotuning (`max_autotune=True`) benchmarks multiple tile-size configurations per kernel and selects the fastest; this adds significant one-time compilation time. `FxGraphCache` persists compiled kernels across processes to amortise this cost.

## Design Rationale

The multi-level lowering (FX → Inductor IR → kernel code) separates backend-agnostic optimisation (fusion, scheduling, constant folding) from backend-specific code generation (Triton vs C++). This allows the same fusion logic to produce GPU Triton kernels or CPU C++ with OpenMP. `AsyncCompile` uses a process pool rather than threads because Python kernel compilation (via `triton.compiler.compile`) releases the GIL but benefits from true parallelism for the C compilation phase. The `FxGraphCache` is keyed on a hash of the graph structure plus compiler config, so changes to `config.py` flags correctly invalidate cached artifacts.
