# `torch/_inductor`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/_inductor` is TorchInductor: the default backend for `torch.compile`. It receives FX graphs from TorchDynamo, lowers them through a multi-pass optimization pipeline, and emits either Triton (GPU) or C++/OpenMP (CPU) kernel code for execution.

## Key Files

| File | Purpose |
|---|---|
| `torch/_inductor/__init__.py` | Public entry: `compile(gm, example_inputs)`, `aot_compile`, `aoti_compile_and_package`, `aoti_load_package` |
| `torch/_inductor/compile_fx.py` | `compile_fx()` — top-level compilation driver; `compile_fx_aot` for AOT path |
| `torch/_inductor/lowering.py` | FX-node-to-Inductor-IR lowering (9480 lines): maps ATen ops to loop-level `TensorBox`/`Pointwise`/`Reduction` nodes |
| `torch/_inductor/scheduler.py` | Fusion and scheduling: decides which loops can be fused, tiled, and pipelined |
| `torch/_inductor/codegen/` | Backend code generators: `cpp.py` for CPU C++/OpenMP, `triton.py` for GPU Triton kernels |
| `torch/_inductor/codecache.py` | Compilation cache keyed on graph hash; avoids recompilation across process restarts |
| `torch/_inductor/config.py` | All tuning flags: `max_autotune`, `triton.cudagraphs`, `loop_ordering_after_fusion`, etc. |
| `torch/_inductor/ir.py` | Inductor IR: `TensorBox`, `Pointwise`, `Reduction`, `Buffer`, `ComputedBuffer` |

## Public Interface

`torch._inductor.compile(gm, example_inputs, options=...)`, `torch._inductor.aot_compile(gm, args, ...)`, `torch._inductor.aoti_compile_and_package(exported_program, ...)`, `torch._inductor.aoti_load_package(path)`, `torch._inductor.list_mode_options()`, `torch._inductor.list_options()`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/_dynamo](torch/_dynamo/ADR.md) | depended-on-by | Dynamo calls `compile_fx()` with captured FX graphs |
| [torch/fx](torch/fx/ADR.md) | depends-on | Receives `GraphModule`; reads `Graph`/`Node` IR for lowering |
| `torch._C` / ATen | depends-on | Op metadata, fake tensor propagation during lowering |

## Runtime Behaviour

`compile_fx()` receives a `GraphModule` from Dynamo and runs it through several passes: (1) joint-graph decompositions to lower composite ops; (2) `lowering.py` maps each FX node to an Inductor IR node (`Pointwise`, `Reduction`, etc.); (3) `scheduler.py` fuses and tiles adjacent pointwise/reduction nodes; (4) the backend codegen in `codegen/cpp.py` or `codegen/triton.py` emits source code; (5) `codecache.py` compiles and caches the result. On CPU, the generated C++ uses OpenMP for multi-threaded parallelism. The compiled callable is returned to Dynamo for installation in its cache.

## Performance Profile

Inductor's compilation cost is amortized across many forward passes; the first call pays full compilation time (seconds for large models). On CPU, the C++ code generator applies loop tiling and OpenMP parallelism — the primary levers for CPU throughput. `codecache.py` persists compiled artifacts to disk keyed on a graph hash, so restarts of the same workload skip recompilation. `max_autotune` triggers multi-configuration profiling at compile time to pick the best tiling/parallelism parameters — expensive at compile time, cheaper at runtime. For CPU-only deployments the Triton codegen path is inactive; only `codegen/cpp.py` is exercised.

## Design Rationale

The two-level IR (FX graph → Inductor loop IR) separates operator semantics from loop structure, enabling fusion and tiling decisions that span multiple ATen ops. `codecache.py` uses a content-addressed cache so identical subgraphs (common across model variants) share compiled artifacts. The `config.py` flag space exposes every tuning knob explicitly so operators can override defaults per deployment, rather than embedding heuristics invisibly in the codegen.
