# `torch/_inductor`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/_inductor` is PyTorch's default compiler backend for FX graphs, as mapped in book chapter 08. It receives functionalized and decomposed FX graphs, lowers ATen and prim operations into Inductor IR, schedules IR nodes into fused kernels, delegates backend-specific source emission to `codegen/`, compiles or loads cached artifacts, and returns callable wrappers. It owns graph lowering, operator lowerings, buffer/layout planning, scheduling, autotuning, code caching, CUDA graph integration, AOTInductor packaging support, and many post-Dynamo graph rewrites.

## Key Files

| File | Purpose |
|---|---|
| `compile_fx.py` | Main backend entry points: `compile_fx`, `compile_fx_inner`, forward/backward compilation, AOT paths, cudagraph decisions, and AOTAutograd integration |
| `graph.py` | `GraphLowering`, an FX interpreter that lowers graph nodes to Inductor IR and orchestrates code generation |
| `lowering.py` | Registry of ATen/prims lowerings, fallback handling, layout constraints, foreach grouping, and IR construction helpers |
| `ir.py` | Inductor IR definitions including `TensorBox`, `StorageBox`, `Buffer`, `Pointwise`, `Reduction`, views, layouts, and extern kernels |
| `scheduler.py` | Dependency analysis, memory planning, fusion decisions, stream assignment, and dispatch to backend scheduling classes |
| `codecache.py` | Local and remote caches for generated Python, C++, Triton, FX graph artifacts, AOTI objects, and cache-key metadata |
| `select_algorithm.py` | Template and extern-kernel choice machinery for autotuned matmul, convolution, and other algorithm families |
| `virtualized.py` | Global virtual handles (`V`, `ops`) used by lowerings, IR, scheduler, and codegen without passing every context explicitly |
| `decomposition.py` | Decomposition table selection that normalizes higher-level ops before lowering |
| `output_code.py` | Runtime representation of compiled artifacts and wrapper metadata returned from compilation |

## Public Interface

`compile_fx.compile_fx()` is the primary backend callable used by Dynamo and `torch.compile(backend="inductor")`. `compile_fx_inner()` performs the core graph-lowering and code-generation work, while `compile_fx_forward()` and `compile_fx_backward()` support AOTAutograd's split graphs. `aot_compile.py` and `compile_fx_aot()` support AOTInductor paths that package compiled graphs for deployment. The module exposes configuration through `torch._inductor.config`, debug helpers through `debug.py`, cache objects through `codecache.py`, and lowerings through registration helpers in `lowering.py`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/fx](torch/fx/ADR.md) | depends-on | `GraphLowering` interprets `GraphModule` nodes and preserves node metadata for lowering and debug output |
| [torch/_dynamo](torch/_dynamo/ADR.md) | depended-on-by | Dynamo invokes Inductor as the default backend for captured FX graphs |
| [torch/_functorch](torch/_functorch/ADR.md) | depends-on | AOTAutograd partitions forward/backward graphs and supplies functionalized graphs to Inductor |
| [torch/_inductor/codegen](torch/_inductor/codegen/ADR.md) | depends-on | Scheduled IR groups become Triton, C++, wrapper, and AOTI code through backend codegen classes |
| [aten/src/ATen/native](aten/src/ATen/native/ADR.md) | depends-on | Lowerings target ATen operator schemas and fall back to eager/external ATen kernels when not fused |
| [torch/csrc/inductor](torch/csrc/inductor/ADR.md) | depended-on-by | AOTI and runtime support consume generated code and metadata |
| [c10/core](c10/core/ADR.md) | depends-on | IR and wrappers preserve dtype, device, layout, symbolic sizes, strides, and storage semantics |

## Runtime Behaviour

A Dynamo-produced FX graph enters `compile_fx.py`, which chooses decompositions, config patches, fake mode, cudagraph policy, and AOTAutograd integration before calling the inner compiler. `GraphLowering` subclasses `torch.fx.Interpreter`; it walks placeholders, calls, attributes, and outputs, maps each supported target through `lowering.py`, and appends `TensorBox`, `Buffer`, `Pointwise`, `Reduction`, view, and `ExternKernel` IR nodes to graph state. The scheduler computes read/write dependencies from IR, groups compatible nodes, decides fusion, plans buffer lifetimes, chooses backend scheduling by device, and asks codegen to emit kernels and wrappers. `codecache.py` then hashes graph structure, device properties, source code, shape environment, and compiler options to reuse compiled artifacts or build new Triton/C++ modules.

Fallbacks remain explicit in the IR. Unsupported or library-preferred operations become `ExternKernel` or fallback nodes, so generated wrappers can call ATen, cuBLAS, cuDNN, oneDNN, or other libraries between fused kernels while preserving dependency and mutation ordering.

## Performance Profile

Inductor's primary speedup comes from fusion: pointwise chains, reduction epilogues, layout conversions, and some templates combine multiple FX operations into fewer memory passes and fewer kernel launches. `scheduler.py` evaluates dependency legality, memory pressure, backend features, and device streams before fusing, while `select_algorithm.py` and template callers benchmark alternative tile shapes and libraries for compute-heavy kernels. Compile latency can be high because the backend runs decompositions, shape reasoning, scheduling, autotuning, source generation, compiler invocation, and cache writes; `codecache.py` and remote caches exist to amortize this cost across calls and processes. Runtime performance depends on cache hits, stable shapes, successful buffer reuse, good layout choices, and whether generated Triton/C++ kernels avoid falling back to eager ATen operations.

## Design Rationale

Inductor lowers FX to its own tensor IR because FX records Python-level graph structure but lacks explicit storage, layout, loop, and fusion information. The IR separates tensor views, storage boxes, buffers, pointwise loops, reductions, and extern kernels so the scheduler can reason about mutation, aliasing, memory reuse, and legal fusion. Scheduling is separate from code generation because the same dependency graph can target Triton, C++, MPS, XPU, MTIA, or AOT wrappers with different backend constraints. Source-code generation and caching make kernels inspectable, debuggable, and reusable while allowing existing compilers and Triton autotuners to perform final low-level optimization.
