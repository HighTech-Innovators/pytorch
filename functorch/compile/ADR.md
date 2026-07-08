# `functorch/compile`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`functorch/compile` is the legacy public compile namespace for functorch's AOTAutograd and graph-partitioning APIs, mapped to book chapters 07 and 13. The directory contains a single shim module that re-exports implementation from `torch/_functorch`; it does not implement compilation itself. It preserves import compatibility for code that still calls `functorch.compile.aot_function`, `aot_module`, `make_boxed_compiler`, `min_cut_rematerialization_partition`, `memory_efficient_fusion`, or related helper compilers.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Re-export surface for AOTAutograd, compiler helpers, decomposition utilities, minifier, partitioners, and Python-key decomposition |
| `torch/_functorch/aot_autograd.py` | Real implementation of `aot_function`, `aot_module`, boxed compiler helpers, graph capture, and AOTAutograd wrapper construction |
| `torch/_functorch/compilers.py` | Real implementation of legacy compiler helper functions such as `debug_compile`, `print_compile`, `ts_compile`, `nnc_jit`, and `memory_efficient_fusion` |
| `torch/_functorch/partitioners.py` | Real implementation of `default_partition`, `draw_graph`, and `min_cut_rematerialization_partition` |
| `torch/_functorch/fx_minifier.py` | Real implementation of the minifier re-exported as `minifier` |
| `torch/_functorch/python_key.py` | Real implementation of `pythonkey_decompose` used by this namespace |

## Public Interface

`functorch.compile` exposes `config`, `aot_function`, `aot_module`, `aot_module_simplified`, `compiled_function`, `compiled_module`, `get_aot_compilation_context`, `get_aot_graph_name`, `get_graph_being_compiled`, `make_boxed_compiler`, `make_boxed_func`, `debug_compile`, `default_decompositions`, `draw_graph_compile`, `memory_efficient_fusion`, `nnc_jit`, `nop`, `print_compile`, `ts_compile`, `minifier`, `default_partition`, `draw_graph`, `min_cut_rematerialization_partition`, and `pythonkey_decompose`. The names are imported directly from `torch._functorch` modules and therefore share their behavior, signatures, and deprecation status.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/_functorch](torch/_functorch/ADR.md) | depends-on | All exported names come from AOTAutograd, compiler, partitioner, minifier, and Python-key modules in `torch._functorch` |
| [torch/fx](torch/fx/ADR.md) | depended-on-by | Re-exported partitioners and compiler helpers operate on FX `GraphModule` objects |
| [torch/_inductor](torch/_inductor/ADR.md) | depended-on-by | Inductor imports `min_cut_rematerialization_partition` from this compatibility namespace in `compile_fx.py` |
| [functorch/_src](functorch/_src/ADR.md) | related | Both directories preserve legacy functorch import paths by forwarding to `torch._functorch` |

## Runtime Behaviour

Importing `functorch.compile` executes `__init__.py` and binds names from `torch._functorch.aot_autograd`, `torch._functorch.compilers`, `torch._functorch.fx_minifier`, `torch._functorch.partitioners`, and `torch._functorch.python_key`. Calling any exported function immediately enters the `torch._functorch` implementation; no wrapper logic, argument rewriting, caching, or tracing code runs in this directory. A call such as `functorch.compile.aot_function(fn, fw_compiler, bw_compiler)` therefore follows the AOTAutograd path that captures a functional FX graph, partitions it, compiles forward/backward graphs, and returns the compiled callable.

The namespace remains observable because downstream code imports from it. `torch/_inductor/compile_fx.py` imports `min_cut_rematerialization_partition` from `functorch.compile`, so this shim still participates in compiler startup and must keep the legacy name stable.

## Performance Profile

The shim adds only Python import-time binding and one module lookup; it has no measurable runtime cost relative to AOTAutograd capture, partitioning, compilation, or generated-kernel execution. Performance behavior of functions imported from this namespace belongs to `torch/_functorch`: graph capture and partitioning cost compile time, while compiled forward/backward graphs can reduce Python overhead and enable Inductor fusion. Keeping the shim thin avoids duplicate dispatch layers and prevents legacy imports from adding extra wrappers around hot compiler functions.

## Design Rationale

The directory exists for compatibility after functorch functionality moved under `torch._functorch` and `torch.func`. Re-exporting preserves older user and internal import paths while allowing one implementation to serve both the legacy and modern APIs. The shim structure also makes deprecation and migration explicit: new implementation work belongs in `torch/_functorch`, while `functorch/compile` should stay a stable forwarding layer. A single forwarding file prevents semantic drift between `functorch.compile` and the compiler stack used by Dynamo, AOTAutograd, and Inductor.
