# `torch/_decomp`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/_decomp` owns Python decompositions that rewrite selected `torch.ops.aten` operators into simpler ATen, reference, primitive, or functional RNG operations. Export, compilation, forward-mode AD, and backend tracing use these tables to replace complex or backend-specific operators with lower-level formulas.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Defines decomposition registries, `register_decomposition()`, `get_decompositions()`, `remove_decompositions()`, and core ATen decomposition selection |
| `decompositions.py` | Registers a large set of post-autograd decompositions such as `tanh_backward`, `sigmoid_backward`, `hardswish`, `baddbmm`, and scaled-dot-product attention helpers |
| `decompositions_for_jvp.py` | Registers and scripts decompositions used by forward-mode AD and JVP fallback paths |
| `decompositions_for_rng.py` | Defines RNG decompositions, `PhiloxState`, and `PhiloxStateTracker` for functionalizing CUDA Philox random operators |

## Public Interface

| Symbol | Description |
|---|---|
| `decomposition_table` | Global post-autograd mapping from `OperatorBase` keys to Python decomposition functions |
| `pre_autograd_decomposition_table` | Registry for pre-autograd decompositions |
| `meta_table` | Registry for meta decompositions |
| `register_decomposition()` | Decorator that converts `out` parameters, expands overload packets, and inserts functions into the selected registry |
| `get_decompositions()` | Returns registered decompositions for requested overloads or overload packets |
| `remove_decompositions()` | Removes selected overloads from an existing decomposition dictionary |
| `core_aten_decompositions()` | Returns the default core ATen decomposition table used by export utilities |
| `PhiloxStateTracker` | Tracks seed and offset tensors while RNG decompositions functionalize CUDA random operations |

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [aten/src/ATen](aten/src/ATen/ADR.md) | depends-on | Uses `torch.ops.aten` overloads as decomposition keys and implements replacements with ATen operators |
| [torch/_prims](torch/_prims/ADR.md) | depends-on | Decompositions import `torch._prims as prims` for primitive building blocks |
| [torch/_export](torch/_export/ADR.md) | depended-on-by | Export uses decomposition tables to produce a stable reduced operator set |
| [torch/_dynamo](torch/_dynamo/ADR.md) | depended-on-by | Dynamo and AOT tracing consume decompositions while building compiler graphs |
| [torch/jit](torch/jit/ADR.md) | depends-on | `decompositions_for_jvp.py` scripts or ignores decomposition functions and registers JIT graphs through `torch.jit._register_decomposition` |

## Runtime Behaviour

Importing `torch._decomp` creates `global_decomposition_table`, exposes three registry views, and then imports `torch._decomp.decompositions` and `torch._refs` to populate entries. `register_decomposition()` validates the requested type, wraps safe functions with `_convert_out_params()`, and inserts every overload from an `OpOverloadPacket` through `_add_op_to_registry()`. `decompositions_for_rng.py` rewrites CUDA random operations such as `aten.rand` and `aten.rand_like` to `torch.ops.rngprims.philox_rand` while `PhiloxStateTracker` advances a functional offset.

## Performance Profile

Registry lookup is dictionary-based, so selecting a decomposition is cheap compared with executing the replacement graph. A decomposition can expand one operator into many tensor operations, which improves backend portability but can increase graph size and compile time. Many formulas use wrappers such as `out_wrapper` and `pw_cast_for_opmath`, adding explicit dtype, output, and opmath behavior so compiled graphs match eager kernels. RNG decompositions compute and return Philox offsets, which adds bookkeeping but makes random traces replayable and shape-dependent.

## Design Rationale

PyTorch keeps decompositions in Python so compiler and export paths can iterate on operator lowering without changing C++ kernels. The registry accepts overload packets and higher-order operators because compiler frontends work at operator granularity, not just Python function granularity. Separate files for general, JVP, and RNG decompositions keep normal lowering rules distinct from forward-mode AD hacks and functional random-state handling.
