# `torch/_prims`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/_prims` defines PrimTorch primitive operations, their metadata functions, ATen implementations, and tracing helpers. These primitives form a small operator vocabulary that references, decompositions, and compilers use when lowering PyTorch programs.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Creates `prims` libraries, defines `TensorMeta`, `_make_prim()`, elementwise, view, shape, reduction, creation, and linear algebra primitives |
| `rng_prims.py` | Registers `rngprims` custom ops such as `philox_rand` and higher-order RNG state operators |
| `context.py` | Defines `torch_to_refs_map()`, `all_prims()`, and `TorchRefsMode` for redirecting torch APIs to references and decompositions |
| `debug_prims.py` | Registers `debugprims::load_tensor` and `load_tensor_reader()` for loading or synthesizing debug tensors |
| `executor.py` | Provides `execute()` and `make_traced()` for tracing functions through refs and executing resulting FX graphs |

## Public Interface

| Symbol | Description |
|---|---|
| `TensorMeta()` | Constructs tensor metadata using `torch.empty_strided()` from a tensor, number, or explicit shape, strides, dtype, and device |
| `_make_prim()` | Registers a primitive schema, fake/meta implementation, ATen implementation, autograd error path, tags, and return type |
| `ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND` | Enum controlling elementwise metadata dtype promotion for default, int-to-float, bool, and complex-to-real cases |
| `abs`, `add`, `broadcast_in_dim`, `as_strided`, `reshape`, `cat`, `sum`, `empty`, `full` | Representative primitive operations exported through `__all__` |
| `register_rng_prims()` / `register_philox_rand()` | Registers stateless Philox RNG primitives and RNG higher-order operations |
| `TorchRefsMode` | `TorchFunctionMode` that redirects torch APIs and Tensor methods to `torch._refs` or `torch._decomp.decomposition_table` |
| `execute()` / `make_traced()` | Prototype execution and tracing helpers built on FX `make_fx` |

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch](torch/ADR.md) | depends-on | Uses `torch.library`, `torch._ops`, tensors, devices, fake tensors, and default device helpers |
| [aten/src/ATen](aten/src/ATen/ADR.md) | depends-on | Primitive ATen implementations call existing tensor operators such as `torch.as_strided`, `torch.rand`, `torch.bmm`, and creation ops |
| [torch/_decomp](torch/_decomp/ADR.md) | depended-on-by | Decomposition rules lower ATen operators into primitive operations |
| [torch/fx](torch/fx/ADR.md) | depends-on | `executor.py` traces with `make_fx` and executes FX `GraphModule` instances |
| [torch/_dynamo](torch/_dynamo/ADR.md) | depended-on-by | Dynamo and compiler pipelines use primitive and reference execution modes while tracing |

## Runtime Behaviour

`_make_prim()` parses a schema, registers either an old-style `torch.library.Library` op or a `torch.library.custom_op`, attaches fake/meta implementations, and installs an autograd implementation that raises through `backwards_not_supported()`. `_prim_impl()` always runs the meta function before the ATen implementation so primitives reject invalid broadcasting, dtype, device, and shape combinations that a broader ATen op might accept. `TorchRefsMode.__torch_function__()` lets primitive calls pass through, maps torch APIs to `torch._refs`, and falls back to `torch._decomp.decomposition_table` for `OpOverload` and `OpOverloadPacket` objects.

## Performance Profile

Primitive execution adds Python-level metadata validation before calling the ATen implementation, which improves compiler correctness and costs extra work in eager execution. Metadata functions such as `_prim_elementwise_meta()`, `_as_strided_meta()`, and `_broadcast_in_dim_meta()` compute output shapes, strides, devices, and dtypes without running device kernels. RNG primitives use `torch.random.fork_rng()` and `CUDARngStateHelper.set_torch_state_tensor()` to produce deterministic Philox output, which trades runtime state manipulation for functional graph replay. `make_traced()` performs a fresh `make_fx` trace on every call and explicitly leaves caching as future work.

## Design Rationale

PrimTorch primitives give PyTorch compilers a small, explicit, and meta-aware operation set instead of forcing every backend to understand the full ATen surface. Registering primitives through `torch.library` keeps them visible to dispatch, fake tensor mode, and tracing while allowing Python to define metadata quickly. The separation between refs, decompositions, and prims lets high-level torch APIs lower step by step: public API to reference formula, reference formula to decomposed ATen, and decomposed ATen to primitives.
