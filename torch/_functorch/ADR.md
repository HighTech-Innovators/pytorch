# `torch/_functorch`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/_functorch` implements PyTorch's composable function transforms: `vmap` (vectorised map), `grad`, `vjp`, `jvp`, `jacrev`, `jacfwd`, and AOT autograd. It is the internal implementation backing `torch.func`, `functorch`, and the AOT compilation pipeline used by TorchInductor.

## Key Files

| File | Purpose |
|---|---|
| `eager_transforms.py` | `grad`, `vjp`, `jvp`, `vmap`, `jacrev`, `jacfwd`, `linearize`, `hessian` — eager-mode functional transforms; uses `torch._C._functorch` C-extension for nesting level management |
| `aot_autograd.py` | `aot_function`, `aot_module`, `aot_module_simplified` — AOT (Ahead-of-Time) autograd: traces the joint forward+backward graph at compile time; called by TorchInductor's `compile_fx` |
| `_aot_autograd/` | AOT autograd internal modules: `autograd_cache.py` (persistent cache), `subclass_parametrization.py` (tensor subclass handling), `runtime_wrappers.py` (runtime dispatch logic) |
| `partitioners.py` | `default_partition`, `min_cut_rematerialization_partition` — partition the joint graph into forward and backward halves; minimises recomputed activations |
| `apis.py` | `vmap` public entry; `_wraps_without_dynamo_attrs` decorator |
| `autograd_function.py` | `custom_function_call` — integrates custom `torch.autograd.Function` subclasses with functorch transforms |
| `compilers.py` | Debugging compilers: `nop`, `print_compile`, `draw_graph_compile`, `ts_compile` |
| `config.py` | Flags: `max_dist_from_bw`, `debug_assert`, `enable_autograd_cache` |
| `_activation_checkpointing/` | Activation checkpointing via `torch.utils.checkpoint`; integrated with AOT autograd |
| `functional_call.py` | `functional_call(module, params, args)` — calls an `nn.Module` with externally-supplied parameters, enabling `grad` to differentiate through module weights |

## Public Interface

| Symbol | Description |
|---|---|
| `torch.func.vmap(fn, in_dims, out_dims, chunk_size)` | Vectorises `fn` over a batch dimension; backed by `eager_transforms.vmap` |
| `torch.func.grad(fn, argnums)` | Returns the gradient of a scalar-valued function |
| `torch.func.vjp(fn, *primals)` | Returns `(output, vjp_fn)` for vector-Jacobian products |
| `torch.func.jvp(fn, primals, tangents)` | Returns `(output, tangents_out)` for Jacobian-vector products |
| `torch.func.jacrev(fn, argnums)` | Jacobian via reverse-mode AD |
| `torch.func.functional_call(module, params, args)` | Stateless module call with external params |
| `torch._functorch.aot_autograd.aot_module_simplified` | Traces the forward+backward as a joint FX graph; called by `compile_fx` |
| `torch._functorch.partitioners.min_cut_rematerialization_partition` | Partitions joint graph into fwd/bwd halves; used by `compile_fx` |

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| `torch._C._functorch` | depends-on | C-extension: `_grad_increment_nesting`, `_grad_decrement_nesting`, `_vmap_increment_nesting`, `_wrap_for_grad`, `is_functorch_wrapped_tensor` — manages the transform nesting stack |
| [torch/fx](torch/fx/ADR.md) | depends-on | `make_fx` in `torch.fx.experimental.proxy_tensor` is used by AOT autograd to trace the joint graph |
| [torch/autograd](torch/autograd/ADR.md) | depends-on | `grad`, `vjp` internally call `torch.autograd.grad`; forward AD uses `torch.autograd.forward_ad` |
| [torch/_dynamo](torch/_dynamo/ADR.md) | depended-on-by | Dynamo calls `aot_autograd` after extracting an FX graph; `compiled_autograd` uses functorch to trace the backward |
| [torch/_inductor](torch/_inductor/ADR.md) | depended-on-by | `compile_fx` calls `aot_module_simplified` and `min_cut_rematerialization_partition` |
| `torch.utils._pytree` | depends-on | `tree_flatten`, `tree_map`, `tree_unflatten` used pervasively for flattening function inputs/outputs |

## Runtime Behaviour

`vmap(fn)(*inputs)` calls `torch._C._functorch._vmap_increment_nesting` to push a new vmap level onto the transform stack, wraps each input tensor in a `BatchedTensor` (a C++ `TensorImpl` subclass that adds a batch dimension), calls `fn` with the wrapped inputs (which dispatches through the `FuncTorchBatched` dispatch key), then calls `_vmap_decrement_nesting` and unwraps the output. `grad(fn)(x)` works similarly: increments the grad-transform nesting level, wraps inputs in `GradTrackingTensor`, runs `fn`, triggers a gradient computation through the transform's mini-autograd engine, and returns the gradient. Composing transforms (e.g., `vmap(grad(fn))`) stacks nesting levels; each level's dispatch key is consulted in sequence during ATen dispatch. AOT autograd's `aot_module_simplified` traces both forward and backward in a single `make_fx` call with `fake_mode` tensors, producing a joint FX graph that Inductor can lower end-to-end.

## Performance Profile

- **Allocation sites**: each `vmap` call allocates `BatchedTensor` wrappers for all inputs — one per tensor per call. For batches of small tensors this wrapper overhead can dominate the compute time.
- **Synchronization costs**: transform nesting level management (`_grad_increment_nesting` etc.) is a C++ thread-local integer increment — essentially free. AOT autograd's `make_fx` trace is a one-time cost at compilation.
- **Data movement**: `vmap` with `chunk_size` processes the batch in sub-batches to bound peak memory; each chunk requires loading the corresponding slice into the batch dimension — proportional I/O to the chunk size.
- **Redundant or repeated work**: `min_cut_rematerialization_partition` performs a min-cut computation on the joint graph to determine which activations to save vs. recompute; for large graphs this is an NP-approximation problem. The partition is computed once per graph shape and cached.

## Design Rationale

Functional transforms use the dispatch key system rather than Python-level wrapping because ATen operations must be interceptable at the C++ level for correct handling of in-place ops, views, and custom kernels. Separate nesting levels for each transform type (grad, vmap, jvp) allow them to be safely composed in any order. AOT autograd's joint-graph approach (tracing forward and backward together) enables Inductor to fuse backward computation with forward without re-running Python code, which is the key enabler for `torch.compile`'s backward performance.
