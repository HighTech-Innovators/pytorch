# `torch/_functorch`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/_functorch` implements PyTorch's functional transforms and AOTAutograd compiler frontend, as mapped across book chapters 07 and 13. It powers `torch.func` transforms such as `vmap`, `grad`, `vjp`, `jvp`, `jacrev`, `jacfwd`, `hessian`, and `functionalize`, and it captures forward/backward graphs for compiler backends through AOTAutograd. The directory manages dynamic-layer interpreter state, batched tensors, grad/jvp nesting, functional tensor wrapping, alias/mutation analysis, graph partitioning, min-cut rematerialization, and compatibility shims for legacy `functorch` APIs.

## Key Files

| File | Purpose |
|---|---|
| `apis.py` | Dynamo-visible wrappers for `vmap`, `grad`, `grad_and_value`, and related `torch.func` entry points |
| `vmap.py` | Input/output dimension validation, BatchedTensor wrapping/unwrapping, nesting management, chunked vmap, and randomness policy |
| `eager_transforms.py` | Eager implementations for `vjp`, `jvp`, `jacrev`, `jacfwd`, `hessian`, `functionalize`, `grad_impl`, and `grad_and_value_impl` |
| `aot_autograd.py` | Public AOTAutograd entry points, AOT config wiring, fake-mode setup, metadata collection, and stage-one/stage-two graph capture/compile |
| `_aot_autograd/` | Decomposed AOTAutograd implementation: schemas, graph capture, graph compilation, wrappers, metadata, runtime wrappers, and cache support |
| `partitioners.py` | Forward/backward graph partitioning, min-cut rematerialization, activation checkpointing, RNG functionalization, and recomputation policy |
| `pyfunctorch.py` | Python dynamic-layer interpreter wrappers for vmap, grad, jvp, and functionalize dispatch through PyDispatcher |
| `make_functional.py` | Functional module wrappers that separate parameters/buffers from module state |
| `compile_utils.py` | FX graph utilities including common subexpression elimination and ATen target normalization |
| `compilers.py` | Reference compiler functions, debug compilers, decomposition defaults, and legacy fusion compiler hooks |

## Public Interface

The stable public APIs are exposed through `torch.func` and legacy `functorch`, not directly through this private package. `apis.py` supplies wrappers for `vmap`, `grad`, and `grad_and_value`; `eager_transforms.py` supplies `vjp`, `jvp`, `jacrev`, `jacfwd`, `hessian`, and `functionalize`; `aot_autograd.py` supplies `aot_function`, `aot_module`, `aot_module_simplified`, `compiled_function`, `compiled_module`, `make_boxed_func`, `make_boxed_compiler`, and AOT graph context helpers. Partitioners expose `default_partition`, `min_cut_rematerialization_partition`, and graph drawing helpers used by compiler integrations.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/fx](torch/fx/ADR.md) | depends-on | AOTAutograd captures, partitions, and returns FX `GraphModule` objects, and partitioners analyze FX nodes |
| [torch/_dynamo](torch/_dynamo/ADR.md) | depends-on | Dynamo traces selected transform wrappers and supplies graphs; Dynamo guards track functorch dynamic-layer state |
| [torch/_inductor](torch/_inductor/ADR.md) | depended-on-by | Inductor receives AOTAutograd forward/backward graphs and uses min-cut partitioning choices |
| [torch/autograd](torch/autograd/ADR.md) | depends-on | Grad transforms call `torch.autograd.grad`, manage forward-ad state, and compile backward graphs |
| [c10/core](c10/core/ADR.md) | depends-on | BatchedTensor, FunctionalTensor, tensor metadata, dispatch keys, and symbolic sizes flow through transform layers |
| [functorch/compile](functorch/compile/ADR.md) | depended-on-by | Legacy compile namespace re-exports AOTAutograd and partitioner APIs from this directory |
| [functorch/_src](functorch/_src/ADR.md) | depended-on-by | Legacy private shim modules re-export selected internals from this directory |

## Runtime Behaviour

`vmap` validates the input pytree and `in_dims`, computes a batch size, increments a vmap nesting level, wraps mapped tensor arguments with `_add_batch_dim`, runs the user function, and removes the batch dimension from outputs according to `out_dims`. Grad-family transforms increment grad or jvp nesting, wrap differentiable tensors, run the function under the correct grad-mode semantics, call autograd or forward-mode AD, and unwrap results back into the user's pytree structure. `functionalize` wraps tensors in functional tensor state, runs the function with mutation tracking, syncs and propagates input mutations when required, and returns mutation-free values to downstream compiler code.

AOTAutograd receives a Python function or module plus example inputs, processes parameters/buffers and fake tensors, captures a functional FX graph, analyzes input mutation, aliasing, saved tensors, RNG, and subclass metadata, then partitions the joint graph into compiled forward and backward graphs. The runtime wrapper runs the compiled forward graph, stores or recomputes needed values, invokes the compiled backward graph during autograd, and applies required epilogues such as copying updated input values back to mutated user tensors.

## Performance Profile

`vmap` improves performance when batching rules push an outer loop into individual operators, replacing Python loops and repeated dispatch with batched kernels. Its overhead comes from pytree flattening, dynamic-layer stack manipulation, BatchedTensor wrapping, and fallback paths for operators without efficient batching rules. AOTAutograd improves compiled training performance by functionalizing mutation, partitioning forward/backward graphs, eliminating dead code, and allowing Inductor to fuse both graphs; its compile-time cost comes from fake execution, alias/mutation metadata analysis, partitioner graph algorithms, and optional min-cut rematerialization. Min-cut partitioning trades memory for recomputation by selecting saved tensors based on graph cost estimates and activation-checkpoint annotations.

## Design Rationale

Function transforms live below `torch.func` because they must coordinate Python APIs with C++ dispatch keys and dynamic-layer interpreter stacks. Transform implementations wrap tensors rather than rewriting user code, so existing PyTorch operations invoke batching, grad, jvp, or functionalization rules through dispatch. AOTAutograd requires fully functional graphs because compiler backends optimize pure dataflow better than graphs with Python-visible mutations and aliases; wrapper epilogues preserve user semantics after compiled graph execution. Partitioning is pluggable because inference, training, memory-constrained training, and activation checkpointing need different save-versus-recompute policies.
