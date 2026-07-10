# `functorch`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`functorch` is a compatibility shim that re-exports the function-transform APIs (`vmap`, `grad`, `jvp`, `vjp`, `jacfwd`, `jacrev`) from their canonical implementation in `torch._functorch`. It exists to preserve backward compatibility with code that imported from `functorch` before the functionality was merged into `torch._functorch`.

## Key Files

| File | Purpose |
|---|---|
| `functorch/__init__.py` | Re-exports `vmap`, `grad`, `grad_and_value`, `jvp`, `vjp`, `jacfwd`, `jacrev`, `hessian`, `functionalize`, `make_functional`, `make_functional_with_buffers` from `torch._functorch.deprecated` |
| `functorch/compile/__init__.py` | Re-exports AOT autograd and compiler utilities (`aot_function`, `aot_module`, `make_fx`, `min_cut_rematerialization_partition`, etc.) from `torch._functorch` |
| `functorch/_src/` | Minimal source retained for compatibility; core implementation lives in `torch/_functorch/` |

## Public Interface

`functorch.vmap`, `functorch.grad`, `functorch.grad_and_value`, `functorch.jvp`, `functorch.vjp`, `functorch.jacfwd`, `functorch.jacrev`, `functorch.hessian`, `functorch.functionalize`, `functorch.make_functional`, `functorch.compile.aot_function`, `functorch.compile.make_fx`. All delegate immediately to `torch._functorch`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| `torch._functorch` | depends-on | All functional-transform implementations; `functorch` is a thin re-export layer over this internal package |
| `torch._C` | depends-on | Dispatch hooks for `vmap`/`grad` are registered in the C extension |
| User code | depended-on-by | Code importing from the pre-merge `functorch` package |

## Runtime Behaviour

Importing `functorch` triggers `torch._functorch.deprecated` imports, which issue `UserWarning` deprecation notices directing users to `torch.func` (the canonical public API). The function objects themselves are identical to those in `torch._functorch`; no extra wrapping occurs after the import-time deprecation warning. `vmap` and `grad` work by pushing a transform-level dispatch key onto `torch._C`'s dispatch key stack; batch/grad dimensions are tracked through the ATen dispatch system without materializing intermediate tensors.

## Performance Profile

`functorch` re-exports add one Python attribute lookup per call (the import redirect); all computation runs in `torch._functorch`. `vmap` overhead is proportional to the number of batched dimensions: each vectorized dimension adds one dispatch-key stack push/pop per operator call. `grad` has the same overhead as `torch.autograd.grad` since it uses the same autograd engine. No allocation occurs in the `functorch` shim itself.

## Design Rationale

`functorch` was merged into PyTorch as `torch._functorch` and surfaced publicly as `torch.func`. The `functorch` package is retained as a compatibility shim with deprecation warnings so existing code continues to work without changes. Keeping the shim as a separate top-level package (rather than a `torch.functorch` submodule) preserves backward compatibility for code that checks `import functorch` separately from `import torch`.
