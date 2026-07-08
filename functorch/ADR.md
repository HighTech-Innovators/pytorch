# `functorch`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`functorch` owns the legacy top-level Python package for composable function transforms. It presents the public namespace for `vmap`, `grad`, Jacobian and Hessian helpers, and functional-module utilities while forwarding implementation into `torch._functorch` and sibling compatibility packages.

## Key Files

| File | Purpose |
|---|---|
| `README.md` | Explains the transform model, installation paths, and the supported APIs such as `vmap`, `grad`, `jacrev`, and `make_functional` |
| `__init__.py` | Re-exports deprecated transform entry points and utility types from `torch._functorch` |
| `compile/__init__.py` | Re-exports AOTAutograd and compiler helpers from `torch._functorch` through the legacy compile namespace |
| `writing_batching_rules.md` | Documents the batching-rule authoring model that underpins `vmap` support |
| `examples` | Contains runnable examples that exercise the public transform surface |

## Public Interface

`functorch.__init__.py` exports `vmap`, `grad`, `grad_and_value`, `vjp`, `jvp`, `jacrev`, `jacfwd`, `hessian`, `functionalize`, `combine_state_for_ensemble`, `make_functional`, `make_functional_with_buffers`, `FunctionalModule`, `FunctionalModuleWithBuffers`, and `make_fx`. The package also exposes `functorch.compile` as a companion namespace for `aot_function`, `aot_module`, `memory_efficient_fusion`, `default_partition`, and other compiler helpers.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/_functorch](torch/_functorch/ADR.md) | depends-on | `__init__.py` and `compile/__init__.py` re-export their implementations from `torch._functorch` |
| [functorch/_src](functorch/_src/ADR.md) | mutual | Both directories preserve legacy functorch import paths after implementation moved into `torch._functorch` |
| [functorch/compile](functorch/compile/ADR.md) | depended-on-by | The top-level package is the namespace users pair with the legacy compile shim when combining transforms and AOTAutograd |

## Runtime Behaviour

Importing `functorch` executes `__init__.py`, binds deprecated transform names from `torch._functorch.deprecated`, binds `FunctionalModule` and `FunctionalModuleWithBuffers` from `torch._functorch.make_functional`, binds `make_fx` from `torch._functorch.python_key`, and sets `__version__ = torch.__version__`. The package itself does not implement transforms; every public call enters `torch._functorch` immediately after the import binding.

`README.md` describes the supported transform combinations and the intended execution model. It shows `vmap(torch.sin)`, nested `grad(grad(...))`, `vmap(grad(...))` for per-sample gradients, and the relationship between eager transforms and ahead-of-time tracing through `make_fx` and AOTAutograd.

## Performance Profile

- **Allocation sites** - The top-level package allocates only module bindings, while real transform allocations happen in `torch._functorch` when batching layers, FX graphs, or functional parameter tuples are created.
- **Synchronization costs** - The namespace adds no extra locking or dispatch layers of its own; synchronization behavior belongs to the underlying autograd, compiler, and backend code reached through the re-exported functions.
- **Data movement** - `make_functional` and `make_functional_with_buffers` expose model parameters and buffers as explicit Python tuples, so transform pipelines can move module state through pure functions instead of mutating modules in place.
- **Redundant or repeated work** - Keeping `functorch` as a thin re-export layer avoids duplicating transform implementations across `functorch` and `torch._functorch`, and it lets legacy imports reuse the same optimized execution paths.

## Design Rationale

PyTorch keeps this package because a large user and test surface still imports `functorch` directly even after transforms moved into core PyTorch. A thin compatibility namespace preserves those imports, keeps migration incremental, and makes `torch._functorch` the single implementation source for transform semantics and performance work.
