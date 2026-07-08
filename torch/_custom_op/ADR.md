# `torch/_custom_op`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/_custom_op` owns the deprecated Python custom-operator API that predates `torch.library.custom_op`. It validates schemas, defines dispatcher entries, and layers Python autograd and meta registrations on top of `torch.library`.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Re-exports `custom_op`, `CustomOp`, and `get_ctx` from the implementation modules. |
| `impl.py` | Defines `custom_op`, `CustomOp`, schema validation helpers, dispatcher registration, and deprecated backend, meta, and autograd registration methods. |
| `autograd.py` | Builds the generated `torch.autograd.Function` bridge used by `impl_backward()` and `impl_save_for_backward()`. |

## Public Interface

The main entry points are `custom_op()`, `CustomOp`, and `get_ctx()`. `CustomOp` exposes `impl()`, `impl_factory()`, `impl_abstract()`, `impl_save_for_backward()`, and `impl_backward()` for registering device kernels, factory kernels, meta kernels, and autograd formulas. Helper hooks such as `custom_op_from_existing()`, `get_op()`, and `get_abstract_impl()` let other subsystems recover dispatcher state from a qualified operator name.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/_library](torch/_library/ADR.md) | depends-on | `impl.py` uses `torch.library.Library`, `library.impl`, `infer_schema`, and `torch._library.fake_impl.set_ctx_getter` to define dispatcher and Meta behavior. |
| [torch/autograd](torch/autograd/ADR.md) | depends-on | `autograd.py` generates `torch.autograd.Function` subclasses in `gen_autograd_function()` and uses `ctx.save_for_backward()` semantics in `construct_autograd_kernel()`. |
| [torch/_subclasses](torch/_subclasses/ADR.md) | depended-on-by | `impl.py` keeps `global_registry` alive specifically so FakeTensor and related tracing paths can discover custom-op registrations by qualified name. |

## Runtime Behaviour

`custom_op()` in `impl.py` parses the qualified name with `parse_qualname()`, infers or validates the schema with `infer_schema()` and `validate_function_matches_schema()`, defines the dispatcher schema through `library.Library.define()`, and returns a `CustomOp` wrapper around the resulting operator handle. `CustomOp.__call__()` bypasses `torch.ops` caching and invokes `_C._dispatch_call_boxed()` directly on the saved `_DispatchOperatorHandle`.

Device registrations call `CustomOp.impl()` or `impl_factory()`, which record Python source locations in `_register_impl()` and then install dispatcher kernels with `library.impl()`. Autograd registrations are staged through `impl_save_for_backward()` and `impl_backward()`, and once both pieces exist `_register_autograd_kernel()` calls `construct_autograd_kernel()` to synthesize a `torch.autograd.Function` based backward path.

## Performance Profile

- **Allocation sites** - `construct_autograd_kernel()` flattens pytrees, allocates namedtuple wrappers with `namedtuple_args_cls()`, and instantiates a generated `torch.autograd.Function` class for each explicit autograd call path.
- **Synchronization costs** - The module itself does not synchronize devices, but `autograd_not_implemented()` branches on `torch.is_grad_enabled()` and tensor `requires_grad` flags on every Autograd dispatch before redispatching below autograd.
- **Data movement** - `CustomOp.__call__()` avoids extra Python namespace lookups by calling the boxed dispatcher handle directly, but `mark_non_differentiable()` and backward result normalization still walk output and gradient pytrees in Python.
- **Redundant or repeated work** - Schema parsing and validation happen once at registration time, while steady-state calls pay only the boxed dispatch and optional autograd indirection branch installed by `autograd_kernel_indirection()`.

## Design Rationale

The directory keeps the old API alive by translating every user-visible concept into the newer `torch.library` machinery instead of maintaining a second dispatcher stack. The split between `impl.py` and `autograd.py` preserves a clear boundary between schema and dispatcher setup on one side and generated backward plumbing on the other.
