# `torch/_library`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/_library` owns the Python infrastructure behind `torch.library`, including custom operator definition, fake implementation registration, autograd bridging, and small side registries that do not live in dispatcher tables. It is the supported replacement for the older `torch._custom_op` stack.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Imports the core registry modules and re-exports helpers such as `register_fake_class`, `triton_op`, and `wrap_triton`. |
| `custom_ops.py` | Implements `torch.library.custom_op`, `CustomOpDef`, backend kernel registration, fake registration, autograd registration, vmap registration, and dispatcher setup. |
| `autograd.py` | Builds generated autograd wrappers with `make_autograd_impl()` and tensor-list support shims. |
| `fake_impl.py` | Implements `FakeImplHolder`, `FakeImplCtx`, symbolic dynamic-size creation, and Meta-kernel wrapping. |
| `simple_registry.py` | Stores per-op fake impls, `__torch_dispatch__` rules, effect markers, and symmetric-memory metadata outside dispatcher tables. |

## Public Interface

The supported operator entry point is `torch.library.custom_op()`, which returns a `CustomOpDef`. `CustomOpDef` exposes `register_kernel()`, `set_kernel_enabled()`, `register_fake()`, `register_effect()`, `register_torch_dispatch()`, `register_autograd()`, `register_vmap()`, and `register_autocast()`. Other public hooks include `torch.library.get_ctx()` for fake kernels, `register_fake_class()`, `capture_triton()`, `triton_op()`, `wrap_triton()`, and the registry objects `SimpleLibraryRegistry`, `SimpleOperatorEntry`, and `singleton`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/autograd](torch/autograd/ADR.md) | depends-on | `autograd.py` dynamically creates `torch.autograd.Function` subclasses and uses `ctx.save_for_backward()` compatible setup hooks. |
| [torch/_subclasses](torch/_subclasses/ADR.md) | depends-on | `fake_impl.py` raises `DynamicOutputShapeException` for unsupported fake kernels and allocates symbolic sizes that FakeTensor consumes during tracing. |
| [torch/_higher_order_ops](torch/_higher_order_ops/ADR.md) | depended-on-by | `effects.py` and HOP tracing consume `simple_registry` effect metadata and `CustomOpDef` schema information. |
| [torch/_custom_op](torch/_custom_op/ADR.md) | depended-on-by | The deprecated custom-op API forwards schema inference, meta behavior, and dispatcher registration into this directory. |

## Runtime Behaviour

`custom_op()` in `custom_ops.py` infers or accepts a schema string, constructs a `CustomOpDef`, and immediately calls `_register_to_dispatcher()`, which parses the schema, defines the dispatcher op, and installs fake, autograd, and ADInplaceOrView handlers. `CustomOpDef.register_kernel()` stores per-device callables in `_backend_fns`, wraps them with contract checks for inplace and out variants, and uses `Library.impl()` or a generated `BackendSelect` router to reach the right backend at runtime.

`register_fake()` stores an abstract implementation on the op definition, while `_register_fake_dispatcher_impl()` installs a Meta kernel that either calls that fake implementation or falls back to `generate_trivial_fake_impl()` for simple cases. `register_autograd()` saves `backward` and `setup_context` callables, and `_register_autograd_dispatcher_impl()` hands them to `autograd.make_autograd_impl()`, which creates a synthetic `autograd.Function` that redispatches below autograd on the forward path and validates backward arity on the reverse path.

`SimpleLibraryRegistry.find()` lazily creates a `SimpleOperatorEntry` for each qualified op name, and that entry keeps holders for fake impls, `__torch_dispatch__` rules, effect markers, and symmetric-memory argument names. `FakeImplHolder.register()` appends the newest fake kernel, registers a dispatcher Meta kernel, and exposes `FakeImplCtx.new_dynamic_size()` so data-dependent output shapes can be represented symbolically.

## Performance Profile

- **Allocation sites** - Registration allocates dispatcher libraries, parsed schemas, generated autograd classes, registration handles, and per-op holder objects, but those costs are paid once when the custom op is defined.
- **Synchronization costs** - The directory itself does not synchronize devices; its hot runtime paths stay inside dispatcher redispatch, version-counter bumps in `_register_mutation_version_bump()`, and Python validation around fake or autograd hooks.
- **Data movement** - Fake kernels avoid touching tensor data, while autograd wrappers flatten list inputs and outputs with `_pytree` only when a traced backward path needs tensor-list support.
- **Redundant or repeated work** - `SimpleLibraryRegistry` keeps fake, effect, and torch-dispatch registrations outside dispatcher tables so lookups are direct dictionary reads, and `register_kernel()` can temporarily disable a backend by editing `_disabled_kernel` instead of tearing down and rebuilding libraries.

## Design Rationale

`torch.library` needs richer semantics than raw dispatch keys alone can represent, especially for fake execution, `__torch_dispatch__` rules, and effect metadata. This directory centralizes those higher-level registrations while still lowering every executable path into the dispatcher, so the supported API stays ergonomic without splitting runtime semantics across multiple subsystems.
