# `torch/numa`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/numa` owns NUMA-aware CPU-affinity utilities for multi-device jobs. It computes CPU sets near accelerator devices, wraps launch commands with `numactl`, and can bind all threads in the current process before work starts.

## Key Files

| File | Purpose |
|---|---|
| `binding.py` | Defines `AffinityMode`, `NumaOptions`, command wrapping, process binding, topology discovery, and error handling for NUMA placement |
| `__init__.py` | Marks the NUMA package root |

## Public Interface

`AffinityMode` and `NumaOptions` are the exported public types. Operational entry points visible in source are `_maybe_wrap_command_args_with_numa_binding`, `_maybe_wrap_with_numa_binding`, `_maybe_apply_numa_binding_to_current_process`, `_assemble_numactl_command_args`, and the topology helpers `_get_validated_logical_cpus_to_bind_to`, `_get_numa_node_index_for_device_index`, and `_bind_all_threads_in_current_process_to_logical_cpus`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/distributed](torch/distributed/ADR.md) | depended-on-by | the module-level docstring explicitly targets `torchrun` and `elastic_launch` integration |
| [torch/xpu](torch/xpu/ADR.md) | depends-on | binding logic maps local ranks to accelerator-local NUMA nodes by querying device counts and device-to-node relationships |

## Runtime Behaviour

`_maybe_wrap_command_args_with_numa_binding()` exits early when `numa_options` is `None`, otherwise it computes the allowed logical CPUs for a `device_index`, builds a `numactl --physcpubind=...` prefix, and emits a `signpost_event` on success. `_maybe_wrap_with_numa_binding()` returns a decorator that calls `_maybe_apply_numa_binding_to_current_process()` before invoking the wrapped function, which is how in-process launchers apply affinity. CPU selection depends on `AffinityMode`: helper functions such as `_node_get_logical_cpus_to_bind_to`, `_socket_get_logical_cpus_to_bind_to`, `_exclusive_get_logical_cpus_to_bind_to`, and `_core_complex_get_logical_cpus_to_bind_to` choose different locality and exclusivity policies. Any exception flows through `_handle_exception()`, which logs structured telemetry and either re-raises or falls back based on `NumaOptions.should_fall_back_if_binding_fails`.

## Performance Profile

NUMA binding spends its cost at startup, not on each training step: topology discovery, CPU-range parsing, and command wrapping all happen before the workload begins. The code is optimized for locality rather than raw launch speed, and its own docstring notes that better locality can improve end-to-end performance when workers stay near their accelerator's memory controllers. Exclusive and core-complex modes trade more topology analysis for lower contention on shared cores and last-level cache. The structured `signpost_event` logging path adds only modest Python overhead and helps diagnose binding failures without instrumenting the steady-state compute path.

## Design Rationale

NUMA policy lives in a dedicated package because placement logic is orthogonal to model code but still needs PyTorch-aware device mapping. Representing policies as `AffinityMode` values plus `NumaOptions` keeps the launcher API explicit while allowing the implementation to evolve as machine topologies change.
