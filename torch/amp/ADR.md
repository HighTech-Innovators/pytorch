# `torch/amp`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/amp` provides the device-generic automatic mixed precision interface for PyTorch training and inference. It owns the public `autocast` context manager, custom autograd decorators, and `GradScaler` dynamic loss-scaling state machine that coordinate Python optimizers with C++ autocast and AMP kernels.

## Key Files

| File | Purpose |
|---|---|
| `torch/amp/__init__.py` | Re-exports `autocast`, `custom_fwd`, `custom_bwd`, `is_autocast_available`, `_enter_autocast`, `_exit_autocast`, and `GradScaler` |
| `torch/amp/autocast_mode.py` | Implements `autocast`, `_UnmanagedAutocast`, tracing helpers, dtype validation, input casting, and custom autograd decorators |
| `torch/amp/grad_scaler.py` | Implements `GradScaler`, `_MultiDeviceReplicator`, `OptState`, gradient unscale/check logic, optimizer-step gating, and scale updates |

## Public Interface

| Symbol | Description |
|---|---|
| `autocast` | Context manager and decorator that sets per-device autocast enabled state, target dtype, nesting count, and weight-cache policy |
| `is_autocast_available(device_type)` | Calls `torch._C._is_autocast_available` to report backend support |
| `custom_fwd` / `custom_bwd` | Decorators for `torch.autograd.Function` methods that preserve or override autocast state across forward and backward |
| `_enter_autocast` / `_exit_autocast` | Internal helpers traced by pre-dispatch export paths through `torch.overrides.handle_torch_function` |
| `GradScaler` | Dynamic loss scaler with `scale`, `unscale_`, `step`, `update`, `state_dict`, `load_state_dict`, and optimizer inf-check helpers |
| `OptState` | Per-optimizer stage enum with `READY`, `UNSCALED`, and `STEPPED` states |

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch](torch/ADR.md) | depends-on | Uses tensor creation, dtype APIs, optimizer types, CUDA availability checks, and public autocast state functions |
| [torch/csrc](torch/csrc/ADR.md) | depends-on | Calls C++ entry points including `_is_autocast_available`, `_get_privateuse1_backend_name`, `_amp_foreach_non_finite_check_and_unscale_`, and `_amp_update_scale_` |
| [torch/autograd](torch/autograd/ADR.md) | depends-on | `custom_fwd` and `custom_bwd` decorate `torch.autograd.Function` methods and `GradScaler` manages scaled backward gradients |
| [torch/fx](torch/fx/ADR.md) | depends-on | `autocast.__enter__` and `__exit__` detect `PreDispatchTorchFunctionMode` to preserve autocast regions during export and tracing |
| [torch/optim](torch/optim/ADR.md) | depends-on | `GradScaler.step` invokes `Optimizer.step`, injects `grad_scale` and `found_inf` for AMP-aware optimizers, and skips updates on non-finite gradients |

## Runtime Behaviour

`torch.amp.__init__` exposes the public namespace by importing from `autocast_mode.py` and `grad_scaler.py`. Entering `autocast` validates the device type, selects `torch.get_autocast_dtype` when no dtype is passed, saves the previous cache and dtype state, calls `torch.set_autocast_enabled`, increments nesting, and clears the autocast cache when the outermost context exits. `GradScaler.scale` lazily creates `_scale` and `_growth_tracker` tensors on the first output device, `unscale_` groups gradients by device and dtype, and `step` calls `optimizer.step()` only when `found_inf_per_device` sums to zero.

## Performance Profile

`autocast` keeps state thread-local and uses the C++ autocast cache to avoid repeated casts of eligible weights inside nested regions. `GradScaler._unscale_grads_` batches gradients by device and dtype, then calls `torch._amp_foreach_non_finite_check_and_unscale_` once per bucket instead of launching a kernel for each parameter. `GradScaler.update` combines per-optimizer `found_inf` tensors on the scale device and calls `torch._amp_update_scale_`; `get_scale()` explicitly synchronizes because it converts the scale tensor to a Python float with `.item()`.

## Design Rationale

`torch/amp` centralizes mixed precision around device-type strings instead of maintaining separate public implementations for CUDA, CPU, XPU, MTIA, MAIA, HPU, and PrivateUse1 backends. The module keeps policy and state transitions in Python while leaving operator allowlists, dtype dispatch, non-finite checks, and scale updates in optimized C++ kernels. `GradScaler` tracks each optimizer with an explicit `OptState` so user errors such as double `unscale_()` or double `step()` produce deterministic exceptions.
