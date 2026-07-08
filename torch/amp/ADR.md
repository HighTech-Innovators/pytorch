# `torch/amp`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/amp` provides automatic mixed precision for training and inference, the performance optimization identified in book Chapter 13 as the `torch.amp` path for FP16/BF16 throughput. It owns device-generic autocast context management, decorators for custom autograd functions, runtime dtype eligibility checks, gradient scaling, non-finite gradient detection, optimizer-step skipping, and scale-state serialization. The package changes execution dtypes and gradient magnitudes without changing model definitions or optimizer algorithms.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Public AMP namespace exporting `autocast`, `custom_fwd`, `custom_bwd`, `is_autocast_available`, and `GradScaler` |
| `autocast_mode.py` | Implements autocast context/decorator behavior, device support checks, dtype selection, cache nesting, pre-dispatch tracing hooks, input casting, and custom Function decorators |
| `grad_scaler.py` | Implements `GradScaler`, lazy scale tensors, per-optimizer state, unscale/non-finite checks, optimizer step gating, scale growth/backoff, and state serialization |

## Public Interface

The package exports `torch.amp.autocast`, `torch.amp.custom_fwd`, `torch.amp.custom_bwd`, `torch.amp.is_autocast_available`, and `torch.amp.GradScaler`. `autocast(device_type, dtype=None, enabled=True, cache_enabled=None)` works as a context manager or decorator around forward and loss computation. `custom_fwd(device_type=..., cast_inputs=...)` and `custom_bwd(device_type=...)` decorate `torch.autograd.Function` methods so custom operations preserve autocast state. `GradScaler` exposes `scale`, `unscale_`, `step`, `update`, `state_dict`, `load_state_dict`, `get_scale`, factor setters/getters, and `is_enabled`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/autograd](torch/autograd/ADR.md) | depends-on | Custom AMP decorators target `torch.autograd.Function`, scaled losses call backward, and gradient scaling manipulates `.grad` tensors before optimizer steps |
| [torch/optim](torch/optim/ADR.md) | depends-on | `GradScaler.step` unscales optimizer gradients, calls `optimizer.step`, and injects `grad_scale`/`found_inf` for fused optimizers |
| [aten/src/ATen/core](aten/src/ATen/core/ADR.md) | depends-on | Autocast state, supported dtype queries, nesting, cache control, and non-finite/unscale kernels are exposed through `torch._C` and ATen operators |
| [torch/nn](torch/nn/ADR.md) | depended-on-by | Models wrap forward and loss computation in `autocast` and use `GradScaler` during training loops |
| [torch/fx](torch/fx/ADR.md) | depends-on | `autocast_mode.py` traces `_enter_autocast` and `_exit_autocast` through pre-dispatch `TorchFunctionMode` for export and proxy modes |
| [torch/cuda](torch/cuda/ADR.md) | depends-on | CUDA AMP checks availability, BF16 support, streams, and FP16 execution constraints for GPU mixed precision |

## Runtime Behaviour

`autocast.__init__` validates `device_type`, resolves the fast dtype from `torch.get_autocast_dtype`, checks backend support through `torch._C._is_autocast_available`, validates supported dtypes, and disables or errors for unsupported CUDA/BF16 and custom-backend cases. Entering autocast saves prior enabled state, dtype, and cache state, sets the requested autocast state through `torch.set_autocast_enabled` and `torch.set_autocast_dtype`, increments nesting, and enables the weight cache; exiting decrements nesting, clears the cache at the outermost level, and restores previous state. `custom_fwd` optionally casts floating-point inputs on the target device and records whether forward used autocast, while `custom_bwd` re-enters the same autocast state for backward. `GradScaler.scale` lazily creates scale and growth-tracker tensors, multiplies losses or output containers, `unscale_` groups gradients by device and dtype and calls `_amp_foreach_non_finite_check_and_unscale_`, `step` skips optimizer updates when any device found inf or NaN, and `update` calls `_amp_update_scale_` to grow or back off the scale.

## Performance Profile

Autocast improves compute-bound workloads by running eligible operations in FP16 or BF16 while leaving numerically sensitive operations in safer dtypes, matching Chapter 13's mixed-precision performance guidance. The context manager stores autocast state in thread-local C++ state and uses a nesting counter plus weight cache, so repeated forwards avoid redundant casts until the outermost context exits. `GradScaler` avoids CPU synchronization in `unscale_` by using foreach non-finite checks per device and dtype; only `get_scale` calls `.item()` and documents the CPU-GPU sync. Fused optimizers consume `grad_scale` and `found_inf` directly, reducing extra passes over gradients during mixed-precision updates.

## Design Rationale

AMP is device-generic because CPU BF16, CUDA FP16/BF16, XPU, HPU, and private-use backends share the same user model: run forward under autocast, scale the loss if needed, step the optimizer safely, and update the scale. Autocast lives as a context manager instead of a model rewrite so users isolate precision-sensitive regions with nested `enabled=False` blocks. Gradient scaling is separate from autocast because BF16 does not need scaling while FP16 training needs dynamic overflow control. The package preserves custom autograd correctness through decorators that bind backward precision to the forward autocast state.
