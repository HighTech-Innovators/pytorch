# `torch/quantization`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/quantization` owns the legacy quantization namespace kept for backward compatibility. It forwards eager-mode, FX-mode, observer, and qconfig APIs to `torch.ao.quantization` while preserving older import paths.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Re-exports the legacy quantization surface, defines `default_eval_fn`, and assembles `__all__` |
| `quantize.py` | Forwards eager-mode quantization entry points such as `prepare`, `convert`, and `quantize_dynamic` to `torch.ao.quantization.quantize` |
| `observer.py` | Forwards observer classes such as `ObserverBase`, `HistogramObserver`, and `MinMaxObserver` |
| `qconfig.py` | Forwards `QConfig` types and default qconfig factories |

## Public Interface

Public symbols include `quantize`, `quantize_dynamic`, `quantize_qat`, `prepare`, `convert`, `prepare_qat`, `quantize_jit`, `QuantStub`, `DeQuantStub`, `QuantWrapper`, `ObserverBase`, `HistogramObserver`, `default_observer`, `default_weight_observer`, `QConfig`, `default_qconfig`, `default_dynamic_qconfig`, `float16_dynamic_qconfig`, `fuse_modules`, and `default_eval_fn`. The forwarded modules preserve older import sites such as `torch.quantization.quantize`, `torch.quantization.observer`, and `torch.quantization.qconfig`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/ao/quantization](torch/ao/quantization/ADR.md) | depends-on | `quantize.py`, `observer.py`, and `qconfig.py` are explicit forwarding shims into the newer quantization implementation |
| [torch/nn](torch/nn/ADR.md) | depended-on-by | quantization entry points transform `nn.Module` graphs and `default_eval_fn` calibrates models by calling `model(data)` |

## Runtime Behaviour

Importing `torch.quantization` performs star imports from the compatibility shims and then publishes a curated `__all__` list that preserves the historical namespace. `quantize.py` simply imports names like `prepare`, `convert`, `prepare_qat`, and `quantize_dynamic` from `torch.ao.quantization.quantize`, so calls land in the new implementation without wrapper logic. `observer.py` and `qconfig.py` do the same for observer classes and qconfig factories, which means bug fixes in `torch.ao.quantization` automatically flow through the legacy namespace. `default_eval_fn()` is the only local behavioral helper: it iterates `calib_data` and runs `model(data)` for each `(data, _target)` pair during calibration.

## Performance Profile

The compatibility layer adds almost no steady-state overhead because the forwarded symbols are imported directly from `torch.ao.quantization`. Calibration through `default_eval_fn()` runs full model forward passes over the provided dataset, so its cost is the model cost rather than any wrapper cost in this package. Importing the legacy namespace does load many symbols eagerly, which slightly increases import work compared with a minimal facade. The design intentionally favors compatibility over minimal namespace size so existing applications do not pay migration churn.

## Design Rationale

The package exists to keep old user code working while the implementation migrates into `torch.ao.quantization`. A thin forwarding layer avoids maintaining duplicate quantization logic and makes the deprecation path explicit in source.
