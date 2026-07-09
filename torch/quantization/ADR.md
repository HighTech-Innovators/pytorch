# `torch/quantization`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/quantization` preserves the legacy quantization import surface while the implementation lives under `torch.ao.quantization`. The package keeps old names such as `quantize`, `prepare`, `convert`, `ObserverBase`, `QConfig`, `QuantStub`, and `DeQuantStub` available from the historical namespace, and it carries a small CUDA weight-layout helper in `_quantized_conversions.py`.

## Key Files

| File | Purpose |
|---|---|
| `torch/quantization/__init__.py` | Re-exports eager, TorchScript, observer, qconfig, stub, and mapping APIs and defines `default_eval_fn`. |
| `torch/quantization/_quantized_conversions.py` | Implements `pack_int4_to_int8`, `unpack_int8_to_int4`, and `quantized_weight_reorder_for_mixed_dtypes_linear_cutlass`. |
| `torch/quantization/quantization_mappings.py` | Compatibility shim that imports mapping tables and lookup helpers from `torch.ao.quantization.quantization_mappings`. |
| `torch/quantization/observer.py` | Compatibility shim that imports observer classes and default observer factories from `torch.ao.quantization.observer`. |

## Public Interface

| Symbol | Description |
|---|---|
| `quantize`, `quantize_dynamic`, `quantize_qat`, `prepare`, `convert`, `prepare_qat` | Legacy eager-mode quantization entry points re-exported by `__init__.py`. |
| `quantize_jit`, `quantize_dynamic_jit`, `_prepare_ondevice_dynamic_jit`, `_convert_ondevice_dynamic_jit`, `_quantize_ondevice_dynamic_jit` | TorchScript quantization names imported through `quantize_jit.py` and listed in `__all__`. |
| `QuantWrapper`, `QuantStub`, `DeQuantStub` | Module wrappers and stubs re-exported through `stubs.py`. |
| `ObserverBase`, `HistogramObserver`, `default_observer`, `default_weight_observer`, `default_per_channel_weight_observer` | Observer types and factories re-exported by `observer.py`. |
| `QConfig`, `default_qconfig`, `default_dynamic_qconfig`, `float16_dynamic_qconfig`, `default_qat_qconfig` | QConfig objects re-exported through `qconfig.py`. |
| `get_default_static_quant_module_mappings`, `get_default_dynamic_quant_module_mappings`, `get_default_qat_module_mappings`, `get_quantized_operator` | Mapping queries re-exported by `quantization_mappings.py`. |
| `default_eval_fn(model, calib_data)` | Runs `model(data)` for each `(data, _target)` pair during calibration. |
| `pack_int4_to_int8`, `unpack_int8_to_int4`, `quantized_weight_reorder_for_mixed_dtypes_linear_cutlass` | Tensor conversion helpers in `_quantized_conversions.py`; the CUTLASS reorder path requires a CUDA `torch.int8` 2D weight and `torch.int8` or `torch.quint4x2` quantized dtype. |

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch](torch/ADR.md) | depends-on | Imports `torch.ao.quantization.*` modules, calls `torch.tensor`, `torch.arange`, `torch.stack`, `torch.zeros_like`, and tensor view/index/scatter operations in `_quantized_conversions.py`. |
| [torch/nn](torch/nn/ADR.md) | depends-on | Re-exported quantization APIs operate on modules and expose `QuantWrapper`, `QuantStub`, and `DeQuantStub` module transformations. |
| [torch/fx](torch/fx/ADR.md) | depends-on | The `torch/quantization/fx` subpackage re-exports FX graph-mode quantization helpers such as `prepare`, `convert`, and `fuse`. |
| [torch/jit](torch/jit/ADR.md) | depends-on | `quantize_jit.py` re-exports TorchScript graph-mode quantization entry points. |

## Runtime Behaviour

Importing `torch.quantization` eagerly imports the compatibility modules named in `__init__.py`, so most public names resolve to objects defined in `torch.ao.quantization` rather than local implementations. `default_eval_fn` performs a simple calibration pass by iterating `calib_data`, discarding `_target`, and calling `model(data)` for each sample. `_quantized_conversions.py` validates rank, dtype, dtypeq, and CUDA placement before it packs int4 lanes, unpacks int8 bytes, transposes weights, permutes columns with `index_copy`, and materializes the CUTLASS layout with `scatter_`.

## Performance Profile

The compatibility shim adds import-time fan-out because `__init__.py` imports fake quantization, observers, qconfig, mappings, quantize, quantize_jit, and stubs even when callers use one symbol. The re-exported quantization algorithms run in `torch.ao.quantization`, so this directory contributes almost no steady-state overhead for `prepare`, `convert`, or observer execution. `pack_int4_to_int8` and `unpack_int8_to_int4` use vectorized tensor bit operations and allocate their result tensors directly. `quantized_weight_reorder_for_mixed_dtypes_linear_cutlass` allocates multiple CUDA index tensors, performs a transpose or int4 pack/unpack path, uses `index_copy` and `scatter_`, and requires row and column divisibility checks before it returns a `torch.uint8` view.

## Design Rationale

The package keeps old `torch.quantization` imports working while the source of truth moves to `torch.ao.quantization`. The local files make the migration explicit in their module docstrings and instruct new entries to be added in `torch/ao/quantization` with a shim import here. The small local CUTLASS conversion helper stays in the legacy package because it is a tensor-layout utility rather than a general quantization workflow.
