# `torch/masked`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/masked` owns PyTorch's masked-value API. It provides masked reductions and normalizations plus the prototype `MaskedTensor` wrapper type for carrying data and boolean validity masks together.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Re-exports masked reductions, normalization ops, `MaskedTensor`, and construction helpers |
| `_ops.py` | Implements masked reductions such as `sum`, `mean`, `amax`, `softmax`, `var`, and `normalize` and synthesizes their docstrings |
| `maskedtensor/core.py` | Defines `MaskedTensor`, validity checks, and wrapper-subclass behavior |

## Public Interface

`MaskedTensor`, `is_masked_tensor`, `masked_tensor`, `as_masked_tensor`, `sum`, `prod`, `cumsum`, `cumprod`, `amax`, `amin`, `argmax`, `argmin`, `mean`, `median`, `logsumexp`, `logaddexp`, `norm`, `normalize`, `softmax`, `log_softmax`, `softmin`, `var`, and `std` are the public symbols. Internally important hooks include `MaskedTensor._from_values`, `_input_mask`, `_output_mask`, `_combine_input_and_mask`, and `_wrap_result`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/nn](torch/nn/ADR.md) | depends-on | `_ops.py` uses `torch.nn.functional` and exposes masked analogues of normalization-style APIs |
| [torch/sparse](torch/sparse/ADR.md) | mutual | `MaskedTensor` validates and operates on `torch.sparse_coo` and `torch.sparse_csr` layouts, while sparse-aware helpers in `_ops.py` preserve masked sparse structure |

## Runtime Behaviour

Importing `torch.masked` pulls reduction functions from `_ops.py` and construction APIs from `maskedtensor.creation`, then publishes them through `__all__`. `_apply_docstring_templates()` mutates each reduction function's `__doc__` at import time and appends the function name to `__all__`, so the public list is generated alongside the docs. `MaskedTensor.__new__()` creates a wrapper subclass with `_make_wrapper_subclass`, and `__init__()` immediately clones and validates the data and mask through `_preprocess_data()` and `_validate_members()`. For sparse layouts, `_preprocess_data()` normalizes inputs with `_sparse_coo_where` or `_sparse_csr_where` so the data and mask share the same sparse structure before downstream ops run.

## Performance Profile

Mask propagation adds real overhead because most operations must broadcast or combine masks in addition to applying the numeric kernel. `MaskedTensor` clones both the data tensor and the mask tensor on construction, which avoids aliasing surprises but increases allocation and copy cost. Sparse support reduces data movement for structured masked inputs, but helper paths such as `_masked_tensor_str()` may materialize dense representations for formatting. The reduction APIs try to avoid filling masked values eagerly by computing with helper functions like `_where`, `_input_mask`, and `_output_mask`, which keeps repeated masked operations closer to ordinary tensor dispatch.

## Design Rationale

The package separates masked semantics from ordinary tensor semantics instead of overloading every `torch` operator with implicit mask behavior. A dedicated wrapper type plus explicit masked reductions makes validity handling observable, testable, and portable across dense and sparse layouts.
