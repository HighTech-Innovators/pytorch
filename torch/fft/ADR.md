# `torch/fft`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/fft` owns the Python namespace for PyTorch spectral transforms. It connects names such as `fft`, `rfft`, `fftn`, and `fftshift` to the `torch._C._fft` builtins while carrying the user-facing documentation for normalization, Hermitian symmetry, output sizing, and CUDA half-precision constraints.

## Key Files

| File | Purpose |
|---|---|
| `torch/fft/__init__.py` | Defines `__all__`, aliases `Tensor = torch.Tensor`, and binds each Python FFT symbol with `_add_docstr(_fft.fft_*)` |

## Public Interface

| Symbol | Description |
|---|---|
| `fft` / `ifft` | One-dimensional complex-to-complex forward and inverse transforms backed by `_fft.fft_fft` and `_fft.fft_ifft` |
| `fft2` / `ifft2` | Two-dimensional transforms over `dim=(-2, -1)` by default, backed by `_fft.fft_fft2` and `_fft.fft_ifft2` |
| `fftn` / `ifftn` | N-dimensional transforms over caller-selected dimensions, backed by `_fft.fft_fftn` and `_fft.fft_ifftn` |
| `rfft`, `rfft2`, `rfftn` | Real-input transforms that return one-sided Hermitian frequency representations |
| `irfft`, `irfft2`, `irfftn` | Inverse transforms that interpret input as half-Hermitian and reconstruct real output |
| `hfft`, `ihfft`, `hfft2`, `ihfft2`, `hfftn`, `ihfftn` | Hermitian-domain transform variants defined with `_add_docstr` bindings in `__init__.py` |
| `fftfreq`, `rfftfreq`, `fftshift`, `ifftshift` | Frequency-bin and spectrum-shifting helpers backed by `_fft.fft_fftfreq`, `_fft.fft_rfftfreq`, `_fft.fft_fftshift`, and `_fft.fft_ifftshift` |

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch](torch/ADR.md) | depends-on | Imports `torch.Tensor` and exposes the `torch.fft` namespace under the root Python package |
| [torch/csrc](torch/csrc/ADR.md) | depends-on | Imports `_add_docstr` and `_fft` from `torch._C`, where the Python-visible C++ bindings live |
| [aten/src/ATen](aten/src/ATen/ADR.md) | depends-on | Executes the actual spectral kernels, shape handling, normalization, and backend dispatch reached through `_fft.fft_*` builtins |

## Runtime Behaviour

Importing `torch.fft` assigns `Tensor = torch.Tensor` and installs Python names by passing each `_fft.fft_*` builtin through `_add_docstr`. Calling `torch.fft.fft`, `torch.fft.rfft2`, or another bound function enters the C++ binding immediately; the Python file does not implement transform loops or shape kernels. The documentation in `__init__.py` defines runtime semantics for `n`, `s`, `dim`, and `norm`, including zero-padding, trimming, inverse normalization, Hermitian one-sided inputs, and the requirement to pass `n` or `s` for odd-length inverse real round trips.

## Performance Profile

The Python layer adds one attribute lookup and one C++ builtin call per FFT invocation, so transform cost lives in ATen and backend libraries rather than in `torch/fft/__init__.py`. The real-input functions `rfft`, `rfft2`, and `rfftn` reduce output storage by omitting negative frequencies in the final transformed dimension. The CUDA documentation in the source states that `torch.half` and `torch.chalf` support requires SM53 or newer and power-of-two signal lengths in every transformed dimension, so dtype and size choices directly select fast or unsupported backend paths.

## Design Rationale

`torch/fft` keeps the public spectral API in a small Python module while delegating all numerical implementation to the dispatcher-backed C++ namespace. This split lets PyTorch publish detailed, NumPy-like documentation and stable Python names without duplicating backend-specific FFT logic in Python. The grouped namespace also separates spectral transforms from the root `torch` API while preserving direct access to `Tensor` and common doc templates.
