# `torch/fft`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/fft` owns the Python namespace for Fourier-transform operators. It binds user-facing names such as `fft`, `fftn`, and `fftshift` to built-in implementations and attaches the API documentation that describes normalization, shape, and dtype behavior.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Connects `torch.fft` symbols to `torch._C._fft` builtins with `_add_docstr` and publishes the namespace in `__all__` |

## Public Interface

`fft`, `ifft`, `fft2`, `ifft2`, `fftn`, `ifftn`, `rfft`, `irfft`, `rfft2`, `irfft2`, `rfftn`, `irfftn`, `hfft`, `ihfft`, `fftfreq`, `rfftfreq`, `fftshift`, and `ifftshift` are the public operators. `Tensor = torch.Tensor` is also exported so the generated docstrings can refer to the concrete tensor type used by these operators.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [aten/src/ATen/native](aten/src/ATen/native/ADR.md) | depends-on | every public symbol in `__init__.py` wraps a corresponding builtin from `torch._C._fft` |
| `torch/_torch_docs.py` (no ADR — single file, not a directory) | depends-on | the module formats operator docstrings with `common_args` and `factory_common_args` |

## Runtime Behaviour

Importing `torch.fft` immediately binds Python names like `fft` and `ifft2` to builtin callables such as `_fft.fft_fft` and `_fft.fft_ifft2` through `_add_docstr`. The Python layer does not implement transform math itself; it only exposes consistent names, signatures, and documentation for the underlying kernels. The docstrings in `__init__.py` encode behavior such as zero-padding, trimming, normalization modes, and the difference between full-spectrum transforms and one-sided real transforms. Calls therefore enter C++ kernels directly once Python argument binding finishes.

## Performance Profile

The package adds negligible overhead because each operator is a direct builtin binding instead of a Python wrapper with extra control flow. Allocation behavior is controlled by the underlying kernels and by the optional `out` argument described in the docstrings, so callers can reuse storage when appropriate. Real-input APIs such as `rfft`, `rfft2`, and `rfftn` reduce data movement by returning one-sided Hermitian outputs instead of full redundant spectra. Error checking, normalization, and shape handling occur below this layer, which keeps the namespace composition cost out of hot loops.

## Design Rationale

PyTorch keeps FFT operations in a dedicated namespace so spectral APIs are discoverable and coherent without crowding the root `torch` module. The file is intentionally declarative because the implementation work belongs in backend kernels, while this package's job is to present a stable Python surface.
