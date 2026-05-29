# Architecture Decision Record: torch/fft

## Architectural Role

`torch/fft` provides Fast Fourier Transform operations using cuFFT (GPU) and FFTPACK (CPU) backends. FFTs are fundamental for signal processing, physics simulations, spectral analysis, and certain deep learning applications (spectral convolutions, neural ODEs). This module is specialized but important for its target use cases.

## Responsibilities

- Implementing FFT operators (torch.fft.fft, torch.fft.ifft, torch.fft.rfft, torch.fft.irfft, torch.fft.fftn, torch.fft.ifftn)
- Integrating with cuFFT for GPU execution and FFTPACK/MKL for CPU
- Supporting batched FFT operations for efficiency
- Handling both real and complex transforms with appropriate normalization
- Managing FFT plan caching to amortize planning cost

## Dependencies

**Inbound** (what depends on torch/fft):
- Signal processing applications
- Physics simulations
- Spectral neural networks and other specialized deep learning
- Time series analysis

**Outbound** (what torch/fft depends on):
- cuFFT (for GPU FFTs)
- FFTPACK or MKL (for CPU FFTs)
- `aten/src/ATen/native` for kernel coordination
- `c10/core` for tensor abstractions

## Trade-offs

**cuFFT/MKL overhead vs. portability**: Using vendor libraries ensures optimal performance but creates dependency on external software. Custom implementations would eliminate this dependency but be slower.

**FFT plan persistence vs. memory overhead**: FFT plans are cached to avoid recomputation, trading memory for speed. The alternative (recompute each time) would be slower.

## Extension Boundaries

- **Custom backends**: Alternative FFT libraries can be integrated.
- **Custom operations**: New FFT-based operators can be built on top of base FFT implementations.

## Runtime Implications

**Plan creation**: First FFT on a given size/dtype creates and caches an execution plan. Subsequent calls reuse the plan.

**In-place transforms**: Many FFT operations support in-place variants for memory efficiency.

**Batch processing**: FFT naturally supports batch dimensions for processing multiple signals in parallel.

## Performance Implications

**cuFFT optimization**: cuFFT is highly optimized for GPU FFTs, providing 10-100x speedup over naive implementations.

**Plan caching**: Reusing FFT plans amortizes plan creation cost, reducing per-call overhead.

**Memory layout**: FFT performance depends on input memory layout; contiguous tensors are fastest.

## Ownership Boundaries

- **torch.fft owns**: operator interface and batching
- **cuFFT/MKL own**: actual FFT computation
- **Plans own**: cached FFT execution plans

## Verification Points

- `torch/fft/__init__.py` — Public API interface
- `torch/fft/` — Implementation directory
