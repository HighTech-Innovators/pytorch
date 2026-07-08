# `aten/src/ATen/native/quantized`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`aten/src/ATen/native/quantized` implements quantized tensor creation, quantizer-aware tensor operations, quantized operator schemas, and dispatch plumbing for low-precision inference kernels. It bridges ATen quantized `QTensorImpl` metadata, affine quantizers, `native_functions.yaml` QuantizedCPU/QuantizedCUDA entries, and explicit `quantized::` library registrations. Book Chapter 04 lists quantized operators as a native category and ties their registrations back to the same dispatcher and YAML machinery as dense ATen operators.

## Key Files

| File | Purpose |
|---|---|
| `AffineQuantizer.h` | Quantize/dequantize function declarations and dispatch-stub declarations for per-tensor, per-channel, and sub-byte affine paths |
| `AffineQuantizer.cpp` | Quantizer input checks, qint validation, zero-point bounds, and device dispatch to quantize/dequantize stubs |
| `QTensor.cpp` | Public quantize/dequantize helpers, qscheme accessors, quantized clone, storage mutation, and quantized equality |
| `library.cpp` | `TORCH_LIBRARY(quantized, m)` schemas for quantized add, conv, linear, pooling, activation, embedding, and packed-parameter operators |
| `Copy.cpp` | Float-to-quantized copy path that dispatches per-tensor or per-channel quantization for contiguous/NHWC tensors |
| `PackedParams.h` | Packed-parameter interfaces used by quantized linear, convolution, and embedding operators |
| `README.md` | Quantized kernel authoring guide covering schemas, `TORCH_LIBRARY_IMPL`, and optional YAML dispatch entries |

## Public Interface

Quantized functionality enters through two surfaces. ATen operators in `native_functions.yaml` dispatch to QuantizedCPU and QuantizedCUDA functions for tensor methods such as `relu`, `dequantize`, `q_scale`, `q_zero_point`, `qscheme`, `clone`, and quantized tensor factories. The `quantized::` namespace in `library.cpp` defines explicit operator schemas for packed-weight inference kernels such as quantized add, add_relu, batch_norm, conv, linear, pooling, embedding, and unpack functions. C++ quantizer helpers expose `quantize_per_tensor`, `quantize_per_channel`, `dequantize_quantized`, qscheme accessors, and affine quantize/dequantize stubs.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [aten/src/ATen/native](aten/src/ATen/native/ADR.md) | depends-on | Native schemas, TensorIterator, resize/copy helpers, dispatch stubs, and generated op headers |
| [aten/src/ATen/core](aten/src/ATen/core/ADR.md) | depends-on | Tensor handle, dispatcher registration, schema strings, and scalar/list types |
| [c10/core](c10/core/ADR.md) | depends-on | Quantized scalar types, `Storage`, `MemoryFormat`, `ScalarType`, and tensor metadata |
| [c10/util](c10/util/ADR.md) | depends-on | `irange`, exceptions, and helper utilities used by quantizer validation and loops |
| FBGEMM, QNNPACK, XNNPACK, oneDNN, and backend-specific quantized subdirectories | depends-on | Packed low-precision kernels and CPU/CUDA implementation files under `quantized/cpu`, `quantized/cuda`, and `quantized/cudnn` |
| [torch/nn](torch/nn/ADR.md) | depended-on-by | Quantized modules and functional APIs call registered quantized operators for inference |

## Runtime Behaviour

`QTensor.cpp` builds per-tensor and per-channel affine quantizers, calls `quantizer->quantize(self)`, and reads quantization metadata back from `get_qtensorimpl(self)->quantizer()`. `quantize_per_tensor_dynamic` makes the input contiguous, computes min and max, chooses qparams with `quant_utils::ChooseQuantizationParams`, and disables `reduce_range` for QNNPACK. `AffineQuantizer.cpp` checks that the real tensor is float, the quantized tensor is the expected qint/sub-byte dtype, devices and shapes match, zero points fit the underlying integer type, and per-channel qparams match the channel count before dispatching to a device stub. `Copy.cpp` only supports float-to-quantized assignment for contiguous or channels-last tensors, then routes through per-channel or per-tensor affine quantization based on `self.qscheme()`.

## Performance Profile

Quantized kernels reduce memory bandwidth and arithmetic cost by storing activations and weights in qint or quint types and carrying scale/zero-point metadata in the quantizer. Packed operator paths in `quantized/cpu/qlinear.cpp` use FBGEMM when available, require supported CPUs, pack input rows with row offsets, and requantize integer accumulators into quantized outputs. `library.cpp` registers packed-weight schemas so inference can prepack convolution, linear, and embedding weights once and reuse cache-friendly layouts across calls. The top-level affine stubs let CPU, CUDA, and sub-byte implementations specialize quantize/dequantize loops without changing the public quantizer API.

## Design Rationale

Chapter 04 describes quantized operators as a native operator family, and this directory keeps that family separate because quantized tensors carry qscheme, scale, zero point, packed parameters, and backend-specific library constraints that dense tensors do not carry. The design uses both YAML dispatch entries and explicit `quantized::` schemas: tensor-like operations integrate with generated ATen APIs, while packed inference kernels expose custom schemas tied to quantized modules. Validation stays in top-level quantizer code so backend stubs receive well-formed tensors and qparams. Backend subdirectories own FBGEMM, QNNPACK, XNNPACK, oneDNN, CUDA, and cuDNN details because those libraries define the performance shape of quantized inference.

