# `aten/src/ATen/native`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`aten/src/ATen/native` is the main ATen operator implementation layer. It owns `native_functions.yaml`, shared native C++ implementations, meta kernels, structured-kernel bodies, dispatch-stub declarations, and category files for pointwise, reduction, linear algebra, convolution, pooling, shape, sparse, and quantized operations. Book Chapter 04 identifies this directory as the tensor library where operator schemas, TensorIterator setup, generated registrations, and backend-specific kernels meet.

## Key Files

| File | Purpose |
|---|---|
| `native_functions.yaml` | Declarative operator registry with schemas, variants, dispatch mappings, tags, structured settings, and autogen metadata |
| `README.md` | Operator authoring guide for schemas, variants, annotations, dispatch entries, and registration patterns |
| `BinaryOps.cpp` | Shared binary-operator meta functions, structured implementations, generated-op includes, and dispatch-stub definitions |
| `DispatchStub.h` | Device and CPU-instruction-set function-pointer dispatch layer used by native kernels |
| `TensorIterator.h` | Local include shim to `ATen/TensorIterator.h` for elementwise and reduction kernel setup |
| `CPUFallback.cpp` | Fallback logic for CPU execution paths when backend-specific registrations need generic handling |
| `LinearAlgebra.cpp` | Native linear algebra operator implementations and shape checking |

## Public Interface

`native_functions.yaml` is the public source of truth for generated ATen APIs: each `func` entry names the operator, overload, argument types, returns, variants, dispatch keys, tags, and structured-kernel relationships. C++ implementations in this directory export functions in `at::native`, and torchgen generates typed wrappers, dispatcher registrations, Python bindings, and autograd wrappers from the YAML. Shared native files call `TensorIterator`, meta helpers, `resize_output`, and dispatch stubs; backend subdirectories provide CPU, CUDA, sparse, and quantized implementations that match the YAML dispatch names.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [aten/src/ATen/core](aten/src/ATen/core/ADR.md) | depends-on | Tensor handle, schemas, generated op headers, dispatcher, boxing, and scalar/list types |
| [c10/core](c10/core/ADR.md) | depends-on | `TensorImpl` metadata, `ScalarType`, `Device`, `Layout`, `DispatchKey`, and symbolic shapes |
| [c10/util](c10/util/ADR.md) | depends-on | `ArrayRef`, `SmallVector`, `MaybeOwned`, `irange`, exceptions, and math utilities |
| [aten/src/ATen/native/cpu](aten/src/ATen/native/cpu/ADR.md) | implemented-by | CPU files register native stubs and provide instruction-set-specific inner loops |
| [aten/src/ATen/native/cuda](aten/src/ATen/native/cuda/ADR.md) | implemented-by | CUDA files register native stubs and provide GPU kernels and library integrations |
| [aten/src/ATen/native/sparse](aten/src/ATen/native/sparse/ADR.md) | implemented-by | Sparse files implement COO, CSR, CSC, BSR, BSC, and sparse math dispatch entries |
| [aten/src/ATen/native/quantized](aten/src/ATen/native/quantized/ADR.md) | implemented-by | Quantized files implement quantized tensor creation, copy, schema, and kernel registrations |
| [torchgen](torchgen/ADR.md) | depended-on-by | Code generation reads `native_functions.yaml` and emits registration and binding code |

## Runtime Behaviour

A generated wrapper for an operator such as `add.Tensor` enters the dispatcher with a schema derived from `native_functions.yaml`; the YAML entry delegates functional and inplace variants to `add.out` and maps sparse, sparse CSR, Mkldnn, nested, and other keys to named native functions. `BinaryOps.cpp` defines meta functions such as `TORCH_META_FUNC2(add, Tensor)` that build a borrowing binary op and validate `alpha` before data computation runs. Structured out kernels use `TensorIteratorBase` inheritance and dispatch stubs so shared shape/type work happens once while CPU, CUDA, sparse, and other backends supply device-specific inner loops. The local README documents the same flow: declare in YAML, choose variants and annotations, then implement matching C++ functions.

## Performance Profile

`TensorIterator` factors broadcasting, dtype promotion, output allocation, stride ordering, and memory-format handling out of each elementwise operator, which keeps hot math kernels focused on their inner loop. `DispatchStub` stores device and CPU-capability function pointers, chooses CPU implementations for AVX2, AVX512, SVE, VSX, or default builds, and calls the selected pointer after registration. Structured kernels split meta and impl work, so the Meta backend can compute output shapes and dtypes without touching data. YAML tags such as `core` and `pointwise` give generated code and compiler tooling stable metadata without scanning C++ bodies.

## Design Rationale

Chapter 04 explains the core decision: operators are declared once in YAML and implemented in layered C++ instead of being registered manually in many places. That design gives torchgen one source for C++ signatures, Python bindings, dispatcher registrations, autograd wrappers, mobile selective builds, and backend dispatch names. Shared native files own semantic validation and TensorIterator setup because those rules are backend-independent. Backend subdirectories own inner loops and library calls because performance depends on device, memory format, instruction set, and vendor libraries.

