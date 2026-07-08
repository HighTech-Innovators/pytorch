# `caffe2/utils`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`caffe2/utils` collects small Caffe2 utility surfaces that PyTorch still uses outside the main tensor core. In this source tree it provides string helpers, protobuf visibility wrappers, fast integer-division helpers, and the legacy CPU thread-pool bridge that ATen mobile, quantized kernels, NNPACK, and XNNPACK still call.

## Key Files

| File | Purpose |
|---|---|
| `string_utils.h` | Declares `split`, `trim`, `editDistance`, `StartsWith`, and `EndsWith` |
| `string_utils.cc` | Implements tokenization, space trimming, and edit-distance computation with rolling work vectors |
| `fixed_divisor.h` | Defines `FixedDivisor<std::int32_t>` for quotient and remainder by a precomputed magic divisor |
| `proto_wrap.h` | Declares protobuf and ONNX wrapper symbols for `GetEmptyStringAlreadyInited` and protobuf shutdown |
| `proto_wrap.cc` | Implements the wrapper symbols in `caffe2`, `ONNX_NAMESPACE`, and `torch` namespaces |
| `threadpool/ThreadPool.h` | Declares the legacy `ThreadPool` interface, global flags, and `getDefaultNumThreads()` |
| `threadpool/ThreadPool.cc` | Implements `ThreadPoolImpl`, task partitioning, platform-specific thread caps, and local execution fallback |
| `threadpool/pthreadpool-cpp.h` | Declares `PThreadPool`, `pthreadpool()`, and `pthreadpool_()` for ATen and external kernel libraries |
| `threadpool/pthreadpool.cc` | Implements tiled 1D/2D/3D/4D adapters over `legacy_pthreadpool_*` callbacks using `FixedDivisor` |
| `threadpool/thread_pool_guard.h` | Declares `_NoPThreadPoolGuard`, the thread-local opt-out for nested pthreadpool execution |

## Public Interface

This directory exports `caffe2::split`, `caffe2::trim`, `caffe2::editDistance`, `caffe2::StartsWith`, and `caffe2::EndsWith` from `string_utils.h`. It exports `caffe2::FixedDivisor<std::int32_t>` with `Div`, `Mod`, and `DivMod`, plus `caffe2::ShutdownProtobufLibrary`, `caffe2::GetEmptyStringAlreadyInited`, `ONNX_NAMESPACE::GetEmptyStringAlreadyInited`, and `torch::GetEmptyStringAlreadyInited` from `proto_wrap.h`. The thread-pool surface includes `caffe2::ThreadPool`, `caffe2::PThreadPool`, `caffe2::pthreadpool()`, `caffe2::pthreadpool_()`, the `legacy_pthreadpool_*` C API in `pthreadpool.h`, and `_NoPThreadPoolGuard`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [caffe2/core](caffe2/core/ADR.md) | depends-on | `ThreadPool.h` includes `caffe2/core/common.h` for Caffe2-wide macros and exported symbol handling |
| [c10/util](c10/util/ADR.md) | depends-on | `ThreadPool.cc`, `proto_wrap.h`, and `pthreadpool.cc` use c10 flags, logging, and exception macros to configure and validate utility behavior |
| [aten/src/ATen/native](aten/src/ATen/native/ADR.md) | depended-on-by | `aten/src/ATen/native/NNPACK.cpp`, `native/xnnpack/Common.h`, and many `native/quantized/cpu/*.cpp` files include `pthreadpool-cpp.h` to share one CPU thread pool with backend libraries |
| [aten/src/ATen/core](aten/src/ATen/core/ADR.md) | depended-on-by | `aten/src/ATen/ParallelNative.cpp` and `aten/src/ATen/ParallelOpenMP.cpp` call `caffe2::pthreadpool()` and `caffe2::pthreadpool_()` on mobile and OpenMP-backed intra-op paths |

## Runtime Behaviour

`ThreadPoolImpl::run` in `threadpool/ThreadPool.cc` decides between inline execution and parallel execution by checking `range < minWorkSize_`, `FLAGS_caffe2_threadpool_force_inline`, and `numThreads_ == 0`; when it runs in parallel it partitions the range into `FnTask` slices and hands them to `WorkersPool::Execute`. `getDefaultNumThreads()` initializes `cpuinfo` when available, applies Android, iOS, and macOS caps, honors `FLAGS_pthreadpool_size`, and then clamps the result to 63 threads for tsan-safe lock limits. `PThreadPool::run` in `pthreadpool-cpp.cc` switches to sequential execution when `_NoPThreadPoolGuard::is_enabled()` is true; otherwise it calls `pthreadpool_parallelize_1d` and preserves a singleton pool across callers, with an `atfork` hook that leaks the inherited pool and recreates it in the child process. `legacy_pthreadpool_compute_2d`, `legacy_pthreadpool_compute_2d_tiled`, and the higher-dimensional variants in `pthreadpool.cc` linearize nested loop spaces and use `FixedDivisor<int32_t>::DivMod` to recover tile coordinates without repeated integer division. `string_utils.cc` tokenizes through `std::getline`, trims leading and trailing spaces with `find_first_not_of` and `find_last_not_of`, and computes edit distance through `editDistanceHelper`, which updates `current`, `previous`, and `previous1` rows while also handling adjacent transpositions. `proto_wrap.cc` forwards every exported wrapper directly to `::google::protobuf::internal::GetEmptyStringAlreadyInited()` or `::google::protobuf::ShutdownProtobufLibrary()` so Caffe2, ONNX, and torch share the same protobuf singleton symbols.

## Performance Profile

- **Allocation sites** — `split` builds a `std::vector<std::string>` and `editDistance` allocates three work vectors sized to `s1.length() + 1`. The thread-pool code keeps long-lived worker state in singleton `PThreadPool` and `WorkersPool` objects rather than allocating per task.
- **Synchronization costs** — `ThreadPoolImpl::run`, `ThreadPoolImpl::withPool`, `PThreadPool::get_thread_count`, `PThreadPool::set_thread_count`, and `PThreadPool::run` all serialize access with `std::mutex`. The thread-local `_NoPThreadPoolGuard_enabled` flag avoids recursive pool submission without cross-thread coordination.
- **Data movement** — `legacy_pthreadpool_*_tiled` adapters in `pthreadpool.cc` keep task arguments in small stack contexts and reconstruct multidimensional indices from one linear task id, which reduces per-task parameter traffic into backend libraries. The string helpers move parsed substrings into the result vector, and `trim` returns a substring view copy rather than mutating the input.
- **Redundant or repeated work** — `FixedDivisor<int32_t>` pays one `CalcSignedMagic()` pass per divisor so repeated `Div` and `DivMod` calls in tiled loops can replace hardware division with multiply-and-shift on non-ROCm builds. `PThreadPool::set_thread_count` recreates the underlying pool when the count changes because the wrapped pthreadpool API is data-parallel and blocking, so it does not support live resizing in place.

## Design Rationale

PyTorch keeps these utilities under `caffe2/utils` because they are small compatibility surfaces that multiple layers still consume, but they do not justify their own larger subsystem. The thread-pool code exists to unify ATen mobile, quantized backends, NNPACK, and XNNPACK around one CPU pool instead of letting each library spawn independent worker sets. The string helpers, protobuf wrappers, and fixed-divisor helper stay simple and header-light so older Caffe2 call sites and backend adapters can reuse them without pulling in the larger ATen runtime.
