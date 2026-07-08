# `caffe2`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`caffe2` owns the remaining Caffe2 compatibility layer that PyTorch still builds and reuses. It aggregates legacy Caffe2 core headers, serialization containers, thread-pool utilities, and selected perf-kernel support around ATen-generated source lists.

## Key Files

| File | Purpose |
|---|---|
| `CMakeLists.txt` | Pulls in shared code generation, selects ATen threading mode, appends generated ATen source lists, and adds Caffe2 subdirectories |
| `core/common.h` | Declares `caffe2::GetBuildOptions()` and includes generated Caffe2 macros plus common namespace aliases |
| `serialize/inline_container.h` | Declares `PyTorchStreamReader`, `PyTorchStreamWriter`, and the aligned ZIP container contract used for serialization |
| `utils/threadpool/ThreadPool.h` | Declares the work-stealing Caffe2 thread-pool interface and work-size controls |
| `perfkernels/common.h` | Houses shared declarations for legacy high-performance kernel code |

## Public Interface

The top-level C++ surface exposes `caffe2::GetBuildOptions()`, `caffe2::ThreadPool::createThreadPool(int)`, `getNumThreads()`, `setNumThreads(size_t)`, `setMinWorkSize(size_t)`, `run(...)`, and `withPool(...)`. The serialization surface exposes `caffe2::serialize::PyTorchStreamReader`, `ChunkRecordIterator`, and `PyTorchStreamWriter`, including `getRecord(...)`, `getRecordOffset(...)`, `getAllRecords()`, and `setAdditionalReaderSizeThreshold(...)`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [caffe2/core](caffe2/core/ADR.md) | depends-on | The directory reuses the core compatibility headers and generated build-option reporting layer |
| [c10/core](c10/core/ADR.md) | depends-on | `serialize/inline_container.h` uses `at::Allocator`, `at::DataPtr`, and backend-related core types from the modern runtime |
| [cmake](cmake/ADR.md) | depends-on | `CMakeLists.txt` includes shared code generation and relies on dependency resolution from `cmake/` |
| [torch](torch/ADR.md) | depended-on-by | Python serialization and packaging code consume the `PyTorchStreamReader` and `PyTorchStreamWriter` container format declared here |

## Runtime Behaviour

`CMakeLists.txt` includes `cmake/Codegen.cmake`, sets `ATEN_THREADING` to `OMP` or `NATIVE`, appends ATen-generated CPU, CUDA, HIP, MPS, XPU, and test source lists into `Caffe2_*` variables, and then adds `core`, `serialize`, `utils`, and `perfkernels` subdirectories. That file also applies allowlists and emits debug source listings when `PRINT_CMAKE_DEBUG_INFO` is enabled.

At runtime, `serialize/inline_container.h` defines the archive contract for PyTorch's ZIP-based container format: `PyTorchStreamWriter` writes uncompressed ZIP64 records aligned to 64-byte boundaries, and `PyTorchStreamReader` offers single-reader, in-place, and multi-reader `getRecord(...)` overloads plus offset queries for mmap-friendly access. `utils/threadpool/ThreadPool.h` defines a work-stealing interface with `executionMutex_` and `minWorkSize_`, so callers can bypass pool overhead for tiny ranges and access the underlying `WorkersPool` through `withPool(...)`.

## Performance Profile

- **Allocation sites** - Serialization allocates archive records and reader buffers, while thread-pool creation allocates worker state only when callers request a pool through `createThreadPool(...)` or `defaultThreadPool()`.
- **Synchronization costs** - `ThreadPool` guards `minWorkSize_` updates with `executionMutex_`, and parallel record loading in `PyTorchStreamReader` coordinates multiple readers around one logical archive.
- **Data movement** - `PyTorchStreamWriter` writes tensor payloads into aligned ZIP records, and `PyTorchStreamReader.getRecordMultiReaders(...)` spreads large reads across multiple adapters to move data faster into destination buffers.
- **Redundant or repeated work** - The container format stores data records uncompressed and aligns them to 64-byte boundaries so repeated loads can mmap or copy raw tensor bytes without an extra decompression step.

## Design Rationale

PyTorch keeps this directory because serialization formats, thread-pool contracts, and legacy include paths still carry the `caffe2` namespace even though the main runtime has moved to c10 and ATen. The component acts as a compatibility shell that reuses modern generated sources and core types while preserving stable Caffe2-facing APIs.
