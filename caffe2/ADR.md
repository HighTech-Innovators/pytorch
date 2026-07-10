# `caffe2`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`caffe2` is the legacy Caffe2 layer retained within the PyTorch repository. It provides the ZIP-based model serialization container (`caffe2/serialize`), SIMD-optimised embedding-lookup kernels (`caffe2/perfkernels`), and a minimal set of shared utilities and macros (`caffe2/core`, `caffe2/utils`) on which both the serialization layer and higher-level PyTorch code depend.

## Key Files

| File | Purpose |
|---|---|
| `caffe2/serialize/inline_container.h` | Declares `PyTorchStreamWriter` / `PyTorchStreamReader` — the ZIP64-based container format used by `torch.save` and `torch.jit.save`; defines alignment constraints (64-byte boundaries for mmap-safe tensor reads) |
| `caffe2/serialize/inline_container.cc` | Implements the ZIP read/write path using miniz; `ChunkRecordIterator` provides chunked streaming reads for large tensors |
| `caffe2/serialize/versions.h` | Defines `kProducedFileFormatVersion` (0xA), `kProducedBytecodeVersion` (0x8), and the min/max supported ranges; the version-bump comment history documents every operator semantic change since format version 1 |
| `caffe2/serialize/file_adapter.h` | `FileAdapter` — RAII `FILE*` wrapper implementing `ReadAdapterInterface` for file-backed deserialization |
| `caffe2/serialize/read_adapter_interface.h` | Abstract `ReadAdapterInterface` that `inline_container` uses to read from files, streams, or in-memory buffers |
| `caffe2/perfkernels/embedding_lookup_idx.h` / `.cc` | Dispatch-based embedding-bag lookup with optional SIMD acceleration; `EmbeddingLookupGenericSlowIdx` is the scalar baseline, with `_avx2` and `_sve` variants in sibling files |
| `caffe2/perfkernels/common.h` / `common_avx.cc` / `common_avx2.cc` / `common_sve.cc` | SIMD capability detection and ISA-specific prologs shared by perf kernels |
| `caffe2/core/common.h` | `GetBuildOptions()` and cross-platform macros (`NOMINMAX`, Android shims); includes `c10/macros/Macros.h` |
| `caffe2/core/timer.h` | `caffe2::Timer` — `std::chrono::high_resolution_clock` wrapper exposing `NanoSeconds()`, `MilliSeconds()`, `Seconds()` |
| `caffe2/utils/string_utils.h` | `split`, `trim`, `editDistance`, `StartsWith`, `EndsWith` — string helpers exported via `TORCH_API` |
| `caffe2/utils/proto_wrap.h` | `ShutdownProtobufLibrary()` and `GetEmptyStringAlreadyInited()` wrappers to avoid duplicate-symbol issues when protobuf is built with hidden visibility |

## Public Interface

**Serialization layer (consumed by `torch/csrc/jit` and `torch/serialization.py` via `torch._C`):**
- `caffe2::serialize::PyTorchStreamWriter` — writes ZIP64 archives with 64-byte-aligned entries
- `caffe2::serialize::PyTorchStreamReader` — reads archives; exposes `getRecord()`, `getRecordWithKey()`, `hasRecord()`
- `caffe2::serialize::ChunkRecordIterator` — streaming read interface for large tensor records
- `caffe2::serialize::ReadAdapterInterface` — abstract adapter base for file, istream, and in-memory backends
- `kProducedFileFormatVersion`, `kProducedBytecodeVersion` — version constants checked at save and load

**Performance kernels (consumed by embedding-bag operators in `aten/src/ATen/native`):**
- `caffe2::EmbeddingLookupIdx<IndexType, InType, OutType>` — embedding lookup with length-offset indexing; dispatches to AVX2 or SVE path at runtime

**Core utilities (consumed throughout `torch/csrc` and `aten`):**
- `caffe2::Timer` — lightweight high-resolution timer
- `caffe2::split`, `caffe2::trim`, `caffe2::editDistance` — string utilities exported as `TORCH_API`
- `caffe2::GetBuildOptions()` — returns compile-time build option map
- `caffe2::ShutdownProtobufLibrary()` — safe protobuf teardown for ASAN / valgrind

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [c10/core](c10/core/ADR.md) | depends-on | `c10::Allocator`, `c10::Backend` used by `inline_container.cc`; `c10::CPUAllocator` included for buffer management |
| [c10/util](c10/util/ADR.md) | depends-on | `c10::Exception`, `c10::Logging`, `c10::irange`, `c10::BFloat16`, `c10::Half`, `c10::string_view` |
| [torch/csrc/jit](torch/csrc/jit/ADR.md) | depended-on-by | JIT serialization (`torch.jit.save`) reads and writes through `PyTorchStreamReader` / `PyTorchStreamWriter` |
| `torch._C` (Python extension) | depended-on-by | `torch.save` / `torch.load` reach `caffe2::serialize` through the C extension |
| `aten/src/ATen/native` (embedding ops) | depended-on-by | `EmbeddingLookupIdx` kernels are called from ATen's `EmbeddingBag` native implementation |

## Runtime Behaviour

`PyTorchStreamWriter` opens (or creates) a ZIP64 archive on construction, writing a `version` record containing the ASCII representation of `kProducedFileFormatVersion`. Every `writeRecord()` call pads the entry to a 64-byte boundary so that tensor data can be directly `mmap`-ed without copying. The writer acquires no global locks; callers are responsible for single-writer access. `PyTorchStreamReader` opens the archive and indexes all records via miniz on construction; subsequent `getRecord()` calls are thread-safe for concurrent readers because they do not mutate shared state beyond the `std::mutex` on the underlying `mz_zip_archive`.

`EmbeddingLookupIdx` performs runtime ISA dispatch on first call: `caffe2/perfkernels/common.h` exposes `cpuinfo`-based capability flags that select the AVX2 (`embedding_lookup_idx_avx2.cc`) or SVE (`embedding_lookup_idx_sve.cc`) path over the generic scalar fallback in `embedding_lookup_idx.cc`. The kernel iterates over embedding-bag offset ranges, accumulates rows with optional per-sample weights, and optionally normalises by segment length. An out-of-bounds index check returns `false`; callers in ATen convert this to a C++ exception via `TORCH_CHECK`.

## Performance Profile

**Allocation sites:** `PyTorchStreamWriter` allocates per-record staging buffers on the heap via `std::vector<uint8_t>` before passing data to miniz. `ChunkRecordIterator::next()` copies into a caller-supplied `void* buf` in chunks; no additional heap allocation occurs during the read loop itself. For large tensor records the chunked path avoids a single large allocation.

**Synchronization costs:** `PyTorchStreamReader` wraps the underlying `mz_zip_archive` handle with a `std::mutex` (visible in `inline_container.h` as `std::mutex ar_mutex_`) to serialise concurrent `getRecord` calls. Writes are unsynchronised by design (single-writer contract). No CPU/GPU barriers exist in this layer — serialization is a CPU-only path even when tensor data originates on CUDA.

**Data movement:** Every `writeRecord()` call copies tensor bytes from the caller's buffer into the miniz staging area, then to the file descriptor — two copies per record for file-backed writers. `FileAdapter` wraps a raw `FILE*` and reads via `fseeko`/`fread`; there is no zero-copy mmap path in the C++ layer (mmap alignment is a layout guarantee for potential external consumers).

**Redundant or repeated work:** `EmbeddingLookupGenericSlowIdx` uses `__builtin_prefetch` to prefetch the next index row one iteration ahead (visible in `embedding_lookup_idx.cc`), mitigating cache-miss cost for large embedding tables. The scalar path performs no vectorisation; the AVX2 and SVE paths in `embedding_lookup_idx_avx2.cc` and `embedding_lookup_idx_sve.cc` process multiple elements per iteration.

## Design Rationale

The directory retains only the subset of Caffe2 that PyTorch still depends on: the ZIP-based serialization container (used by both `torch.save` and TorchScript), SIMD-tuned embedding kernels, and a thin layer of shared utilities. The `caffe2::serialize` namespace is intentionally kept separate from `torch/csrc/jit/serialization` to allow the container format to be used without pulling in the JIT stack. Build options (`GetBuildOptions()`, `macros.h.in`) are generated by CMake and expose the exact flags active at compile time, enabling runtime capability queries without recompilation. The ISA dispatch in `perfkernels` delegates the dispatch decision to `cpuinfo` rather than compile-time `#ifdef` guards, allowing a single binary to exploit AVX2 or SVE when available.
