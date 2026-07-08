# `caffe2/serialize`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`caffe2/serialize` owns the C++ ZIP-container reader and writer that TorchScript, libtorch, and related loaders use for archive persistence. It also owns the reader adapter layer and the versioning constants that gate backward and forward compatibility.

## Key Files

| File | Purpose |
|---|---|
| `inline_container.h` | Declares `PyTorchStreamReader`, `PyTorchStreamWriter`, `ChunkRecordIterator`, and alignment helpers for the archive format |
| `inline_container.cc` | Implements ZIP64 archive initialization, version checks, aligned record offsets, multithreaded reads, and final archive writing |
| `read_adapter_interface.h` | Defines the abstract `ReadAdapterInterface` consumed by `PyTorchStreamReader` |
| `file_adapter.h` | Declares `FileAdapter`, which reads archives through `FILE*` APIs and tracks file size |
| `istream_adapter.h` | Declares `IStreamAdapter`, which wraps `std::istream` sources for archive reads |
| `in_memory_adapter.h` | Defines `MemoryReadAdapter`, which reads from caller-owned memory buffers with OOB clamping |
| `versions.h` | Defines `kMinSupportedFileFormatVersion`, `kMaxSupportedFileFormatVersion`, `kProducedFileFormatVersion`, and bytecode version bounds |
| `inline_container_test.cc` | Verifies aligned offsets, chunked reads, multireader reads, and error handling for missing records |

## Public Interface

The primary entry points are `caffe2::serialize::PyTorchStreamReader` and `caffe2::serialize::PyTorchStreamWriter` from `inline_container.h`. `PyTorchStreamReader` exposes `getRecord`, `getRecordMultiReaders`, `getRecordOffset`, `getRecordSize`, `getAllRecords`, `hasRecord`, and `createChunkReaderIter`, while `PyTorchStreamWriter` exposes `writeRecord`, `writeEndOfFile`, `setMinVersion`, `archiveName`, and `serializationId`. The adapter layer exports `ReadAdapterInterface`, `FileAdapter`, `IStreamAdapter`, and `MemoryReadAdapter` so callers can load the same archive format from files, streams, or in-memory buffers.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [c10/core](c10/core/ADR.md) | depends-on | `inline_container.h` and `inline_container.cc` use `at::Allocator`, `at::DataPtr`, `c10::GetCPUAllocator`, and backend allocator hooks when materializing records |
| [c10/util](c10/util/ADR.md) | depends-on | `inline_container.cc`, `file_adapter.cc`, and `istream_adapter.cc` rely on `TORCH_CHECK`, `CAFFE_THROW`, `hash_combine`, and string-view helpers for validation and metadata |
| [torch/csrc/jit](torch/csrc/jit/ADR.md) | depended-on-by | `torch/csrc/jit/serialization/import_read.cpp`, `export.cpp`, `import.cpp`, and multiple mobile compatibility loaders include `caffe2/serialize/inline_container.h` to read and write program archives |
| [torch/csrc/api](torch/csrc/api/ADR.md) | depended-on-by | `torch/csrc/api/src/serialize/output-archive.cpp` reaches `jit::ExportModule`, which writes C++ frontend archives through `PyTorchStreamWriter` |

## Runtime Behaviour

Each `PyTorchStreamReader` constructor in `inline_container.cc` installs a different `ReadAdapterInterface` implementation, then `init()` reads the first bytes to reject the preview magic number, initializes `miniz`, discovers the archive root directory from the first ZIP entry, loads `.data/serialization_id` when present, and parses the `version` or `.data/version` record against the bounds in `versions.h`. `getRecord` can either decompress a whole record into a CPU-allocator buffer with `mz_zip_reader_extract_to_mem` or, when `additionalReaders` exceed `additional_reader_size_threshold_`, split the raw record into chunks that multiple `ReadAdapterInterface` instances pull in parallel and reassemble into one destination buffer. `PyTorchStreamWriter::writeRecord` prefixes every name with `archive_name_plus_slash_`, computes aligned padding through `detail::getPadding`, and adds the entry with `mz_zip_writer_add_mem_ex_v2` so tensor payloads stay 64-byte aligned by default. `writeEndOfFile` injects missing `version`, `byteorder`, and `.data/serialization_id` records, then finalizes the ZIP64 central directory and closes the underlying stream.

## Performance Profile

- **Allocation sites** — `getRecord` allocates one `at::DataPtr` sized to `stat.m_uncomp_size` unless the caller supplies an existing destination buffer. Writer-side state grows mainly in `files_written_`, the reusable `padding_` string, and the `std::ofstream` owned by `PyTorchStreamWriter`.
- **Synchronization costs** — Reader operations serialize metadata access with `reader_lock_`, and multireader fetches pay thread creation and `join()` costs in `getRecordMultiReaders`. The single-writer path has no explicit locking because one `PyTorchStreamWriter` instance owns its `mz_zip_archive` and output stream.
- **Data movement** — The format keeps each record as a separate ZIP member, and `getRecordOffset` plus 64-byte alignment let callers mmap or directly seek to tensor payloads. Large reads can bypass decompression work distribution by splitting raw byte ranges across independent readers, while standard reads decompress one record straight into the final buffer.
- **Redundant or repeated work** — `getRecordOffset` rereads each local ZIP header to compute filename and extra-field sizes before deriving the payload offset. `writeSerializationId` walks every written record name once at finalize time to combine record-name hashes with accumulated CRC32 state into a deterministic archive identifier.

## Design Rationale

Book Chapter 11 describes a ZIP-based format because PyTorch needs random access to individual tensors, portable metadata, and room for format evolution, and this directory implements that contract directly in C++. The adapter abstraction keeps `PyTorchStreamReader` independent of storage medium while still exposing archive-aware helpers such as `getRecordOffset` and `ChunkRecordIterator`. The writer always emits an archive-name subdirectory, version markers, byte-order metadata, and a serialization id so higher layers such as TorchScript import, libtorch save/load, and mobile compatibility tooling can share one durable container format.
