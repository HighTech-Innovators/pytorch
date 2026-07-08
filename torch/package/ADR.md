# `torch/package`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/package` owns hermetic packaging for Python modules, pickled objects, and resources. It exports packages into self-contained archives and imports them back through a private module namespace that avoids accidental reliance on the ambient Python environment.

## Key Files

| File | Purpose |
|---|---|
| `package_exporter.py` | Defines `PackageExporter`, dependency scanning, export actions, error reporting, and archive writing |
| `package_importer.py` | Defines `PackageImporter`, local module loading, extern-module validation, and pickle deserialization |
| `importer.py` | Defines the abstract `Importer` protocol, `sys_importer`, and `OrderedImporter` composition |
| `find_file_dependencies.py` | Extracts source-level module references for package dependency resolution |
| `__init__.py` | Marks the package root for the packaging namespace |

## Public Interface

`PackageExporter`, `PackageImporter`, `Importer`, `OrderedImporter`, `ObjNotFoundError`, and `ObjMismatchError` are the main source-visible interfaces. `PackageExporter` uses actions such as `extern`, `mock`, `intern`, and `deny`, while `PackageImporter` exposes `import_module`, `load_binary`, `load_text`, and `load_pickle` for archive contents.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/csrc](torch/csrc/ADR.md) | depends-on | exporter and importer use `torch._C.PyTorchFileWriter`, `torch._C.PyTorchFileReader`, and `torch._C.ScriptModuleSerializer` for archive I/O and storage handling |
| [torch/serialization](torch/serialization/ADR.md) | depends-on | importer uses `_get_restore_location`, `_maybe_decode_ascii`, and deserialization storage helpers while exporter tags storages with serialization metadata |
| [torch/nn](torch/nn/ADR.md) | depended-on-by | model code and modules are common packaged payloads for `PackageExporter` and `PackageImporter` |

## Runtime Behaviour

`PackageExporter.__init__()` creates a `PyTorchFileWriter`, a dependency `DiGraph`, a `ScriptModuleSerializer`, and ordered hook tables for extern, mock, and intern actions before any source is written. During export, source files are scanned with `find_files_source_depends_on`, matched against `_ModuleProviderAction` patterns, and either copied into the archive or recorded as external dependencies. `PackageImporter.__init__()` opens the archive or directory reader, loads `extern_modules`, validates them through `module_allowed`, builds an in-memory package tree, and replaces `__import__` inside `patched_builtins` so packaged code resolves modules through the importer. `load_pickle()` reconstructs storages with a local `DeserializationStorageContext`, reuses archive records when possible, and imports whatever modules are needed to rebuild the pickled objects.

## Performance Profile

Export cost is dominated by dependency scanning, archive writing, and object serialization rather than by the small Python orchestration layer. Import reuses `self.modules` as a private module cache, which prevents re-executing already loaded packaged modules and keeps repeated `import_module()` calls cheap. `load_pickle()` reuses storages through `DeserializationStorageContext`, which avoids duplicating tensor payload reads when multiple objects reference the same underlying storage. The packaging model intentionally prefers hermetic correctness over minimal overhead, so it does extra dependency resolution work to catch implicit imports that would otherwise fail later.

## Design Rationale

Packaging code and loading code live in the same namespace because they share the same importer model, mangling rules, and storage format. The explicit `Importer` abstraction is central: packaged modules must round-trip by name without leaking back into global `sys.modules` semantics.
