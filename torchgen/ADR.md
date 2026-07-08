# `torchgen`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torchgen` is the ATen operator code-generation control plane. It reads `aten/src/ATen/native/native_functions.yaml` and `tags.yaml`, parses each YAML entry into immutable schema objects, builds per-dispatch-key backend indices, groups functional/in-place/out variants, and emits the generated C++ and YAML artifacts that make the dispatcher, C++ API, tensor methods, native kernel declarations, backend registrations, static dispatch entry points, and Python/autograd binding generators agree on one operator schema. Book chapter 03 describes the runtime dispatcher path through `at::_ops::*::call`; this directory creates those `at::_ops` structs and registration fragments from the declarative operator registry.

## Key Files

| File | Purpose |
|---|---|
| `gen.py` | Main ATen generator: parses native YAML, validates schemas, groups operators, and writes headers/sources such as `Functions.h`, `Operators.cpp`, `Register*.cpp`, and `Declarations.yaml` |
| `model.py` | Lossless immutable data model for `NativeFunction`, `FunctionSchema`, `DispatchKey`, `BackendIndex`, structured groups, ufunc metadata, arguments, returns, aliases, and operator names |
| `context.py` | Context managers and decorators that attach native-function state while API and destination generators render code |
| `native_function_generation.py` | Synthesizes generated functional/out variants and composite kernels required by aliasing and structured operator conventions |
| `code_template.py` | Small templating engine used by `FileManager` and generator modules to substitute scalar and list values into C++ templates |
| `utils.py` | File emission, sharding, template loading, target enums, namespace helpers, and error-context utilities shared across generators |
| `gen_backend_stubs.py` | Backend-extension generator for dispatch-key registration and native-function stubs outside the in-tree backend set |

## Public Interface

The primary entry point is `python -m torchgen.gen`, which accepts source, install, template, selected-operator, static-dispatch, output-dependency, and dry-run options. Programmatic users call `parse_native_yaml()` to obtain `ParsedYaml(native_functions, backend_indices)`, then pass those objects into `gen_headers()`, `gen_source_files()`, `gen_declarations_yaml()`, and related helpers. `model.py` exposes the stable in-memory representation: `NativeFunction.from_yaml()`, `FunctionSchema.parse()`, `BackendIndex.get_kernel()`, `NativeFunctionsGroup.from_dict()`, and dispatch-key predicates. `FileManager` in `utils.py` provides the write-on-change and sharded-file interface used by ATen, autograd, JIT, and backend generators.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torchgen/api](torchgen/api/ADR.md) | depends-on | Converts parsed schemas into C++ API, dispatcher API, native API, structured, meta, ufunc, and signature models |
| [torchgen/dest](torchgen/dest/ADR.md) | depends-on | Renders destination-specific C++ declarations, wrapper kernels, dispatcher registrations, lazy IR, and ufunc kernels |
| [aten/src/ATen/core](aten/src/ATen/core/ADR.md) | generates-for | Emits dispatcher-facing `at::_ops` wrappers and registration code consumed by the core dispatcher described in book chapter 03 |
| [aten/src/ATen/native](aten/src/ATen/native/ADR.md) | depends-on/generates-for | Reads `native_functions.yaml` and generates `NativeFunctions.h` declarations for native kernel implementations |
| [tools/autograd](tools/autograd/ADR.md) | depended-on-by | Supplies parsed native schemas and API signatures to the autograd generator that implements book chapter 05 wrappers |
| [tools/jit](tools/jit/ADR.md) | depended-on-by | Supplies native schemas and selector logic for generated JIT unboxing wrappers |

## Runtime Behaviour

At generation time, `gen.py` loads `tags.yaml` with `LineLoader`, parses `native_functions.yaml` through `NativeFunction.from_yaml()`, validates cross-entry invariants such as structured delegates and reserved Python keywords, and builds a `BackendIndex` per dispatch key. It then groups related schemas with `get_grouped_native_functions()` and `get_grouped_by_view_native_functions()`, routes groups through API translators and destination renderers, and writes only changed files through `FileManager` so incremental builds avoid unnecessary rebuilds. The generated `Operators.cpp` code calls `Dispatcher::singleton().findSchemaOrThrow(...).typed<schema>()` and then `op.call()` or `op.redispatch()`, directly matching the dispatch flow in book chapter 03.

## Performance Profile

The generator pays most of its build-time cost in YAML parsing, schema validation, per-operator API translation, and text emission across thousands of ATen operators. `parse_native_yaml()` and `parse_tags_yaml()` cache parsed results in module-level dictionaries, and `FileManager._write_if_changed()` preserves timestamps when output content does not change. Large generated files use `write_sharded()` so C++ compilation can run in parallel; the autograd template note for sharded files documents that monolithic generated sources were a major incremental-build bottleneck. The generated runtime path favors unboxed dispatcher signatures and static `at::_ops` handles, so codegen increases build complexity to keep operator calls O(1) at runtime as described in book chapter 03.

## Design Rationale

PyTorch puts operator truth in `native_functions.yaml` instead of hand-writing bindings because each operator must surface consistently in C++, Python, dispatcher registration, autograd, selective build, tracing, static dispatch, and backend extension code. `model.py` uses frozen dataclasses and lossless string round-tripping so generator decisions flow from explicit schema data instead of reparsing ad hoc strings. `BackendIndex` separates per-operator schema from per-backend kernel metadata, which lets in-tree and external backends provide different kernels without duplicating the operator model. The generator emits typed `at::_ops` wrappers because book chapter 03's dispatcher requires a fast unboxed call path and a redispatch path for middleware keys such as Autograd, Functionalize, and Python.
