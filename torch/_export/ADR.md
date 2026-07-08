# `torch/_export`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/_export` contains private implementation pieces for PyTorch export, serialization, verifier, pass infrastructure, trace wrappers, and legacy TorchScript-to-ExportedProgram conversion, as mapped in book chapter 11. The public `torch.export` package owns `ExportedProgram`, `export`, `save`, and `load`; this directory supplies supporting machinery that enforces the export IR contract, serializes/deserializes structured graph data, wraps submodule calls during capture, lowers TorchScript graphs, and supports AOT compile/load compatibility paths. Export differs from `torch.compile()` by requiring a complete graph with explicit input/output signatures and dynamic-shape constraints instead of graph breaks.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Private AOT export compile/load helpers, export-specific Dynamo config, and deprecated `aot_compile`/`aot_load` compatibility |
| `converter.py` | TorchScript graph conversion utilities and `TS2EPConverter` support for converting traced/scripted graphs to export graphs |
| `verifier.py` | Export IR verifier, dialect registry, allowed op checks, metadata checks, signature checks, and `SpecViolationError` |
| `pass_base.py` | Deprecated export pass base with interpreter/tracer helpers that preserve node metadata and fake tensor values |
| `wrappers.py` | Higher-order export tracepoint operator, submodule input/output wrapping, flat-apply support, and strict/exportable subclass helpers |
| `serde/serialize.py` | `ExportedProgramSerializer` and `ExportedProgramDeserializer` for structured schema-based graph serialization |
| `serde/schema.py` | Dataclass schema for serialized `ExportedProgram`, graph, node, argument, tensor metadata, constraints, and schema versions |
| `utils.py` | Export graph utilities for constants, parameters, buffers, CIA decomposition handling, placeholder naming, and signature cleanup |
| `passes/` | Export-specific graph passes such as replacing quantized ops and fixing export IR details |
| `non_strict_utils.py` | Helpers for non-strict export paths and graph-input handling for modules |

## Public Interface

This package is private, but its pieces back public `torch.export` APIs. `torch.export.export()` returns an `ExportedProgram` defined in `torch/export/exported_program.py`; `torch.export.save()` and `torch.export.load()` use the serde layer here to persist structured graph data. `_export.aot_compile()` traces or exports a callable, hands the graph to Inductor's AOT compiler, and returns a shared-library path, while `_export.aot_load()` loads that library through AOTI model container runners. `Verifier` and `load_verifier()` enforce dialect-specific graph invariants before exported programs are saved or consumed. `_wrap_submodules()` records module-call input/output pytree specs and tracepoints for preserved signatures.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/fx](torch/fx/ADR.md) | depends-on | Exported programs store FX `GraphModule` objects and use FX nodes, metadata, tracers, passes, and codegen |
| [torch/_dynamo](torch/_dynamo/ADR.md) | depends-on | Export uses strict Dynamo tracing and dynamic-shape processing to obtain complete graphs |
| [torch/_inductor](torch/_inductor/ADR.md) | depends-on | AOT compile paths send exported graphs to Inductor and AOTI packaging |
| [torch/_functorch](torch/_functorch/ADR.md) | depends-on | Export relies on functionalization and decompositions so graph IR avoids user-visible mutation |
| [torch/csrc/jit](torch/csrc/jit/ADR.md) | depends-on | `converter.py` reads TorchScript graphs and JIT passes for TS-to-export conversion |
| [aten/src/ATen/core](aten/src/ATen/core/ADR.md) | depends-on | Verifier accepts ATen `OpOverload` targets as the export dialect's operator contract |
| [c10/core](c10/core/ADR.md) | depends-on | Serialization records dtype, device, layout, memory format, symbolic dimensions, and tensor metadata |

## Runtime Behaviour

Strict export starts from a module or callable and example inputs, runs Dynamo/export tracing without graph breaks, records an FX `GraphModule`, collects graph signature entries for parameters, buffers, user inputs, constants, custom objects, tokens, and outputs, and attaches range constraints for symbolic dimensions. `wrappers.py` can install module forward hooks that insert `_export_tracepoint` higher-order operator calls around submodule inputs and outputs; those tracepoints preserve module-call pytree signatures in the exported program. `verifier.py` then walks every graph module, checks each node has valid `meta["val"]` data, restricts op targets to allowed builtins, `OpOverload`, and higher-order operators, validates `get_attr` targets, and verifies graph signatures and module call graphs.

Serialization converts the exported program into schema dataclasses in `serde/schema.py`, maps tensors, dtypes, layouts, memory formats, devices, symbolic expressions, ranges, pytree specs, and node targets into stable records, and writes versioned data for `torch.export.save()`. Deserialization reverses the schema into FX graph modules, graph signatures, verifier selection, state dictionaries, constants, and constraints for `torch.export.load()`.

## Performance Profile

Export is an ahead-of-time capture path, so its cost is paid at export time rather than per inference call. Runtime of the exported program's module remains an FX/ATen graph unless a downstream runtime such as AOTInductor compiles it; the export contract focuses on completeness, correctness, and portability rather than immediate eager speed. Capture-time cost comes from strict Dynamo tracing, fake tensor propagation, dynamic-shape constraint solving, functionalization, verifier walks, and serialization of graph metadata and tensor state. The `.pt2` structured format avoids pickling the computation graph and supports safer loading, while tensor data and constants still dominate file size for large models.

## Design Rationale

Export uses an explicit `ExportedProgram` instead of a raw `GraphModule` because deployment needs graph code, state, input/output kinds, module-call signatures, dynamic-shape constraints, verifier dialect, and serialization metadata as one unit. The verifier exists because exported graphs serve as a contract for downstream runtimes; unsupported Python, mutable ops, missing metadata, or invalid attributes must fail before deployment. Structured serde exists because pickle is not a stable or safe graph format for portable programs. Submodule tracepoints preserve call boundaries and pytree specs without requiring the core FX IR to grow a dedicated module-call-signature node type.
