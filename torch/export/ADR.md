# `torch/export`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/export` captures `torch.nn.Module` programs into a normalized ahead-of-time graph representation with explicit inputs, outputs, state, constants, and dynamic-shape constraints. The package exposes the stable user API in `__init__.py` while delegating tracing, dynamic shape processing, graph signatures, unflattening, and `ExportedProgram` behavior to specialized implementation files.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Defines the public `export()`, `save()`, `load()`, and `register_dataclass()` APIs and re-exports export data structures |
| `_trace.py` | Implements `_export()` and `_export_for_training()` using Dynamo, fake tensors, AOTAutograd, guards, and export passes |
| `exported_program.py` | Defines `ExportedProgram`, `ModuleCallSignature`, `ModuleCallEntry`, decomposition handling, validation, and module reconstruction |
| `dynamic_shapes.py` | Defines `Dim`, `dims()`, `ShapesCollection`, additional input constraints, and suggested-fix refinement |
| `unflatten.py` | Reconstructs module hierarchy from flat exported graphs through `UnflattenedModule`, `FlatArgsAdapter`, and interpreter modules |
| `graph_signature.py` | Defines `InputKind`, `OutputKind`, argument specs, `ExportGraphSignature`, and `ExportBackwardSignature` |
| `_unlift.py` | Handles state and constant unlifting plus input-constraint pre-hooks for executable exported modules |

## Public Interface

`__init__.py` exports `export()`, `save()`, `load()`, `register_dataclass()`, `ExportedProgram`, `ExportGraphSignature`, `ExportBackwardSignature`, `ModuleCallEntry`, `ModuleCallSignature`, `Dim`, `dims`, `AdditionalInputs`, `Constraint`, `ShapesCollection`, `CustomDecompTable`, `default_decompositions`, `FlatArgsAdapter`, `unflatten`, and `UnflattenedModule`. `ExportedProgram` exposes `graph_module`, `graph`, `graph_signature`, `state_dict`, `parameters()`, `buffers()`, `constants`, `range_constraints`, `module()`, `run_decompositions()`, and `validate()`. Dynamic-shape users interact with `Dim.AUTO`, `Dim.DYNAMIC`, `Dim.STATIC`, named `Dim("name", min=..., max=...)`, `dims()`, and `refine_dynamic_shapes_from_suggested_fixes()`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch](torch/ADR.md) | depends-on | Public APIs require `torch.nn.Module`, tensors, fake tensors, state dicts, custom ops, and serialized torch objects |
| [torch/_dynamo](torch/_dynamo/ADR.md) | depends-on | Strict export routes tracing through Dynamo and consumes `UserError`, guard production, and graph capture behavior |
| [torch/_export](torch/_export/ADR.md) | depends-on | `_trace.py` and `exported_program.py` rely on internal export passes, verifiers, wrappers, utilities, and serialization helpers |
| [torch/fx](torch/fx/ADR.md) | depends-on | Export represents programs as `torch.fx.GraphModule` graphs and uses FX pass infrastructure and pytree codegen metadata |
| [torch/_functorch](torch/_functorch/ADR.md) | depends-on | AOTAutograd export, graph signatures, functional calls, and parameter/buffer fakification come from functorch integration |
| [torch/utils](torch/utils/ADR.md) | depends-on | The package uses `torch.utils._pytree` `TreeSpec`, flattening, key paths, and serialization of input/output structures |

## Runtime Behaviour

`export()` validates that the input is an `nn.Module`, rejects `torch.jit.ScriptModule`, resolves any `dynamic_spec` decorator into `dynamic_shapes`, and calls `_trace._export()`. `_trace.py` constructs fake inputs, processes dynamic shape specifications, runs Dynamo or non-strict tracing, applies runtime assertion and placeholder naming passes, lifts parameters, buffers, and constants into graph inputs, and records an `ExportGraphSignature`. `ExportedProgram.module()` reconstructs an executable `GraphModule`, checks input constraints through `_unlift.py`, and can run decompositions before validation. `save()` and `load()` serialize and deserialize exported programs through zip-based files and schema-aware export serialization code.

## Performance Profile

Export has high capture-time overhead because it runs tracing, fake tensor propagation, dynamic-shape solving, guard generation, AOTAutograd integration, graph rewrites, and verifier passes. The resulting `ExportedProgram` moves runtime work into a functional ATen graph with lifted state and explicit signatures, which avoids repeated Python module traversal when the exported graph is executed or consumed by downstream tools. Dynamic shapes add guard and runtime-assert cost, while fake tensors reduce capture memory and prevent real kernel execution during shape analysis. `unflatten.py` can run reconstructed modules through an interpreter path for better debugging, which favors debuggability over raw execution speed.

## Design Rationale

`torch/export` separates user-facing AOT capture from both eager execution and lower-level compiler internals. The design records enough structure in `ExportGraphSignature`, `ModuleCallSignature`, range constraints, and pytree specs for downstream serialization, transformation, and module reconstruction to remain sound. Keeping dynamic-shape APIs, graph signatures, unflattening, tracing, and exported-program execution in separate files makes the invariants explicit while preserving a compact public namespace in `__init__.py`.
