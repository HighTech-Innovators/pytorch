# `torch/_export`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/_export` implements the model export pipeline: it captures a `torch.export.ExportedProgram` — a portable, serialisable representation of a PyTorch model with verified input/output signatures — that can be lowered to ONNX, AOT Inductor, or flatbuffer formats for deployment without a Python runtime.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Internal export implementation; `_export` function that drives Dynamo + AOT autograd tracing |
| `converter.py` | ONNX converter: lowers `ExportedProgram` to an ONNX `ModelProto` |
| `verifier.py` | `ExportedProgramVerifier` — validates that an `ExportedProgram` satisfies the export contract (no graph breaks, no data-dependent shapes without symbolic annotation) |
| `serde/` | Serialisation: `serialize.py` / `deserialize.py` — JSON serialisation of `ExportedProgram` graph signature, graph, and constants |
| `passes/` | Graph transformation passes applied after export: constant folding, decomposition, dead-code elimination |
| `pass_infra/` | `PassManager` — ordered pipeline executor; `PassBase` interface |
| `non_strict_utils.py` | Non-strict export mode: relaxes data-dependent shape constraints for models with dynamic control flow |
| `db/` | Operator decomposition database: maps complex ATen ops to their decompositions for export targets that do not support them |

## Public Interface

| Symbol | Description |
|---|---|
| `torch.export.export(fn, args, kwargs, dynamic_shapes)` | Public entry; calls `torch._export._export`; returns `ExportedProgram` |
| `torch.export.ExportedProgram` | Container: `graph_module` (FX `GraphModule`), `graph_signature` (`ExportGraphSignature`), `state_dict`, `constants` |
| `torch.export.Dim(name, min, max)` | Symbolic dimension specification for dynamic shapes |
| `torch.export.dynamic_shapes` | Dict/tuple of `Dim` objects controlling which tensor dimensions are dynamic |
| `torch._export.verifier.ExportedProgramVerifier` | `verify(ep)` — validates that `ep` satisfies all export constraints |
| `torch._export.serde.serialize` | Serialises `ExportedProgram` to a JSON-compatible dict |
| `torch._export.passes.PassManager` | Runs an ordered list of `PassBase` transformations on the exported graph |

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/_dynamo](torch/_dynamo/ADR.md) | depends-on | `torch.export.export` uses Dynamo in strict mode to capture the graph; `torch._dynamo.export` is the underlying call |
| [torch/_functorch](torch/_functorch/ADR.md) | depends-on | AOT autograd traces the forward+backward for export; `make_fx` produces the FX `GraphModule` |
| [torch/fx](torch/fx/ADR.md) | depends-on | `ExportedProgram.graph_module` is a `torch.fx.GraphModule`; pass infrastructure uses FX node APIs |
| [aten/src/ATen](aten/src/ATen/ADR.md) | depends-on | Exported graphs reference ATen operator schemas; decomposition pass maps ops to ATen primitives |
| [torch/_inductor](torch/_inductor/ADR.md) | depended-on-by | Inductor's `aoti_compile_and_package` takes an `ExportedProgram` and compiles it to a shared library |

## Runtime Behaviour

`torch.export.export(fn, args)` calls `torch._export._export`, which runs Dynamo in export mode (no graph breaks allowed) to capture the full computation graph as an FX `GraphModule`. The `ExportGraphSignature` records which nodes are user inputs, parameters, buffers, and constants. If dynamic shapes are specified via `dynamic_shapes`, the corresponding tensor dimensions are annotated with `SymInt` symbolic variables. `verifier.verify` checks the captured graph against the export contract: all operations must have a registered ATen schema, no `call_function` nodes may reference Python closures, and all shapes must be either concrete or annotated symbolic. Serialisation via `serde/serialize.py` produces a JSON representation of the graph, signature, and a zip archive for the state dict.

## Performance Profile

- **Allocation sites**: export is a one-time compilation step; allocations during export (FX nodes, fake tensors, serialisation buffers) do not affect inference latency.
- **Synchronization costs**: none at export time — export runs synchronously in a single Python process. Loading a serialised `ExportedProgram` from disk involves deserialising the JSON graph and loading the state dict, which is I/O-bound.
- **Data movement**: the state dict is serialised to a zip archive using `torch.save`; for large models this involves reading all parameter tensors into memory. `ExportedProgram.state_dict` returns a reference to the in-memory dict without copying.
- **Redundant or repeated work**: `verifier.verify` re-traverses the entire graph after export; for very large graphs this adds measurable latency to the export call. Decomposition passes in `passes/` may introduce redundant ops that a subsequent dead-code elimination pass removes.

## Design Rationale

`ExportedProgram` is a separate type from `GraphModule` because export imposes additional semantic constraints (verified input/output signatures, no graph breaks, explicit parameter lifting) that a plain `GraphModule` does not guarantee. Strict mode (Dynamo export with no graph breaks) is the primary export path because it guarantees a complete, verifiable program — non-strict mode exists as a fallback for models with legitimately data-dependent structure. Serialisation to a portable JSON+zip format (rather than relying on Python pickle) allows `ExportedProgram` to be loaded in C++ environments and version-checked against schema changes.
