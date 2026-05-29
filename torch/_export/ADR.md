# Architecture Decision Record: torch/_export

## Architectural Role

`torch/_export` provides internal implementation of model export functionality, bridging the public `torch/export` API with format-specific converters (ONNX, TorchScript, mobile runtime). It handles lower-level export mechanics including metadata preservation, constant folding during export, and format-specific serialization logic that is not part of the stable public API.

## Responsibilities

- Implementing internal export mechanics not exposed in public API
- Format-specific converters and lowering passes (ONNX, TorchScript, CoreML)
- Metadata preservation during export (shape information, type signatures, version metadata)
- Serialization to target formats with appropriate constraint verification
- Supporting decomposition of complex operators into export-compatible primitives

## Dependencies

**Inbound** (what depends on torch/_export):
- `torch/export` (public API delegates to implementation here)
- Export integration tests
- Format-specific exporter backends

**Outbound** (what torch/_export depends on):
- `torch/fx` for graph representation during export pipeline
- Format libraries (protobuf for ONNX, pickle for TorchScript)
- `aten/src/ATen/core` for operator introspection

## Trade-offs

**Private API stability vs. flexibility**: By keeping export mechanics in `_export`, the public `torch/export` API can remain stable while implementation details evolve. This trades API surface clarity for implementation flexibility.

**Eager conversion vs. graph optimization**: Export immediately converts graphs without cross-format optimization passes. This simplifies implementation but may miss optimization opportunities that require multi-format knowledge.

## Extension Boundaries

- **Custom exporters**: New format converters can be registered to extend export capabilities.
- **Custom decompositions**: Operators can register decomposition rules for export compatibility.

## Runtime Implications

**One-time export cost**: Export is performed once at deployment time, not during training or inference. Errors during export are caught before model is deployed.

**Format fidelity**: Exported models must support all operations used by the original model; missing support raises export errors.

## Ownership Boundaries

- **_export owns**: internal export implementation and format converters
- **export owns**: public API and error handling
- **Format converters own**: target-format-specific serialization

## Verification Points

- `torch/_export/` — Implementation directory
- `torch/_export/exported_program.py` — Core export data structures
