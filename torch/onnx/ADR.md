# `torch/onnx`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/onnx` exports PyTorch models to the ONNX (Open Neural Network Exchange) interchange format. It provides two export paths: a legacy TorchScript-based exporter that traces through `torch.jit.script`/`torch.jit.trace` and maps ops to ONNX symbolic functions per opset version, and a modern Dynamo-based exporter (default since PyTorch 2.x) that consumes an `ExportedProgram` from `torch.export` and uses `onnxscript` to translate FX-level ATen ops to ONNX IR.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Public API: `torch.onnx.export`, `ONNXProgram`, `ExportableModule`, `OnnxExporterError`; routes to dynamo or TorchScript path based on `dynamo=` flag |
| `errors.py` | Structured error hierarchy (`OnnxExporterError` and subclasses) used by both export paths |
| `_internal/exporter/_core.py` | Dynamo-path export engine (~1,780 lines): orchestrates `ExportedProgram` capture strategies, FX passes, ONNX IR construction via `onnxscript.ir`, and `ONNXProgram` assembly |
| `_internal/exporter/_registration.py` | Op-to-ONNX-function registry for the dynamo path; maps `torch.ops.aten.*` overloads to `onnxscript`-decorated translation functions |
| `_internal/exporter/_dispatching.py` | Dispatch logic for the dynamo path: selects the registered ONNX function for each FX node, handles overload resolution and fallback |
| `_internal/torchscript_exporter/registration.py` | `SymbolicRegistry` and `_SymbolicFunctionGroup`: per-opset symbolic function registry for the TorchScript path; supports `register_custom_op_symbolic` / `unregister_custom_op_symbolic` |
| `_internal/torchscript_exporter/symbolic_helper.py` | ~100 helper functions used by symbolic opset modules: shape inference, type casting, list/sequence packing, pooling parameter extraction |
| `symbolic_opset9.py` … `symbolic_opset20.py` | Per-opset symbolic functions (TorchScript path); each file registers Python functions that emit ONNX graph nodes for a specific opset version |
| `_internal/exporter/_onnx_program.py` | `ONNXProgram`: result object from the dynamo path, wrapping the serialized ONNX IR model and providing `save()` |

## Public Interface

| Symbol | Purpose |
|---|---|
| `torch.onnx.export(model, args, f, *, dynamo, opset_version, ...)` | Top-level export function; `dynamo=True` (default) uses the `ExportedProgram` path; `dynamo=False` uses TorchScript tracing |
| `ONNXProgram` | Result of dynamo export: holds the ONNX IR model; `save(path)` writes `.onnx` |
| `ExportableModule` | Protocol type for modules compatible with the dynamo exporter |
| `OnnxExporterError` | Base exception for all export failures |
| `register_custom_op_symbolic(ns::op, fn, opset)` | TorchScript path: registers a custom symbolic function for an operator not in the built-in opset |
| `unregister_custom_op_symbolic(ns::op, opset)` | Removes a previously registered custom symbolic |

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/fx](torch/fx/ADR.md) | depends-on | Dynamo path reads `torch.fx.Graph` nodes from the `ExportedProgram`; FX passes (`_fx_passes.py`) transform the graph before ONNX IR construction |
| [torch/_export](torch/_export/ADR.md) | depends-on | Dynamo path receives `torch.export.ExportedProgram`; `_capture_strategies.py` calls `torch.export.export` and `torch.export.unflatten` |
| [torch/_decomp](torch/_decomp/ADR.md) | depends-on | Dynamo path applies decompositions during export to lower complex ATen ops to primitives that have ONNX translations |
| [torch/csrc/jit](torch/csrc/jit/ADR.md) | depends-on | TorchScript path relies on `torch.jit.trace` / `torch.jit.script` to produce the IR graph that symbolic functions annotate |
| `torch._C._onnx` | depends-on | C++ extension module providing `OperatorExportTypes`, `TensorProtoDataType`, `TrainingMode` enums and `PRODUCER_VERSION` constant |
| `onnxscript` | depends-on | Dynamo path uses `onnxscript.ir` for ONNX IR construction and `onnxscript.evaluator` for op translation; this is an external Python package |

## Runtime Behaviour

On a dynamo-path export call, `torch.onnx.export` delegates to `_internal/exporter/_core.py`. The exporter first attempts to obtain an `ExportedProgram` from the supplied model using a sequence of capture strategies defined in `_capture_strategies.py` (strict export, then non-strict, then scripting fallback). Once captured, the `ExportedProgram.graph` FX graph passes through a series of FX passes (`_fx_passes.py`) that decompose higher-level ATen ops, flatten control flow, and inline constants. The dispatch loop in `_dispatching.py` then iterates over FX nodes and resolves each `call_function` node to a registered ONNX translation function via `_registration.py`. Translation functions emit `onnxscript.ir` nodes; the completed IR is assembled into an `ONNXProgram`. On the TorchScript path, `torch.jit.trace` or `torch.jit.script` produces a `torch.Graph`; symbolic functions in `symbolic_opsetN.py` are invoked for each node to construct the ONNX protobuf graph. Both paths are single-threaded; no shared mutable state is mutated during export beyond the `onnxscript.ir` graph being built within the call.

## Performance Profile

Export is a compile-time operation; latency concerns are compilation rather than inference throughput. The TorchScript path is bounded by JIT tracing or scripting cost plus one symbolic-function call per graph node. The dynamo path adds `torch.export.export` overhead (Dynamo tracing + AOTAutograd decomposition) plus onnxscript IR construction. Neither path allocates device tensors during graph construction; `_tensors.py` in the dynamo path uses metadata-only tensor wrappers. The largest cost for large models is the FX graph traversal in `_core.py` and the serialization of tensor initializers to ONNX protobuf format. No per-node caching or incremental export is implemented; each call reprocesses the full graph.

## Design Rationale

The two-path architecture (TorchScript legacy vs. Dynamo `ExportedProgram`) reflects PyTorch's compiler evolution. The TorchScript path predates FX and uses JIT graph tracing with per-opset symbolic functions registered as Python callables per-operator-name; this makes it extensible via `register_custom_op_symbolic` but tightly coupled to the JIT IR. The Dynamo path consumes the higher-level `ExportedProgram` representation and delegates ONNX op mapping to `onnxscript`, separating graph capture from translation. The `dynamo=True` default in `export()` signals the intended long-term direction; the TorchScript path remains for backward compatibility and is kept under `_internal/torchscript_exporter/` as an implementation detail rather than a primary interface.
