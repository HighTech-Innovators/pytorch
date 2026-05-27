# Architecture Decision Record: torch._export (Model Export and Serialization)

## Architectural Role

`torch._export` is **PyTorch's model export infrastructure**, converting trained models into portable, graph-based representations suitable for deployment. It enables:

1. **ONNX export**: Converting models to ONNX format
2. **Serialization**: Saving model graphs for deployment
3. **Graph normalization**: Standardizing model representation
4. **Backend-agnostic format**: Decoupling from PyTorch runtime

Key insight: `torch._export` is **model serialization and standardization**. It captures the model as a computation graph (via torch.fx), normalizes it, and exports to standardized formats. This enables deployment on devices/frameworks without PyTorch installed.

## Responsibilities

### What This Subsystem Owns

1. **Export API** (`torch._export.export()`)
   - Tracing models to static graphs
   - Handling dynamic shapes and control flow
   - Creating exportable graph

2. **Graph Standardization**
   - Lowering high-level operations
   - Canonicalizing representations
   - Handling backend-specific quirks

3. **Serialization Formats**
   - ONNX export (via `torch.onnx`)
   - MobileNet format
   - Other deployment formats

### What This Subsystem Does NOT Own

- **Graph capture**: torch._dynamo and torch.fx own capture
- **Operation execution**: Backend kernels execute
- **Model training**: torch.nn and torch.optim own this

## Dependencies

- **Upstream**: User code exporting models
- **Downstream**: torch.fx, torch.onnx, deployment platforms

## Trade-offs and Design Decisions

### Static Graph vs. Dynamic Execution

**Decision**: Export to static graphs for better deployment compatibility.

**Trade-off**:
- ✅ **Advantage**: Deployed model doesn't need PyTorch runtime
- ✅ **Advantage**: Graph optimizations possible offline
- ❌ **Disadvantage**: Can't support arbitrary Python code
- ❌ **Disadvantage**: Dynamic shapes require special handling

**Evidence**: Static graph capture in torch._export.export().

## Key Implementation Files

| File | Purpose |
|---|---|
| `torch/_export/export.py` | Export API and logic |

Last Verified: 2026-05-27
