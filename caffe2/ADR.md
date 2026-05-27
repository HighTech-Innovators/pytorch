# Architecture Decision Record: Caffe2 — Legacy Framework Core

**Location:** `./src/caffe2/`  
**Last Updated:** 2026-05-27  
**Classification:** Legacy Framework, Maintenance-Only, Deprecated

---

## Executive Summary

Caffe2 is a legacy deep learning framework that preceded PyTorch. It is maintained in PyTorch as a compatibility layer and for users who still depend on Caffe2 code. Modern development should use PyTorch exclusively.

---

## Architectural Role

Caffe2 serves as a **legacy compatibility layer** in PyTorch. Its role is:

1. **Backward Compatibility** — Support existing Caffe2 code and models
2. **Migration Path** — Allow gradual migration from Caffe2 to PyTorch
3. **Type Definitions** — Caffe2 type system (mostly subsumed by c10/core)
4. **Minimal Implementation** — Provide stubs and basic utilities

**Status:** DEPRECATED. New code should not use Caffe2. PyTorch is the standard interface.

---

## Responsibilities

### What Caffe2 Does

- **Type System** — Define scalar types (int32, float32, etc.) — now mostly in c10/core
- **Utilities** — Timer, macros, platform abstractions
- **Serialization** — Legacy model format support

### What Caffe2 Does NOT Do

- **Core Operations** — Operations are in ATen, not Caffe2
- **Autograd** — Gradient computation in torch/autograd, not Caffe2
- **JIT** — Compilation in torch/jit, not Caffe2
- **Active Development** — Caffe2 is maintenance-only; no new features

---

## Dependencies

### What Depends On Caffe2

- **Legacy Caffe2 Code** — Existing projects using Caffe2 API (being migrated)
- **PyTorch Compatibility** — Caffe2 type system integrated with c10/core

### What Caffe2 Depends On

- **Standard C++ Library** — Minimal dependencies
- **c10/core** — For shared type system

---

## Status and Migration Path

### Why Caffe2 Still Exists

1. **Backward Compatibility** — Some users still depend on Caffe2 code
2. **Model Loading** — Caffe2 models can be loaded and run in PyTorch
3. **Gradual Migration** — Users transitioning from Caffe2 to PyTorch

### Migration Strategy

**For Caffe2 Users:**
1. Convert Caffe2 models to TorchScript format (use ONNX as intermediate)
2. Rewrite Caffe2 code using PyTorch API (torch.nn, torch.optim)
3. Migrate training loops to torch.autograd

**Official Guidance:** See PyTorch migration documentation

---

## Testing and Validation

### Testing Status

- Minimal testing; maintained for compatibility only
- No active performance optimization
- Bug fixes only, no new features

---

## Related Systems

- **c10/core/** — Type system shared with Caffe2
- **PyTorch** — Modern framework replacing Caffe2

---

## References

- `caffe2/core/` — Caffe2 core infrastructure (minimal)
- PyTorch Migration Guide (official documentation)

---

## Summary

Caffe2 is a legacy component maintained for backward compatibility. All new development should use PyTorch. Caffe2 will be gradually deprecated as users migrate to PyTorch.

**Recommendation:** Do not extend or develop Caffe2 features. Focus on PyTorch.
