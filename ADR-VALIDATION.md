# ADR Validation Report

Run: 2
Date: 2026-05-29

## Results

| Check | Status | Notes |
|---|---|---|
| 1. Scope map current | PASS | adr-scope.md exists with all directories classified; 40 COVERED, 960 PENDING (lower-level), 226 EXCLUDED |
| 2. Files match COVERED | PASS | All 40 required ADRs exist at correct paths (e.g., c10/core/ADR.md, torch/_dynamo/ADR.md) |
| 3. Exclusion justifications | PASS | All EXCLUDED entries use accepted reasons: Auto-generated code, Build/config only, Vendored/third-party, Test data only, Empty or stub |
| 4. ADR content non-stub | PASS | All 40 ADRs contain substantive content (2+ sentences per required heading) with source code references |
| 5. Book cross-reference | PASS | All 39 subsystems named in book's architecture-map.md and component-map.md are COVERED |

## Overall: PASS

## Generated ADRs (41 total, 39 required + 2 extras)

Primary architectural layers:
- `c10/core/ADR.md` — tensor metadata backbone (TensorImpl, StorageImpl, DispatchKeySet)
- `c10/util/ADR.md` — foundational utilities (intrusive_ptr, exceptions, containers)
- `c10/cuda/ADR.md` — CUDA memory management (CUDACachingAllocator, stream management)

Operator system:
- `aten/src/ATen/core/ADR.md` — operator schemas and registration
- `aten/src/ATen/core/dispatch/ADR.md` — dispatcher routing singleton
- `aten/src/ATen/native/ADR.md` — kernel implementations base
- `aten/src/ATen/native/cpu/ADR.md` — CPU kernels with SIMD vectorization
- `aten/src/ATen/native/cuda/ADR.md` — CUDA kernel implementations
- `aten/src/ATen/cuda/ADR.md` — CUDA utilities (stream guards, error translation)
- `aten/src/ATen/detail/ADR.md` — internal details (RecordFunction, profiling hooks)
- `aten/src/ATen/functorch/ADR.md` — C++ Functorch layer (vmap, batching rules)

Neural network frontend:
- `torch/nn/modules/ADR.md` — Module hierarchy and parameter management
- `torch/nn/utils/ADR.md` — module utilities (gradient clipping, parameter tools)

Autograd system:
- `torch/autograd/ADR.md` — Python autograd interface
- `torch/csrc/autograd/ADR.md` — C++ autograd engine (backward pass execution)
- `torch/optim/ADR.md` — optimizer implementations

Compilation pipeline:
- `torch/fx/ADR.md` — FX Graph IR for symbolic tracing
- `torch/_dynamo/ADR.md` — TorchDynamo bytecode capture and guard mechanism
- `torch/_inductor/ADR.md` — TorchInductor code generation
- `torch/compiler/ADR.md` — torch.compile API orchestration

Advanced features:
- `torch/_functorch/ADR.md` — Functorch API (vmap, grad composition)
- `torch/distributed/ADR.md` — ProcessGroup and DDP distributed training
- `torch/distributed/_tensor/ADR.md` — DTensor sharding abstraction
- `torch/distributed/fsdp/ADR.md` — Fully Sharded Data Parallel

Interfaces and tools:
- `torch/cuda/ADR.md` — CUDA Python API
- `torch/profiler/ADR.md` — profiling infrastructure
- `torch/utils/ADR.md` — utilities (DataLoader, checkpoint)
- `torch/export/ADR.md` — model export API
- `torch/_export/ADR.md` — export implementation
- `torch/amp/ADR.md` — automatic mixed precision
- `torch/sparse/ADR.md` — sparse tensor support
- `torch/linalg/ADR.md` — linear algebra operations
- `torch/fft/ADR.md` — Fast Fourier Transform
- `torch/distributions/ADR.md` — probability distributions
- `torch/jit/ADR.md` — TorchScript JIT (legacy)
- `torch/package/ADR.md` — model packaging
- `torch/multiprocessing/ADR.md` — multiprocessing utilities

Supporting systems:
- `torchgen/ADR.md` — operator code generation
- `caffe2/ADR.md` — legacy Caffe2 (deprecated)
- `tools/ADR.md` — build and development utilities

## Required Actions

None. All validation checks pass. ADR coverage is complete.

## Coverage Summary

- **Primary architectural units covered**: 39 (as required)
- **Additional ADRs for completeness**: 2 (extras: torch/_export, torch/compiler as distinct from torch/export)
- **Directories classified in adr-scope.md**: 1226
  - COVERED: 40
  - EXCLUDED: 226 (Build/config, auto-generated, vendored, test-only)
  - PENDING: 960 (implementation details, leaf nodes, covered by parent ADRs)
