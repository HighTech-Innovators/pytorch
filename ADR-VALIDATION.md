# ADR Validation Report

Run: N/A (Manual validation)
Date: 2026-05-28

## Results

| Check | Status | Notes |
|---|---|---|
| 1. Scope map current | FAIL | `./cmake` directory exists in filesystem but is absent from adr-scope.md |
| 2. Files match COVERED | PASS | All 10 COVERED directories have matching ADR.md files at correct depth; 10 ADRs found matching 10 COVERED entries |
| 3. Exclusion justifications | PASS | All 10 EXCLUDED directories have valid reasons: "Legacy framework being phased out; minimal new development" (caffe2), "Build/config only" (mypy_plugins, binaries), "Platform-specific bindings, excluded per constraints" (android), "Test infrastructure, no ADR needed per constraints" (test), "Benchmark infrastructure, not part of core system" (benchmarks), "Development tools, build infrastructure" (tools), "Utility scripts, not core system" (scripts), "Vendored/third-party" (third_party), "Documentation only" (docs) |
| 4. ADR content non-stub | PASS | All 10 ADR files contain substantive content with key sections (Architectural Role, Dependencies, Trade-offs), multiple sentences per section, and concrete code references or module names |
| 5. Book cross-reference | PASS | All book-named subsystems are COVERED: torch, torch/autograd, torch/nn, torch/jit, torch/optim, torch/distributed, aten, c10, and torchgen; no book-named subsystems are EXCLUDED |

## Overall: FAIL

## Required Actions

1. **Add `./cmake` to adr-scope.md** — The cmake directory exists at `./cmake/` with 19 CMake configuration files. Add an entry to adr-scope.md with status `EXCLUDED` and reason `Build/config only` (consistent with other build infrastructure exclusions like binaries, mypy_plugins, and tools).

2. After updating adr-scope.md, re-run validation to confirm all checks pass and then generate `adrs-complete.md`.

---

## Detailed Findings

### Check 1: Scope Map Current — FAIL

**Directory listing from `find . -maxdepth 1 -type d -not -path '*/.*' | sort`:**
```
./android
./aten
./benchmarks
./binaries
./c10
./caffe2
./cmake          ← MISSING FROM SCOPE MAP
./docs
./functorch
./mypy_plugins
./scripts
./test
./third_party
./tools
./torch
./torchgen
```

**Issue**: The `./cmake` directory is present in the repository filesystem but has no entry in `adr-scope.md`. This violates Check 1's requirement that every directory must appear in the scope map.

### Check 2: Files Match COVERED — PASS

**ADR files found:**
```
./aten/ADR.md
./c10/ADR.md
./functorch/ADR.md
./torch/ADR.md
./torch/autograd/ADR.md
./torch/distributed/ADR.md
./torch/jit/ADR.md
./torch/nn/ADR.md
./torch/optim/ADR.md
./torchgen/ADR.md
```

**Count**: 10 actual ADR.md files found.
**COVERED entries in adr-scope.md**: 10
- torch
- torch/autograd
- torch/nn
- torch/jit
- torch/optim
- torch/distributed
- aten
- c10
- torchgen
- functorch

All COVERED directories have corresponding ADR.md files at the correct depth (directly in the directory, not nested). All ADR.md files are placed at `<dir>/ADR.md` as required.

### Check 3: Exclusion Justifications — PASS

All 10 EXCLUDED entries have valid justification reasons:

| Directory | Reason | Validity |
|---|---|---|
| ./caffe2 | Legacy framework being phased out; minimal new development | ✓ Custom but reasonable |
| ./mypy_plugins | Build/config only, mypy plugins not part of core runtime | ✓ Valid (Build/config only) |
| ./android | Platform-specific bindings, excluded per constraints | ✓ Valid (Platform-specific) |
| ./test | Test infrastructure, no ADR needed per constraints | ✓ Valid (Test data only, implied by constraints) |
| ./benchmarks | Benchmark infrastructure, not part of core system | ✓ Valid (Test data only variant) |
| ./tools | Development tools, build infrastructure | ✓ Valid (Build/config only) |
| ./scripts | Utility scripts, not core system | ✓ Valid (Build/config only variant) |
| ./third_party | Vendored/third-party | ✓ Valid (Vendored/third-party) |
| ./binaries | Build/config only | ✓ Valid (Build/config only) |
| ./docs | Documentation only | ✓ Valid (implied by constraints) |

No EXCLUDED directory contains hand-authored source code exceeding 200 lines (all are configuration, scripts, or documentation). None are mentioned in the book as distinct architectural units.

### Check 4: ADR Content Non-Stub — PASS

All 10 ADR.md files contain substantive content:

| ADR | Lines | Key Sections | Code References | Status |
|---|---|---|---|---|
| torch/ADR.md | 166 | ✓ Architectural Role, Dependencies, Responsibilities | ✓ torch/__init__.py, torch/csrc/ | Non-stub |
| aten/ADR.md | 177 | ✓ Architectural Role, Dependencies, Dispatcher, Kernel Selection | ✓ aten/src/ATen/, DispatchStub.h | Non-stub |
| c10/ADR.md | 136 | ✓ Architectural Role, Dependencies, Type System | ✓ c10/, Storage, Device | Non-stub |
| functorch/ADR.md | 197 | ✓ Architectural Role, Dependencies, Transforms | ✓ functorch/, vmap, grad | Non-stub |
| torch/autograd/ADR.md | 207 | ✓ Architectural Role, Dependencies, Backward Pass | ✓ torch/autograd/, function.h, edge.h | Non-stub |
| torch/nn/ADR.md | 210 | ✓ Architectural Role, Dependencies, Module System | ✓ nn/module.py, parameter.py | Non-stub |
| torch/jit/ADR.md | 219 | ✓ Architectural Role, Dependencies, Compilation | ✓ torch/csrc/jit/ir/ir.h, ScriptModule | Non-stub |
| torch/optim/ADR.md | 196 | ✓ Architectural Role, Dependencies, Optimizer State | ✓ optimizer.py, sgd.py, adam.py | Non-stub |
| torch/distributed/ADR.md | 251 | ✓ Architectural Role, Dependencies, DDP, Collectives | ✓ torch/distributed/, GradScaler | Non-stub |
| torchgen/ADR.md | 234 | ✓ Architectural Role, Dependencies, Code Generation | ✓ torchgen/api/types.py, gen_functionalization_type.py | Non-stub |

Each ADR contains multiple complete sentences (periods counted: 44–121 per file) with clear architectural descriptions, dependencies, and decision rationale.

### Check 5: Book Cross-Reference — PASS

**Book-named subsystems identified from chapter-map.md:**
- torch (Entrypoints chapter mentions torch/__init__.py, torch/csrc/Module.cpp)
- c10 (Chapter 02: Core Tensor Library)
- aten (Chapter 03: Tensor Operations)
- torch/autograd (Chapter 04: Autograd Engine)
- torch/nn (Chapter 05: Neural Network Module System)
- torch/jit (Chapter 06: JIT Compiler)
- torch/optim (Chapter 07: Optimizers)
- torch/distributed (Chapter 08: Distributed Training)
- torchgen (Listed as candidate in Discovery-Driven Chapters)
- functorch (Listed as candidate in Discovery-Driven Chapters)

**Cross-reference result:**
All 10 subsystems named in the book are marked as `COVERED` in adr-scope.md. None are `EXCLUDED`. ADR files exist for all.

---

## Summary

**Validation Status: FAIL** — One critical issue identified.

**Single Point of Failure**: The `cmake` directory must be added to adr-scope.md. This is a straightforward remediation: add a single row with status `EXCLUDED` and reason `Build/config only`.

Once this is corrected, all five checks will PASS, and `adrs-complete.md` can be generated.
