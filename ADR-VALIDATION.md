# ADR Validation Report

Run: 1  
Date: 2026-05-28

## Results

| Check | Status | Notes |
|---|---|---|
| 1. Scope map current | FAIL | ./src/cmake is missing from adr-scope.md (found in repository but not listed in scope) |
| 2. Files match COVERED | PASS | 9 ADR.md files found, 9 COVERED entries in scope map, all at correct depth |
| 3. Exclusion justifications | FAIL | ./src/binaries has 1445 lines of hand-authored C++ source code (.cc files) but marked as "Built artifacts"; exceeds 200-line threshold for exclusion |
| 4. ADR content non-stub | PASS | All 9 ADRs contain substantive content with minimum 2 sentences per section; all reference actual repository files |
| 5. Book cross-reference | PASS | All book-named subsystems (c10, ATen, torch.core/autograd/nn/jit/distributed) map to COVERED directories |

## Overall: FAIL

## Required Actions

1. **Add missing directory to scope map**: Add `./src/cmake` to adr-scope.md with appropriate status (COVERED or EXCLUDED with valid reason). If excluded, provide justification from: Auto-generated code, Build/config only, Vendored/third-party, Test data only, Empty or stub, or Leaf with no architectural boundary.

2. **Fix binaries exclusion**: ./src/binaries contains 1,445 lines of hand-authored C++ binaries (aot_model_compiler.cc, compare_models_torch.cc, speed_benchmark_torch.cc, etc.). This exceeds the 200-line exclusion threshold. Either:
   - Mark as COVERED and write ADR for ./src/binaries, or
   - Move binaries to a subdirectory or mark individual .cc files as auto-generated if that classification is accurate

3. **Verify cmake directory status**: Determine if ./src/cmake (containing CMakeLists.txt, Dependencies.cmake, Codegen.cmake, etc.) should be:
   - COVERED (write ADR for build system architecture), or
   - EXCLUDED with valid reason (e.g., "Build/config only" if true, though it contains code generation logic)

**Evidence for binaries analysis**:
```
./src/binaries/aot_model_compiler.cc     141 lines
./src/binaries/at_launch_benchmark.cc     94 lines
./src/binaries/compare_models_torch.cc   326 lines
./src/binaries/core_overhead_benchmark.cc 58 lines
./src/binaries/dump_operator_names.cc     85 lines
./src/binaries/lite_interpreter_model_load.cc 33 lines
./src/binaries/load_benchmark_torch.cc    93 lines
./src/binaries/optimize_for_mobile.cc    107 lines
./src/binaries/parallel_info.cc            41 lines
./src/binaries/record_function_benchmark.cc 127 lines
./src/binaries/speed_benchmark_torch.cc  340 lines
TOTAL: 1445 lines
```

## Validation Checklist Verification

**Check 1 — Scope map exists and is current**: FAIL
- adr-scope.md exists at ./src/adr-scope.md ✓
- Discovered directories: 16 total
- Listed in scope: 15 (missing: ./src/cmake)
- Pending status: None ✓

**Check 2 — Actual ADR files match COVERED entries**: PASS
- ADR.md files found: 9 (android, aten, c10, caffe2, functorch, mypy_plugins, tools, torch, torchgen)
- COVERED entries: 9
- All files at correct depth (directly in covered directory) ✓
- Count matches ✓

**Check 3 — Exclusion justifications are valid**: FAIL
- binaries: "Built artifacts" — contains 1445 lines of hand-authored C++ code, exceeds 200-line limit ✗
- benchmarks: "Benchmark scripts only" — 236 lines, boundary case but appears to be scripts ✓
- docs: "Documentation only" — 0 lines ✓
- scripts: "Build scripts and config only" — 310 lines (acceptable for build scripts) ✓
- test: "Test data and fixtures" — 284659 lines (justified as test data) ✓
- third_party: "Vendored/third-party" — 314 lines (acceptable for vendor code) ✓

**Check 4 — ADR content is non-stub**: PASS
- android/ADR.md: 61 lines, has Architectural Role, Dependencies, Trade-offs sections, references JNI, gradle, pytorch_android ✓
- aten/ADR.md: 192 lines, comprehensive sections with 5+ subsections, references Dispatcher.h, DispatchKey.h, native_functions.yaml ✓
- c10/ADR.md: 167 lines, references TensorImpl.h, intrusive_ptr.h, DispatchKey semantics ✓
- caffe2/ADR.md: 55 lines, has all required sections ✓
- functorch/ADR.md: 177 lines, references batching_utils.py, functional_functions.py ✓
- mypy_plugins/ADR.md: 55 lines, references torch type system ✓
- tools/ADR.md: 67 lines, has all required sections ✓
- torch/ADR.md: 259 lines (longest), covers torch.core, torch.autograd, torch.nn, torch.jit, torch.distributed, references torch/__init__.py, Module.cpp ✓
- torchgen/ADR.md: 153 lines, references native_functions.yaml, RegisterSchema.cpp ✓

**Check 5 — Book subsystem cross-reference**: PASS
- Book identifies 7 distinct architectural units across chapters: c10, ATen, torch.autograd, torch.nn, torch.core, torch.jit, torch.distributed
- c10 (./src/c10): COVERED ✓
- ATen (./src/aten): COVERED ✓
- torch subsystems (./src/torch): COVERED (single ADR covers all torch.* components) ✓
- Additional COVERED: android, caffe2, functorch, mypy_plugins, tools, torchgen (not mentioned in book as distinct units, appropriately included) ✓
