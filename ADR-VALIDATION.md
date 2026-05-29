# ADR Validation Report

Run: validation-phase
Date: 2026-05-29

## Results

| Check | Status | Notes |
|---|---|---|
| 1. Scope map current | FAIL | 960 PENDING directories remain (not COVERED or EXCLUDED); 1 root directory (.) not in scope |
| 2. Files match COVERED | PASS | All 40 COVERED directories have ADR.md at correct paths; count matches exactly |
| 3. Exclusion justifications | PASS | All 226 EXCLUDED directories have valid reasons |
| 4. ADR content non-stub | FAIL | 33 of 40 ADR files lack specific file/module references from the source repository |
| 5. Book cross-reference | FAIL | 960 PENDING directories block comprehensive cross-reference validation |

## Overall: FAIL

The ADR coverage is **incomplete**. Phase 1 (book generation) has been completed successfully, and Phase 2 has produced 40 ADRs for core architectural units. However, the **scope map contains 960 PENDING directories** that must be classified as either COVERED or EXCLUDED before ADR coverage can be validated as complete.

## Required Actions

1. **Resolve all PENDING directories in adr-scope.md**
   - Every directory in the scope must have status COVERED (with a corresponding ADR.md file) or EXCLUDED (with a valid exclusion reason from the approved list)
   - Current PENDING count: 960 directories
   - Recommended approach: Review directories by subtree (android/*, aten/*, torch/*, etc.) and classify systematically

2. **Add file/module references to 33 ADR files**
   - The following ADR.md files are missing specific references to actual source files or modules:
     - torchgen/ADR.md
     - aten/src/ATen/detail/ADR.md
     - aten/src/ATen/cuda/ADR.md
     - aten/src/ATen/native/cuda/ADR.md
     - aten/src/ATen/functorch/ADR.md (Note: mentions "aten/src/ATen/functorch/" directory but needs specific file paths)
     - [27 additional files with similar gaps]
   - Each ADR must reference at least one actual file or module from the repository (e.g., "aten/src/ATen/cuda/CUDAStreamGuard.h" or "c10/cuda/CUDAAllocator.cpp")

3. **Validate book-to-ADR cross-reference after resolving PENDING directories**
   - Once all directories are classified, verify that every subsystem mentioned in the book chapters (e.g., "torch/_dynamo", "torch/optim") is either COVERED or has a documented exclusion

## Validation Context

- **Total directories in src/**: 1,227
- **Directories in scope**: 1,226
- **COVERED**: 40 (with matching ADR.md files)
- **EXCLUDED**: 226 (with valid justifications)
- **PENDING**: 960 (requires classification)

The 40 COVERED directories with complete ADRs are:
- aten/src/ATen/core, aten/src/ATen/core/dispatch, aten/src/ATen/cuda, aten/src/ATen/detail, aten/src/ATen/functorch, aten/src/ATen/native, aten/src/ATen/native/cpu, aten/src/ATen/native/cuda
- c10/core, c10/cuda, c10/util
- caffe2
- tools
- torch/_dynamo, torch/_export, torch/_functorch, torch/_inductor
- torch/amp, torch/autograd, torch/compiler, torch/csrc/autograd, torch/cuda
- torch/distributed, torch/distributed/_tensor, torch/distributed/fsdp
- torch/distributions
- torch/export, torch/fft, torch/fx, torch/jit
- torch/linalg
- torch/multiprocessing
- torch/nn/modules, torch/nn/utils
- torch/optim
- torch/package
- torch/profiler
- torch/sparse
- torch/utils
- torchgen

