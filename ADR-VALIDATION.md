# ADR Validation Report

Run: Validation Pass
Date: 2026-05-28

## Results

| Check | Status | Notes |
|---|---|---|
| 1. Scope map current | PASS | All 16 first-level directories accounted for; 11 COVERED, 5 EXCLUDED, 0 PENDING |
| 2. Files match COVERED | PASS | All 11 COVERED directories have ADR.md files at correct depth; no misplaced files |
| 3. Exclusion justifications | FAIL | Three EXCLUDED directories exceed 200-line code threshold |
| 4. ADR content non-stub | PASS | All 11 ADRs contain substantive content with required sections and file references |
| 5. Book cross-reference | PASS | All book-named subsystems verified in ADR scope as COVERED |

## Overall: FAIL

## Critical Issues in Check 3 — Exclusion Justifications

**Three EXCLUDED directories exceed the 200-line hand-authored source code threshold:**

The validation specification requires:
> "FAIL if: the directory contains more than 200 lines of hand-authored source code (.py, .cpp, .h, .cu, .cc, .cxx, .hpp)"

**Violations found:**

| Directory | Status | Reason | Lines | Over limit |
|---|---|---|---|---|
| ./src/scripts | EXCLUDED | Build/config only | 310 | +110 |
| ./src/test | EXCLUDED | Test data only | 284,659 | +284,459 |
| ./src/third_party | EXCLUDED | Vendored/third-party | 314 | +114 |

### Analysis

**1. ./src/scripts (310 lines)**
- Reason: "Build/config only"
- Files at maxdepth 1: lintrunner.py, setup_hooks.py
- Interpretation: These are build/configuration scripts for the lintrunner and git hooks. The 310-line threshold violation is marginal (+110 lines). The exclusion reason is accurate — these scripts support the build process rather than defining architectural boundaries.
- **Assessment**: Marginally exceeds threshold but exclusion reason is valid and the code is truly build infrastructure.

**2. ./src/test (284,659 lines)**
- Reason: "Test data only"
- Description: Comprehensive test suite including conftest.py and many test_*.py files
- Interpretation: Test files are being counted at maxdepth 1 (files directly in test/ directory). This includes hundreds of individual test files.
- **Assessment**: Massive violation, but explanation is that these are test implementations, not test data. The exclusion reason text "Test data only" may need clarification if test code (not just data) exists here.

**3. ./src/third_party (314 lines)**
- Reason: "Vendored/third-party"
- Files at maxdepth 1: generate-xnnpack-wrappers.py, generate-cpuinfo-wrappers.py
- Interpretation: These are generator/wrapper scripts for managing vendored dependencies. The code is not original PyTorch architecture code — it manages third-party integrations.
- **Assessment**: Marginal violation. The exclusion reason "Vendored/third-party" is accurate because the subdirectories contain actual vendored code; the scripts at root are just wrappers.

## Detailed Check Results

### Check 1: Scope Map Exists and Is Current ✓ PASS

- **File**: adr-scope.md exists at ./src/adr-scope.md ✓
- **Directory count**: 16 top-level directories found in ./src
- **Scope entries**: 16 total entries (11 COVERED + 5 EXCLUDED)
- **PENDING status**: None (all complete)
- **Completeness**: All 16 directories accounted for in scope map

**Directory Inventory:**
- COVERED (11): android, aten, benchmarks, binaries, c10, caffe2, functorch, mypy_plugins, tools, torch, torchgen
- EXCLUDED (5): cmake, docs, scripts, test, third_party

### Check 2: Actual ADR Files Match COVERED Entries ✓ PASS

- **ADR files found**: 11 (one per COVERED directory)
- **COVERED directories**: 11
- **Count match**: ✓
- **File locations**: All ADRs placed at correct path (./dirname/ADR.md, not deeper)
- **All locations verified**:
  - android/ADR.md ✓
  - aten/ADR.md ✓
  - benchmarks/ADR.md ✓
  - binaries/ADR.md ✓
  - c10/ADR.md ✓
  - caffe2/ADR.md ✓
  - functorch/ADR.md ✓
  - mypy_plugins/ADR.md ✓
  - tools/ADR.md ✓
  - torch/ADR.md ✓
  - torchgen/ADR.md ✓

### Check 3: Exclusion Justifications Are Valid ✗ FAIL

**Reason text validation**: All exclusion reasons match one of the valid set:
- cmake: "Build/config only" ✓
- docs: "Empty or stub" ✓
- scripts: "Build/config only" ✓
- test: "Test data only" ✓
- third_party: "Vendored/third-party" ✓

**Code line threshold validation**: FAIL — Three directories exceed 200-line threshold:
- scripts: 310 lines (exceeds by 110)
- test: 284,659 lines (exceeds by 284,459)
- third_party: 314 lines (exceeds by 114)

**Assessment**: The validation spec fails on the code line threshold. However:
- **scripts** (Build/config only): The script files (lintrunner.py, setup_hooks.py) are build infrastructure, not architectural code. The marginal overage is due to legitimate build scripts.
- **test** (Test data only): The reason "Test data only" may be misleading since these are test implementations. However, test code should not define architectural boundaries for the main codebase.
- **third_party** (Vendored/third-party): The generator scripts manage vendored dependencies; the actual code is in subdirectories. Marginal overage is acceptable for wrapper/management code.

**Verdict**: Check 3 FAILS on the strict 200-line threshold, but the exclusions are architecturally justified.

### Check 4: ADR Content Is Non-Stub ✓ PASS

All 11 ADRs verified as substantive:

| Directory | Lines | Arch | Deps | Trade | Refs |
|---|---|---|---|---|---|
| android | 82 | ✓ | ✓ | ✓ | src/main/java/org/pytorch/ |
| aten | 165+ | ✓ | ✓ | ✓ | aten/src/ATen/ |
| benchmarks | 62+ | ✓ | ✓ | ✓ | benchmarks/ |
| binaries | 65+ | ✓ | ✓ | ✓ | 26+ file refs |
| c10 | 99+ | ✓ | ✓ | ✓ | c10/core/, c10/util/ |
| caffe2 | 28+ | ✓ | ✓ | ✓ | caffe2/ |
| functorch | 108+ | ✓ | ✓ | ✓ | functorch/ |
| mypy_plugins | 18+ | ✓ | ✓ | ✓ | mypy_plugins/ |
| tools | 29+ | ✓ | ✓ | ✓ | CMakeLists.txt, BUCK.bzl |
| torch | 165+ | ✓ | ✓ | ✓ | torch/csrc/ |
| torchgen | 88+ | ✓ | ✓ | ✓ | torchgen/ |

**All ADRs meet minimum requirements:**
- Architectural role section: Present with substantive description ✓
- Dependencies section: Present with technical details ✓
- Trade-offs section: Present with design decisions ✓
- File/module references: All ADRs reference actual code ✓

### Check 5: Book Subsystem Cross-Reference ✓ PASS

Book chapters covering:
- **c10** (Chapter 02: c10 Core Abstractions) → COVERED ✓
- **ATen** (Chapter 03: ATen Tensor Operations) → COVERED as 'aten' ✓
- **torch** (Chapters 04-09: autograd, nn, jit, distributed, observability) → COVERED ✓
- **torchgen** → COVERED ✓
- **benchmarks** (Chapter 10: Performance and Scalability) → COVERED ✓

All book-named subsystems are properly COVERED with ADRs.

---

## Summary

**Validation Status: FAIL**

- **Checks passed**: 4 of 5 (1, 2, 4, 5)
- **Checks failed**: 1 of 5 (3)

**Primary blocker**: Check 3 (Exclusion justifications) fails due to code line threshold violations in three EXCLUDED directories:
1. scripts: 310 lines (Build/config only) — marginal overage
2. test: 284,659 lines (Test data only) — massive overage, but test code is excluded from architecture
3. third_party: 314 lines (Vendored/third-party) — marginal overage, wrapper code only

**Note on test directory**: The 284K-line count includes all test files at the top level of the test/ directory. These are test implementations, not "test data". The interpretation of "Test data only" may need clarification.

**Architectural Assessment**: All exclusions are architecturally justified. The failures are due to strict interpretation of the 200-line threshold, not due to missing ADRs or architectural gaps.

## Recommendations

To achieve PASS status, options include:

1. **Update validation to account for context**: Modify the validation spec to recognize that:
   - Vendored/third-party wrapper code should be exempted
   - Build/config scripts may exceed threshold marginally
   - Test implementations (not data) have different threshold expectations

2. **Move build scripts to a covered directory**: Create an explicit architecture for build infrastructure and ADR it separately (not recommended)

3. **Create directory structure to reduce root-level file count**: Reorganize scripts/ to put large scripts in subdirectories (changes architecture)

4. **Accept FAIL verdict**: Document that test coverage is architecturally complete but validation spec threshold is exceeded by test implementation files

---

**No `adrs-complete.md` written** — validation did not pass due to Check 3 threshold violations.
