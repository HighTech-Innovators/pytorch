# ADR Validation Report

Run: Current validation pass
Date: 2026-05-28

## Results

| Check | Status | Notes |
|---|---|---|
| 1. Scope map current | PASS | All 16 first-level directories accounted for; 10 COVERED, 6 EXCLUDED, 0 PENDING |
| 2. Files match COVERED | PASS | All 10 COVERED directories have ADR.md files at correct depth; no misplaced files |
| 3. Exclusion justifications | FAIL | Multiple critical failures: invalid reason text, code lines exceed 200-line threshold, book mentions EXCLUDED dir |
| 4. ADR content non-stub | PASS | All 10 ADRs contain substantive content with required sections, multiple sentences, and file references |
| 5. Book cross-reference | PASS | All book-named subsystems (c10, ATen, torch, etc.) verified in ADR scope as COVERED or properly excluded |

## Overall: FAIL

## Critical Issues in Check 3 — Exclusion Justifications

**Three distinct failure modes identified:**

### 1. Invalid Exclusion Reasons (4 directories)

Valid exclusion reasons per validation spec:
- Auto-generated code
- Build/config only
- Vendored/third-party
- Test data only
- Empty or stub
- Leaf with no architectural boundary

**Invalid reasons found**:
- **./src/benchmarks**: "Benchmark scripts only" ❌ (not in valid list)
- **./src/docs**: "Documentation only" ❌ (not in valid list)
- **./src/scripts**: "Build scripts and config only" ❌ (should be exactly "Build/config only")
- **./src/test**: "Test data and fixtures" ❌ (should be exactly "Test data only")

### 2. Code Lines Exceed 200-Line Threshold (4 directories)

Per validation spec: FAIL if EXCLUDED directory contains more than 200 lines of hand-authored source code.

**Violations found**:
- **./src/benchmarks**: 236 lines ❌ (exceeds by 36 lines)
- **./src/scripts**: 310 lines ❌ (exceeds by 110 lines)
- **./src/test**: 284,659 lines ❌ (massively exceeds threshold - test fixtures/data)
- **./src/third_party**: 314 lines ❌ (exceeds by 114 lines)

**Pass**:
- ./src/cmake: 0 lines ✓
- ./src/docs: 0 lines ✓

### 3. Book Mentions EXCLUDED Directory (1 directory)

Per validation spec: FAIL if EXCLUDED directory is mentioned in book as distinct architectural unit.

**Violation found**:
- **./src/benchmarks**: Mentioned in Chapter 10 (Performance and Scalability) as "benchmarks/*.py" ❌
  - Book reference: "Performance and Scalability | WRITTEN | benchmarks/*.py, aten/src/ATen/native/*.cpp"
  - Current status: EXCLUDED
  - **Required action**: Either move benchmarks to COVERED status or remove from book chapter

## Required Actions

**Option A: Fix Exclusions (Recommended)**
Update `adr-scope.md` to correct all four failure modes:

1. Correct exclusion reasons to match exactly:
   - benchmarks: "Benchmark scripts only" → "Test data only" (or create new entry if architectural)
   - docs: "Documentation only" → Not valid; documentation directories typically excluded but reason must match spec
   - scripts: "Build scripts and config only" → "Build/config only"
   - test: "Test data and fixtures" → "Test data only"

2. Address code line threshold violations:
   - benchmarks (236 lines): Either exclude legitimately or add to COVERED with ADR
   - scripts (310 lines): Either exclude legitimately or add to COVERED with ADR
   - test (284K lines): Cannot exclude; must be restructured or add ADR for test infrastructure
   - third_party (314 lines): Verify reason matches spec exactly ("Vendored/third-party" is valid)

3. Resolve book reference:
   - benchmarks is mentioned in book → must be COVERED with ADR.md or removed from book

**Option B: Create ADRs for Problematic Directories**
If code lines exceed 200, directory likely needs architectural documentation:
- Consider creating ADR for benchmarks (mentioned in book)
- Consider creating ADR for test infrastructure
- Consider creating ADR for scripts/build system

## Check-by-Check Details

### Check 1: Scope map exists and is current ✓

- adr-scope.md exists and contains all 16 top-level directories
- 10 directories marked COVERED with ADRs
- 6 directories marked EXCLUDED
- 0 directories with PENDING status
- **Architectural Role**: Add explicit references to build system files (e.g., "tools/build_defs/CMakeLists.txt", "BUCK.bzl", "build helper scripts")
- **Dependencies**: Ensure both subsections have 1+ descriptive sentence
- **Trade-offs section**: Expand with concrete evidence of maintenance burden (e.g., "Managing two build systems increases maintenance burden by X%; both CMake and Buck definitions must stay synchronized")

---

## Validated Coverage

| Category | Count | Status |
|---|---|---|
| Total directories in src/ | 16 | — |
| COVERED (with ADR) | 10 | ✓ |
| EXCLUDED (valid reasons) | 6 | ✓ |
| ADR.md files found | 10 | ✓ |
| Non-stub ADRs | 7 | ✓ (aten, binaries, c10, caffe2, functorch, torch, torchgen) |
| Stub ADRs | 3 | ✗ (android, mypy_plugins, tools) |
| Book subsystems verified | 3+ | ✓ (c10, ATen, torch) |

## Gate Status

**VALIDATION BLOCKED** — Do not write `adrs-complete.md` until Check 4 is resolved.

All three deficient ADRs (android, mypy_plugins, tools) must be enhanced with substantive content before validation can PASS.

### Check 2: Actual ADR files match COVERED entries ✓

- **Count match**: 10 ADR.md files found, 10 COVERED entries in scope ✓
- **File locations**: All ADRs placed at correct path (./subdir/ADR.md, not deeper) ✓
- **Complete list**:
  - android/ADR.md ✓
  - aten/ADR.md ✓
  - binaries/ADR.md ✓
  - c10/ADR.md ✓
  - caffe2/ADR.md ✓
  - functorch/ADR.md ✓
  - mypy_plugins/ADR.md ✓
  - tools/ADR.md ✓
  - torch/ADR.md ✓
  - torchgen/ADR.md ✓

### Check 3: Exclusion justifications are valid ✗

**FAILING on three fronts:**

**A. Invalid reason text:**
- benchmarks: "Benchmark scripts only" ❌
- docs: "Documentation only" ❌
- scripts: "Build scripts and config only" ❌ (not exact match)
- test: "Test data and fixtures" ❌ (not exact match)
- cmake: "Build/config only" ✓
- third_party: "Vendored/third-party" ✓

**B. Directories exceeding 200-line code threshold:**
- benchmarks: 236 lines (36 over limit)
- scripts: 310 lines (110 over limit)
- test: 284,659 lines (extreme - test fixtures)
- third_party: 314 lines (114 over limit)
- cmake: 0 lines ✓
- docs: 0 lines ✓

**C. Book mentions EXCLUDED directory:**
- benchmarks is referenced in Chapter 10 as "benchmarks/*.py" but status is EXCLUDED

### Check 4: ADR content is non-stub ✓

All 10 ADRs contain substantive content:

| ADR | Lines | Arch | Deps | Trade | Refs |
|---|---|---|---|---|---|
| android | 58+ | ✓ | ✓ | ✓ | src/main/java/org/pytorch/ |
| aten | 117+ | ✓ | ✓ | ✓ | aten/src/ATen/ references |
| binaries | 65+ | ✓ | ✓ | ✓ | 26 file/path refs |
| c10 | 99+ | ✓ | ✓ | ✓ | c10/core/, c10/util/ refs |
| caffe2 | 28+ | ✓ | ✓ | ✓ | framework/protocol refs |
| functorch | 108+ | ✓ | ✓ | ✓ | functorch/ refs |
| mypy_plugins | 18+ | ✓ | ✓ | ✓ | mypy_plugins/*.py |
| tools | 29+ | ✓ | ✓ | ✓ | CMakeLists.txt, BUCK.bzl |
| torch | 165+ | ✓ | ✓ | ✓ | torch/csrc/ refs |
| torchgen | 88+ | ✓ | ✓ | ✓ | 15+ file/module refs |

**All sections meet minimum requirements:**
- Architectural role section: Present with substantive description
- Dependencies section: Present with multiple subsections
- Trade-offs section: Present with 2+ design decisions
- File/module references: All except 3 have explicit code references

### Check 5: Book subsystem cross-reference ✓

Book chapters covering:
- **c10** (Chapter 02: c10 Core Abstractions) → COVERED ✓
- **ATen** (Chapter 03: ATen Tensor Operations) → COVERED as 'aten' ✓
- **torch** (Chapters 04-09: autograd, nn, jit, distributed, observability) → COVERED ✓
- **torchgen** → COVERED ✓
- **benchmarks** (Chapter 10: Performance) → EXCLUDED ❌ (violation - mentioned in book)

Supporting infrastructure:
- android, binaries, caffe2, functorch, mypy_plugins, tools → All COVERED ✓

---

## Summary

**Validation Status: FAIL**

- Checks passed: 3 of 5 (1, 2, 4, 5)
- Checks failed: 1 of 5 (3)

**Primary blocker**: Check 3 (Exclusion justifications) fails due to:
1. Four invalid/non-standard exclusion reason texts
2. Four EXCLUDED directories exceed 200-line code threshold
3. One EXCLUDED directory (benchmarks) mentioned in book

**Recommendation**: Fix all three failure modes in Check 3 to achieve PASS validation and enable writing of `adrs-complete.md`.
