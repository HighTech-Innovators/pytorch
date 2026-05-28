# ADR Validation Report

Run: (validation pass)
Date: 2026-05-28

## Results

| Check | Status | Notes |
|---|---|---|
| 1. Scope map current | PASS | All 16 first-level directories accounted for; 10 COVERED, 6 EXCLUDED, 0 PENDING |
| 2. Files match COVERED | PASS | All 10 COVERED directories have ADR.md files at correct depth; no misplaced files |
| 3. Exclusion justifications | PASS | All 6 exclusions use valid reasons (Build/config, Vendored, Documentation, Test data, Benchmark scripts) |
| 4. ADR content non-stub | FAIL | 3 ADRs have insufficient substantive content: android, mypy_plugins, tools. Each must have 2+ complete sentences per required section heading |
| 5. Book cross-reference | PASS | All book-named subsystems (c10, ATen, torch) verified in ADR scope as COVERED |

## Overall: FAIL

## Required Actions

**Check 4 — ADR Content Deficiency:**

The following 3 ADRs do not meet the minimum content standard (2+ complete sentences per section heading):

### 1. android/ADR.md (61 lines)
**Current state**: Architectural Role section has 2 sentences (minimum met), but Dependencies and Trade-offs sections lack substantive content.

**Required improvements**:
- **Dependencies section**: Expand both "Build-Time Dependencies" and "Runtime Dependencies" subsections to include minimum 2 complete sentences each explaining Android NDK, Gradle, Java SDK requirements and ART runtime
- **Trade-offs section**: Must contain 2+ complete sentences analyzing JNI overhead trade-off with API ergonomics

**Example structure needed**:
```
### 1. JNI Overhead
**Decision**: [1-2 sentences]
**Rationale**: [Minimum 2-3 complete sentences]
**Trade-off**: [Minimum 1-2 sentences with impact]
```

### 2. mypy_plugins/ADR.md (55 lines)
**Current state**: Architectural Role has only ~1 sentence; Dependencies and Trade-offs sections exist but lack depth.

**Required improvements**:
- **Architectural Role**: Expand from 1 to minimum 2 complete sentences explaining mypy_plugins' role in the ecosystem
- **Dependencies section**: Add both "Build-Time" (mypy, Python) and "Runtime" (none) subsections with 1+ sentence each
- **Trade-offs section**: Expand to 2+ complete sentences analyzing static typing for dynamic language

### 3. tools/ADR.md (67 lines)
**Current state**: Sections exist but lack file/module references and substantive sentences.

**Required improvements**:
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
