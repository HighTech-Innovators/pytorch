# ADR Validation Report

Run: 3
Date: 2026-07-10

## Results

| Check | Status | Notes |
|---|---|---|
| 1. Scope map current | FAIL | depth-1 directory `adr/` absent from adr-scope.md |
| 2. Files match COVERED | PASS | 26 ADR files, 26 COVERED entries, all at correct paths, no double-nesting |
| 3. Exclusion justifications | FAIL | `caffe2/` uses "Leaf with no architectural boundary" but is named in book/01-system-overview.md as a distinct component |
| 4. ADR content non-stub | PASS | all 26 ADRs have valid headings, section indexes, Key Files tables, Dependencies, Runtime Behaviour ≥2 sentences, Performance Profile ≥2 sentences; no relative-path links; all link targets exist |
| 5. Book cross-reference | FAIL | `caffe2/` is EXCLUDED with no COVERED ancestor but is named as a distinct architectural concern in book/_generated/architecture-health.md (Concern #8) and book/01-system-overview.md |

## Overall: FAIL

## Required Actions

### Check 1 — Add `adr/` to adr-scope.md

- `adr/` is a depth-1 directory present at `./src/adr/` (created by the ADR generation process to hold helper tooling under `adr/_tools/`). Every depth-1 directory must appear explicitly in `adr-scope.md` with no exemption. Add the following row to `adr-scope.md`:

  ```
  | `adr` | yes (Python tooling) | EXCLUDED | Build/config only |
  ```

### Check 3 — Fix invalid exclusion reason for `caffe2/`

- `caffe2/` is currently EXCLUDED with reason `Leaf with no architectural boundary`. This reason is not valid for `caffe2/` because the directory is named as a distinct architectural component in `book/01-system-overview.md` (row: "`caffe2/` | C++ | Legacy Caffe2 components (increasingly deprecated)") and discussed as Concern #8 in `book/_generated/architecture-health.md`. The "Leaf with no architectural boundary" reason is explicitly prohibited for any directory named in the book.
- Choose one of the following resolutions:
  - Change `caffe2/` status to `COVERED` and write `./src/caffe2/ADR.md` documenting the legacy Caffe2 layer, its C++ proto/tensor types, the ongoing migration to PyTorch-native equivalents, and its dependency direction relative to `c10/` and `aten/`.
  - Change the exclusion reason to an honestly applicable valid phrase. The only candidates that could apply given the directory contains C++ legacy components are none of the remaining six; "Leaf with no architectural boundary" is the one that was used and is now disqualified. If no valid reason applies, the correct path is `COVERED`.

### Check 5 — `caffe2/` EXCLUDED while named in book

- `caffe2/` is EXCLUDED (with no COVERED ancestor) but is named as a distinct architectural unit in `book/_generated/architecture-health.md` Concern #8 and in `book/01-system-overview.md`. This is the same root issue as Check 3; resolving Check 3 (either COVERED + ADR, or a valid exclusion reason) will also resolve this finding.
