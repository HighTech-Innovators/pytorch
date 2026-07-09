# ADR Validation Report

Run: 3
Date: 2026-07-09

## Results

| Check | Status | Notes |
|---|---|---|
| 1. Scope map current | PASS | adr-scope.md exists; all 1252 directories classified (explicitly or by implicit ancestor rule); no depth-1 directory missing; no PENDING entries |
| 2. Files match COVERED | PASS | 48 COVERED entries, 48 ADR.md files; all COVERED dirs have ADR.md at exact path `./src/<dir>/ADR.md`; no double-nesting under `./src/src/`; count matches |
| 3. Exclusion justifications | PASS | All 24 EXCLUDED entries use a valid reason; all line-count limits satisfied (Build/config only ≤2000 lines, Leaf with no architectural boundary ≤200 lines, Empty or stub ≤50 lines); no EXCLUDED leaf dir named as a distinct architectural unit |
| 4. ADR content non-stub | PASS | All 48 ADR.md files pass: H1 title with backtick path, bare bullet section index, all 7 required sections present, Key Files as table with real file paths, Dependencies table or explicit no-dep statement, Runtime Behaviour ≥2 sentences, Performance Profile ≥2 sentences; 0 `../` relative links; 0 broken ADR link targets |
| 5. Book cross-reference | PASS | All 39 book-named subsystems (from architecture-map.md and component-map.md) are COVERED directly or via a COVERED ancestor; no EXCLUDED directory with no COVERED ancestor is named as a distinct architectural unit |

## Overall: PASS

## Required Actions

None.
