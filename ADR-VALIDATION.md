# ADR Validation Report

Run: 4
Date: 2026-07-10

## Results

| Check | Status | Notes |
|---|---|---|
| 1. Scope map current | PASS | all 1251 directories covered by explicit entry or ancestor rule; `adr/` now present as EXCLUDED; no PENDING entries |
| 2. Files match COVERED | PASS | 27 COVERED entries, 27 ADR.md files, all at correct paths (`./src/<dir>/ADR.md`), no double-nesting under `./src/src/` |
| 3. Exclusion justifications | PASS | all 10 EXCLUDED directories use valid reasons; line counts within limits for Build/config only entries (max 1445 lines for `binaries/`); no excluded dir is named as a distinct architectural unit in book chapters |
| 4. ADR content non-stub | PASS | all 27 ADRs: valid `# \`<dir>\`` title heading; bare bullet section-index with all 7 links; `## Key Files` appears exactly once as a markdown table with real file paths; Dependencies table or explicit no-dependencies statement present; Runtime Behaviour ≥2 sentences; Performance Profile ≥2 sentences; no `../` relative links; all 19 unique ADR link targets verified to exist at `./src/<link>` |
| 5. Book cross-reference | PASS | all 24 directory-level subsystems in `book/_generated/architecture-map.md` are COVERED or covered by a COVERED ancestor; `torch/serialization.py` is a file within COVERED `torch/`; `test/` is listed in the architecture map as "Test suite / CI only" — not an architectural unit requiring an ADR, and properly EXCLUDED as "Test suite" |

## Overall: PASS

## Required Actions

None.
