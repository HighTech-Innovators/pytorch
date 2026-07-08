# ADR Validation Report

Run: 1
Date: 2026-07-08

## Results

| Check | Status | Notes |
|---|---|---|
| 1. Scope map current | PASS | No PENDING entries; all 2000+ directories satisfy explicit coverage, EXCLUDED-ancestor, or COVERED-ancestor rules |
| 2. Files match COVERED | PASS | 94 ADR.md files found; count equals 94 COVERED entries; no orphans; no double-nesting under ./src/src/ |
| 3. Exclusion justifications | PASS | All 61 EXCLUDED entries use valid reasons; all line-count limits respected |
| 4. ADR content non-stub | PASS | No `../` links; no broken dependency links; all ADRs have title heading, section index, Key Files table, Dependencies, Runtime Behaviour (≥2 sentences), Performance Profile (≥2 sentences), and source references |
| 5. Book cross-reference | PASS | All 64 directory-like subsystems named in book chapters and architecture-map are COVERED or have a COVERED ancestor |

## Overall: PASS

## Required Actions

None.
