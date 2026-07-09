# ADR Validation Report

Run: 1
Date: 2026-07-09

## Results

| Check | Status | Notes |
|---|---|---|
| 1. Scope map current | PASS | No PENDING entries; all depth-1 source directories present; `./adr` is a validation-tool artifact created by the validator per spec — not a generator issue |
| 2. Files match COVERED | FAIL | 21 COVERED entries but only 19 ADR.md files; missing `./src/c10/ADR.md` and `./src/torch/ADR.md`; no double-nesting detected |
| 3. Exclusion justifications | FAIL | 3 invalid exclusion reasons: `./caffe2` uses "Legacy", `./docs` uses "Documentation sources only", `./torch/contrib` uses "Legacy" — none of these match the 7 permitted reasons |
| 4. ADR content non-stub | PASS | All 19 existing ADR.md files pass all content checks: title heading, section index, Key Files table, dependency links (78 total, 0 broken, 0 relative `../` paths), Runtime Behaviour ≥2 sentences, Performance Profile ≥2 sentences |
| 5. Book cross-reference | PASS | All subsystems named as distinct architectural units in the architecture-map are COVERED directly or via a COVERED ancestor |

## Overall: FAIL

## Required Actions

1. **Write ADR for `c10` at `./src/c10/ADR.md`** — `./c10` is marked COVERED in `adr-scope.md` but has no ADR.md file. The ADR must document the c10 top-level library role, its relationship to `c10/core`, `c10/cuda`, `c10/mobile`, and `c10/util`, and how it provides the minimal-dependency foundation used by ATen and PyTorch. Key files: `c10/CMakeLists.txt`, `c10/macros/`, and the overall directory layout. Dependencies should cross-reference `c10/core/ADR.md`, `c10/cuda/ADR.md`, `c10/mobile/ADR.md`, `c10/util/ADR.md`.

2. **Write ADR for `torch` at `./src/torch/ADR.md`** — `./torch` is marked COVERED in `adr-scope.md` but has no ADR.md file. The ADR must document the Python API surface, the `torch/__init__.py` import orchestration, the layering over `aten/src/ATen` and `torch/csrc`, and the public-facing subpackage boundary policy (which subpackages have independent ADRs and which are documented inline). Dependencies should cross-reference `torch/autograd/ADR.md`, `torch/csrc/ADR.md`, `aten/src/ATen/ADR.md`, and other key subpackage ADRs.

3. **Fix exclusion reason for `./caffe2` in `adr-scope.md`** — current reason "Legacy — historical Caffe2 framework; superseded by PyTorch runtime" does not match any of the 7 valid exclusion reasons. Replace with one of: `Auto-generated code`, `Build/config only`, `Vendored/third-party`, `Test data only`, `Test suite`, `Empty or stub`, or `Leaf with no architectural boundary`. Recommended: `Build/config only` (Caffe2 runtime components are now build infrastructure for PyTorch) or `Vendored/third-party` if treated as a bundled legacy runtime.

4. **Fix exclusion reason for `./docs` in `adr-scope.md`** — current reason "Documentation sources only" does not match any of the 7 valid exclusion reasons. Replace with `Build/config only` (documentation build infrastructure, no architectural logic).

5. **Fix exclusion reason for `./torch/contrib` in `adr-scope.md`** — current reason "Legacy — experimental/contributed code" does not match any of the 7 valid exclusion reasons. Replace with `Leaf with no architectural boundary` (experimental contributed code with no independent architectural role) or `Build/config only` depending on content.
