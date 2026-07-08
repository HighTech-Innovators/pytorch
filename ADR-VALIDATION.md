# ADR Validation Report

Run: 1
Date: 2026-07-08

## Results

| Check | Status | Notes |
|---|---|---|
| 1. Scope map current | FAIL | Depth-1 directory `adr` (pipeline-created tools dir) absent from adr-scope.md; no PENDING entries found |
| 2. Files match COVERED | PASS | 94 actual ADR.md files match 94 COVERED entries; no ./src/src/ wrong-depth files found |
| 3. Exclusion justifications | PASS | All 60 EXCLUDED entries use valid reasons; all line-count checks pass; no excluded dir named as distinct architectural unit in book chapters |
| 4. ADR content non-stub | FAIL | 17 broken dependency links; `tools/stats` Dependencies section lacks table and explicit no-deps statement |
| 5. Book cross-reference | PASS | All 36 architecture-map subsystems are COVERED (directly or via ancestor); no excluded dir is a book-named architectural unit |

## Overall: FAIL

## Required Actions

### Check 1 — Scope map missing entry

- Add `adr` to `adr-scope.md` as EXCLUDED with reason `Build/config only` (it is a pipeline-created directory for validation scripts, not a PyTorch source component). This will also implicitly cover `adr/_tools`.

### Check 4 — Broken dependency links (17 total)

The following ADR dependency links point to files that do not exist. Each must be corrected by either (a) removing the link, (b) replacing with the correct existing ADR path, or (c) replacing with a prose reference if no ADR exists for that dependency.

1. `src/c10/metal/ADR.md` line 35: `aten/src/ATen/mps/ADR.md` does not exist — `aten/src/ATen/mps` is not in adr-scope.md; remove link or replace with prose reference
2. `src/c10/metal/ADR.md` line 36: `aten/src/ATen/native/mps/ADR.md` does not exist — same; remove link or replace with prose reference
3. `src/c10/xpu/ADR.md` line 36: `aten/src/ATen/xpu/ADR.md` does not exist — `aten/src/ATen/xpu` is not in adr-scope.md; remove link or replace with prose reference
4. `src/functorch/dim/ADR.md` line 35: `torch/_C/ADR.md` does not exist — `torch/_C` is EXCLUDED (Empty or stub); remove link or replace with prose reference
5. `src/functorch/dim/ADR.md` line 37: `torch/_tensor/ADR.md` does not exist — `torch/_tensor` is not in adr-scope.md; remove link or replace with prose reference
6. `src/torch/cpu/ADR.md` line 31: `torch/csrc/ADR.md` does not exist — `torch/csrc` is not in adr-scope.md (only sub-directories are COVERED); replace with `torch/csrc/autograd/ADR.md` or `torch/csrc/dynamo/ADR.md` as appropriate
7. `src/torch/fft/ADR.md` line 30: `torch/_torch_docs/ADR.md` does not exist — `torch/_torch_docs` is not in adr-scope.md; remove link or replace with prose reference
8. `src/torch/futures/ADR.md` line 30: `torch/csrc/ADR.md` does not exist — same as item 6; replace with `torch/csrc/distributed/ADR.md`
9. `src/torch/linalg/ADR.md` line 30: `torch/_torch_docs/ADR.md` does not exist — same as item 7; remove link or replace with prose reference
10. `src/torch/nativert/ADR.md` line 33: `torch/export/ADR.md` does not exist — `torch/export` is not in adr-scope.md; replace with `torch/_export/ADR.md` (which is COVERED)
11. `src/torch/nativert/ADR.md` line 35: `torch/csrc/ADR.md` does not exist — same as item 6; replace with `torch/csrc/api/ADR.md`
12. `src/torch/nested/ADR.md` line 33: `torch/csrc/ADR.md` does not exist — same as item 6; replace with `torch/csrc/autograd/ADR.md`
13. `src/torch/package/ADR.md` line 33: `torch/csrc/ADR.md` does not exist — same as item 6; replace with `torch/csrc/jit/ADR.md`
14. `src/torch/package/ADR.md` line 34: `torch/serialization/ADR.md` does not exist — `torch/serialization` is a file (`.py`), not a directory; remove link or replace with prose reference (serialization is documented in `caffe2/serialize/ADR.md`)
15. `src/torch/quantization/ADR.md` line 32: `torch/ao/quantization/ADR.md` does not exist — `torch/ao` is EXCLUDED; remove link or replace with prose reference
16. `src/torchgen/operator_versions/ADR.md` line 30: `torch/jit/ADR.md` does not exist — `torch/jit` is not in adr-scope.md; replace with `torch/csrc/jit/ADR.md` (which is COVERED)
17. `src/torchgen/operator_versions/ADR.md` line 31: `torch/csrc/jit/mobile/ADR.md` does not exist — `torch/csrc/jit/mobile` is not in adr-scope.md; remove link or replace with prose reference

### Check 4 — Dependencies section: missing table or explicit no-deps statement

- `src/tools/stats/ADR.md` Dependencies section: states "No single src-local ADR dependency dominates" but does not have a dependency table and does not contain an explicit "no notable dependencies" statement. Add either a table row for any notable src-local dependency or change the prose to explicitly state "There are no notable src-local ADR-tracked dependencies."
