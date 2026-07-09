# ADR Validation Report

Run: 2
Date: 2026-07-09

## Results

| Check | Status | Notes |
|---|---|---|
| 1. Scope map current | FAIL | `./adr` is a depth-1 directory absent from `adr-scope.md`; no PENDING entries; all other depth-1 source directories present |
| 2. Files match COVERED | PASS | 21 COVERED entries, 21 ADR.md files found; all COVERED dirs have `ADR.md` at exact path; no double-nesting under `./src/src/`; counts match |
| 3. Exclusion justifications | FAIL | All 7 exclusion reasons are syntactically valid; 27 line-count violations (26 × `Leaf with no architectural boundary` >200 lines, 1 × `Build/config only` >2000 lines); 2 `Leaf` dirs additionally named in book with distinct roles |
| 4. ADR content non-stub | PASS | All 21 ADR.md files pass: title heading, section index with all 7 links, Key Files table (no placeholders), Dependencies table, dependency links (0 broken, 0 `../` relative paths), Runtime Behaviour ≥2 sentences, Performance Profile ≥2 sentences |
| 5. Book cross-reference | PASS | All 21 book-named subsystems from chapter-map and component-map are COVERED directly or via a COVERED ancestor; no EXCLUDED directory with no COVERED ancestor is named as a distinct architectural unit |

## Overall: FAIL

## Required Actions

### Check 1 — Scope map

1. **Add `./adr` to `src/adr-scope.md` as EXCLUDED** — it is a depth-1 directory (ADR validation tooling created by the validator per spec) absent from the scope map. Add a row: `| ./adr | no | EXCLUDED | Build/config only — ADR generation and validation tooling; not a PyTorch architectural component |`. No ADR file is required.

### Check 3 — `tools` line-count violation (Build/config only, limit 2000 lines)

2. **Reclassify `./tools` in `src/adr-scope.md` or write an ADR** — `find tools -maxdepth 1 \( -name '*.py' ... \) | xargs wc -l` = 4282 lines, which exceeds the 2000-line limit for `Build/config only`. Either write `./src/tools/ADR.md` documenting the build tooling architecture and mark `./tools` as COVERED, or change the exclusion reason to one that does not impose a line-count limit if the content genuinely warrants it (e.g., `Auto-generated code` is not appropriate here; `Vendored/third-party` is not appropriate; the only valid alternative without a limit is to document it).

### Check 3 — `Leaf with no architectural boundary` line-count violations (limit 200 lines)

Each directory below uses `Leaf with no architectural boundary` but has more than 200 lines of source at `maxdepth 1`. For each: either (a) write an `ADR.md` at `./src/<dir>/ADR.md` and change the status to `COVERED` in `adr-scope.md`, or (b) change the exclusion reason to a valid reason whose line-count limit is not exceeded. No valid alternative without a line limit exists for first-party non-test Python code (Auto-generated/Vendored/Test suite do not apply), so writing an ADR is the correct action for most.

3. `./torch/utils` — 18413 lines. **Write `./src/torch/utils/ADR.md`** covering DataLoader, checkpoint, mobile deployment bridge, and `data.distributed` patterns. Mark COVERED.
4. `./torch/_higher_order_ops` — 18625 lines. **Write `./src/torch/_higher_order_ops/ADR.md`** covering higher-order operator dispatch, `cond`, `while_loop`, `associative_scan` semantics. Mark COVERED.
5. `./torch/sparse` — 12622 lines. **Write `./src/torch/sparse/ADR.md`** covering sparse tensor formats (COO, CSR, CSC, BSR, BSC) and operation dispatch. Mark COVERED.
6. `./torch/export` — 12426 lines. **Write `./src/torch/export/ADR.md`** covering the public `torch.export.export()` API surface, `ExportedProgram`, and its relationship to `torch/_export`. Mark COVERED.
7. `./torch/cuda` — 10488 lines. **Write `./src/torch/cuda/ADR.md`** covering Python-level CUDA device management, memory stats, stream/event API, and CUDAGraph bindings. Mark COVERED.
8. `./torch/distributions` — 9914 lines. **Write `./src/torch/distributions/ADR.md`** covering the probability distributions library (Constraint, Transform, Distribution base class, RSample/log_prob design). Mark COVERED.
9. `./torch/_decomp` — 7490 lines. **Write `./src/torch/_decomp/ADR.md`** covering operator decomposition rules, `@register_decomposition`, core/extra decomp tables, and their role in AOT autograd / export. Mark COVERED.
10. `./c10/metal` — 7157 lines. **Write `./src/c10/metal/ADR.md`** covering the Apple Metal device abstraction layer and its relationship to `c10/cuda`. Mark COVERED, or use `Leaf with no architectural boundary` only if content is relocated to parent ADR and line count confirmed below limit.
11. `./torch/_prims` — 3897 lines. **Write `./src/torch/_prims/ADR.md`** covering primitive operation semantics, device-generic lowering, and the contract with `torch/_refs` and `torch/_functorch`. Mark COVERED.
12. `./c10/xpu` — 3886 lines. **Write `./src/c10/xpu/ADR.md`** covering the Intel XPU device abstraction and its relationship to `c10/cuda`. Mark COVERED.
13. `./torchgen/api` — 4916 lines (also named in book chapter 06 directory table as "Type translation and signature generation"). **Write `./src/torchgen/api/ADR.md`** covering the type-translation and signature-generation pipeline (dispatcher.py, native.py, structured.py, cpp.py, autograd.py, meta.py). Mark COVERED.
14. `./torchgen/dest` — 2435 lines (also named in book chapter 06 directory table as "Code generation targets"). **Write `./src/torchgen/dest/ADR.md`** covering native-function registration output, dispatch-key registration, and lazy IR generation. Mark COVERED.
15. `./torch/linalg` — 3174 lines. **Write `./src/torch/linalg/ADR.md`** covering the linear algebra operation library, its operator schema, and relationship to ATen kernels. Mark COVERED.
16. `./torch/compiler` — 1670 lines. **Write `./src/torch/compiler/ADR.md`** covering the `torch.compile()` public API shim, `_dynamo` wiring, and compiler options. Mark COVERED.
17. `./torch/fft` — 1442 lines. **Write `./src/torch/fft/ADR.md`** covering the FFT operation library and its ATen-dispatch design. Mark COVERED.
18. `./torch/amp` — 1261 lines. **Write `./src/torch/amp/ADR.md`** covering automatic mixed precision (autocast context manager, GradScaler, dtype promotion policy). Mark COVERED.
19. `./torch/multiprocessing` — 1238 lines. **Write `./src/torch/multiprocessing/ADR.md`** covering shared-memory tensor semantics, spawn/fork context, and `Pool` integration. Mark COVERED.
20. `./torch/_custom_op` — 1032 lines. **Write `./src/torch/_custom_op/ADR.md`** covering the custom operator registration API, schema inference, and interaction with torchscript/export. Mark COVERED.
21. `./torch/accelerator` — 870 lines. **Write `./src/torch/accelerator/ADR.md`** covering the device-agnostic accelerator API, current-device abstraction, and its backends. Mark COVERED.
22. `./torch/quantization` — 589 lines. **Write `./src/torch/quantization/ADR.md`** covering the quantisation API, QConfig, observer, and quantise/dequantise lifecycle. Mark COVERED.
23. `./torch/nested` — 522 lines. **Write `./src/torch/nested/ADR.md`** covering nested tensor semantics, ragged-size representation, and supported operations. Mark COVERED.
24. `./torch/futures` — 335 lines. **Write `./src/torch/futures/ADR.md`** covering async future primitives, `Future[T]`, and RPC interaction. Mark COVERED.
25. `./torch/func` — 296 lines. **Write `./src/torch/func/ADR.md`** covering the public `vmap`/`grad`/`jacrev`/`hessian` API surface and its delegation to `torch/_functorch`. Mark COVERED.
26. `./torch/cpu` — 252 lines. **Write `./src/torch/cpu/ADR.md`** covering CPU-thread and memory management APIs. Mark COVERED.
27. `./torchgen/aoti` — 232 lines. **Write `./src/torchgen/aoti/ADR.md`** covering AOT Inductor code-generation targets and their role in the `torch.export` → compiled-artifact pipeline. Mark COVERED.
28. `./torch/_dispatch` — 204 lines. **Write `./src/torch/_dispatch/ADR.md`** covering Python-level dispatch utilities and their role in operator testing and dispatch tracing. Mark COVERED.
