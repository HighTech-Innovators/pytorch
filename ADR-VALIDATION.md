# ADR Validation Report

Run: 2
Date: 2026-07-10

## Results

| Check | Status | Notes |
|---|---|---|
| 1. Scope map current | FAIL | 20 PENDING entries; see details below |
| 2. Files match COVERED | FAIL | 5 COVERED dirs missing ADR.md; 13 ADR.md files at PENDING (not COVERED) paths; count mismatch (14 actual vs 6 COVERED) |
| 3. Exclusion justifications | FAIL | 9 of 10 exclusion reasons do not match any of the 7 valid phrases exactly |
| 4. ADR content non-stub | FAIL | 5 broken dependency links across 5 ADR files |
| 5. Book cross-reference | FAIL | `test` is EXCLUDED but appears as a row in the architecture-map.md layer table |

## Overall: FAIL

## Required Actions

### Check 1 — Scope map: resolve all PENDING entries

The following 13 directories already have `ADR.md` files and must be updated to `COVERED` in `adr-scope.md`:

- `c10/core` → `./src/c10/core/ADR.md` exists — mark COVERED
- `c10/util` → `./src/c10/util/ADR.md` exists — mark COVERED
- `c10/cuda` → `./src/c10/cuda/ADR.md` exists — mark COVERED
- `aten/src/ATen` → `./src/aten/src/ATen/ADR.md` exists — mark COVERED
- `aten/src/ATen/native` → `./src/aten/src/ATen/native/ADR.md` exists — mark COVERED
- `aten/src/ATen/native/cpu` → `./src/aten/src/ATen/native/cpu/ADR.md` exists — mark COVERED
- `torch/csrc` → `./src/torch/csrc/ADR.md` exists — mark COVERED
- `torch/csrc/autograd` → `./src/torch/csrc/autograd/ADR.md` exists — mark COVERED
- `torch/csrc/jit` → `./src/torch/csrc/jit/ADR.md` exists — mark COVERED
- `torch/csrc/api` → `./src/torch/csrc/api/ADR.md` exists — mark COVERED
- `torch/autograd` → `./src/torch/autograd/ADR.md` exists — mark COVERED
- `torch/nn` → `./src/torch/nn/ADR.md` exists — mark COVERED
- `torch/nn/parallel` → `./src/torch/nn/parallel/ADR.md` exists — mark COVERED

The following 7 directories are PENDING with no `ADR.md` — write an ADR for each and mark COVERED:

- `torch/distributed` — write `./src/torch/distributed/ADR.md` covering process-group API, collective operations, FSDP/DDP integration, and the `_C._distributed_c10d` C++ bridge
- `torch/_dynamo` — write `./src/torch/_dynamo/ADR.md` covering graph-capture via symbolic bytecode tracing (`symbolic_convert.py`), guard generation (`guards.py`), and the `torch.compile` entry point
- `torch/_inductor` — write `./src/torch/_inductor/ADR.md` covering Triton/C++ code generation, loop tiling, and the lowering pipeline from FX graphs
- `torch/fx` — write `./src/torch/fx/ADR.md` covering proxy-based tracing, the `Graph`/`Node` IR, and transformation pass API
- `torch/profiler` — write `./src/torch/profiler/ADR.md` covering the `profile()` context manager, Kineto integration, and the chrome-trace/memory-profiler output paths
- `torch/jit` — write `./src/torch/jit/ADR.md` covering the `torch.jit.script` entry point (`_script.py`, 1806 lines), the Python-to-TorchScript compiler, and the `torch.jit.trace` path
- `tools/autograd` — write `./src/tools/autograd/ADR.md` covering `derivatives.yaml`, the codegen pipeline that emits `generated/autograd/Functions.h`, and the relationship to `torchgen/`

### Check 2 — Files match COVERED entries

In addition to promoting the 13 PENDING entries above, the following 5 currently-COVERED directories are missing their `ADR.md`:

- Write `./src/aten/ADR.md` covering the top-level ATen layout (operator YAML, `TensorIterator`, dispatch table bootstrap) and how it relates to the `aten/src/ATen`, `aten/src/ATen/native`, and `aten/src/ATen/native/cpu` sub-ADRs
- Write `./src/functorch/ADR.md` covering the function-transform frontend (vmap, grad, jvp) and its relationship to `torch/_C` dispatch hooks
- Write `./src/tools/ADR.md` covering the overall `tools/` layout, distinguishing build-tooling directories from the architecturally significant `tools/autograd/` sub-tree (with a cross-reference to `tools/autograd/ADR.md`)
- Write `./src/torch/ADR.md` covering the top-level Python public API, module-import bootstrap (`__init__.py`), and the role of `torch._C` as the C-extension bridge, with cross-references to major sub-ADRs
- Write `./src/torchgen/ADR.md` covering the YAML-driven operator code-generation pipeline (`gen.py`, `model.py`) and the output artifacts consumed by `aten/` and `tools/autograd/`

### Check 3 — Exclusion reasons: use exact valid phrases

Replace free-form reasons in `adr-scope.md` with one of the seven permitted phrases:

- `android` → change reason to `Build/config only` (or to a valid phrase; "Android platform bindings; out of CPU-only Linux deployment scope" is not a valid reason — closest valid options are `Build/config only` or `Leaf with no architectural boundary`)
- `benchmarks` → change reason to `Build/config only` ("Benchmark scripts, not production architecture" is not a valid reason)
- `binaries` → change reason to `Build/config only` ("Standalone benchmark/tool entry-point executables" is not a valid reason)
- `caffe2` → change reason to `Build/config only` or `Leaf with no architectural boundary` ("Legacy Caffe2 subsystem, deprecated" is not a valid reason; if caffe2 is a substantial deprecated codebase it may need `COVERED` treatment — verify line count)
- `cmake` → change reason to `Build/config only` ("Build configuration only" is not an exact match)
- `docs` → change reason to `Build/config only` ("Documentation" is not a valid reason)
- `mypy_plugins` → change reason to `Build/config only` ("Type-checker plugins (tooling)" is not a valid reason)
- `scripts` → change reason to `Build/config only` ("CI/tooling scripts" is not a valid reason)
- `third_party` → change reason to `Vendored/third-party` ("Vendored third-party code" is not an exact match)

Note: `test` already uses the exact valid phrase `Test suite`.

### Check 4 — Broken dependency links

Fix the following broken ADR cross-links (each target ADR must be written before these links can resolve):

- `aten/src/ATen/ADR.md` line 40: link `torchgen/ADR.md` — `./src/torchgen/ADR.md` does not exist; write `./src/torchgen/ADR.md` (see Check 2 action above)
- `torch/csrc/ADR.md` line 40: link `torchgen/ADR.md` — `./src/torchgen/ADR.md` does not exist; write `./src/torchgen/ADR.md` (see Check 2 action above)
- `torch/csrc/autograd/ADR.md` line 38: link `tools/autograd/ADR.md` — `./src/tools/autograd/ADR.md` does not exist; write `./src/tools/autograd/ADR.md` (see Check 1 action above)
- `torch/csrc/jit/ADR.md` line 41: link `torch/jit/ADR.md` — `./src/torch/jit/ADR.md` does not exist; write `./src/torch/jit/ADR.md` (see Check 1 action above)
- `torch/nn/parallel/ADR.md` line 35: link `torch/distributed/ADR.md` — `./src/torch/distributed/ADR.md` does not exist; write `./src/torch/distributed/ADR.md` (see Check 1 action above)

### Check 5 — Book cross-reference

- `test/` appears as a row in `book/_generated/architecture-map.md`'s layer table but is marked `EXCLUDED` in `adr-scope.md` with no COVERED ancestor. Either: (a) add a note in `adr-scope.md` for `test` clarifying it is test infrastructure (not an architectural unit) and keep it EXCLUDED with reason `Test suite`; or (b) if the book explicitly frames the test architecture as a distinct architectural unit, change its status to `COVERED` and write `./src/test/ADR.md`. Option (a) is the expected path given the valid reason `Test suite`.
