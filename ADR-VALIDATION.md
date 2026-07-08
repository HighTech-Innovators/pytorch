# ADR Validation Report

Run: 2
Date: 2026-07-08

## Results

| Check | Status | Notes |
|---|---|---|
| 1. Scope map current | FAIL | 41 PENDING entries; 6 depth-1 dirs absent (aten, c10, caffe2, functorch, tools, torch); 494 unclassified directories |
| 2. Files match COVERED | FAIL | 0 COVERED entries in adr-scope.md; 41 ADR.md files exist but scope not updated from PENDING; count mismatch 41 vs 0 |
| 3. Exclusion justifications | FAIL | 105 of 106 EXCLUDED entries use reasons not in the 7 approved values; torch/_decomp and torch/onnx EXCLUDED but named in book as distinct architectural units |
| 4. ADR content non-stub | FAIL | Broken link in ./src/c10/core/ADR.md line 43: c10/macros/ADR.md does not exist |
| 5. Book cross-reference | FAIL | torch/_decomp EXCLUDED but named as distinct unit in chapter 09; torch/onnx EXCLUDED but named as distinct unit in chapter 11 |

## Overall: FAIL

## Required Actions

### Check 1 — Scope map must be current

1. Add the following 6 depth-1 directories to `adr-scope.md` with an appropriate status (EXCLUDED is likely correct for the top-level parent dirs that are not themselves architectural units):
   - `aten` — EXCLUDED (parent container; architectural units are its subdirectories already in scope)
   - `c10` — EXCLUDED (parent container; architectural units are its subdirectories already in scope)
   - `caffe2` — EXCLUDED (parent container; `caffe2/core` is separately COVERED)
   - `functorch` — EXCLUDED (parent container; `functorch/compile` and `functorch/_src` are separately COVERED)
   - `tools` — EXCLUDED (parent container; `tools/autograd`, `tools/code_analyzer`, `tools/jit` are separately COVERED)
   - `torch` — EXCLUDED (parent container; architectural units are its subdirectories already in scope)

2. Once depth-1 parents are added with EXCLUDED status, their unclassified descendants (e.g., `aten/src`, `aten/src/ATen`, `aten/src/ATen/accelerator`, `aten/src/ATen/core/boxing`, and ~490 others) will be implicitly covered by the EXCLUDED ancestor rule and no longer require explicit entries. Verify by re-running `validate_scope.py`.

3. Change all 41 PENDING entries to COVERED — ADR.md files have been written for every one of them:
   - `aten/src/ATen/core` → COVERED
   - `aten/src/ATen/native` → COVERED
   - `aten/src/ATen/native/cpu` → COVERED
   - `aten/src/ATen/native/cuda` → COVERED
   - `aten/src/ATen/native/quantized` → COVERED
   - `aten/src/ATen/native/sparse` → COVERED
   - `c10/core` → COVERED
   - `c10/cuda` → COVERED
   - `c10/mobile` → COVERED
   - `c10/util` → COVERED
   - `caffe2/core` → COVERED
   - `functorch/_src` → COVERED
   - `functorch/compile` → COVERED
   - `tools/autograd` → COVERED
   - `tools/code_analyzer` → COVERED
   - `tools/jit` → COVERED
   - `torch/_dynamo` → COVERED
   - `torch/_export` → COVERED
   - `torch/_functorch` → COVERED
   - `torch/_inductor` → COVERED
   - `torch/_inductor/codegen` → COVERED
   - `torch/amp` → COVERED
   - `torch/autograd` → COVERED
   - `torch/csrc/api` → COVERED
   - `torch/csrc/autograd` → COVERED
   - `torch/csrc/distributed` → COVERED
   - `torch/csrc/dynamo` → COVERED
   - `torch/csrc/inductor` → COVERED
   - `torch/csrc/jit` → COVERED
   - `torch/csrc/profiler` → COVERED
   - `torch/cuda` → COVERED
   - `torch/distributed` → COVERED
   - `torch/fx` → COVERED
   - `torch/nn` → COVERED
   - `torch/nn/modules` → COVERED
   - `torch/optim` → COVERED
   - `torch/profiler` → COVERED
   - `torch/utils` → COVERED
   - `torchgen` → COVERED
   - `torchgen/api` → COVERED
   - `torchgen/dest` → COVERED

### Check 3 — Exclusion reasons must use approved values

Replace every non-conforming exclusion reason with exactly one of: `Auto-generated code`, `Build/config only`, `Vendored/third-party`, `Test data only`, `Test suite`, `Empty or stub`, `Leaf with no architectural boundary`. The 105 entries requiring correction and their suggested replacements:

- `benchmarks`: 'Benchmark scripts, no core logic' → `Test suite`
- `binaries`: 'Binary entrypoints' → `Build/config only`
- `cmake`: 'Build configuration' → `Build/config only`
- `docs`: 'Documentation' → `Build/config only`
- `android`: 'Mobile/device-specific, non-core' → `Build/config only`
- `scripts`: 'Scripts' → `Build/config only`
- `third_party`: 'Vendored code' → `Vendored/third-party`
- `mypy_plugins`: 'Type-checking plugins' → `Build/config only`
- `c10/benchmark`: 'Benchmarks' → `Test suite`
- `c10/hip`: 'ROCm variant' → `Build/config only`
- `c10/macros`: 'Header macros only' → `Auto-generated code`
- `c10/metal`: 'Metal GPU variant' → `Build/config only`
- `c10/test`: 'Tests' → `Test suite`
- `c10/xpu`: 'XPU variant' → `Build/config only`
- `aten/src/ATen/templates`: 'Code-gen templates' → `Auto-generated code`
- `aten/src/ATen/test`: 'Tests' → `Test suite`
- `aten/src/ATen/benchmarks`: 'Benchmarks' → `Test suite`
- `aten/tools`: 'ATen build tooling' → `Build/config only`
- `tools/alerts`: 'CI alerting' → `Build/config only`
- `tools/amd_build`: 'ROCm build' → `Build/config only`
- `tools/build_defs`: 'Build definitions' → `Build/config only`
- `tools/code_coverage`: 'Coverage tooling' → `Build/config only`
- `tools/coverage_plugins_package`: 'Coverage plugins' → `Build/config only`
- `tools/dynamo`: 'Dynamo dev tooling' → `Build/config only`
- `tools/experimental`: 'Experimental tooling' → `Build/config only`
- `tools/gdb`: 'Debugger scripts' → `Build/config only`
- `tools/github`: 'CI/GitHub tooling' → `Build/config only`
- `tools/iwyu`: 'include-what-you-use config' → `Build/config only`
- `tools/linter`: 'Linters' → `Build/config only`
- `tools/lite_interpreter`: 'Lite interpreter tooling' → `Build/config only`
- `tools/lldb`: 'Debugger scripts' → `Build/config only`
- `tools/packaging`: 'Packaging' → `Build/config only`
- `tools/pyi`: 'Stub generation' → `Auto-generated code`
- `tools/setup_helpers`: 'Setup helpers' → `Build/config only`
- `tools/shared`: 'Shared build helpers' → `Build/config only`
- `tools/stats`: 'CI stats' → `Build/config only`
- `tools/test`: 'Tests' → `Test suite`
- `tools/testing`: 'Test infra' → `Test suite`
- `tools/vendoring`: 'Vendoring' → `Vendored/third-party`
- `functorch/benchmarks`: 'Benchmarks' → `Test suite`
- `functorch/docs`: 'Documentation' → `Build/config only`
- `functorch/examples`: 'Examples' → `Build/config only`
- `functorch/experimental`: 'Experimental' → `Build/config only`
- `functorch/op_analysis`: 'Analysis scripts' → `Build/config only`
- `functorch/dim`: 'Named-dim prototype' → `Leaf with no architectural boundary`
- `functorch/einops`: 'einops shim' → `Leaf with no architectural boundary`
- `caffe2/perfkernels`: 'Legacy perf kernels' → `Leaf with no architectural boundary`
- `caffe2/serialize`: 'Legacy serialization' → `Leaf with no architectural boundary`
- `caffe2/utils`: 'Legacy utils' → `Leaf with no architectural boundary`
- `torch/_awaits`: 'Thin JIT await wrapper' → `Leaf with no architectural boundary`
- `torch/backends`: 'Backend config flags' → `Leaf with no architectural boundary`
- `torch/_C`: 'Native extension stub' → `Empty or stub`
- `torch/_C_flatbuffer`: 'Native stub' → `Empty or stub`
- `torch/compiler`: 'Thin compiler facade' → `Leaf with no architectural boundary`
- `torch/contrib`: 'Contrib extras' → `Leaf with no architectural boundary`
- `torch/cpu`: 'Thin CPU facade' → `Leaf with no architectural boundary`
- `torch/_custom_op`: 'Legacy custom-op shim' → `Leaf with no architectural boundary`
- `torch/_decomp`: 'Decomposition tables' → **must be COVERED** (see Check 5 below)
- `torch/_dispatch`: 'Python dispatch helpers' → `Leaf with no architectural boundary`
- `torch/fft`: 'Thin functional wrapper' → `Leaf with no architectural boundary`
- `torch/func`: 'Thin wrapper over _functorch' → `Leaf with no architectural boundary`
- `torch/futures`: 'Thin futures wrapper' → `Leaf with no architectural boundary`
- `torch/headeronly`: 'Header-only shims' → `Leaf with no architectural boundary`
- `torch/_higher_order_ops`: 'HOP definitions' → `Leaf with no architectural boundary`
- `torch/_lazy`: 'Lazy tensor backend' → `Leaf with no architectural boundary`
- `torch/legacy`: 'Legacy code' → `Leaf with no architectural boundary`
- `torch/lib`: 'Prebuilt libs' → `Vendored/third-party`
- `torch/_library`: 'Library registration helpers' → `Leaf with no architectural boundary`
- `torch/linalg`: 'Thin functional wrapper' → `Leaf with no architectural boundary`
- `torch/_logging`: 'Logging config' → `Leaf with no architectural boundary`
- `torch/masked`: 'Masked tensor prototype' → `Leaf with no architectural boundary`
- `torch/monitor`: 'Monitoring hooks' → `Leaf with no architectural boundary`
- `torch/mps`: 'MPS device variant' → `Build/config only`
- `torch/mtia`: 'MTIA device variant' → `Build/config only`
- `torch/multiprocessing`: 'MP wrappers' → `Leaf with no architectural boundary`
- `torch/_native`: 'Native shim' → `Leaf with no architectural boundary`
- `torch/nativert`: 'Native runtime prototype' → `Leaf with no architectural boundary`
- `torch/nested`: 'Nested tensor wrapper' → `Leaf with no architectural boundary`
- `torch/numa`: 'NUMA bindings' → `Leaf with no architectural boundary`
- `torch/_numpy`: 'NumPy compat layer' → `Leaf with no architectural boundary`
- `torch/onnx`: 'ONNX export (self-contained)' → **must be COVERED** (see Check 5 below)
- `torch/package`: 'Packaging' → `Leaf with no architectural boundary`
- `torch/_prims`: 'Primitive op refs' → `Leaf with no architectural boundary`
- `torch/_prims_common`: 'Prim helpers' → `Leaf with no architectural boundary`
- `torch/quantization`: 'Legacy quant shim (see torch/ao)' → `Leaf with no architectural boundary`
- `torch/_refs`: 'Reference decompositions' → `Leaf with no architectural boundary`
- `torch/signal`: 'Signal windows' → `Leaf with no architectural boundary`
- `torch/sparse`: 'Thin sparse wrapper' → `Leaf with no architectural boundary`
- `torch/special`: 'Thin functional wrapper' → `Leaf with no architectural boundary`
- `torch/_strobelight`: 'Profiling integration' → `Leaf with no architectural boundary`
- `torch/_subclasses`: 'Fake/functional tensor subclasses' → `Leaf with no architectural boundary`
- `torch/testing`: 'Test utilities' → `Test suite`
- `torch/_vendor`: 'Vendored code' → `Vendored/third-party`
- `torch/xpu`: 'XPU device variant' → `Build/config only`
- `torch/accelerator`: 'Accelerator facade' → `Leaf with no architectural boundary`
- `torch/ao`: 'Quant/sparsity (large, self-contained)' → `Leaf with no architectural boundary`
- `torch/distributions`: 'Probability distributions' → `Leaf with no architectural boundary`
- `torchgen/aoti`: 'AOTInductor codegen' → `Auto-generated code`
- `torchgen/_autoheuristic`: 'Autoheuristic' → `Leaf with no architectural boundary`
- `torchgen/decompositions`: 'Decomp codegen' → `Auto-generated code`
- `torchgen/fuse`: 'Fusion codegen' → `Auto-generated code`
- `torchgen/operator_versions`: 'Operator versioning' → `Leaf with no architectural boundary`
- `torchgen/selective_build`: 'Selective build' → `Build/config only`
- `torchgen/shape_functions`: 'Shape function codegen' → `Auto-generated code`
- `torchgen/static_runtime`: 'Static runtime codegen' → `Auto-generated code`

### Check 4 — ADR content

- Broken link in `./src/c10/core/ADR.md` line 43: `c10/macros/ADR.md` does not exist — `c10/macros` is EXCLUDED and has no ADR. Remove or replace this link; there is no ADR to link to for `c10/macros`. Replace the row with a prose note in the Dependencies section or remove the entry.

### Check 5 — Book cross-reference

- `torch/_decomp` is cited in book chapter 09 (`09-fx-intermediate-representation.md`, lines 131, 182, 233) as a distinct architectural unit (`torch/_decomp/` — operator decomposition rules). It must not be EXCLUDED. Change its status in `adr-scope.md` to COVERED and write `./src/torch/_decomp/ADR.md` covering: decomposition registry, operator lowering to primitive ops, `torch.ops.aten` dispatch, and `decompositions.py` structure.
- `torch/onnx` is cited in book chapter 11 (`11-serialization-and-export.md`, lines 115, 120, 266) as a distinct architectural unit (`torch/onnx/` — ONNX export). It must not be EXCLUDED. Change its status in `adr-scope.md` to COVERED and write `./src/torch/onnx/ADR.md` covering: the `torch.onnx.export` API, ONNX opset versioning, symbolic function registry, and export pipeline.
