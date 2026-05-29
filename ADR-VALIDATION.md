# ADR Validation Report

Run: 1
Date: 2026-05-29

## Results

| Check | Status | Notes |
|---|---|---|
| 1. Scope map current | FAIL | `src/adr-scope.md` does not exist |
| 2. Files match COVERED | FAIL | 0 ADR.md files found; architecture map lists ~40 architectural units with no coverage |
| 3. Exclusion justifications | FAIL | Cannot evaluate — scope map absent; no EXCLUDED entries exist |
| 4. ADR content non-stub | FAIL | No ADR.md files exist anywhere in the repository |
| 5. Book cross-reference | FAIL | Book names 39 distinct architectural units; none appear as COVERED in adr-scope.md (absent) |

## Overall: FAIL

## Required Actions

- Create `src/adr-scope.md` covering all 1227 directories found under `src/`. Every directory must be classified as `COVERED` or `EXCLUDED` with an accepted exclusion reason; no `PENDING` entries may remain.

- Write ADR.md for each of the following architectural units named in the book (`book/_generated/architecture-map.md` and `book/_generated/component-map.md`) that must be `COVERED`:
  - `c10/core` at `c10/core/ADR.md` — tensor primitives, TensorImpl, Storage, DispatchKeySet
  - `c10/util` at `c10/util/ADR.md` — intrusive_ptr, exception utilities
  - `c10/cuda` at `c10/cuda/ADR.md` — CUDA device primitives, CUDACachingAllocator
  - `aten/src/ATen/core` at `aten/src/ATen/core/ADR.md` — operator schemas, dispatch tables
  - `aten/src/ATen/core/dispatch` at `aten/src/ATen/core/dispatch/ADR.md` — Dispatcher implementation
  - `aten/src/ATen/native` at `aten/src/ATen/native/ADR.md` — CPU kernel implementations
  - `aten/src/ATen/native/cpu` at `aten/src/ATen/native/cpu/ADR.md` — vectorized CPU kernels
  - `aten/src/ATen/native/cuda` at `aten/src/ATen/native/cuda/ADR.md` — CUDA kernel implementations
  - `aten/src/ATen/cuda` at `aten/src/ATen/cuda/ADR.md` — CUDA infrastructure
  - `aten/src/ATen/detail` at `aten/src/ATen/detail/ADR.md` — ATen internal details
  - `aten/src/ATen/functorch` at `aten/src/ATen/functorch/ADR.md` — Functorch C++ layer
  - `torch/nn/modules` at `torch/nn/modules/ADR.md` — nn.Module hierarchy
  - `torch/nn/utils` at `torch/nn/utils/ADR.md` — module utilities
  - `torch/autograd` at `torch/autograd/ADR.md` — Python autograd interface
  - `torch/csrc/autograd` at `torch/csrc/autograd/ADR.md` — C++ autograd engine
  - `torch/optim` at `torch/optim/ADR.md` — optimizer implementations
  - `torch/fx` at `torch/fx/ADR.md` — FX Graph IR
  - `torch/_dynamo` at `torch/_dynamo/ADR.md` — TorchDynamo bytecode capture, guard mechanism
  - `torch/_inductor` at `torch/_inductor/ADR.md` — TorchInductor code generation, scheduler
  - `torch/_functorch` at `torch/_functorch/ADR.md` — Functorch/AOTAutograd
  - `torch/distributed` at `torch/distributed/ADR.md` — ProcessGroup, DDP, comm hooks
  - `torch/distributed/_tensor` at `torch/distributed/_tensor/ADR.md` — DTensor abstraction
  - `torch/distributed/fsdp` at `torch/distributed/fsdp/ADR.md` — Fully Sharded Data Parallel
  - `torch/cuda` at `torch/cuda/ADR.md` — CUDA Python interface
  - `torch/profiler` at `torch/profiler/ADR.md` — profiling infrastructure, Kineto
  - `torch/utils` at `torch/utils/ADR.md` — utilities including data loading
  - `torch/export` at `torch/export/ADR.md` — model export interface
  - `torch/_export` at `torch/_export/ADR.md` — export internals
  - `torch/compiler` at `torch/compiler/ADR.md` — compiler interface
  - `torch/amp` at `torch/amp/ADR.md` — automatic mixed precision
  - `torch/sparse` at `torch/sparse/ADR.md` — sparse tensor support
  - `torch/linalg` at `torch/linalg/ADR.md` — linear algebra
  - `torch/fft` at `torch/fft/ADR.md` — Fourier transforms
  - `torch/distributions` at `torch/distributions/ADR.md` — probability distributions
  - `torch/jit` at `torch/jit/ADR.md` — TorchScript JIT
  - `torch/package` at `torch/package/ADR.md` — model packaging
  - `torch/multiprocessing` at `torch/multiprocessing/ADR.md` — multiprocessing utilities
  - `torchgen` at `torchgen/ADR.md` — code generation tools
  - `caffe2` at `caffe2/ADR.md` — legacy Caffe2 (or classify as EXCLUDED with accepted reason if warranted)
  - `tools` at `tools/ADR.md` — build and development tools (or classify as EXCLUDED `Build/config only` if warranted)

- Each ADR.md must contain substantive content (minimum 2 complete sentences per heading) under: Architectural role, Dependencies, and Trade-offs, plus at least one reference to an actual file or module from the source repository.

- All remaining 1000+ lower-level directories must be classified in `adr-scope.md`; those classified as `EXCLUDED` must use one of the seven accepted exclusion reasons and pass the applicable line-count thresholds.
