# `tools`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`tools` is the build-time tooling root. It contains the autograd codegen subsystem (`tools/autograd/`, architecturally significant) plus a collection of build scripts, CI helpers, and code analyzers that are build/CI infrastructure, not runtime architecture.

## Key Files

| File | Purpose |
|---|---|
| `tools/autograd/` | Architecturally significant: gradient formula YAML + codegen pipeline (see [tools/autograd/ADR.md](tools/autograd/ADR.md)) |
| `tools/build_libtorch.py` | Entry point for building standalone LibTorch |
| `tools/build_pytorch_libs.py` | Builds individual PyTorch C++ libraries |
| `tools/code_analyzer/` | Analyzes operator usage for selective build (mobile/lite builds) |

## Public Interface

`tools` has no runtime Python API. `tools/autograd/gen_autograd.py` is invoked by the build system. Everything else in `tools/` is build or CI infrastructure.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [tools/autograd](tools/autograd/ADR.md) | contains | Architecturally significant sub-tree; see child ADR |
| [torchgen](torchgen/ADR.md) | depends-on | `tools/autograd/gen_autograd.py` imports `torchgen/model.py` types |
| Build system (CMake/Ninja) | depended-on-by | Build rules invoke scripts in `tools/` |

## Runtime Behaviour

Nothing in `tools/` is imported or executed at runtime. All activity is at build time: CMake invokes `tools/autograd/gen_autograd.py` (which in turn calls into `torchgen`) to produce generated C++ source files before the C++ compiler runs. The remaining directories (`tools/build_defs`, `tools/amd_build`, `tools/code_coverage`, etc.) contain CI scripts and build helpers run by the PyTorch CI infrastructure.

## Performance Profile

Build-time cost only. The Python code in `tools/` runs during the build phase and has no runtime overhead.

## Design Rationale

Collecting build scripts, CI tooling, and the autograd codegen under a single `tools/` root keeps them out of both the runtime `torch/` package and the `torchgen/` codegen library. `tools/autograd/` is the one sub-tree with architectural significance — it owns the gradient formula declarations that define the differentiability surface of ATen. The rest of `tools/` is deployment-specific build infrastructure and is excluded from the ADR scope.
