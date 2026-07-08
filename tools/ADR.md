# `tools`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`tools` owns the repository's shared Python build, code generation, maintenance, and developer-automation scripts. It also acts as the package root for specialized subcomponents such as `tools/autograd`, `tools/code_analyzer`, and `tools/jit`, which already carry their own ADRs.

## Key Files

| File | Purpose |
|---|---|
| `README.md` | Describes the directory as shared build-process and developer-tooling infrastructure |
| `BUCK.bzl` | Defines Buck Python libraries and binaries for substitute, JIT generation, autograd generation, and code-analysis helpers |
| `build_pytorch_libs.py` | Builds the native library portion of PyTorch by preparing an environment and calling the CMake wrapper |
| `generate_torch_version.py` | Derives the version string from git, `PKG-INFO`, `version.txt`, and build environment variables, then writes `torch/version.py` |
| `update_masked_docs.py` | Regenerates `torch/masked/_docs.py` by importing `torch.masked._ops` and rendering docstrings |

## Public Interface

Top-level scripts expose `define_tools_targets(...)`, `build_pytorch(...)`, `_create_build_env()`, `get_sha()`, `get_tag()`, `get_torch_version()`, and `update_masked_docs.main()`. The directory also exposes runnable modules and scripts such as `tools.jit.gen_unboxing`, `tools.code_analyzer.gen_oplist`, `tools.setup_helpers.generate_code`, `build_libtorch.py`, `nightly.py`, and `clean.py`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [tools/autograd](tools/autograd/ADR.md) | depends-on | `BUCK.bzl` packages autograd generators and `generate_code` depends on the autograd library target |
| [tools/code_analyzer](tools/code_analyzer/ADR.md) | depends-on | `BUCK.bzl` exports `gen_oplist` and `gen_operators_yaml` from this subcomponent |
| [tools/jit](tools/jit/ADR.md) | depends-on | `BUCK.bzl` builds the `jit` library and `gen_unboxing_bin` from the JIT generation scripts |
| [cmake](cmake/ADR.md) | depended-on-by | `Codegen.cmake` and other build modules invoke scripts from `tools/` during configure and code generation |
| [torch](torch/ADR.md) | depended-on-by | `generate_torch_version.py` writes `torch/version.py` and `update_masked_docs.py` rewrites `torch/masked/_docs.py` |

## Runtime Behaviour

`BUCK.bzl` packages Python sources into Buck targets like `:jit`, `:autograd`, `:gen_oplist_lib`, and `:generate_code`, then exposes matching binaries such as `gen_unboxing_bin` and `generate_code_bin`. `build_pytorch_libs.py` prepares the native build environment with `_create_build_env()`, overlays MSVC variables on Windows when Ninja is active, runs `cmake.generate(...)`, optionally executes `BUILD_CUSTOM_STEP`, and then calls `cmake.build(my_env)`.

`generate_torch_version.py` checks `.git`, `.hg`, `PKG-INFO`, `version.txt`, and `PYTORCH_BUILD_VERSION` / `PYTORCH_BUILD_NUMBER` in order, validates the final string with `packaging.version.Version`, and writes the generated constants into `torch/version.py`. `update_masked_docs.py` imports `torch`, iterates `torch.masked._ops.__all__`, calls `torch.masked._generate_docstring(...)`, and overwrites `torch/masked/_docs.py` only when content changed.

## Performance Profile

- **Allocation sites** - Codegen and versioning scripts allocate large in-memory strings for generated files, and Buck target definitions aggregate sizable source and resource lists for downstream packaging.
- **Synchronization costs** - `build_pytorch_libs.py` is a thin orchestration layer, but it synchronizes the native build behind `cmake.generate(...)`, optional custom shell steps, and `cmake.build(...)` so later phases see a consistent library graph.
- **Data movement** - The directory writes generated files into `torch/` and the build tree, and it shells out to `git`, `hg`, and CMake to move repository metadata into generated Python or C++ artifacts.
- **Redundant or repeated work** - `generate_torch_version.py` re-checks repository metadata on each invocation so version strings stay correct for source, sdist, and release builds, and `update_masked_docs.py` re-renders all masked-operation docstrings to keep the generated file deterministic.

## Design Rationale

PyTorch keeps these scripts under one package root so CMake, Buck, packaging, and maintenance tooling can import shared helpers instead of duplicating logic. The top level stays broad while deeper domains such as autograd, code analysis, and JIT generation each live in their own subdirectories with focused implementations and their own ADRs.
