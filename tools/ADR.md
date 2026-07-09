# `tools`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`tools` contains repository-level automation for building, release maintenance, source checkout management, generated assets, and developer diagnostics. The directory keeps operational scripts such as `nightly.py`, `gen_vulkan_spv.py`, `build_pytorch_libs.py`, and `generate_torch_version.py` outside the import-time `torch` package while still letting them coordinate PyTorch's C++ and Python build products.

## Key Files

| File | Purpose |
|---|---|
| `nightly.py` | Command-line workflow for checking out or pulling nightly PyTorch binaries into a local virtual environment |
| `gen_vulkan_spv.py` | Generates Vulkan SPIR-V C++ source and header data from shader definitions and YAML metadata |
| `build_pytorch_libs.py` | Invokes the CMake wrapper in `tools/setup_helpers/cmake.py` to build PyTorch libraries |
| `generate_torch_version.py` | Computes the version string and git SHA used in generated torch version metadata |
| `create_worktree.py` | Creates and removes local git worktrees with submodules cloned from local object stores |
| `stale_issues.py` | Drives GitHub issue listing, counting, subscription save/restore, and collaborator checks through `gh` |

## Public Interface

The public interface is script-oriented rather than package-oriented. `nightly.py` exposes `checkout` and `pull` subcommands through `make_parser()` and `main()`, backed by `Venv`, `PipSource`, `install()`, `checkout_nightly_version()`, and `pull_nightly_version()`. `gen_vulkan_spv.py` exposes `invoke_main()` and `main(argv)` for SPIR-V generation, with `SPVGenerator`, `ShaderInfo`, `preprocess()`, `genCppFiles()`, and `generateSpvBinStr()` as the implementation surface. Build and metadata consumers call `build_pytorch_libs.build_pytorch()`, `generate_torch_version.get_torch_version()`, `get_sha()`, and `get_tag()`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [aten/src/ATen](aten/src/ATen/ADR.md) | depended-on-by | ATen build and Vulkan paths consume generated sources and libraries coordinated by `build_pytorch_libs.py` and `gen_vulkan_spv.py` |
| [torch/csrc](torch/csrc/ADR.md) | depended-on-by | Python binding libraries are built through `build_pytorch_libs.build_pytorch()` and receive version metadata from `generate_torch_version.py` |
| [c10/core](c10/core/ADR.md) | depended-on-by | The library build includes c10 artifacts that are driven through the CMake helper invoked by `build_pytorch_libs.py` |
| [torchgen](torchgen/ADR.md) | depended-on-by | Code-generation workflows under `tools` operate alongside torchgen-generated operator and binding sources |

## Runtime Behaviour

Most files in `tools` run as command-line entry points and delegate heavy work to subprocesses. `nightly.py` shells out through `subprocess`, manages a `Venv`, records logs under a per-run logging directory, installs packages, and moves nightly files into the checkout. `gen_vulkan_spv.py` reads shader/YAML inputs, preprocesses template environments, runs external shader tooling, and writes generated `spv.h` and `spv.cpp` content. `build_pytorch_libs.py` overlays the MSVC environment on Windows when Ninja is selected, then calls the `CMake` helper with build options supplied by setup.

## Performance Profile

`tools` code is not on PyTorch tensor execution hot paths; its costs occur during checkout, build, release, and diagnostics workflows. Runtime is dominated by subprocess latency, compiler invocations, package installation, filesystem traversal, and generated-file writes. `gen_vulkan_spv.py` does linear work over shader inputs and serializes binary SPIR-V blobs into C++ strings, while `nightly.py` spends most wall time in virtual-environment creation, package installation, and file moves. `build_pytorch_libs.py` intentionally delegates parallelism and incremental rebuild behavior to CMake and Ninja instead of reimplementing scheduler logic in Python.

## Design Rationale

PyTorch keeps these workflows as standalone scripts so release, build, and source-maintenance operations can run before `torch` is importable from the working tree. The scripts centralize platform-specific logic such as Windows compiler environment overlay, nightly binary placement, and Vulkan source generation without coupling those concerns to runtime modules. The directory also preserves a clear layering boundary: generated artifacts and build commands serve `c10`, ATen, and `torch/csrc`, but the runtime libraries do not import `tools`.
