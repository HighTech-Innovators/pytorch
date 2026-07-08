# `cmake`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`cmake` owns the reusable CMake modules that configure, generate, and assemble the PyTorch build. It translates environment variables, dependency choices, and code generation inputs into concrete build targets and configured headers.

## Key Files

| File | Purpose |
|---|---|
| `Dependencies.cmake` | Discovers CUDA, BLAS, Python, Eigen, sanitizers, and other build dependencies and updates `Caffe2_*` library lists |
| `Codegen.cmake` | Configures generated headers, installs public headers, and drives `torchgen` dry runs plus `add_custom_command()` generation steps |
| `EnvVarForwarding.cmake` | Mirrors `BUILD_*`, `USE_*`, `CMAKE_*`, and selected aliases from the environment into cache variables |
| `public/cuda.cmake` | Supplies CUDA-specific package discovery used by `Dependencies.cmake` |
| `TorchConfig.cmake.in` | Publishes the installed consumer-facing CMake package configuration |

## Public Interface

The directory exports CMake macros and variables such as `disable_ubsan`, `enable_ubsan`, `filter_list`, `filter_list_exclude`, `CAFFE2_USE_CUDA`, `Caffe2_CUDA_DEPENDENCY_LIBS`, `BLAS`, and `Python_EXECUTABLE`. Other build files consume these modules by including `cmake/Codegen.cmake`, including `cmake/public/cuda.cmake`, and depending on the generated `ATEN_CPU_FILES_GEN_TARGET` and `ATEN_CUDA_FILES_GEN_TARGET` targets.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [tools](tools/ADR.md) | depends-on | `Codegen.cmake` runs `tools.generate_torch_version` and `Dependencies.cmake` invokes `tools/optional_submodules.py` |
| [aten/src/ATen/native](aten/src/ATen/native/ADR.md) | depends-on | `Codegen.cmake` drives `torchgen` from `native_functions.yaml` and `tags.yaml` |
| [third_party](third_party/ADR.md) | depends-on | `Dependencies.cmake` points include paths and fallback package resolution at vendored source trees under `third_party/` |
| [caffe2](caffe2/ADR.md) | depended-on-by | `caffe2/CMakeLists.txt` includes `cmake/Codegen.cmake` and relies on the dependency variables defined here |

## Runtime Behaviour

`EnvVarForwarding.cmake` maps explicit aliases like `CUDNN_LIB_DIR=CUDNN_LIBRARY`, forwards named passthrough variables such as `BLAS` and `TORCH_CUDA_ARCH_LIST`, then enumerates the full process environment with `cmake -E environment` to cache every `BUILD_*`, `USE_*`, and `CMAKE_*` variable. `Dependencies.cmake` toggles `CAFFE2_USE_CUDA`, populates `Caffe2_CUDA_DEPENDENCY_LIBS`, finds `Threads`, chooses a BLAS backend, locates Python, and falls back to `third_party/eigen` after calling `tools/optional_submodules.py checkout_eigen` when a system Eigen install is missing.

`Codegen.cmake` computes `COMMIT_SHA` by running `tools.generate_torch_version.get_sha`, configures `caffe2/core/macros.h`, bootstraps dynamic output lists with `torchgen.gen --dry-run`, includes generated `.cmake` manifests, and then creates `add_custom_command()` rules for headers, sources, declarations, and unboxing artifacts. The module finishes by grouping those outputs into `ATEN_CPU_FILES_GEN_TARGET`, `ATEN_CUDA_FILES_GEN_TARGET`, and optional XPU generation targets so downstream libraries can depend on generated files explicitly.

## Performance Profile

- **Allocation sites** - Configure-time helpers materialize large CMake lists for generated headers, generated sources, and dependency libraries, and `Codegen.cmake` writes multiple generated `.cmake` manifests to describe dynamic outputs.
- **Synchronization costs** - Build ordering is explicit: `ATEN_CPU_FILES_GEN_LIB` and `ATEN_CUDA_FILES_GEN_LIB` serialize consumers behind code generation so CUDA and CPU compilation never race missing generated headers.
- **Data movement** - `torchgen` writes generated C++ and header trees into `${CMAKE_BINARY_DIR}/aten/src/ATen` and `${CMAKE_BINARY_DIR}/torch/headeronly/core`, and `configure_file()` copies configured macros into the build tree.
- **Redundant or repeated work** - `Codegen.cmake` intentionally runs `torchgen` once in `--dry-run` mode to discover dynamic outputs and again in the actual custom command to materialize them, trading extra configure work for a correct incremental dependency graph.

## Design Rationale

PyTorch splits build logic into focused CMake modules because dependency discovery, environment translation, and code generation each evolve independently. The directory keeps those policies centralized so top-level `CMakeLists.txt` files can declare targets while `cmake/` owns how the project finds optional backends and regenerates operator code.
