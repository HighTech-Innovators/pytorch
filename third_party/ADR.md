# `third_party`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`third_party` owns the vendored dependency source trees that PyTorch builds against. It also owns wrapper-generation scripts that adapt upstream source layouts into the include and build patterns PyTorch expects.

## Key Files

| File | Purpose |
|---|---|
| `README.md` | States that the directory contains vendored third-party libraries |
| `generate-cpuinfo-wrappers.py` | Generates guarded wrapper sources for `cpuinfo` files based on `CPUINFO_SOURCES` |
| `generate-xnnpack-wrappers.py` | Parses XNNPACK CMake source lists and emits guarded wrapper files plus `.bzl` source-definition files |
| `xnnpack_src_defs.bzl` | Stores generated XNNPACK source lists used by Buck-based builds |
| `xnnpack_wrapper_defs.bzl` | Stores generated wrapper paths used by Buck-based builds |

## Public Interface

The directory exposes vendored source roots such as `XNNPACK`, `FP16`, `fbgemm`, `cpuinfo`, `protobuf`, `pybind11`, and `googletest` to the build system. Its scripting interface exposes `CPUINFO_SOURCES`, `WRAPPER_SRC_NAMES`, `SRC_NAMES`, `update_sources()`, and `gen_wrappers()` for wrapper regeneration.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [cmake](cmake/ADR.md) | depended-on-by | `Dependencies.cmake` and package-specific CMake modules point include paths and `add_subdirectory()` calls at vendored libraries here |
| [caffe2](caffe2/ADR.md) | depended-on-by | `caffe2/CMakeLists.txt` and its subdirectories build against vendored math, threading, and serialization dependencies |
| [aten/src/ATen/native/cpu](aten/src/ATen/native/cpu/ADR.md) | depended-on-by | CPU backends consume libraries such as XNNPACK, cpuinfo, and fbgemm that are staged under this directory |

## Runtime Behaviour

The directory mostly participates at build time. `generate-cpuinfo-wrappers.py` iterates `CPUINFO_SOURCES`, creates `cpuinfo/wrappers/...` directories on demand, and writes `#if`-guarded wrapper files so architecture-specific sources only compile on matching platforms.

`generate-xnnpack-wrappers.py` reads `XNNPACK/CMakeLists.txt` and nested generated CMake fragments, collects named source sets like `PROD_AVX2_MICROKERNEL_SRCS` and `PROD_NEON_MICROKERNEL_SRCS`, then emits wrapper files under `xnnpack_wrappers/` and updates `xnnpack_wrapper_defs.bzl` and `xnnpack_src_defs.bzl`. The rest of the directory contributes raw upstream trees that PyTorch's CMake and Buck rules compile directly.

## Performance Profile

- **Allocation sites** - Wrapper generators build in-memory maps of source lists with `collections.defaultdict(list)` and materialize one output file per wrapped upstream source.
- **Synchronization costs** - The generation scripts are single-threaded and avoid cross-process coordination; synchronization only appears later when the outer build compiles the vendored sources they enumerate.
- **Data movement** - Regeneration writes many small wrapper files and `.bzl` manifests, while the build consumes large vendored source trees like XNNPACK, protobuf, and googletest directly from this directory.
- **Redundant or repeated work** - `generate-xnnpack-wrappers.py` intentionally re-parses XNNPACK CMake fragments and re-emits wrapper manifests so Buck and CMake stay aligned with the current upstream snapshot.

## Design Rationale

Vendoring keeps PyTorch's build reproducible across platforms and CI images without requiring system installs for every dependency. The wrapper scripts exist because upstream projects organize sources around their own build systems, while PyTorch needs uniform, platform-guarded source lists that its CMake and Buck integrations can consume.
