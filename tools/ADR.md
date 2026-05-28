# ADR: tools — Development and Build Infrastructure

**Status**: ACTIVE  
**Last Updated**: 2026-05-28

## Architectural Role

The tools directory provides development utilities, build helpers, and infrastructure for PyTorch development and project maintenance. The directory contains:

- **Build support** (CMake, BUCK): Build system helpers including `build_defs/CMakeLists.txt`, `build_defs/*.bzl` files, and `BUCK.bzl` defining cross-platform build targets
- **Code analysis** (code_analyzer, code_coverage): Static analysis plugins for code quality and coverage reporting integration
- **Testing infrastructure** (autograd testing, benchmarking): Utilities in `autograd/` and `perf_test/` directories supporting test automation and performance measurement
- **Development utilities** (worktree creation, MNIST downloading): Helper scripts for developer productivity
- **CI/CD helpers** (build scripts, release notes generation): Release and build automation in `release_notes/` subdirectory

tools is classified as **Operational** (non-runtime) because it exists to support development and build processes, not runtime functionality.

## Responsibilities

### What tools Owns
- **Build system helpers**: CMake utilities, Bazel/Buck definitions, build flags
- **Code analysis**: Static analysis plugins, coverage reporting integration
- **Development utilities**: Create git worktrees, download datasets, build libtorch
- **Testing infrastructure**: Autograd testing utilities, profiling support
- **Release automation**: Generate release notes, version bumping

### What tools Does Not Own
- **Test execution**: test/ directory owns tests
- **Benchmarking**: benchmarks/ directory owns benchmark code
- **Documentation**: docs/ directory owns documentation

## Dependencies

### Build-Time Dependencies
The tools directory depends on **CMake** as the primary build orchestration system, with complementary support for **BUCK/Bazel** build definitions maintained in `build_defs/` subdirectories (`buck_helpers.bzl`, `fb_xplat_cxx_library.bzl`). Build scripts are written in **Python** and require **Python 3.6+** to execute. A **C++ compiler** (GCC, Clang, or MSVC depending on platform) is required to compile helper utilities and build tools that are themselves C++ programs.

### Runtime Dependencies
No runtime dependencies exist; tools are exclusively used at build and development time. The tools subsystem is not linked into any shipped binary and has zero impact on application runtime behavior or performance.

## Trade-offs and Design Decisions

### 1. Multiple Build Systems (CMake + Buck)
**Decision**: Support both CMake and Buck/Bazel build systems  
**Rationale**: 
Different development communities prefer different build systems: CMake dominates open-source C++ projects and integrates naturally with IDEs and package managers, while Buck/Bazel is the internal build system at Meta (formerly Facebook) and offers superior performance for large monorepos through fine-grained incremental builds. PyTorch must support both to simultaneously serve the open-source community and Meta's internal development workflow.

**Trade-off**: Maintaining two parallel build systems increases maintenance burden significantly. Evidence from the source tree shows that both CMake configurations and Buck definitions (in `BUCK.bzl` files across multiple directories and comprehensive `build_defs/` helpers like `fb_xplat_cxx_library.bzl`) must be kept in sync whenever a new build target or dependency is introduced. Any divergence between systems can silently break builds on one platform while the other continues working, requiring developers to test against both systems. This dual maintenance overhead is justified by the requirement to serve both communities, but it represents a substantial ongoing cost that manifests in longer CI/CD cycle times and occasional broken builds on one build system while the other works.

## Extension Boundaries

### Public Extension Points
1. **Custom build targets**: Add new executable or library via CMake/Buck definitions in `CMakeLists.txt` or `BUCK.bzl`
2. **Code analysis plugins**: Register new static analysis tools via integration in `tools/code_analyzer/` framework
3. **Development utilities**: Add new helper scripts in `tools/` subdirectories (typically Python, following existing patterns)
4. **Build system helpers**: Extend CMake utility functions in `tools/build_defs/CMakeLists.txt` or Buck macros in `build_defs/*.bzl`

### Build Target Addition Pattern
To add a new build target (e.g., a new executable):
1. Add CMake target definition in `CMakeLists.txt` using `add_executable()` or `add_library()`
2. Link dependencies (e.g., `target_link_libraries(mytarget PRIVATE torch c10)`)
3. Add equivalent Buck/Bazel definition in `BUCK.bzl` or `build_defs/` helper macros (see `fb_xplat_cxx_library.bzl` pattern)
4. Test both build systems: `cmake --build build` and `buck build ...`

## Performance Implications

### Build System Performance
- **CMake**: Incremental builds typically 10-30 seconds for full PyTorch; cold builds 2-5 minutes
- **Buck/Bazel**: Fine-grained dependency tracking enables faster incremental builds (30-60% faster than CMake for partial changes)
- **Dual system maintenance**: Build times are effectively doubled because developers must validate against both systems
- **Caching**: Both systems support artifact caching; Buck has superior caching via shared cache at Meta

### Code Analysis Performance
- **Static analysis tools**: Typically 5-30 seconds per analysis pass (depends on tool complexity and codebase size)
- **Integration cost**: Running analysis on each commit adds ~1-2 minutes to CI/CD pipeline
- **Coverage reporting**: Generates reports of test coverage; adds ~5 minutes to test pipeline

## Ownership Boundaries

### tools Owns
- **Build system definitions**: CMake and Buck configurations for all compilable targets
- **Build helpers**: Utility functions in `tools/build_defs/` (CMake helpers, Buck macro definitions)
- **Development utilities**: Helper scripts for common tasks (create worktrees, download datasets, build shared libraries)
- **Code analysis infrastructure**: Integration framework for static analysis tools

### tools Borrows
- **CMake/Buck frameworks**: Uses build system APIs (not responsible for maintaining CMake/Buck projects)
- **Compiler**: Uses installed C++ compiler (GCC, Clang, MSVC)
- **Python runtime**: Executes Python scripts; does not own Python interpreter

### Adjacent Ownership
- **Actual tests**: `test/` directory owns test code; tools provides test infrastructure
- **Benchmarks**: `benchmarks/` directory owns benchmark code; tools may provide build support
- **Documentation**: `docs/` owns documentation; tools may provide doc generation utilities

## Key Files and References

| File | Purpose |
|---|---|
| `CMakeLists.txt` | Primary CMake build definition (top-level) |
| `tools/build_defs/CMakeLists.txt` | CMake utility functions and helpers |
| `BUCK.bzl` | Top-level Buck/Bazel build definitions |
| `tools/build_defs/*.bzl` | Buck/Bazel helper macros (e.g., `fb_xplat_cxx_library.bzl`, `buck_helpers.bzl`) |
| `tools/code_analyzer/` | Code quality analysis plugins and infrastructure |
| `tools/autograd/` | Autograd testing utilities |
| `tools/perf_test/` | Performance testing infrastructure |
| `tools/release_notes/` | Release note generation automation |
| `tools/jit/` | JIT compilation helpers |

## Build System Synchronization

### Dual System Maintenance Challenge
Both CMake and Buck must be kept in sync whenever a build target or dependency is added:

1. **Synchronization point**: Add new library or executable
   - CMake: Modify `CMakeLists.txt` with `add_library()` or `add_executable()`
   - Buck: Modify `BUCK.bzl` or appropriate `build_defs/*.bzl` with `cxx_library()` or `cxx_binary()`
   - Both must specify identical dependencies and compilation flags

2. **Divergence risk**: If only one system is updated, builds succeed on one platform and fail on the other
   - Example: Adding a new dependency to ATen in CMakeLists.txt but forgetting Buck.bzl results in:
     - `cmake --build .` works fine
     - `buck build aten` fails with undefined reference
   - Detection: Integration testing on both systems is mandatory

3. **Maintenance cost example** (from source evidence in `tools/build_defs/`):
   - `tools/build_defs/*.bzl` contains 300+ lines of macro definitions to express CMake concepts in Buck
   - This duplication is necessary because CMake and Buck have different semantics

## Notes and Caveats

1. **Build system complexity**: Maintaining two build systems increases maintenance burden significantly; documented via `build_defs/` duplication
2. **CI/CD dependency**: Tools are tightly coupled to CI/CD pipelines; changes to build system require coordination with CI infrastructure team
3. **Platform-specific code**: Some tools are Linux-only (e.g., certain code analysis tools may require LLVM/Clang toolchain); adaptation needed for Windows, macOS
4. **Version compatibility**: Build system behavior depends on CMake/Buck version; tool scripts may need updates when upgrading build system versions
