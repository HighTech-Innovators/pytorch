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
1. **Custom build targets**: Add new executable or library via CMake/Buck definitions
2. **Code analysis plugins**: Register new static analysis tools
3. **Development utilities**: Add new helper scripts

## Notes and Caveats

1. **Build system complexity**: Managing two build systems increases maintenance burden
2. **CI/CD dependency**: Tools are tightly coupled to CI/CD pipelines
3. **Platform-specific code**: Some tools are Linux-only (may require adaptation for Windows, macOS)
