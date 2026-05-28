# ADR: tools — Development and Build Infrastructure

**Status**: ACTIVE  
**Last Updated**: 2026-05-28

## Architectural Role

The tools directory provides development utilities, build helpers, and infrastructure for PyTorch development:

- **Build support** (CMake, BUCK): Build system helpers and definitions
- **Code analysis** (code_analyzer, code_coverage): Static analysis and coverage tools
- **Testing infrastructure** (autograd testing, benchmarking)
- **Development utilities** (worktree creation, MNIST downloading)
- **CI/CD helpers** (build scripts, release notes generation)

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
- **CMake**: Build orchestration
- **BUCK/Bazel**: Alternative build system
- **Python**: Build scripts are Python-based
- **C++ compiler**: For compilation

### Runtime Dependencies
- None; tools are used at build/development time only

## Trade-offs and Design Decisions

### 1. Multiple Build Systems (CMake + Buck)
**Decision**: Support both CMake and Buck/Bazel build systems  
**Rationale**:
- **Flexibility**: Different teams prefer different build systems
- **Facebook integration**: Facebook uses Buck internally
- **Open source**: CMake is standard for open source projects

**Trade-off**: Maintenance burden of two build systems; easy to desynchronize

**Evidence**: tools/build_defs/ contains both CMake and Buck definitions. BUCK.bzl files in many directories.

## Extension Boundaries

### Public Extension Points
1. **Custom build targets**: Add new executable or library via CMake/Buck definitions
2. **Code analysis plugins**: Register new static analysis tools
3. **Development utilities**: Add new helper scripts

## Notes and Caveats

1. **Build system complexity**: Managing two build systems increases maintenance burden
2. **CI/CD dependency**: Tools are tightly coupled to CI/CD pipelines
3. **Platform-specific code**: Some tools are Linux-only (may require adaptation for Windows, macOS)
