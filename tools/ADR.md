# Architecture Decision Record: tools

## Architectural Role

`tools` contains build and development utilities: CMake configuration files, Python code generators, benchmark scripts, static analysis tools, and CI/CD integration scripts. These are build-time, development-time, and deployment-time tools that are not Runtime Critical for training/inference itself, but essential for PyTorch development and deployment infrastructure.

## Responsibilities

- Maintaining CMake build configuration for all platforms (Linux, macOS, Windows)
- Code generation for operators and bindings (via torchgen)
- Benchmarking utilities for performance measurement
- Static analysis and code quality tools
- CI/CD integration (GitHub Actions configuration, lint scripts)
- Development utilities (autocompletion, debugging helpers)

## Dependencies

**Inbound** (what depends on tools):
- Build system (for building PyTorch from source)
- CI/CD pipelines (for automated testing and deployment)
- Development workflows (for developers working on PyTorch)

**Outbound** (what tools depends on):
- CMake build system
- Python tooling
- GitHub Actions and CI platforms

## Trade-offs

**Build configuration centralization vs. per-target tuning**: CMake files attempt to provide good defaults for all platforms, trading per-platform optimization for maintenance simplicity.

**Generated code via torchgen vs. manual coding**: Operators are generated from specifications rather than manually coded, trading implementation flexibility for consistency and automatic updates.

## Extension Boundaries

- **Custom build targets**: Users can add custom CMake targets for specialized use cases.
- **Custom code generators**: New generator plugins can be added to torchgen.
- **Custom benchmarks**: New benchmark scripts can be added to measure domain-specific performance.

## Runtime Implications

**No runtime impact**: Tools are used during build/deployment, not during inference or training.

**Build-time code generation**: Build includes code generation step (via torchgen), adding 30-60 seconds to build time.

## Performance Implications

**Build-time only**: Tools only affect build time, not runtime performance.

**Generated code quality**: Generated code quality affects runtime performance; tool improvements can indirectly improve runtime efficiency.

## Ownership Boundaries

- **tools own**: build configuration, code generation, CI/CD scripts
- **Built artifacts own**: resulting binaries and libraries

## Verification Points

- `tools/cmake/` — CMake build configuration (enables cross-platform builds: Windows, Linux, macOS)
- `tools/scripts/` — Python utility scripts for development
- `torchgen/` — Operator code generator (generates operator registration and bindings)
- `tools/build/` — Platform-specific build helpers
- `CMakeLists.txt` at repository root — Main build definition using tools/cmake configurations
