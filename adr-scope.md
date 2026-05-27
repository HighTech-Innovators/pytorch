# ADR Scope

| Directory | Source files present | Status | Reason (if EXCLUDED) |
|---|---|---|---|
| ./torch | yes | COVERED | Main Python module and API surface |
| ./torch/autograd | yes | COVERED | Automatic differentiation engine |
| ./torch/nn | yes | COVERED | Neural network module system |
| ./torch/jit | yes | COVERED | JIT compiler and TorchScript |
| ./torch/optim | yes | COVERED | Optimization algorithms |
| ./torch/distributed | yes | COVERED | Multi-process/multi-node training |
| ./aten | yes | COVERED | Tensor operations library |
| ./c10 | yes | COVERED | Core tensor library |
| ./torchgen | yes | COVERED | Code generation system |
| ./functorch | yes | COVERED | Function transforms (vmap, grad, etc.) |
| ./caffe2 | yes | EXCLUDED | Legacy framework being phased out; minimal new development |
| ./mypy_plugins | yes | EXCLUDED | Build/config only, mypy plugins not part of core runtime |
| ./android | yes | EXCLUDED | Platform-specific bindings, excluded per constraints |
| ./test | yes | EXCLUDED | Test infrastructure, no ADR needed per constraints |
| ./benchmarks | yes | EXCLUDED | Benchmark infrastructure, not part of core system |
| ./tools | yes | EXCLUDED | Development tools, build infrastructure |
| ./scripts | yes | EXCLUDED | Utility scripts, not core system |
| ./third_party | yes | EXCLUDED | Vendored/third-party |
| ./binaries | yes | EXCLUDED | Build/config only |
| ./docs | yes | EXCLUDED | Documentation only |
