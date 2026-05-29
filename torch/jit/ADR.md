# Architecture Decision Record: torch/jit

## Architectural Role

`torch/jit` provides TorchScript JIT compilation (legacy system, largely superseded by `torch.compile`). TorchScript enables compiling Python models to static computational graphs without full eager execution, useful for deployment on edge devices, mobile platforms, and in production environments with strict latency budgets. While `torch.compile` is preferred for modern development, TorchScript remains important for deployed models and certain specialized use cases.

## Responsibilities

- Implementing torch.jit.trace() for trace-based compilation (executing model once on example input and capturing operations)
- Implementing torch.jit.script() for annotation-based compilation (analyzing Python source code with type hints)
- Generating static computational graphs in TorchScript IR
- Providing code generation to native executables or portable serialized format
- Supporting partial JIT (selectively compiling parts of models)

## Dependencies

**Inbound** (what depends on torch/jit):
- Legacy code and models (TorchScript is deprecated for new code)
- Edge/mobile deployment via torch.jit.trace
- Production inference systems requiring compiled models

**Outbound** (what torch/jit depends on):
- `torch/fx` for graph representation
- LLVM for native code generation (optional)
- Python's AST for script analysis

## Trade-offs

**Static graphs vs. dynamic execution**: TorchScript produces static graphs, trading control flow flexibility for deployment simplicity. Models with complex control flow (data-dependent loops, dynamic shapes) are difficult to script.

**Trace-based vs. script-based**: Tracing captures actual execution (simpler but misses control flow), scripting analyzes code (preserves control flow but requires type annotations).

## Extension Boundaries

- **Custom operators**: Users can register custom C++ operators for TorchScript via TORCH_LIBRARY.
- **Custom backends**: TorchScript can target different backends (native code, mobile runtime, etc.).

## Runtime Implications

**Compilation**: torch.jit.trace() executes the model once to capture the graph. torch.jit.script() analyzes Python source without execution.

**Serialization**: Compiled models can be serialized with `model.save()` and deserialized without Python dependency.

**Deployment**: Compiled TorchScript models can run on edge devices, mobile platforms, and production servers without full PyTorch runtime.

## Performance Implications

**Compilation overhead**: Tracing typically takes 100ms-1s; scripting is faster (seconds for complex models).

**Execution speedup**: 1.5-3x speedup typical for compiled models due to graph optimization and native code generation.

**Deployment benefit**: Compiled models avoid Python overhead, enabling inference in resource-constrained environments (mobile, microcontrollers).

## Ownership Boundaries

- **torch.jit owns**: compilation and serialization API
- **Code generators own**: native code generation
- **Model owns**: the original Python code

## Verification Points

- `torch/jit/__init__.py` — Public API interface
- `torch/jit/trace.py` — Trace-based compilation
- `torch/jit/script.py` — Script-based compilation
