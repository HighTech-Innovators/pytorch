# Architecture Decision Record: torch/compiler

## Architectural Role

`torch/compiler` provides the high-level `torch.compile()` API, orchestrating compilation of PyTorch models via TorchDynamo (bytecode capture) and TorchInductor (code generation) backends. It is the user-facing entry point for PyTorch's compilation system, enabling automatic performance optimization without model rewriting.

## Responsibilities

- Implementing `torch.compile()` decorator and context manager
- Managing compilation configuration and backend selection (eager, reduce-overhead, max-autotune modes)
- Coordinating TorchDynamo for frame capture and guard generation
- Dispatching to TorchInductor for code generation or fallback backends
- Providing compilation mode selection and performance tuning options
- Managing compiled graph caching to avoid recompilation on repeated calls

## Dependencies

**Inbound** (what depends on torch/compiler):
- User code calling `torch.compile(model)`
- Research frameworks and ML libraries

**Outbound** (what torch/compiler depends on):
- `torch/_dynamo` for bytecode capture and guard mechanism
- `torch/_inductor` for code generation
- `torch/fx` for graph representation

## Trade-offs

**Backend abstraction vs. backend-specific tuning**: `torch.compile()` abstracts different backends (Inductor, TorchScript), trading per-backend optimization opportunities for user-friendly API stability.

**Eager compilation vs. lazy compilation**: Models are compiled on first call, adding latency to first execution. Lazy compilation would reduce startup latency but complicate error handling.

## Extension Boundaries

- **Custom backends**: New compilation backends can be registered to replace or supplement TorchInductor.
- **Custom compilation modes**: Advanced users can define custom compilation strategies.

## Runtime Implications

**First-call compilation**: On first invocation, `torch.compile()` triggers bytecode capture, graph lowering, and code generation (typically 100ms-1s).

**Cached execution**: Subsequent calls reuse cached compiled graphs if input signatures match (shape, dtype, device).

**Guard checking**: Before executing compiled code, guards are checked; if any guard fails, execution falls back to eager mode.

## Performance Implications

**Compilation overhead**: Typical compilation latency is 100ms-10s depending on model size and complexity.

**Execution speedup**: 2-5x speedup possible for models with fusion opportunities, 0-20% for compute-limited models.

**Memory overhead**: Compiled code is cached in memory; large model compilations can consume significant memory.

## Ownership Boundaries

- **torch.compiler owns**: compilation API and orchestration
- **TorchDynamo owns**: bytecode capture
- **TorchInductor owns**: code generation

## Verification Points

- `torch/compiler/__init__.py` — Public API interface
- `torch/compiler/compile_fx.py` — Compilation orchestration
