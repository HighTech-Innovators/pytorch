# Architecture Decision Record: torch/_inductor

## Architectural Role

`torch/_inductor` (TorchInductor) is the code generation backend for torch.compile, converting FX graphs to optimized C++/CUDA code. It implements operator fusion, memory optimization, and scheduling for GPU and CPU backends. TorchInductor is Runtime Critical for compilation performance; the quality of generated code directly determines compiled model performance.

## Responsibilities

- Implementing the scheduler (determining fusion boundaries, operation order)
- Performing kernel fusion (combining multiple operations into single kernels)
- Generating C++ code for CPU execution
- Generating CUDA/Triton kernels for GPU execution
- Implementing loop optimization and tiling strategies
- Managing memory layout and aliasing analysis
- Implementing fallback kernels for operations without custom implementations

## Dependencies

**Inbound** (what depends on torch/_inductor):
- `torch/_dynamo` for graph conversion
- User code (via torch.compile)

**Outbound** (what torch/_inductor depends on):
- `torch/fx` for graph representation
- CUDA/Triton for GPU code generation
- Standard compilation tools (C++ compiler, CUDA compiler)

## Trade-offs

**Heuristic-based scheduling**: The scheduler uses heuristics (operation cost, memory pressure) to decide fusion boundaries. Optimal scheduling is NP-hard; heuristics are practical but may miss opportunities.

**Graph-level optimization vs. kernel-level**: TorchInductor optimizes at the graph level (whole model), enabling global fusion. The trade-off is compilation time (graph optimization is expensive).

## Extension Boundaries

- **Custom kernels**: Users can register custom CUDA/Triton kernels to TorchInductor.
- **Scheduling strategies**: Advanced users can customize fusion heuristics.

## Runtime Implications

**Compilation pipeline**: FX graph → Inductor lowering → scheduler → code generation → compilation → execution.

**Code generation**: Generated code is compiled at runtime (or cached), adding latency on first call.

**Optimization opportunities**: Fusion can combine multiple operations into single kernels, reducing kernel launch overhead and improving cache locality.

## Performance Implications

**Compilation overhead**: Typically 100ms-10s depending on model size and compilation complexity.

**Kernel fusion benefit**: 2-5x speedup possible for models with fusion opportunities, 0-20% for memory-bandwidth-limited models.

**Code generation quality**: Generated code quality varies; complex fusion patterns may generate suboptimal code.

## Ownership Boundaries

- **Inductor owns**: scheduling decisions, code generation, optimization
- **Model owns**: the original FX graph
- **Compiler owns**: C++/CUDA compilation

## Verification Points

- `torch/_inductor/scheduler.py` — Scheduling heuristics
- `torch/_inductor/codegen/` — Code generation for CPU and GPU
- `torch/_inductor/compile_fx.py` — Main compilation entry point
