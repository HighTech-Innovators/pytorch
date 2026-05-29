# Architecture Decision Record: torch/fx

## Architectural Role

`torch/fx` provides the FX Graph IR (Intermediate Representation)—a graph-based representation of PyTorch models that enables symbolic tracing and transformation. It is the foundation for torch.compile, enabling model optimization, cross-framework export, and programmatic code transformation. FX IR is Runtime Critical for compilation pipelines.

## Responsibilities

- Implementing graph capture via `torch.fx.symbolic_trace()` (tracing a model's execution)
- Representing models as directed acyclic graphs (DAGs) with nodes for operations and placeholders for inputs
- Providing graph transformation APIs (rewriting, constant folding, dead code elimination)
- Supporting code generation (converting graphs back to Python code via `fx.symbolic_trace().graph_module.code`)
- Implementing graph printing and visualization utilities

## Dependencies

**Inbound** (what depends on torch/fx):
- `torch/compile` for compilation pipeline
- TorchScript export
- Model visualization and analysis tools

**Outbound** (what torch/fx depends on):
- Dispatcher for operation introspection
- Python's `inspect` module for code analysis

## Trade-offs

**Symbolic tracing vs. static typing**: FX trace graphs are "loose" (same operation can have different tensor shapes in different traces). This is flexible but can miss optimizations that require shape knowledge.

**Graph representation overhead**: Storing a full graph representation (nodes, edges) uses memory but enables sophisticated transformations.

## Extension Boundaries

- **Custom graph transformations**: Users can write custom passes that traverse and modify FX graphs.
- **Custom operations**: Users can define custom nodes in FX graphs.

## Runtime Implications

**Tracing**: Calling `torch.fx.symbolic_trace(model)` executes the model once on dummy inputs to capture the graph, adding one-time overhead.

**Graph-based execution**: FX enables graph-level optimizations (fusion, memory optimization) that are impossible in pure tensor operations.

## Performance Implications

**Graph capture overhead**: Tracing a model is typically 10-100ms depending on model size, but this is a one-time cost.

**Optimization potential**: Graph-level optimizations can provide 2-5x speedups for models with heavy operator fusion opportunities.

## Ownership Boundaries

- **FX graph owns**: operation DAG and node metadata
- **GraphModule owns**: the graph and module-level state

## Verification Points

- `torch/fx/graph_module.py` — Main GraphModule class
- `torch/fx/graph.py` — Graph representation
- `torch/fx/symbolic_trace.py` — Graph tracing
