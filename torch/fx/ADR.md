# Architecture Decision Record: torch.fx (Functional Transformation)

## Architectural Role

The `torch.fx` subsystem provides **functional program transformation** — the ability to capture PyTorch models as symbolic graphs and transform them. Unlike JIT (which compiles to a fixed IR), FX enables:

1. **Symbolic Execution**: Execute Python code with symbolic (not concrete) tensors to build a graph
2. **Graph Transformation**: Inspect and modify the captured graph
3. **Custom Backends**: Rewrite the graph to target alternative execution engines

Key insight: FX is for **model analysis and optimization at the graph level** — it's lower-level than JIT (exposes the graph for user manipulation) but higher-level than ATen (abstracts away operation details).

## Responsibilities

### What This Subsystem Owns

1. **Symbolic Tracing** (`tracer.py`)
   - Execute Python code with symbolic Tensor proxies
   - Record attribute accesses and method calls
   - Build GraphModule representing the computation

2. **Graph Module API** (`graph_module.py`)
   - GraphModule class: represents transformed model
   - Graph class: DAG of nodes representing operations
   - Node classes: represent operations, inputs, outputs

3. **Graph Transformation API** (`node.py`, `graph.py`)
   - Insert, delete, replace nodes
   - Modify graph topology
   - Preserve semantic equivalence or intentionally change behavior

4. **Backends and Executors** (`interpreter.py`, `_compile.py`)
   - Interpreter: execute graph by walking nodes
   - Custom backends: target-specific execution (Vulkan, TVM, etc.)

5. **Utilities and Passes** (various modules)
   - Dead code elimination: remove unused nodes
   - Graph visualization
   - Debugging and introspection

### What This Subsystem Does NOT Own

- **Operation Implementation**: ATen/torch operations own kernels
- **Compilation**: torch.jit handles IR compilation; FX does transformation
- **Automatic Differentiation**: torch.autograd handles gradients
- **Memory Management**: c10 allocators handle memory

## Dependencies

### Upstream Dependencies (What Uses This)

- **Model Optimization**: Custom optimization passes on captured graphs
- **Model Deployment**: Convert to alternative formats (ONNX, custom IR)
- **Model Inspection**: Analyze model structure and data flow
- **Quantization**: Instrument graphs for quantization-aware training

### Downstream Dependencies (What This Uses)

- **torch.nn.Module**: Wraps modules in GraphModule
- **torch operations**: Calls torch.add, torch.matmul, etc. which trigger tracing
- **Python ast module**: For source-level inspection

### Dependency Direction

```
Model Optimization Code
    ↓
torch.fx.symbolic_trace()
    ↓
FX Tracer (symbolic execution)
    ↓
torch operations (traced via proxies)
    ↓
GraphModule (captured representation)
```

## Trade-offs and Design Decisions

### Symbolic vs. Eager Execution

**Decision**: FX uses symbolic execution (Tensor proxies) rather than eager execution.

**Trade-off**:
- ✅ **Advantage**: Graph captured without running real computation (faster)
- ✅ **Advantage**: Graph valid for any input shape (shape-polymorphic)
- ❌ **Disadvantage**: Cannot capture data-dependent control flow (loops, conditionals on values)
- ❌ **Disadvantage**: Symbolic execution misses some Python semantics

**Evidence**: Chapter 10 describes symbolic execution capturing all paths through control flow simultaneously.

### Graph Mutability

**Decision**: GraphModule allows inspection and modification of captured graph.

**Trade-off**:
- ✅ **Advantage**: Enables custom transformations (no need to rewrite original code)
- ✅ **Advantage**: Facilitates analysis and debugging
- ❌ **Disadvantage**: Easy to create broken graphs (invalid topology)
- ❌ **Disadvantage**: Users must understand graph semantics

**Evidence**: `graph.py` exposes `insert_node()`, `replace_node()`, `delete_node()` methods.

### Lazy Evaluation Interpretation

**Decision**: When executing GraphModule, trace through captured graph (not original code).

**Trade-off**:
- ✅ **Advantage**: Execution order follows graph structure (deterministic)
- ✅ **Advantage**: Can apply graph-level optimizations
- ❌ **Disadvantage**: Python side effects in original code don't execute
- ❌ **Disadvantage**: If graph captures incorrect semantics, execution diverges from original

**Evidence**: Interpreter in `interpreter.py` walks graph nodes; never calls original Python forward.

## Extension Boundaries

### Symbolic Tracing Custom Modules

**Boundary**: Call `torch.fx.symbolic_trace(module)` on any `torch.nn.Module`.

```python
from torch.fx import symbolic_trace
model = MyModel()
gm = symbolic_trace(model)  # Capture as GraphModule
```

**Evidence**: `tracer.py` contains tracing logic.

### Custom Graph Transformations

**Boundary**: Subclass or use GraphModule to transform captured graph.

```python
gm = symbolic_trace(model)
for node in gm.graph.nodes:
    if node.op == 'call_function' and node.target == torch.nn.functional.relu:
        # Replace all ReLUs with Sigmoid
        node.replace_all_uses_with(new_node)
```

**Evidence**: GraphModule exposes graph for manipulation.

### Custom Execution Backends

**Boundary**: Subclass `Interpreter` or implement custom execution for GraphModule.

```python
class MyBackend(Interpreter):
    def call_function(self, fn, args, kwargs):
        # Custom execution logic
        pass
```

**Evidence**: `Interpreter` in `interpreter.py` defines interface for custom backends.

## Runtime Implications

### Lifecycle and Initialization

1. **Module Definition**: User defines torch.nn.Module
2. **Symbolic Trace**: Call `torch.fx.symbolic_trace(module)` to capture GraphModule
3. **Transformation** (optional): Modify captured graph
4. **Execution**: Call GraphModule forward as if it were original module
5. **Inspection**: Print graph, visualize, analyze

### Concurrency Behavior

**Thread Safety**:
- **Tracing**: Thread-safe; each trace independent
- **Execution**: Graph execution thread-safe (delegates to ATen)
- **Modification**: Graph modification not thread-safe; serialize access

**Evidence**: No explicit locking in FX; graph safety relies on ATen thread safety.

### Failure Behavior

1. **Untraceable Code**: If Python code uses unsupported features (data-dependent control flow, print, etc.), tracing fails
2. **Shape Errors**: Graph execution may fail if input shape different from traced shape
3. **Broken Topology**: If user modifies graph incorrectly, execution may fail or produce wrong results
4. **Backend Unavailable**: If target backend not available, fallback to interpreter

**Evidence**: Common failures when tracing models with dynamic control flow.

## Performance Implications

### Known Hotspots

1. **Symbolic Tracing Overhead**: Tracing adds latency (but less than JIT compilation)
2. **Graph Interpretation**: Walking nodes slower than eager execution, but graph-level optimizations compensate
3. **Memory**: Captured graph stored in memory; large models create large graphs

### Optimization Opportunities

- **Custom Passes**: User can write passes to optimize graph for specific hardware
- **Fused Operations**: Fuse operations at graph level
- **Memory Optimization**: Reorder operations to improve cache locality

## Ownership Boundaries

### State Owned by FX

1. **Captured Graph**: GraphModule and node structure
2. **Transformation History**: Modified nodes and connections
3. **Execution State**: Variables bound during execution

### State Borrowed from Modules

1. **Parameters**: Module parameters; FX only reads via proxies
2. **Buffers**: Non-learnable state
3. **Forward Logic**: FX captures forward, not modifies it

### State Owned by Users

1. **Original Module**: User-defined module
2. **Graph Modifications**: User transforms captured graph
3. **Backend Configuration**: User chooses execution backend

## Key Implementation Files

| File | Purpose |
|---|---|
| `tracer.py` | Symbolic tracing infrastructure |
| `graph_module.py` | GraphModule class and execution |
| `graph.py` | Graph structure and node management |
| `node.py` | Node representation |
| `interpreter.py` | Reference interpreter for executing graphs |

## Verification

All claims in this ADR are grounded in:

1. **Source Files**: `src/torch/fx/` — checked for symbolic tracing, graph transformation, execution
2. **Book Chapter**: Chapter 10 "Functional Transformation with FX" provides architectural understanding
3. **Code Flow**: Traced from user `torch.fx.symbolic_trace()` through graph capture and execution

Last Verified: 2026-05-27
