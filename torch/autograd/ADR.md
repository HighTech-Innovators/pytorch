# Architecture Decision Record: Autograd Engine (torch.autograd)

## Architectural Role

**torch.autograd** is PyTorch's automatic differentiation system — the machinery that computes gradients for training neural networks. It records a dynamic computation graph during forward pass and uses reverse-mode automatic differentiation (backpropagation) to compute gradients during the backward pass. When training code calls `loss.backward()`, autograd traces backward through the computation graph applying the chain rule.

**Location**: `torch/csrc/autograd/` and `torch/autograd/` | **Language**: C++17 + Python | **Dependencies**: ATen, c10

## Responsibilities

**autograd owns**:
- Computation graph construction (`Node`, `Edge` graph structure)
- Forward pass instrumentation (hooks into ATen to record operations)
- Backward pass orchestration (backward pass engine)
- Gradient accumulation (`SavedVariable`, `InputBuffer`)
- Backward functions for all operations (manually written in FunctionsManual.cpp + generated)
- Leaf node gradient accumulation (`AccumulateGrad`)
- Context managers for training/inference mode (`torch.enable_grad()`, `torch.no_grad()`)

**autograd does not own**:
- Forward pass kernel execution (ATen owns that)
- Tensor creation or storage (c10 owns that)
- Optimization algorithms (torch.optim owns that)
- Parameter management (torch.nn owns that)

## Dependencies

### Inbound Dependencies
- **torch.nn** uses autograd to track gradient through neural network layers
- **torch.optim** uses autograd to access gradients during optimization
- **Training code** calls `.backward()` on loss to trigger backpropagation
- **Forward pass** (every ATen operation) is intercepted by autograd

### Outbound Dependencies
- **ATen** provides forward kernels that autograd hooks into and wraps
- **c10** provides tensor types and dispatch keys that autograd uses
- **torch/csrc/autograd** C++ source provides core autograd implementation

## Trade-offs and Design Decisions

### 1. Dynamic Graph Construction (Not Static)
**Decision**: Computation graph built at runtime as operations execute, not predefined before forward pass.

**Rationale**: 
- Enables control flow (if/while/for) in computation graph
- Supports variable-length sequences, conditional branches
- Simpler programming model vs explicit graph definition

**Alternative**: Static graph definition (TorchScript/graph mode) — used for optimization/export, but not default.

**Trade-off**: Dynamic graph construction has overhead compared to static graphs, but simplicity outweighs cost for research.

### 2. Reverse-Mode Automatic Differentiation
**Decision**: Use backpropagation (reverse-mode AD) rather than forward-mode AD.

**Rationale**: 
- For neural networks: output is scalar (loss), inputs are high-dimensional (parameters)
- Reverse mode is O(1) in number of outputs, forward mode is O(number of inputs)
- Reverse mode scales to millions of parameters; forward mode does not

**Trade-off**: Cannot easily compute directional derivatives or second derivatives without multiple passes.

### 3. Separate Node/Edge Graph Structure
**Decision**: Computation graph represented as DAG of `Node` objects connected by `Edge` objects, separate from TensorImpl.

**Rationale**: 
- Tensors (TensorImpl) are immutable after creation; graph info would clutter them
- Multiple tensors can share computation (e.g., `z = x + x`); separate graph structure avoids duplication
- Backward pass traverses graph without touching original tensors

**Evidence**: function.h and edge.h define Node and Edge types; graph_task.h defines backward traversal.

### 4. SavedVariable for Backward Dependencies
**Decision**: Operations save only the *specific values* needed for gradient computation, not entire intermediate tensors.

**Rationale**: 
- Reduces memory overhead; e.g., `add` gradient needs only input shape, not values
- Enables checkpointing: save activations selectively, recompute others during backward
- Supports in-place operations that would otherwise corrupt intermediate tensors

**Evidence**: saved_variable.h implements selective tensor saving.

**Trade-off**: Backward functions must explicitly specify what to save; can lead to bugs if incorrect.

### 5. AccumulateGrad for Leaf Nodes
**Decision**: Leaf tensors (parameters, inputs with requires_grad=True) use special `AccumulateGrad` node that accumulates gradients into `.grad` attribute.

**Rationale**: 
- Leaf tensors are created by users, not by operations, so they have no backward function
- `.grad` attribute accumulates gradients from multiple backward passes (e.g., in loops)
- Separates user-facing `.grad` attribute from internal gradient accumulation

**Evidence**: accumulate_grad.h implements leaf node gradient collection.

### 6. Dispatch Key Integration for Autograd
**Decision**: Autograd is integrated into ATen's dispatcher via `AutogradCPU`, `AutogradCUDA`, etc. dispatch keys.

**Rationale**: 
- When a tensor has requires_grad=True, its DispatchKeySet includes AutogradCPU/AutogradCUDA
- Dispatcher routes to autograd wrapper which records operation, calls actual kernel, returns new tensor with graph edge
- No branching in forward pass code; autograd is transparent to kernel implementations

**Evidence**: VariableTypeManual.cpp registers autograd dispatch keys.

**Trade-off**: Small dispatch overhead for all operations on requires_grad tensors; negligible in practice.

### 7. Manual + Generated Backward Functions
**Decision**: Some backward functions hand-written in FunctionsManual.cpp (~291K lines); others auto-generated from native_functions.yaml.

**Rationale**: 
- Simple operations generated automatically to avoid duplication
- Complex operations (convolution, RNNs, custom ops) hand-written for correctness/efficiency
- Separation allows evolution of backward implementations without regenerating simple cases

**Evidence**: FunctionsManual.cpp is massive; torchgen also generates backward functions.

## Extension Boundaries

**Custom operations**: Register backward function via `Function` subclass; called when custom operation is used in differentiable context.

**Higher-order derivatives**: Set `requires_grad=True` on gradients themselves; backward pass can differentiate through backward pass.

**Custom autograd functions**: torch.autograd.Function enables users to define custom forward/backward pairs for non-differentiable operations.

## Runtime Implications

### Initialization
- Autograd initialization happens when C extension loads (torch/csrc/autograd/init.cpp)
- Registration of backward functions for all ATen operations at startup
- Minimal per-operation overhead — registrations are table lookups

### Forward Pass with Autograd
1. Operation called on tensor(s) with requires_grad=True
2. Dispatcher routes to AutogradCPU/AutogradCUDA dispatch key
3. Autograd wrapper:
   - Creates output tensor
   - If requires_grad, creates Node for this operation
   - Saves inputs/metadata in SavedVariable
   - Adds Edge from output Node to input Nodes
4. Actual kernel executes
5. Returns new tensor with requires_grad=True

**Overhead**: 10-20% for non-specialized tensors; less for bulk operations that amortize overhead.

### Backward Pass
1. `.backward()` called on loss tensor
2. GraphTask created representing this backward pass
3. Backward engine traverses graph in reverse topological order (BFS with dependency counting)
4. For each node, call its `backward()` method with incoming gradients
5. Accumulate gradients into leaf nodes' `.grad` attribute

**Traversal**: Linear in number of nodes visited; can prune unreachable subgraphs.

### Memory
- Graph structure: ~60 bytes per Node (metadata + edges list)
- Saved tensors: Operation-specific; can range from 0 bytes (gradient only needs shape) to full tensor
- Checkpointing: Users can trade compute for memory by not saving intermediate activations

### Concurrency
- **Not thread-safe** for concurrent mutations to graph structure
- **Safe** for reading completed graph from multiple threads
- **Forward pass**: Thread-safe if each thread operates on independent tensors
- **Backward pass**: Not thread-safe; single-threaded backward per graph

### Lifecycle
- Graph lifetime is from forward pass through backward pass (or `detach()`)
- After `.backward()`, graph is deallocated (gradients stored in `.grad` attribute)
- Multiple backward passes: `.backward()` can be called again if `retain_graph=True`

## Performance Implications

### Known Hotspots
1. **Graph traversal during backward**: Linear in number of operations, O(N) for N operations
2. **Dispatch overhead**: Small per operation (~1-2% for simple ops)
3. **Memory allocation for SavedVariable**: Depends on operation; 0-100% of forward memory
4. **Gradient accumulation**: Atomic operations to accumulate into `.grad`

### Optimization Opportunities
1. **Checkpointing**: Save activations selectively; recompute others during backward (trade memory for compute)
2. **Graph optimization**: Fuse operations, eliminate dead branches before backward
3. **In-place operations**: Dangerous with autograd; `.backward()` may fail if in-place operation corrupts SavedVariable
4. **Detaching**: Use `.detach()` to stop gradient flow; saves memory and compute

### Synchronization Costs
- Backward pass is single-threaded; no explicit synchronization needed
- Backend (CUDA, HIP) may synchronize for stream ordering

## Ownership Boundaries

**autograd owns**:
- Computation graph structure (Node, Edge, GraphTask)
- Backward function implementations for all operations
- Gradient accumulation logic
- Graph traversal and backward pass execution

**autograd delegates to ATen**:
- Forward kernel execution
- Operation dispatch (through AutogradCPU/AutogradCUDA dispatch keys)

**autograd references but does not own**:
- torch.nn (uses autograd to track gradients, but autograd doesn't know about modules)
- torch.optim (uses gradients from autograd, but autograd doesn't optimize)
- User parameters (torch.nn.Module owns parameters, autograd just computes gradients for them)

**Parent/peer systems own**:
- torch/csrc/autograd/ C++ implementation
- torch/autograd/ Python API (torch.autograd.Function, enable_grad, etc.)
