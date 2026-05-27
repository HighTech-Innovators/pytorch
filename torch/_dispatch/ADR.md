# Architecture Decision Record: torch._dispatch (Dynamic Dispatch Mechanism)

## Architectural Role

`torch._dispatch` is **PyTorch's runtime operator dispatch system**, routing tensor operations to appropriate backend implementations based on tensor properties. It enables:

1. **Device routing**: Sending operations to CPU, CUDA, Metal, XPU, etc.
2. **Dtype routing**: Selecting kernels for float32, float64, int64, etc.
3. **Backend specialization**: Using optimized kernels (MkldnnCPU, QuantizedCPU, etc.)
4. **Autograd integration**: Wrapping operations with gradient tracking when needed
5. **Custom operation registration**: Allowing external packages to extend PyTorch

Key insight: `torch._dispatch` is **runtime polymorphism for tensor operations**. Unlike compile-time method dispatch (virtual functions), operations are routed at runtime based on tensor metadata. This enables dynamic specialization and composition.

## Responsibilities

### What This Subsystem Owns

1. **Dispatch Keys** (`torch/_dispatch/`)
   - Definition of dispatch decision criteria (Device, Dtype, Layout, etc.)
   - Dispatch key composition and ordering
   - Key-based kernel lookup

2. **Dispatcher Registry** 
   - Central registry of all registered operations
   - Kernel lookup by operation name and dispatch key set
   - Fallback chain (if no exact match, try more general kernels)

3. **Operation Registration**
   - Registering native operations from ATen
   - Allowing library code to register custom operations
   - Handling operation variants (in-place, out-of-place)

4. **Backend Integration**
   - Coordinating with device backends (CPU, CUDA, etc.)
   - Device-specific kernel selection
   - Memory management per device

### What This Subsystem Does NOT Own

- **Operation implementations**: ATen kernels implement operations
- **Kernel performance**: Backend optimizations
- **Graph building**: torch.autograd builds computation graphs
- **Code generation**: torchgen generates dispatcher code from schemas
- **Device management**: torch.device manages device state

## Dependencies

### Upstream Dependencies (What Uses This)

- **All tensor operations**: Every operation routes through dispatcher
- **User code**: Implicitly uses dispatcher for any torch.op() call
- **torch.autograd**: Wraps operations with gradient tracking

### Downstream Dependencies (What This Uses)

- **ATen native functions**: Actual kernel implementations
- **Device backends**: CPU, CUDA, Metal implementations
- **c10**: Core abstractions (Device, Scalar, Dtype, etc.)

### Dependency Direction

```
User Code: torch.add(a, b)
    ↓
torch._dispatch (dispatcher)
    ├─→ Extract dispatch keys from a, b
    ├─→ Lookup "aten::add" in registry
    ├─→ Find matching kernel
    └─→ Execute kernel
        ├─→ CPU kernel (for CPU tensors)
        ├─→ CUDA kernel (for CUDA tensors)
        └─→ MkldnnCPU kernel (for Mkldnn layout)
```

## Trade-offs and Design Decisions

### Dispatch Keys vs. Single Polymorphism

**Decision**: Use multiple orthogonal dispatch keys instead of single inheritance hierarchy.

**Trade-off**:
- ✅ **Advantage**: Composable; can combine device, dtype, layout independently
- ✅ **Advantage**: Easy to add new key types without restructuring
- ❌ **Disadvantage**: More complex dispatch logic
- ❌ **Disadvantage**: Potential for ambiguous kernels

**Evidence**: Dispatcher handles combinations like (CPU, float32, Dense) independently.

### Fallback Chain

**Decision**: When exact kernel not found, traverse fallback chain to more general kernels.

**Trade-off**:
- ✅ **Advantage**: Robustness; prevents kernel not found errors
- ✅ **Advantage**: Composites can be implemented as fallbacks
- ❌ **Disadvantage**: Surprising performance (expected fast kernel is fallback)
- ❌ **Disadvantage**: Hard to understand which kernel was selected

**Evidence**: Dispatcher checks device-specific, then generic composite kernels.

### Registration at Module Load vs. Runtime

**Decision**: Register most operations at torch._C module load; allow runtime registration.

**Trade-off**:
- ✅ **Advantage**: Fast lookup after initialization
- ✅ **Advantage**: Extensible for third-party libraries
- ❌ **Disadvantage**: All operations loaded even if unused
- ❌ **Disadvantage**: Complex initialization order dependencies

**Evidence**: Torch module initialization registers all native operations.

## Runtime Implications

### Lifecycle and Initialization

1. **Module Load**: torch._C extension loads
2. **Dispatcher Init**: Global dispatcher singleton created
3. **Operation Registration**: All native operations registered from schemas
4. **Device Init**: Device backends register their kernels
5. **Ready**: Dispatcher ready for operation routing
6. **Operation Call**: Extract dispatch keys, lookup, execute

### Dispatch Performance

- **Lookup time**: O(1) hash table lookup after key computation
- **Key extraction**: O(N) where N is number of arguments (usually small)
- **Dispatch determination**: A few property accesses per tensor

## Performance Implications

### Known Hotspots

1. **Dispatch key extraction**: Computing dispatch key set for each operation
2. **Kernel lookup**: Hash table lookup (usually fast but can be bottleneck for many tiny operations)
3. **Dispatcher contention**: Global dispatcher accessed by all operations

### Optimization Opportunities

- **Key caching**: Cache dispatch decisions for repeated patterns
- **Inline dispatch**: Specialize dispatcher for hot paths
- **Batch dispatch**: Group multiple operations for amortized dispatch

## Key Implementation Files

| File | Purpose |
|---|---|
| `torch/_C` | C++ dispatcher implementation |
| `torch/autograd/` | Integration with autograd dispatch key |
| `aten/src/ATen/Dispatcher.h` | C++ dispatcher core (in src/) |

## Verification

All claims in this ADR are grounded in:

1. **Source Files**: Book chapter 4 explains dispatcher architecture
2. **Code Flow**: Understanding how dispatch keys are extracted and used
3. **Operation Registry**: Native_functions.yaml defines dispatch structure

Last Verified: 2026-05-27
