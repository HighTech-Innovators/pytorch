# Architecture Decision Record: torch/_dynamo

## Architectural Role

`torch/_dynamo` (TorchDynamo) is a bytecode compiler that captures Python-level model execution and converts it to graph-form for compilation. It sits at the entry point of torch.compile, using Python's sys.settrace hook to intercept function calls and bytecode instructions. TorchDynamo enables torch.compile to work with Python control flow (loops, conditionals) without requiring model rewriting. It is Runtime Critical for the torch.compile pipeline.

## Responsibilities

- Implementing bytecode capture via Python frame introspection (sys.settrace)
- Managing guards (conditions under which compiled code is valid, e.g., "tensor shape is known")
- Handling Python control flow (if, loops, function calls) by extracting "trace regions" that are amenable to compilation
- Integrating with TorchInductor for code generation
- Providing caching infrastructure for compiled graphs (avoiding recompilation on repeated calls)

## Dependencies

**Inbound** (what depends on torch/_dynamo):
- `torch.compile()` entry point
- User code

**Outbound** (what torch/_dynamo depends on):
- `torch/fx` for graph representation
- `torch/_inductor` for code generation
- Python runtime (frame introspection, bytecode)

## Trade-offs

**Frame introspection overhead**: Using sys.settrace adds 5-10% overhead to traced code even when not compiling. This is the price of dynamic bytecode capture.

**Graph fallback on unhandled Python**: When TorchDynamo encounters Python operations it cannot handle (file I/O, network requests, complex control flow), it falls back to eager execution. This flexibility trades compilation completeness for robustness.

## Extension Boundaries

- **Custom bytecode handlers**: Advanced users can register handlers for custom Python operations.
- **Custom guard types**: New guard types can be defined for domain-specific compilation validity conditions.

## Runtime Implications

**Bytecode capture**: Entering compiled code triggers bytecode capture if not cached. First call compiles; subsequent calls use cached graphs.

**Guard checking**: Before executing compiled code, guards are checked. If any guard fails (e.g., shape changed), fallback to eager execution.

**Control flow handling**: Loops and conditionals in Python are handled by extracting separate trace regions for each branch.

## Performance Implications

**Frame overhead**: 5-10% overhead even when not compiling, as sys.settrace adds overhead.

**Cache hit rate**: Compilation benefit depends on cache hit rate. Workloads with variable input shapes have lower cache hit rates and see less benefit.

**First-run latency**: First call to a compiled function incurs compilation overhead (typically 100ms-1s depending on model size).

## Ownership Boundaries

- **TorchDynamo owns**: bytecode capture, guard management, cache
- **Model owns**: Python code (TorchDynamo only captures it)
- **TorchInductor owns**: code generation from traced graphs

## Verification Points

- `torch/_dynamo/convert_frame.py` — Bytecode capture
- `torch/_dynamo/guards.py` — Guard generation and checking
- `torch/_dynamo/cache.py` — Compiled graph caching
