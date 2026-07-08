# `torch/csrc/jit`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/csrc/jit` implements the legacy TorchScript compiler, IR, optimizer, serializer, mobile runtime, and C++ execution API. It is the C++ half of the scripting and tracing stack described alongside Python bindings in book Chapter 06 (`book/06-python-bindings.md`): Python APIs enter C++ through `torch._C`, while this directory owns the `torch::jit` module/object model, graph representation, execution plans, and archive format. Modern `torch.compile` uses Dynamo and Inductor, but TorchScript remains the serialized-program runtime for scripted modules, mobile models, backend lowering, and long-lived JIT tests.

## Key Files

| File | Purpose |
|---|---|
| `api/module.h` | Public `torch::jit::Module` wrapper with methods for `forward`, attributes, parameters, buffers, child modules, and save/load helpers |
| `api/compilation_unit.h` | Owns named TorchScript functions and class definitions compiled from source or loaded archives |
| `ir/ir.h` | Defines the TorchScript graph, blocks, nodes, values, and type-carrying SSA IR used by frontend and optimization passes |
| `frontend/ir_emitter.h` | Lowers parsed TorchScript syntax into IR using resolvers, source ranges, and schema matching |
| `runtime/graph_executor.h` | Builds and caches `ExecutionPlan` objects and runs optimized graphs with the bytecode interpreter |
| `runtime/interpreter.h` | Defines stack-based bytecode execution for optimized TorchScript graphs |
| `serialization/export.cpp` | Writes TorchScript archives, tensor records, and ONNX export records |
| `mobile/interpreter.h` | Smaller mobile bytecode interpreter for exported lite modules |
| `backends/backend_interface.h` | Defines the pluggable backend interface used by NNAPI, XNNPACK, and other TorchScript backend paths |

## Public Interface

The primary C++ interface is `torch::jit::Module`, which exposes `forward`, `get_method`, `register_parameter`, `register_buffer`, `register_attribute`, `named_parameters`, `named_buffers`, recursive module traversal, and archive save/load operations. `CompilationUnit` exposes `define` and function lookup for compiled source. `GraphExecutor` exposes `run`, `runAsync`, `getPlanFor`, `getInputIndependentPlan`, and debug state access. Backend registration and mobile import/export APIs let downstream runtimes consume a stable TorchScript program instead of Python bytecode.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [c10/core](c10/core/ADR.md) | depends-on | Uses `IValue`, `ClassType`, `FunctionSchema`, `QualifiedName`, devices, tensors, and intrusive ownership for scripted objects |
| [c10/util](c10/util/ADR.md) | depends-on | Uses `ArrayRef`, `Exception`, `irange`, optional-like utilities, and low-level containers across IR and runtime code |
| [aten/src/ATen/core](aten/src/ATen/core/ADR.md) | depends-on | Calls dispatcher schemas and ATen operators from emitted graphs and registered operations |
| [torch/csrc/autograd](torch/csrc/autograd/ADR.md) | depends-on | Handles `Variable`, symbolic autograd export, differentiable graph execution, and gradient-aware scripted modules |
| [torch/csrc/api](torch/csrc/api/ADR.md) | depended-on-by | The C++ frontend exposes `torch::jit::compile` and module APIs by including JIT headers |

## Runtime Behaviour

TorchScript source or traced Python code enters the frontend, where the lexer/parser and `ir_emitter` create an SSA `Graph` with typed `Value` objects and operator schemas. The `GraphExecutor` runs required passes, builds an `ExecutionPlan`, and executes bytecode on a stack; profiling mode records runtime types and shapes so later plans can specialize. `Module` stores parameters, buffers, attributes, and submodules as `c10::ivalue::Object` state, and `forward` dispatches to the scripted `Method` named `forward`. Serialization writes tensor payloads as archive records and pickled program metadata, while mobile import lowers the same program concept into a smaller bytecode and interpreter path.

## Performance Profile

The hot path is execution-plan selection plus interpreter dispatch, so `GraphExecutor` caches plans by `ArgumentSpec` and reuses optimized code graphs when the global flag allows it. Profiling execution pays extra warmup cost to collect type and shape information, then amortizes that cost through specialized plans, fusion groups, and backend-specific lowering paths such as oneDNN, fuser, NNAPI, and XNNPACK. Mobile runtime strips the general compiler surface and runs a compact bytecode interpreter to reduce binary size and startup overhead. Archive export copies tensor bytes into little-endian records, so serialization cost scales with model parameter size rather than graph size.

## Design Rationale

TorchScript keeps a C++ IR and runtime so serialized PyTorch programs can execute without the Python interpreter. The design separates frontend parsing, IR transformation, plan selection, and bytecode interpretation because each phase has different stability and performance requirements. `IValue` provides one tagged value representation for tensors, scalars, lists, tuples, objects, and futures, which lets the interpreter and serializer share one calling convention. The legacy JIT remains distinct from Dynamo because TorchScript optimizes a restricted, serializable program representation, while Chapter 07's Dynamo captures ordinary CPython frames for ahead-of-time backend compilation.
