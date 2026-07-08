# `tools/jit`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`tools/jit` contains JIT-oriented code generation helpers for ATen operator integration. Its active generator, `gen_unboxing.py`, reads `native_functions.yaml` through torchgen and emits boxed JIT operator registrations plus generated unboxing functions that pop `c10::IValue` arguments from a `Stack`, convert them to typed C++ arguments, call ATen, and pack results. This supports mobile and selective builds where a fixed operator set can use generated static unboxing instead of template-heavy runtime boxing machinery.

## Key Files

| File | Purpose |
|---|---|
| `gen_unboxing.py` | Generates `RegisterCodegenUnboxedKernels.cpp`, `UnboxingFunctions.h`, and sharded `UnboxingFunctions.cpp` for selected ATen operators |
| `templates/aten_schema_declarations.cpp` | Template that embeds generated JIT schema declarations in a C++ raw string |
| `templates/external_functions_codegen_template.cpp` | Template for TensorExpr external function wrappers and registrations |
| `test/test_gen_unboxing.py` | Unit tests for allowlist precedence and YAML allowlist handling in `gen_unboxing.main()` |

## Public Interface

The command-line entry point is `python -m tools.jit.gen_unboxing`. It accepts `--source-path`, `--install-dir`, `--output-dependencies`, `--dry-run`, `--op-selection-yaml-path`, `--op-registration-allowlist`, and `--TEST-ONLY-op-registration-allowlist-yaml-path`. Programmatic users call `gen_unboxing(native_functions=..., cpu_fm=..., selector=...)` or instantiate `ComputeUnboxingFunctions(Target.DECLARATION|Target.DEFINITION, selector)` and `ComputeCodegenUnboxedKernels(selector)`. The generated C++ exposes `at::unboxing::<op>(Stack&)` wrappers and registers boxed JIT `OperatorGenerator` entries.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torchgen](torchgen/ADR.md) | depends-on | Parses native schemas, builds custom-build selectors, escapes schema strings, and provides file emission/sharding utilities |
| [torchgen/api](torchgen/api/ADR.md) | depends-on | Uses C++ signature groups, `api.unboxing.convert_arguments()`, and `translate()` to convert stack values into faithful C++ API calls |
| [tools/code_analyzer](tools/code_analyzer/ADR.md) | depends-on | Consumes selected-operator YAML or allowlists produced for mobile/custom builds |
| [torch/csrc/jit](torch/csrc/jit/ADR.md) | generates-for | Emits JIT `RegisterOperators` code, schema declarations, and boxed stack wrappers for the JIT runtime |
| [aten/src/ATen/core](aten/src/ATen/core/ADR.md) | generates-for | Generates wrappers around ATen operator schemas and calls into the same dispatcher-backed C++ API described in book chapter 03 |

## Runtime Behaviour

At generation time, `main()` chooses an explicit allowlist over a test YAML allowlist, builds a `SelectiveBuilder`, parses `aten/src/ATen/native/native_functions.yaml`, creates a `FileManager`, and calls `gen_unboxing()`. `ComputeUnboxingFunctions` emits declarations only for root operators selected by the selector; definitions use `convert_arguments()` to generate `peek()`, type conversion, `drop()`, ATen call, and `pack()` code. `ComputeCodegenUnboxedKernels` emits JIT `OperatorGenerator` entries that record the schema, construct `c10::Argument` defaults, call `RECORD_FUNCTION`, and delegate to `at::unboxing::<op>(stack)`.

## Performance Profile

Generated unboxing shifts stack conversion from generic C++ template metaprogramming into explicit per-operator code. For fewer than 100 selected operators, `gen_unboxing()` emits one `UnboxingFunctions.cpp` shard; larger selections use five unboxing shards and ten registration shards to speed compilation. The generated runtime path still accepts a boxed JIT `Stack`, but after `IValue` conversion it calls faithful C++ APIs directly and avoids repeated dynamic schema interpretation for a fixed mobile operator set. Selective generation keeps both source size and JIT registry initialization proportional to selected root operators.

## Design Rationale

The JIT runtime registers boxed operators, while ATen kernels and generated `at::_ops` prefer typed unboxed calls. `tools/jit` bridges that mismatch for selected builds by emitting explicit stack adapters from the same schemas used by the dispatcher. It uses faithful C++ signatures to avoid packing and unpacking `TensorOptions` solely for wrapper convenience. The selector dependency keeps mobile binaries small, and the generated `OperatorGenerator` entries preserve the schema strings required by JIT alias analysis and profiling while delegating actual computation to ATen.
