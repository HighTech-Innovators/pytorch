# `torchgen/api`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torchgen/api` translates parsed operator schemas into typed API models used by PyTorch's generated C++, Python binding, dispatcher, autograd, lazy, functionalization, and ufunc code. It defines semantic objects such as `PythonSignature`, `Derivative`, `ForwardDerivative`, `NamedCType`, and synthesized `Expr` lists so destination generators can emit code without reinterpreting YAML schemas each time.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Package marker for the API translation modules |
| `python.py` | Python binding and `.pyi` model generation, including `PythonArgument`, `PythonSignature`, `signature_from_schema`, `dispatch_lambda_args`, and `dispatch_lambda_exprs` |
| `autograd.py` | Autograd model dataclasses and matching logic, including `DifferentiabilityInfo`, `Derivative`, `ForwardDerivative`, `dispatch_strategy`, and `match_differentiability_info` |
| `translate.py` | Program-synthesis engine that converts in-scope `Binding` or `Expr` values into requested `NamedCType` goals |
| `cpp.py` | C++ API signature and type mapping helpers consumed by Python, dispatcher, autograd, and destination generators |
| `functionalization.py` | Functionalization naming, attribute, constructor, and reverse-operation API helpers |

## Public Interface

| Symbol | Description |
|---|---|
| `torchgen.api.python.PythonArgument` | Models one Python parser argument and renders parser and `.pyi` argument strings |
| `torchgen.api.python.PythonSignature` / `PythonSignatureDeprecated` | Represents generated Python signatures, output arguments, tensor-options arguments, and deprecated parser variants |
| `torchgen.api.python.signature()` / `signature_from_schema()` | Build a `PythonSignature` from a `NativeFunction` or `FunctionSchema` |
| `torchgen.api.python.dispatch_lambda_args()` / `dispatch_lambda_exprs()` | Produce C++ lambda argument declarations and PythonArgParser output expressions for generated bindings |
| `torchgen.api.autograd.DifferentiabilityInfo` | Stores matched derivative formulas, saved inputs, saved outputs, forward derivatives, and output differentiability |
| `torchgen.api.autograd.match_differentiability_info()` | Attaches derivative metadata to each `NativeFunction`, including generated foreach derivative info |
| `torchgen.api.autograd.gen_differentiable_outputs()` | Selects differentiable returns using `cpp.return_names()` and `is_differentiable()` |
| `torchgen.api.translate.translate()` | Synthesizes ordered expressions for requested `NamedCType` goals from available bindings |

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torchgen](torchgen/ADR.md) | depends-on | Uses `torchgen.model`, `torchgen.api.types`, `torchgen.context`, `torchgen.local`, and `torchgen.utils` as the schema and type-system substrate |
| [torchgen/dest](torchgen/dest/ADR.md) | depended-on-by | Destination generators call API helpers such as `translate`, `cpp.arguments`, `DispatcherSignature`, and `StructuredImplSignature` before emitting C++ |
| [torch/csrc](torch/csrc/ADR.md) | depended-on-by | Generated Python binding code targets `PythonArgParser`, `wrap()`, `pybind11::gil_scoped_release`, and C++ dispatch lambdas used by the Python extension |
| [aten/src/ATen](aten/src/ATen/ADR.md) | depended-on-by | Generated dispatcher and native signatures call `at::` functions and use ATen C++ types such as `Tensor`, `TensorOptions`, `Scalar`, and `SymInt` |

## Runtime Behaviour

Code generation calls this package after `torchgen.model` has parsed native schemas and derivative metadata. `python.py` builds `PythonSignature` objects, inserts scattered TensorOptions fields such as `dtype`, `device`, `layout`, `pin_memory`, and `requires_grad`, then maps parser outputs like `_r.tensor(0)` or `_r.toSymInt(1)` to dispatch lambda expressions. `autograd.py` matches native functions against `derivatives.yaml` schemas by exact schema, functional signature, generated variants, and foreach reference functions, then rewrites in-place forward-AD formulas so `result` becomes `self_p` and reused out-of-place formulas update `self_t_raw`. `translate.py` resolves requested C++ semantic types by direct lookup, reference/value conversions, TensorOptions packing and unpacking, SymInt and IntArrayRef conversions, and explicit failure through `UnsatError` when no rule exists.

## Performance Profile

The modules run at build-time code generation, so their hot paths are schema list traversals, dataclass construction, string formatting, and dictionary lookups rather than tensor computation. `translate()` bounds search cost by using mostly direct mutually exclusive rules and by rejecting unresolved goals with `UnsatError` instead of performing unbounded backtracking. `python.py` explicitly enumerates supported parser unpacking methods and supported return types, which makes generation predictable and catches unsupported signatures before C++ compilation. `autograd.py` performs extra passes for foreach and in-place forward AD formulas, but it stores results in `NativeFunctionWithDifferentiabilityInfo` so downstream generators reuse the matched metadata.

## Design Rationale

The package separates API semantics from output formatting so every destination generator can share one interpretation of `native_functions.yaml`. `NamedCType` carries both C++ type and semantic name, which prevents accidental substitution of unrelated values that share a primitive type such as `bool` or `int64_t`. Python binding generation keeps `PythonSignature` distinct from C++ signatures because parser schemas, `.pyi` signatures, TensorOptions scattering, deprecated signatures, and out-variant grouping each have different rules. Autograd matching stays in this layer because derivative formulas need schema variants, C++ type names, saved attribute metadata, and code-generation policy before any destination file can emit VariableType code.
