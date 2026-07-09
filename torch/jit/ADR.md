# `torch/jit`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/jit` provides TorchScript: a statically-typed subset of Python that compiles `nn.Module` subclasses and standalone functions to a portable IR. It supports two capture modes — `script` (AST-based compilation) and `trace` (execution-based graph recording) — and serialises compiled programs to `torch.package` / flatbuffer / ZIP archives that can be loaded in C++ without a Python interpreter.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Package entry; re-exports `script`, `trace`, `freeze`, `load`, `save`, `fork`, `wait`, `export`, `ignore`, `unused`, `is_scripting`, `ScriptModule`, `ScriptFunction` |
| `_script.py` | `script()` — entry point for AST compilation; `ScriptModule` and `RecursiveScriptModule`; `CompilationUnit`; `infer_methods_to_compile` |
| `_recursive.py` | Recursive module compilation: traverses `nn.Module` hierarchies, compiles submodules in dependency order |
| `_serialization.py` | `save(module, path)` / `load(path)` — ZIP-archive serialisation of `ScriptModule`; `jit_module_from_flatbuffer` for mobile |
| `_freeze.py` | `freeze(module)` — inlines constants, folds batch-norm parameters, removes training-only branches |
| `_fuser.py` | NNC/nvFuser kernel fusion: `fuser('fuser1')` / `fuser('fuser2')` context manager; `set_fusion_strategy` |
| `_ir_utils.py` | `_InsertPoint` — context manager for inserting nodes at a specific graph position |
| `_passes/` | Graph optimization passes: CSE, dead-code elimination, loop unrolling, inlining |
| `_async.py` | `fork(fn, *args)` / `wait(future)` — async task execution in the TorchScript runtime |

## Public Interface

| Symbol | Description |
|---|---|
| `torch.jit.script(fn_or_module)` | Compiles a Python function or `nn.Module` to `ScriptFunction` / `ScriptModule` using AST analysis |
| `torch.jit.trace(fn, example_inputs)` | Records operations on `example_inputs` and produces a `ScriptFunction` |
| `torch.jit.ScriptModule` | Compiled module with a typed IR; callable from Python or C++ |
| `torch.jit.ScriptFunction` | Compiled function; `torch.jit.save(fn, path)` serialises to a ZIP archive |
| `torch.jit.freeze(module)` | Constant-folds a `ScriptModule`; reduces dispatch overhead at inference time |
| `torch.jit.save(obj, path)` | Serialises `ScriptModule` or `ScriptFunction` to a ZIP archive |
| `torch.jit.load(path)` | Deserialises from a ZIP archive; returns a `ScriptModule` |
| `torch.jit.fork(fn, *args)` | Schedules async execution; returns a `Future` |
| `torch.jit.is_scripting()` | Returns `True` inside TorchScript context; used to write dual Python/TorchScript code |
| `torch.jit.export` | Decorator marking a method as callable from outside the module in TorchScript |
| `torch.jit.ignore` | Decorator excluding a method from compilation |

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| `torch._C` (torch/csrc/jit) | depends-on | C++ IR (`torch::jit::Graph`, `IValue`, `CompilationUnit`, `GraphExecutor`) accessed via `torch._C._jit_*` |
| [torch/nn](torch/nn/ADR.md) | depends-on | `script` traverses `nn.Module` hierarchies; `ScriptModule` wraps C++ modules |
| [torch/fx](torch/fx/ADR.md) | mutual | FX provides decomposition primitives; JIT passes can lower to FX; some decompositions use `_jit_internal` |
| [aten/src/ATen](aten/src/ATen/ADR.md) | depends-on | TorchScript IR operations map to ATen operator schemas |
| [torch/_export](torch/_export/ADR.md) | depended-on-by | Export pipeline can serialise TorchScript-based modules alongside `torch.export` programs |

## Runtime Behaviour

`torch.jit.script(module)` calls `_script.py`'s `script()`, which inspects the module's `__init__` and `forward` source via `inspect.getsource`, parses it into a TorchScript AST using `torch._C._parse_source_def`, type-checks it against registered operator schemas, and produces a `torch::jit::Graph` in C++. `_recursive.py`'s `infer_methods_to_compile` determines which methods need compilation by tracing `forward`, then compiles each in dependency order. `ScriptModule.__call__` invokes `GraphExecutor::run` in C++, which optimises and executes the IR. `torch.jit.trace` runs the function once with `example_inputs`, recording all ATen dispatches into a `Graph` without source analysis — it cannot capture data-dependent control flow. `freeze` runs constant propagation, dead-code elimination, and batch-norm folding passes in C++; after freezing, `ScriptModule.forward` has no Python overhead.

## Performance Profile

- **Allocation sites**: `script` compilation allocates C++ `IValue` objects for constants and a `GraphExecutor` per compiled function. At inference time, frozen `ScriptModule` execution is allocation-free for fixed-shape inputs.
- **Synchronization costs**: the C++ `GraphExecutor` is thread-safe; multiple threads can call the same `ScriptModule` concurrently without Python GIL held.
- **Data movement**: serialisation via `torch.jit.save` pickles tensors in ZIP format; large models with many parameters require proportional I/O. `jit_module_from_flatbuffer` loads from a memory-mapped flatbuffer, enabling zero-copy load.
- **Redundant or repeated work**: `trace`-based capture runs the function once extra at compilation time; for models with non-trivial forward passes (data loading, augmentation) this can be slow. `freeze` runs a fixed set of graph passes regardless of whether they apply; the cost scales with graph size.

## Design Rationale

Two capture modes (script vs trace) exist because neither is universal: `script` handles data-dependent control flow but requires the code to use TorchScript's type system; `trace` handles arbitrary Python but silently bakes in the control-flow path taken on the example inputs. `freeze` is separate from compilation because it is only safe at inference time (it destroys training-only branches); performing it at script time would make training impossible. Serialisation to a ZIP archive rather than a custom binary format means the serialised artifact can be inspected with standard tools and extended with additional files (e.g., extra state dicts).
