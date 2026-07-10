# `torch/jit`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/jit` is the Python surface of TorchScript: the system that compiles Python/PyTorch code into a portable, serializable IR executable without the Python runtime. It exposes `torch.jit.script` (static compilation) and `torch.jit.trace` (execution-trace capture).

## Key Files

| File | Purpose |
|---|---|
| `torch/jit/_script.py` | `torch.jit.script` entry point (1806 lines): `script()`, `ScriptModule`, `ScriptFunction`, `RecursiveScriptModule`; calls into `torch.jit.frontend` and `torch/_C._jit_get_operation` |
| `torch/jit/__init__.py` | Re-exports the public API; `torch.jit.trace`, `torch.jit.load`, `torch.jit.save`, `torch.jit.fork`, `torch.jit.wait` |
| `torch/jit/_recursive.py` | `_compile_and_register_class`, `infer_methods_to_compile`: recursively compiles submodules of an `nn.Module` |
| `torch/jit/_serialization.py` | `save`, `load` wrappers over the C++ serialization backend |
| `torch/jit/frontend.py` | AST→TorchScript IR frontend: `get_jit_def`, `get_jit_class_def` |
| `torch/jit/_builtins.py` | Registry of Python builtins supported in TorchScript (`_register_builtin`) |
| `torch/jit/_freeze.py` | `freeze()`: specializes a `ScriptModule` by inlining constants and removing training-time branches |

## Public Interface

`torch.jit.script(obj)`, `torch.jit.trace(mod, example_inputs)`, `torch.jit.save(m, path)`, `torch.jit.load(path)`, `torch.jit.fork(fn, *args)`, `torch.jit.wait(future)`, `torch.jit.freeze(module)`, `torch.jit.export`, `torch.jit.is_scripting()`, `ScriptModule`, `ScriptFunction`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/csrc/jit](torch/csrc/jit/ADR.md) | depends-on | C++ TorchScript compiler, IR, interpreter, serialization backend |
| `torch._C` | depends-on | `_jit_get_operation`, `_get_model_id`, JIT compilation primitives |
| `torch.nn` | depends-on | `nn.Module` is the primary object scripted via `RecursiveScriptModule` |

## Runtime Behaviour

`torch.jit.script(fn_or_module)` in `_script.py` resolves the input type: for a plain function it calls `get_jit_def` (in `frontend.py`) to parse the Python AST into a TorchScript definition, then compiles it via the C++ compiler in `torch/csrc/jit/`. For an `nn.Module`, `infer_methods_to_compile` in `_recursive.py` determines which methods to compile and recursively scripts submodules. The result is a `ScriptModule` (wrapping a C++ `torch::jit::Module`) whose `forward` runs the compiled IR interpreter — not Python. `torch.jit.trace` instead executes the module with example inputs, recording `call_function` nodes, and produces a `ScriptModule` from the trace graph.

## Performance Profile

Scripted modules bypass the Python interpreter for every forward call: execution runs in the C++ TorchScript interpreter in `torch/csrc/jit/`. Scripting cost is paid once at `jit.script()`  time (AST parsing, type inference, IR compilation). `freeze()` eliminates training-time branches and inlines module attributes, reducing interpreter dispatch overhead further. `torch.jit.fork`/`wait` allow async parallel execution of independent TorchScript tasks. The main runtime cost in a scripted model is the ATen kernel dispatch within the interpreter — the same cost as eager mode — so scripting primarily eliminates Python overhead, not operator-level cost.

## Design Rationale

TorchScript compiles a Python-like subset rather than arbitrary Python: the restricted type system makes type inference tractable and enables AOT serialization. The split between `_script.py` (Python entry point and `nn.Module` recursion) and `torch/csrc/jit/` (compiler and interpreter) keeps the compilation logic in C++ for performance while exposing a Pythonic API. `freeze()` is a separate post-compilation step rather than a compiler flag so users can inspect the unspecialized module first. The `trace` path trades completeness (no data-dependent branches captured) for generality (any Python callable can be traced).
