# `torch/csrc/api`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/csrc/api` is the Python-free C++ frontend (LibTorch). It mirrors PyTorch's Python API in C++: `torch::nn::Module` and standard layers, `torch::optim` optimizers, a data-loading API, and TorchScript loading. It lets C++ programs build, train, and run models using ATen/autograd directly, with no CPython dependency.

## Key Files

| File | Purpose |
|---|---|
| `torch/csrc/api/include/torch/nn/` | `torch::nn::Module`, `Linear`, `Conv`, containers, functional API |
| `torch/csrc/api/src/nn/` | Implementations of the C++ `nn` modules |
| `torch/csrc/api/include/torch/optim/` | `SGD`, `Adam`, and other optimizers |
| `torch/csrc/api/src/optim/` | Optimizer implementations |
| `torch/csrc/api/include/torch/data.h`, `src/data/` | Dataset/DataLoader C++ API |
| `torch/csrc/api/include/torch/serialize.h`, `src/serialize.cpp` | Archive save/load |
| `torch/csrc/api/src/jit.cpp` | Loading TorchScript modules from C++ |

## Public Interface

`torch::nn::Module`, `torch::nn::Linear`, `torch::nn::Sequential`, `register_module()`, `torch::optim::SGD`, `torch::optim::Adam`, `torch::save()`, `torch::load()`, `torch::data::datasets`, `torch::data::make_data_loader()`, and the umbrella header `<torch/torch.h>`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [aten/src/ATen](aten/src/ATen/ADR.md) | depends-on | Uses `at::Tensor` and all operators for module math |
| [torch/csrc/autograd](torch/csrc/autograd/ADR.md) | depends-on | Training relies on the C++ autograd engine |
| [c10/core](c10/core/ADR.md) | depends-on | Tensor, device, and dtype types |
| [torch/csrc/jit](torch/csrc/jit/ADR.md) | depends-on | Loading and running scripted models from C++ |
| LibTorch C++ consumers | depended-on-by | External C++ inference/training programs |

## Runtime Behaviour

A C++ `struct Net : torch::nn::Module` registers submodules via `register_module("fc", torch::nn::Linear(...))`, which stores them in the module's ordered child registry — mirroring the Python `_modules` dict — so that `parameters()`, `to()`, and state-dict traversal recurse consistently. Calling `module->forward(x)` executes ATen ops directly and, when gradients are enabled, records nodes on the same C++ autograd graph the Python path uses. Optimizers walk `parameters()` and apply updates in `step()`. Serialization writes/reads the same archive format used by the Python side.

## Performance Profile

Because there is no Python interpreter or GIL in the loop, the C++ frontend avoids the per-op Python↔C++ bridge cost that dominates small-op eager Python execution — the same ATen kernels run with far lower per-call dispatch overhead. Module composition, parameter iteration, and optimizer steps are ordinary C++ traversals over intrusive-pointer-held tensors. The performance ceiling is therefore the underlying ATen kernels and allocator, not language-boundary overhead, making this frontend attractive for latency-sensitive CPU inference.

## Design Rationale

A maintained C++ frontend exists because several deployment paths must run PyTorch models without a Python runtime (embedded inference, C++ services). Mirroring the Python API's semantics (module registration, state dict, optimizers) keeps mental models portable between the two frontends and lets the same checkpoints load in both. Reusing ATen and the C++ autograd engine avoids duplicating the numeric core, so the frontend is a thin, ergonomic C++ surface over the shared lower layers rather than a parallel implementation.
