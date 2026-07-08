# `torch/csrc/api`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/csrc/api` implements the C++ frontend, also known as libtorch. It maps the user-facing concepts from book Chapter 06 (`book/06-python-bindings.md`) into C++: modules, parameters, optimizers, data loading, serialization, device helpers, and JIT entry points operate directly on ATen tensors without crossing the Python C API. The directory provides a C++ API that mirrors the Python `torch.nn`, `torch.optim`, `torch.data`, and `torch.jit` surface while preserving static typing and RAII ownership.

## Key Files

| File | Purpose |
|---|---|
| `include/torch/torch.h` | Umbrella header for the public C++ frontend |
| `include/torch/nn/module.h` | Base `torch::nn::Module` class with parameter, buffer, submodule, traversal, cloning, and device conversion APIs |
| `include/torch/nn/modules/linear.h` | Representative module header defining options and implementation classes for a concrete neural-network layer |
| `include/torch/optim/optimizer.h` | Base optimizer options, parameter groups, state, serialization, and cloning interfaces |
| `include/torch/data/dataloader/base.h` | Threaded C++ `DataLoaderBase` with job/result queues, workers, prefetching, and ordered sequencing |
| `include/torch/serialize/archive.h` | Input/output archive abstractions for saving C++ modules and optimizer state |
| `src/nn/module.cpp` | Runtime implementation for module registration, traversal, cloning, and recursive state operations |
| `src/optim/adam.cpp` | Concrete optimizer implementation that mutates parameter tensors and optimizer state |
| `src/jit.cpp` | Implements `torch::jit::compile` by creating a `CompilationUnit` from TorchScript source |

## Public Interface

The public surface is the installed header tree under `include/torch`. Users construct `torch::nn::Module` subclasses, register parameters with `register_parameter`, register buffers with `register_buffer`, compose submodules with `register_module`, move state with `to`, and serialize with input/output archives. Optimizers expose parameter groups, typed options such as `AdamOptions` and `SGDOptions`, `step`, `zero_grad`, and state serialization. Data APIs expose datasets, samplers, transforms, iterators, and dataloaders. Device helpers expose `torch::cuda`, `torch::mps`, and `torch::xpu` wrappers, while `torch::jit::compile` links C++ users to the TorchScript compiler.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [c10/core](c10/core/ADR.md) | depends-on | Uses devices, scalar types, intrusive pointers, optional values, and exception utilities through ATen and C++ frontend headers |
| [aten/src/ATen/core](aten/src/ATen/core/ADR.md) | depends-on | Uses `at::Tensor`, dispatcher-backed operators, archives, and tensor options as the computational substrate |
| [torch/csrc/autograd](torch/csrc/autograd/ADR.md) | depends-on | Optimizers and modules operate on gradient-bearing tensors and call autograd-aware tensor methods |
| [torch/csrc/jit](torch/csrc/jit/ADR.md) | depends-on | `src/jit.cpp` and public JIT headers expose TorchScript compilation and scripted module integration |
| [torch/nn](torch/nn/ADR.md) | related | Mirrors the Python module and options model in C++ so libtorch users get the same architectural vocabulary |

## Runtime Behaviour

A C++ module constructor registers parameters, buffers, and child modules into ordered dictionaries owned by `torch::nn::Module`; recursive traversal APIs then apply operations such as `to`, `clone`, `train`, `eval`, and serialization across that tree. Forward methods call ATen functions directly, so operator execution uses the same dispatcher and tensor implementations as Python code. Optimizer `step` implementations read parameter groups, access or initialize per-parameter state, and update tensors in place. `DataLoaderBase` creates jobs, pushes them to worker threads, collects results through a shuttle, preserves sequence order when configured, and joins workers in its destructor.

## Performance Profile

The C++ frontend adds little runtime overhead over ATen because module and optimizer calls dispatch directly to C++ tensor operators instead of passing through Python argument parsing or the GIL. Template-heavy headers increase compile time for C++ applications, but they keep options and module wrappers type-safe at runtime. `DataLoaderBase` overlaps host-side dataset fetching with model execution by prefetching jobs and using worker threads when `workers > 0`; the in-order sequencer trades some latency for deterministic batch order. Optimizer implementations store state in C++ objects and mutate tensors in place, which avoids Python object churn in tight training loops.

## Design Rationale

Libtorch exists so production inference, embedded systems, and C++ training stacks can use PyTorch without embedding Python. The API mirrors Python names because users move models and concepts between Python and C++ frequently, but it uses C++ ownership, templates, and explicit options objects to fit the language. The module tree keeps parameters and buffers registered centrally so serialization, cloning, and device conversion do not need per-module custom traversal code. Data loading and optimizers live in this directory because they form the minimum application framework needed around ATen tensors.
