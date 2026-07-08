# `torch/csrc/profiler`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/csrc/profiler` implements the C++ profiler backend for PyTorch observability. Book Chapter 12 (`book/12-observability.md`) identifies `RecordFunction` and Kineto as the central instrumentation path; this directory collects `RecordFunction` callbacks, tensor metadata, memory events, Python stacks, Kineto activities, execution traces, ITT/NVTX ranges, and exported profiler results. It is the native implementation behind `torch.profiler`, legacy autograd profiler paths, and low-level tracing integrations.

## Key Files

| File | Purpose |
|---|---|
| `collection.h` | Defines profiler event types, tensor metadata, extra fields, allocation records, and profiler state structures |
| `collection.cpp` | Implements event collection, post-processing, tensor-id calculation, and profiler result construction |
| `kineto_shim.h` | Thin interface around libkineto activity collection, correlation ids, metadata, and trace start/stop |
| `kineto_shim.cpp` | Kineto integration implementation and fallback behavior when Kineto is unavailable |
| `orchestration/observer.h` | Defines profiler configuration, activity types, profiler states, and lifecycle coordination |
| `orchestration/python_tracer.h` | Python call tracing support for stack and module attribution |
| `python/init.cpp` | Pybind surface for `_C._profiler`, `RecordFunctionFast`, captured tracebacks, and profiler configuration |
| `standalone/nvtx_observer.cpp` | NVTX standalone observer for CUDA range annotation |
| `standalone/itt_observer.cpp` | ITT standalone observer for CPU tooling integration |
| `unwind/unwind.h` | Native stack unwinding support for profiler stack collection |

## Public Interface

The native public surface exposes profiler configuration objects, activity enums, experimental config, profiler lifecycle functions, captured traceback objects, event result types, and `RecordFunctionFast` through `torch._C._profiler`. It also exposes autograd profiler helpers such as `profilerStep`, Kineto availability checks, metadata injection, and activity support queries. Standalone observer APIs emit NVTX, ITT, PrivateUse1, or execution-trace annotations without requiring the full Kineto event pipeline.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [c10/core](c10/core/ADR.md) | depends-on | Reads tensor impl addresses, devices, scalar types, and memory-reporting hooks for event metadata |
| [c10/util](c10/util/ADR.md) | depends-on | Uses approximate clocks, flat hash maps, strong types, exceptions, and utility containers |
| [aten/src/ATen/core](aten/src/ATen/core/ADR.md) | depends-on | Hooks into `RecordFunction`, operator scopes, `at::Tensor`, and dispatcher-level activity data |
| [torch/csrc/autograd](torch/csrc/autograd/ADR.md) | depends-on | Shares autograd profiler activity types, sequence numbers, backward scopes, and Python traceback handling |
| [torch/profiler](torch/profiler/ADR.md) | depended-on-by | Python profiler API starts, stops, schedules, and exports data collected by this native backend |

## Runtime Behaviour

Starting a profiler creates a `ProfilerConfig`, prepares Kineto if requested, installs `RecordFunction` callbacks, and begins collecting CPU operator, backend, allocation, Python call, Python C call, Kineto, and Python GC events. Each operator callback records name, overload, scope, sequence number, correlation id, start and end times, optional input tensor metadata, optional stacks/modules, and optional performance counters. Kineto correlation ids connect CPU-side operator spans to CUDA or other device activities, and post-processing builds trees, flows, tensor identities, memory timelines, and exportable traces. Python bindings expose both the full profiler state and the faster `RecordFunctionFast` context manager for lightweight manual ranges.

## Performance Profile

When profiling is disabled, the runtime cost stays near the `RecordFunction` callback check already present in the dispatcher. Enabling shape recording, stack capture, memory profiling, Python tracing, or performance counters increases overhead because the collector retains tensor metadata, walks Python or native stacks, and post-processes event graphs. `RecordFunctionFast` exists because the Python `record_function` context manager costs about 14 microseconds in cited source comments, while the C++ object path avoids dispatcher overhead and reduces manual-range overhead by roughly 0.2 to 0.4 microseconds per context. Trace-only mode skips event transfer, tree construction, and materialization on exit when callers only need exported traces.

## Design Rationale

The profiler centers on `RecordFunction` because every dispatched operator already passes through that hook point, giving backend-independent coverage without rewriting kernels. Kineto stays behind `kineto_shim` so CUDA/CUPTI and other hardware activity collectors do not leak into the core event data model. The event schema stores raw collection data first and performs tree construction and tensor-id analysis later, which keeps callbacks small and lets expensive attribution run after profiling stops. Standalone observers support NVTX, ITT, and execution traces because many users need external tooling ranges even when they do not need a full PyTorch profiler table.
