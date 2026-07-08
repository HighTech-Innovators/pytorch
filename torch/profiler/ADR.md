# `torch/profiler`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/profiler` is the Python profiler API over the native Kineto and `RecordFunction` backend. Book Chapter 12 (`book/12-observability.md`) describes this layer as the user-facing observability entry point for CPU operator timelines, CUDA activity, memory profiling, stacks, FLOP estimates, schedules, and Chrome trace export. The package turns a context manager into coordinated native profiling, trace handling, tensorboard export, ITT annotation, memory timeline construction, and optimizer-step tracking.

## Key Files

| File | Purpose |
|---|---|
| `profiler.py` | Defines `profile`, `_KinetoProfile`, schedules, tensorboard trace handlers, activity parsing, and execution-trace observer orchestration |
| `__init__.py` | Public exports for `profile`, `schedule`, `ProfilerActivity`, `record_function`, Kineto availability, and optimizer-step hooks |
| `_memory_profiler.py` | Builds categorized memory profiles and timelines from native allocation and tensor metadata events |
| `_chrome_trace_export.py` | Chrome trace formatting and export helpers |
| `_trace_validator.py` | Trace validation utilities for profiler outputs |
| `_utils.py` | Shared profiler utility functions |
| `itt.py` | Python ITT range, mark, and availability wrapper over `torch._C._itt` with a stub when unavailable |
| `python_tracer.py` | Python tracing utilities used when profiler stack and Python call attribution are enabled |
| `_cupti/monitor.py` | CUPTI monitor support for CUDA profiler activity collection |
| `_cupti/pm_sampling.py` | Performance-monitor sampling helpers for CUPTI-based profiling |

## Public Interface

The main public API is `torch.profiler.profile`, a context manager with activities, `record_shapes`, `profile_memory`, `with_stack`, `with_flops`, `with_modules`, `schedule`, `on_trace_ready`, experimental config, execution-trace observer, accumulation, custom trace id, and post-processing timeout options. The package also exposes `schedule`, `tensorboard_trace_handler`, `supported_activities`, `ProfilerAction`, `ProfilerActivity`, `record_function`, `ExecutionTraceObserver`, and `itt.range_push`/`range_pop`/`mark`. Profile objects export Chrome traces, memory timelines, stacks, key averages, and raw events through methods implemented in `profiler.py` and native results.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/csrc/profiler](torch/csrc/profiler/ADR.md) | depends-on | Starts native profiling, reads `_C._profiler` events, configures Kineto, and exports native profiler results |
| [torch/autograd](torch/autograd/ADR.md) | depends-on | Reuses autograd profiler types, `record_function`, Kineto step tracking, and supported activity discovery |
| [torch/cuda](torch/cuda/ADR.md) | related | CUDA activities, CUPTI monitoring, CUDA graphs, and memory timelines depend on CUDA runtime data when enabled |
| [torch/optim](torch/optim/ADR.md) | related | Registers an optimizer post-step hook in selected Kineto daemon modes to increment optimizer profiler steps |
| [torch/utils](torch/utils/ADR.md) | related | Uses common Python utilities and emits traces consumed by tensorboard and other tooling |

## Runtime Behaviour

Entering a `profile` context parses requested activities, creates a `_KinetoProfile`, configures native profiler options, and applies schedule state such as wait, warmup, active, and repeat. During active windows, native callbacks collect operator, device, allocation, Python, and Kineto events while Python code advances profiler steps through `step`. When a trace becomes ready, `on_trace_ready` handlers can write Chrome JSON or gzipped traces, and tensorboard handlers place traces in worker-specific files. Memory profiling consumes native allocation and tensor metadata events, categorizes tensors as inputs, parameters, gradients, activations, temporaries, optimizer state, or autograd detail, and builds timeline data for export.

## Performance Profile

The schedule API limits overhead by collecting only selected training steps instead of profiling an entire long-running job. `record_shapes`, `with_stack`, `profile_memory`, Python tracing, and FLOP estimation increase overhead because native and Python layers retain tensors, collect stacks, process allocation graphs, and compute derived metadata. CUPTI and Kineto activity collection add device-side correlation cost, especially for CUDA kernels and memory copies. Post-processing can dominate very large traces, so `post_processing_timeout_s` allows partial results and `trace_only` skips event materialization when callers only need the trace file.

## Design Rationale

The Python API keeps profiling ergonomic with a context manager and schedule function while delegating low-level timing and device correlation to native `RecordFunction` and Kineto code. The package separates trace collection from trace handlers so users can export Chrome traces, tensorboard files, memory timelines, or execution traces without changing model code. Memory profiling lives beside the user API because categorization requires Python-visible semantics such as parameters, gradients, optimizer state, and autograd nodes. ITT wrappers remain small and optional so builds without ITT still import cleanly and fail only when users call unavailable functions.
