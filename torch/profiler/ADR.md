# `torch/profiler`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/profiler` provides PyTorch's primary performance observability interface: the `profile` context manager for collecting CPU and GPU operation traces, a `schedule`-based step-triggered recording model, memory profiling, and export to Chrome trace format and TensorBoard. It wraps Kineto (a profiling library backed by CUPTI and perf_event) via `torch._C._autograd`.

## Key Files

| File | Purpose |
|---|---|
| `profiler.py` | `profile` context manager — `_KinetoProfile`, `schedule`, `tensorboard_trace_handler`, `ProfilerAction`, `ExecutionTraceObserver` |
| `__init__.py` | Public re-exports; conditionally registers `_optimizer_post_hook` via `register_optimizer_step_post_hook` to track optimizer steps |
| `_memory_profiler.py` | `MemoryProfile`, `MemoryProfileTimeline` — snapshot-based memory tracing; records allocation and free events with stack traces |
| `_chrome_trace_export.py` | Serialises Kineto trace to Chrome JSON (`chrome://tracing`) format |
| `_cupti/` | CUPTI (CUDA Profiling Tools Interface) integration: lazy reinit for CUDA 12 graphs |
| `python_tracer.py` | Python-level event tracer used as fallback when Kineto is not available |
| `_utils.py` | Utility helpers: `_parse_legacy_trace`, event filtering |

## Public Interface

| Symbol | Description |
|---|---|
| `torch.profiler.profile(activities, schedule, on_trace_ready, ...)` | Context manager; collects events for `ProfilerActivity.CPU` and/or `.CUDA`; `with profile(...) as p: ...` |
| `torch.profiler.schedule(wait, warmup, active, repeat)` | Returns a callable that maps step index to `ProfilerAction` (NONE, WARMUP, RECORD, RECORD_AND_SAVE) |
| `torch.profiler.tensorboard_trace_handler(dir_name)` | Callback for `on_trace_ready`; writes Chrome JSON traces into `dir_name` |
| `torch.profiler.ProfilerActivity.CPU` | Record CPU-side operator events |
| `torch.profiler.ProfilerActivity.CUDA` | Record CUDA kernel events via CUPTI |
| `torch.profiler.record_function(name)` | Context manager that annotates a code region; appears in Chrome traces as a named span |
| `torch.profiler.ExecutionTraceObserver` | Records a JSON execution trace for `torch.distributed` and model serving analysis |
| `MemoryProfile` | Snapshot memory allocations and frees with call-stack attribution |
| `kineto_available()` | Returns `True` if the Kineto profiling library is compiled in |

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| `torch._C._autograd` | depends-on | `_supported_activities`, `kineto_available`, `DeviceType` — C++ Kineto integration; `_enable_profiler`, `_disable_profiler` |
| `torch._C._profiler` | depends-on | `_ExperimentalConfig`, `ProfilerActivity`, `RecordScope`, `_add_execution_trace_observer` |
| `torch.autograd.profiler` | depends-on | Legacy profiler: `KinetoStepTracker`, `record_function`; still in the call chain for backward compatibility |
| [torch/optim](torch/optim/ADR.md) | depends-on | `register_optimizer_step_post_hook` in `torch.optim.optimizer` is called by the profiler's `__init__` to track optimizer steps |
| [aten/src/ATen](aten/src/ATen/ADR.md) | depends-on | `RECORD_FUNCTION(name, inputs)` macro in ATen kernels fires events consumed by the C++ profiler |
| [torch/_dynamo](torch/_dynamo/ADR.md) | depended-on-by | Dynamo's `ChromiumEventLogger` writes profiler events; `dynamo_timed` uses `record_function` |

## Runtime Behaviour

`profile.__enter__` calls `torch._C._autograd._enable_profiler(ProfilerConfig, activities)` which activates Kineto's event collection for the specified activities. Every ATen operation fires `RECORD_FUNCTION` which calls `at::RecordFunction::before` in C++; if the profiler is active this records the event with a timestamp from `ApproximateClock::getTime()`. `profile.__exit__` calls `_disable_profiler()` which collects all events, correlates CPU and CUDA events by thread ID and correlation ID, and returns a `torch.autograd.profiler.profile` result object. The `schedule` callable maps the step index to a `ProfilerAction`; `profile.step()` must be called at each training step boundary to advance the schedule. When `ProfilerAction.RECORD_AND_SAVE` fires, `on_trace_ready` is called with the current `profile` object.

## Performance Profile

- **Allocation sites**: the profiler allocates one event record per ATen operation per call; at high operation rates (>10,000 ops/sec) this generates significant allocation pressure in the profiler's event buffer.
- **Synchronization costs**: CUDA event timestamps require `cudaEventRecord` at kernel launch and `cudaEventSynchronize` at profiler stop; the synchronisation at `profile.__exit__` can take hundreds of milliseconds for long traces. CPU events use `ApproximateClock::getTime()` (vDSO), which is fast.
- **Data movement**: execution trace observer serialises events to JSON files; for distributed training this generates one file per rank, potentially gigabytes of trace data.
- **Redundant or repeated work**: when `schedule` sets `ProfilerAction.NONE`, the profiler records nothing; the `RECORD_FUNCTION` hook checks a thread-local flag and returns immediately. Warmup steps (`ProfilerAction.WARMUP`) collect events but discard them, allowing CUDA JIT and cache effects to stabilise before measuring.

## Design Rationale

The `schedule`-based recording model separates training steps from profiler control, allowing the profiler to capture a specific window of steps without requiring manual start/stop in user code. This is necessary because profiling the first steps (which include JIT compilation and cache warming) would pollute performance measurements. `record_function` is retained as a separate API from the profiler context manager so that library code (ATen, distributed) can annotate operations without knowing whether profiling is active — the cost is a thread-local flag check when inactive.
