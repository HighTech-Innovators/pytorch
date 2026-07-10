# `torch/profiler`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/profiler` is the Python profiling API. It wraps the Kineto-backed `torch._C._profiler` extension to provide the `profile()` context manager, scheduling callbacks, Chrome trace export, memory profiling, and execution-trace observation.

## Key Files

| File | Purpose |
|---|---|
| `torch/profiler/profiler.py` | `profile` context manager (1585 lines); `schedule()` for step-based activation; `tensorboard_trace_handler`, `ProfilerActivity`, `ProfilerAction` |
| `torch/profiler/_memory_profiler.py` | `MemoryProfile`, `MemoryProfileTimeline`: allocation-site tracking and timeline visualization |
| `torch/profiler/_chrome_trace_export.py` | Chrome JSON trace serialization |
| `torch/profiler/python_tracer.py` | Python-level call-stack capture during profiling |
| `torch/profiler/_trace_validator.py` | Validates captured trace data structure before export |

## Public Interface

`torch.profiler.profile(activities=..., schedule=..., on_trace_ready=..., record_shapes=..., profile_memory=...)`, `torch.profiler.schedule(wait, warmup, active, repeat)`, `torch.profiler.tensorboard_trace_handler(dir_name)`, `torch.profiler.ProfilerActivity`, `torch.profiler.ExecutionTraceObserver`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| `torch._C._profiler` | depends-on | C++ Kineto integration: `_add_execution_trace_observer`, `_enable_execution_trace_observer`, `_ExperimentalConfig` |
| `torch.autograd` | depends-on | `kineto_available`, `ProfilerActivity` imported from `torch.autograd` |
| User training code | depended-on-by | `with torch.profiler.profile(...) as p:` wraps training steps |

## Runtime Behaviour

`profile.__enter__` starts the Kineto profiler via `torch._C._profiler` bindings, activating the selected `ProfilerActivity` set (CPU, CUDA, XPU). The `schedule()` callback drives a state machine across steps: `WAIT` (profiler off) → `WARMUP` → `ACTIVE` (recording) → `RECORD_AND_SAVE`; `profile.step()` advances this machine. On `RECORD_AND_SAVE`, `on_trace_ready` is called with the `profile` object, which exposes `key_averages()`, `events()`, and `export_chrome_trace()`. `_memory_profiler.py` instruments the allocator to capture allocation-site stacks when `profile_memory=True`. `ExecutionTraceObserver` records the execution trace as a JSON artifact for post-hoc analysis.

## Performance Profile

Profiling adds per-op event-recording overhead proportional to the number of operations executed while active. `record_shapes=True` adds shape-capture overhead on each operator call. The `schedule` mechanism lets production code amortize this by activating the profiler only for a short window of steps out of a longer run. `_memory_profiler.py` with `profile_memory=True` adds a stack-capture call at each allocation site — significant cost when allocation rates are high. On CPU-only deployments CUDA activities are absent; only CPU activity recording is active, keeping overhead lower than in mixed CPU/GPU runs.

## Design Rationale

The `schedule` state machine separates profiling policy (which steps to record) from mechanism (the Kineto API), so training loops can declare a recording window declaratively rather than wrapping code in conditionals. Delegating the heavy recording to `torch._C._profiler` (Kineto) keeps Python-layer overhead minimal and reuses the same record format as the C++ profiler. `tensorboard_trace_handler` as a first-class `on_trace_ready` callback integrates with the standard TensorBoard visualization workflow without requiring manual trace management.
