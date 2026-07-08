# `torch/_strobelight`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/_strobelight` integrates PyTorch with Meta's Strobelight profiling tools. It wraps `strobeclient` process management for generic Python functions and adds compile-time profiling hooks keyed by compiler frame identifiers.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Marks the package and exposes the profiler modules for import. |
| `cli_function_profiler.py` | Implements `StrobelightCLIFunctionProfiler`, subprocess control, and the `strobelight()` decorator. |
| `compile_time_profiler.py` | Implements `StrobelightCompileTimeProfiler`, frame filtering, shared profiling state, and profile URL generation. |

## Public Interface

The main classes are `StrobelightCLIFunctionProfiler` and `StrobelightCompileTimeProfiler`. Supporting entry points are `strobelight()`, `get_fburl()`, and `get_strobelight_url()`, plus methods such as `StrobelightCompileTimeProfiler.enable()` and `StrobelightCompileTimeProfiler.profile_compile_time()`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/_dynamo](torch/_dynamo/ADR.md) | depends-on | `StrobelightCompileTimeProfiler.get_frame()` reads `CompileContext.current_trace_id()` so compile-time profiles can be keyed to Dynamo frame identifiers. |

## Runtime Behaviour

`StrobelightCLIFunctionProfiler.profile()` enforces a process-wide lock, calls `_start_strobelight()` to launch `strobeclient run --async`, executes the wrapped work function, and then stops the run and optionally fetches result snippets through `_stop_strobelight_no_throw()`. The helper methods `_wait_for_running()`, `_stop_run()`, and `_get_results()` poll `strobeclient getRunStatus` until the external profiler transitions through `PREPARING`, `RUNNING`, and `SUCCESS` states.

`StrobelightCompileTimeProfiler.enable()` verifies that `strobeclient` exists, creates a shared identifier in `_cls_init()`, and builds a reusable `StrobelightCLIFunctionProfiler` configured with compile-time sample tags. `profile_compile_time()` filters by `COMPILE_STROBELIGHT_FRAME_FILTER`, prevents recursive profiling with `inside_profile_compile_time`, and then delegates the actual phase timing to the embedded CLI profiler around the supplied callback.

## Performance Profile

- **Allocation sites** - The profiling paths allocate subprocess argument lists, regex match objects, log records, and small result lists, but the dominant cost is external profiling rather than Python object creation.
- **Synchronization costs** - A class-level `Lock` serializes concurrent profiling attempts, and every run blocks on repeated `strobeclient` status polling before and after the wrapped function executes.
- **Data movement** - The module does not move tensor data; it moves profiler metadata between subprocess stderr, Python regex parsers, and log messages.
- **Redundant or repeated work** - `StrobelightCompileTimeProfiler` keeps counters such as `success_profile_count` and `failed_profile_count` at the class level so repeated compile phases reuse one profiler configuration instead of rebuilding it for every frame.

## Design Rationale

The directory keeps the generic subprocess protocol separate from compile-specific policy because the latter needs compiler frame IDs, recursion suppression, and shared run tags. That split also makes it possible to reuse the CLI profiler as a standalone function wrapper while still providing a compiler-oriented integration point.
