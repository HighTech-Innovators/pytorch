# `torch/_logging`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/_logging` owns PyTorch's structured and component-scoped logging system. It maps `TORCH_LOGS` settings to logger state, manages artifact loggers, and emits structured trace records that compiler tooling can recover with `tlparse`.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Re-exports the public logging API such as `set_logs`, `trace_structured`, `getArtifactLogger`, and `warning_once`. |
| `_internal.py` | Implements logger registration, environment parsing, handler setup, `LazyTraceHandler`, structured trace emission, and warning filters. |
| `structured.py` | Converts stacks and file references into compact structured payloads using string interning and trace emission helpers. |
| `_registrations.py` | Registers component aliases and artifact names that `set_logs()` and `getArtifactLogger()` understand. |

## Public Interface

The public API consists of `set_logs()`, `trace_structured()`, `dtrace_structured()`, `trace_structured_artifact()`, `getArtifactLogger()`, `get_structured_logging_overhead()`, `hide_warnings()`, `warning_once()`, `LazyString`, and `DEFAULT_LOGGING`. Artifact producers also use `structured.intern_string()`, `structured.get_user_stack()`, and `structured.get_framework_stack()` when building JSON payloads.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/_dynamo](torch/_dynamo/ADR.md) | depended-on-by | Dynamo debugging and compile profiling use `set_logs()`, artifact loggers, and `trace_structured()` for graph breaks, guards, and bytecode diagnostics. |
| [torch/_native](torch/_native/ADR.md) | depended-on-by | `instrumentation.py` emits `CompileEvent` records through `trace_structured()` and logger names registered here. |
| [torch/_subclasses](torch/_subclasses/ADR.md) | depended-on-by | FakeTensor paths call `dtrace_structured()` and artifact loggers to report mismatched fake kernels and hierarchical compile data. |

## Runtime Behaviour

`_init_logs()` resets previously configured logger state, applies overrides from `TORCH_LOGS` and `TORCH_LOGS_OUT`, configures handlers for registered component loggers, and then installs a special `LazyTraceHandler` on the synthetic `trace_log`. `LazyTraceHandler.emit()` defers file creation until the first structured trace record, picks the destination directory from `TORCH_TRACE`, `TORCH_DTRACE`, or the compile debug directory, and emits a `torch_version` artifact right after opening the trace file.

`trace_structured()` validates reserved field names, captures compile-context metadata such as `frame_id`, `frame_compile_id`, and `attempt`, then forwards JSON-compatible metadata and payloads to `log_trace_structured_event`. `structured.py` supports that path by interning repeated strings through `intern_string()`, dumping generated source files with `dump_file()`, and trimming user or framework stacks into compact frame dictionaries.

## Performance Profile

- **Allocation sites** - Logger and artifact registration allocate handler objects and formatter state once, while each structured trace event allocates only the metadata and payload objects produced by the deferred callables.
- **Synchronization costs** - The logging stack does not synchronize tensors or devices, but file-backed tracing still serializes writes through Python logging handlers and can pay filesystem latency the first time `LazyTraceHandler` opens its stream.
- **Data movement** - `trace_structured()` avoids eager formatting by accepting `metadata_fn` and `payload_fn`, and `LazyString` postpones string construction until a logger actually renders the record.
- **Redundant or repeated work** - Downstream callers check whether `trace_log.handlers` is non-empty before building expensive payloads, and `_logging` tracks per-compile structured logging overhead in `structured_logging_overhead` so trace cost can be accounted for explicitly.

## Design Rationale

The subsystem separates regular module logs from structured trace artifacts because production compiler jobs often need machine-readable traces even when stderr is suppressed. Central registration in `_registrations.py` and `_internal.py` keeps user-facing configuration consistent across Dynamo, Inductor, native DSLs, and fake tensor diagnostics.
