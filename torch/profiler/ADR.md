# Architecture Decision Record: torch/profiler

## Architectural Role

`torch/profiler` provides profiling infrastructure (torch.profiler.profile) for performance analysis and optimization. It enables measurement of kernel execution time, operator latency, and memory usage. This module is important for performance optimization but not Runtime Critical for training itself.

## Responsibilities

- Implementing profiler context manager (torch.profiler.profile)
- Collecting performance data (operator timing, kernel launching, memory allocation)
- Integrating with Kineto (Intel profiler backend)
- Exporting profiles to visualization formats (Chrome trace, TensorBoard)

## Dependencies

**Inbound**: User code, analysis scripts
**Outbound**: `aten/src/ATen/detail` for RecordFunction hooks, `c10/cuda` for GPU timing

## Trade-offs

**Profiler overhead**: Enabled profilers add 5-20% overhead to execution. This is acceptable for offline profiling but too expensive for always-on production monitoring.

## Runtime Implications

**Context activation**: Within profiler context, RecordFunction callbacks are activated to collect timing data.

**Export**: Profiles can be exported to Chrome trace format for browser-based visualization.

## Performance Implications

**5-20% overhead when enabled**: Disabled by default to avoid production overhead.

## Ownership Boundaries

- **Profiler owns**: measurement and export logic
- **Kineto owns**: GPU-level profiling

## Verification Points

- `torch/profiler/__init__.py` — Public interface
- `torch/profiler/profiler.py` — Main profiler implementation
