# Architecture Decision Record: aten/src/ATen/detail

## Architectural Role

`aten/src/ATen/detail` provides internal implementation details and utilities used throughout ATen that are not part of the public API. It houses low-level abstractions like the InlineEvent system, record functions, and type mapping utilities that enable observability and performance analysis without cluttering public interfaces.

## Responsibilities

- Managing RecordFunction infrastructure for operator profiling hooks
- Providing InlineEvent for high-performance event tracking
- Type utilities and introspection (dtype mapping, type helpers)
- TensorIterator and shape utilities
- Internal profiling and observability hooks

## Dependencies

**Inbound**: ATen operators, profiler, tracing systems
**Outbound**: c10/core, CUDA libraries

## Trade-offs

**RecordFunction overhead**: Every operator execution goes through RecordFunction callbacks, adding 1-2% overhead even when not profiling. Profiling requires minimal code but has constant baseline cost.

## Runtime Implications

**Profiling hooks**: Enabled at library load time; can be configured dynamically to reduce overhead in production.

**Event tracking**: InlineEvent provides low-latency event recording for profiler integration.

## Ownership Boundaries

- **RecordFunction owns**: callback registry and hook invocation
- **InlineEvent owns**: event metadata and timestamps

## Verification Points

- `aten/src/ATen/record_function.h` — Profiling hook infrastructure
- `aten/src/ATen/detail/` — Implementation directory
