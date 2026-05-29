# Architecture Decision Record: caffe2

## Architectural Role

`caffe2` is a legacy deep learning framework (now deprecated) that is maintained within PyTorch for backward compatibility. It represents the older Caffe2 architecture prior to the PyTorch integration. Modern PyTorch development should use torch APIs, but caffe2 remains supported for existing codebases that depend on Caffe2-style APIs and implementations.

## Responsibilities

- Maintaining legacy Caffe2 operator implementations (largely superseded by torch operators)
- Providing backward compatibility layer for old Caffe2 code
- Supporting some specialized operators not yet ported to PyTorch
- Integrating Caffe2 mobile/deployment infrastructure where needed

## Dependencies

**Inbound** (what depends on caffe2):
- Very limited; primarily legacy codebases
- Some specialized mobile deployment paths

**Outbound** (what caffe2 depends on):
- Minimal; largely isolated from modern PyTorch
- Some integration with torch/utils for checkpoint loading

## Trade-offs

**Backward compatibility vs. code maintenance**: Maintaining caffe2 code ensures existing projects don't break but requires ongoing maintenance for diminishing returns.

**Duplicate implementations vs. code sharing**: Some Caffe2 operators are reimplemented rather than shared with PyTorch versions, trading code duplication for isolation and independent maintenance.

## Extension Boundaries

- **Custom operators**: Custom Caffe2 operators are minimal; users should migrate to torch if possible.

## Runtime Implications

**Limited use**: Caffe2 APIs are called only by legacy code; modern code uses torch APIs.

**Operator execution**: Caffe2 operators execute on available hardware (CPU/GPU) via internal dispatch.

## Ownership Boundaries

- **caffe2 owns**: legacy operator implementations
- **torch owns**: modern PyTorch APIs (replaces caffe2)

## Verification Points

- `caffe2/` — Legacy code directory
- `caffe2/proto/` — Serialization format definitions
