# Architecture Decision Record: torch/export

## Architectural Role

`torch/export` provides stable APIs for exporting PyTorch models to other formats (ONNX, mobile, edge devices) without losing model information. It enables deployment of PyTorch models in non-PyTorch environments.

## Responsibilities

- Implementing model export interface (torch.export.export)
- Supporting multiple export formats (ONNX, TorchScript, mobile)
- Preserving model architecture and metadata during export
- Validating exports for correctness

## Dependencies

**Inbound**: Deployment code
**Outbound**: `torch/fx` for graph representation

## Trade-offs

**Lossless export vs. simplicity**: Exporting all model information enables faithful re-implementation but complicates export logic.

## Runtime Implications

**Export cost**: One-time cost at model export time, not during training/inference.

## Ownership Boundaries

- **torch.export owns**: export orchestration
- **Format-specific exporters own**: conversion to target format

## Verification Points

- `torch/export/__init__.py` — Public interface
