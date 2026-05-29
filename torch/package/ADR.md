# Architecture Decision Record: torch/package

## Architectural Role

`torch/package` provides model packaging infrastructure for portable model distribution. It enables bundling a PyTorch model with all its code dependencies (custom layers, utility functions) into a single portable package that can be shared, deployed, and executed on different machines without complex dependency management. This is important for reproducible model sharing and deployment.

## Responsibilities

- Implementing package creation from models with code dependencies
- Package extraction and execution on target systems
- Dependency detection and bundling (finding all code needed by model)
- Serialization of packages to standardized format (ZIP with metadata)
- Package validation and integrity checking

## Dependencies

**Inbound** (what depends on torch/package):
- Model distribution and sharing (Hugging Face Hub integration)
- Production deployment
- Research reproducibility

**Outbound** (what torch/package depends on):
- ZIP file format for serialization
- Python's pickle and importlib for code bundling
- `torch/jit` for model serialization

## Trade-offs

**Automatic dependency detection vs. manual specification**: Packages attempt to automatically find and bundle dependencies, trading false negatives (missing dependencies) for user convenience. Manual specification would be more reliable but require more user effort.

**Single package vs. modular bundles**: Packages are self-contained units, trading flexibility for simplicity (no need to manage separate dependency files).

## Extension Boundaries

- **Custom importers**: Users can register custom code importers for domain-specific modules.

## Runtime Implications

**Package creation**: Scanning model code to find dependencies takes 1-10 seconds depending on model size.

**Package loading**: Loading a package from disk takes 1-5 seconds depending on package size.

**Execution**: Packaged models execute normally after extraction; no performance overhead compared to standard models.

## Ownership Boundaries

- **torch.package owns**: packaging and extraction logic
- **Packages own**: bundled code and model state

## Verification Points

- `torch/package/__init__.py` — Public API interface
- `torch/package/` — Implementation directory
- `torch/package/package_importer.py` — Import machinery
