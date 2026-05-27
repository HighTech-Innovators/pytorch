# Architecture Decision Record: torch.quantization (Model Quantization)

## Architectural Role

`torch.quantization` is **PyTorch's model quantization framework**, enabling conversion of floating-point models to lower-precision integer representations. It enables:

1. **Post-training quantization**: Quantizing trained models without retraining
2. **Quantization-aware training**: Training models that remain quantizable
3. **Dynamic quantization**: Computing quantization parameters at runtime
4. **Custom backends**: Supporting various quantization schemes and hardware
5. **Model compression**: Reducing model size and memory footprint

Key insight: `torch.quantization` is **precision reduction via integer arithmetic**. Unlike AMP (which uses lower floating-point precision), quantization uses integers with a scale factor. This provides better memory savings and sometimes better performance on integer-only hardware.

## Responsibilities

### What This Subsystem Owns

1. **Quantization Schemes** (`torch/quantization/qconfig.py`)
   - Defining how to quantize activations and weights
   - Per-channel vs. per-layer quantization
   - Symmetric vs. asymmetric quantization

2. **Quantization Preparation** (`torch/quantization/quantize.py`)
   - Fusing operations for quantization (Conv+BatchNorm → Conv)
   - Inserting fake quantization for training
   - Calibrating quantization parameters

3. **Calibration** (`torch/quantization/observer.py`)
   - Collecting statistics on activation ranges
   - Computing optimal scale and zero-point
   - Multiple calibration strategies (min-max, percentile, entropy)

4. **Quantized Operations** (`torch/nn/quantized/`)
   - Quantized versions of common modules (Linear, Conv)
   - Quantized operation implementations
   - Kernel execution on integer hardware

5. **Backend Support**
   - Quantization for different hardware (ARM, x86, NVIDIA)
   - Backend-specific optimization options

### What This Subsystem Does NOT Own

- **Model definitions**: torch.nn owns module definitions
- **Training**: torch.optim and torch.autograd own training loops
- **Kernel execution**: Backend kernels execute quantized operations
- **Memory management**: c10 allocators handle memory
- **Serialization**: torch.save/torch.load handle persistence

## Dependencies

### Upstream Dependencies (What Uses This)

- **Model optimization workflows**: Reducing model size/latency
- **Edge deployment**: Mobile and embedded inference
- **Research on quantization**: Academic studies on compression

### Downstream Dependencies (What This Uses)

- **torch.nn**: Creating quantized module versions
- **torch.autograd**: Fake quantization for training
- **ATen**: Executing quantized operations
- **torch.jit**: Converting quantized models for export

### Dependency Direction

```
Training Code or Pre-trained Model
    ↓
torch.quantization (quantization framework)
    ├─→ Quantization preparation
    ├─→ Calibration
    ├─→ Fake quantization (for training)
    └─→ Quantized module conversion
        ↓
    Quantized model with integer arithmetic
```

## Trade-offs and Design Decisions

### Post-Training vs. Quantization-Aware Training

**Decision**: Support both post-training and QAT.

**Trade-off**:
- ✅ **Advantage**: Post-training is fast; no retraining needed
- ✅ **Advantage**: QAT provides better accuracy
- ❌ **Disadvantage**: Complexity of supporting both
- ❌ **Disadvantage**: Users must choose which approach to use

**Evidence**: `torch.quantization.quantize()` for post-training; fake quantization for QAT.

### Symmetric vs. Asymmetric Quantization

**Decision**: Support both; let users choose via QConfig.

**Trade-off**:
- ✅ **Advantage**: Flexibility; symmetric is simpler, asymmetric is more accurate
- ✅ **Advantage**: Users can tune for their hardware
- ❌ **Disadvantage**: Complexity; more options to tune
- ❌ **Disadvantage**: Performance implications not obvious

**Evidence**: QConfig specifies symmetric/asymmetric via observer types.

### Fake Quantization for Training

**Decision**: Insert fake quantization during training to simulate integer precision.

**Trade-off**:
- ✅ **Advantage**: Training adapts to quantization noise early
- ✅ **Advantage**: Better final model accuracy
- ❌ **Disadvantage**: Training slower (additional operations)
- ❌ **Disadvantage**: More complex training code

**Evidence**: `torch.quantization.QuantStub` inserted during model preparation.

## Runtime Implications

### Lifecycle and Initialization

1. **Model preparation**: Fusing operations, inserting fake quantization
2. **Calibration**: Running training/validation data to collect statistics
3. **QAT (optional)**: Fine-tuning model with fake quantization
4. **Quantization conversion**: Replacing float modules with quantized versions
5. **Export**: Converting to deployment format (ONNX, mobile, etc.)
6. **Inference**: Running quantized model with integer arithmetic

### Calibration Process

- **Min-max calibration**: Compute scale from observed min/max values
- **Percentile calibration**: Ignore outliers; use percentiles
- **Entropy calibration**: Minimize KL divergence between original and quantized distributions

## Performance Implications

### Speedup Opportunities

1. **Integer arithmetic**: Faster than floating-point on many devices
2. **Memory bandwidth**: 4x reduction (float32 to int8) reduces memory bottleneck
3. **Model size**: 4x smaller models fit better in cache
4. **Quantized kernels**: Specialized integer kernels on hardware

### Accuracy Considerations

- **Post-training quantization**: May lose 1-5% accuracy
- **QAT**: Can achieve near-original accuracy
- **Per-channel quantization**: Better accuracy than per-layer

## Key Implementation Files

| File | Purpose |
|---|---|
| `torch/quantization/qconfig.py` | Quantization schemes and configs |
| `torch/quantization/quantize.py` | Quantization pipeline |
| `torch/quantization/observer.py` | Calibration statistics |
| `torch/nn/quantized/` | Quantized modules and operations |

## Verification

All claims in this ADR are grounded in:

1. **Source Files**: Examined `torch/quantization/` implementation
2. **Documentation**: PyTorch quantization guide
3. **Code Flow**: Understanding preparation, calibration, conversion

Last Verified: 2026-05-27
