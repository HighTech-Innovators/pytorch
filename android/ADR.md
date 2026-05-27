# Architecture Decision Record: Android Integration

**Location:** `./src/android/`  
**Last Updated:** 2026-05-27  
**Classification:** Platform Integration, Mobile Deployment

---

## Executive Summary

Android integration provides PyTorch runtime for Android devices via the PyTorch Android API. It packages the PyTorch C++ core (ATen, autograd, dispatch) and Python interpreter into an Android library that Java applications can call to load and execute models on mobile devices.

---

## Architectural Role

Android integration serves as the **mobile deployment layer** for PyTorch. Its responsibilities are:

1. **JNI Bindings** — Bridge between Java and C++ PyTorch core
2. **Model Loading** — Load serialized models (.pt files) on device
3. **Inference** — Run forward pass on Android devices
4. **Device Selection** — Route to CPU, GPU, or accelerators (NNAPI, Hexagon)
5. **Memory Management** — Handle memory constraints of mobile devices
6. **Gradle Build** — Package PyTorch for Android distribution

Because android integration is the bridge to mobile execution, **model inference on devices depends on correct JNI bindings and device handling**.

---

## Responsibilities

### What Android Integration Does

- **JNI Layer** — Java-C++ interface for calling PyTorch from Android
- **Module Loading** — Load TorchScript modules from .pt files
- **Forward Inference** — Run model forward pass on device
- **Tensor Conversion** — Convert Java arrays to/from Tensors
- **GPU Support** — Support GPU acceleration if available
- **Performance Profiling** — Measure inference latency and memory usage
- **TorchVision Bindings** — Bindings for torchvision models (separate library)

### What Android Integration Does NOT Do

- **Model Training** — Android only does inference (training is CPU/GPU intensive)
- **Distributed Execution** — Single-device only
- **Advanced Operations** — Subsets of operations supported (not full PyTorch)
- **Dynamic Graphs** — Only static compiled models (.pt files)

---

## Dependencies

### What Depends On Android Integration

- **Android Applications** — Apps using PyTorch for on-device inference
- **Mobile ML Platforms** — Systems deploying models to Android devices

### What Android Integration Depends On

- **PyTorch Core** — c10, ATen, autograd (C++ libraries)
- **Python** — CPython runtime embedded or cross-compiled
- **Android NDK** — C++ compilation tools for ARM architecture
- **Android SDK** — Java/Kotlin libraries for Android
- **Gradle** — Build system for Android packages

**Dependency Direction**: Applications depend on Android integration; integration depends downward on PyTorch core and Android SDK.

---

## Trade-Offs and Design Decisions

### Decision 1: Subset of Operations for Mobile

**What**: Not all PyTorch operations are available on Android. Focus on inference operations commonly used in mobile models (conv2d, linear, relu, etc.).

**Why**:
- Binary Size: Full PyTorch is 100+ MB; mobile is restricted (< 50 MB typical app)
- Performance: Mobile CPUs are slower; operations are optimized for inference only
- Memory: Mobile devices have limited RAM (< 4GB typical); avoid training operations

**Alternatives**:
- Full PyTorch — Would make app too large and slow
- Ultra-Minimal — Might break compatibility

**Trade-offs**:
- **Pro**: Reasonable app size, acceptable performance
- **Con**: Can't run arbitrary PyTorch code; only inference

---

### Decision 2: TorchScript-Only Models

**What**: Android only supports TorchScript-compiled models (.pt files), not live Python code.

**Why**:
- Static Deployment: Models are fixed at deployment time; no dynamic graph changes
- Smaller Runtime: Don't need Python interpreter for arbitrary code
- Security: No ability to load untrusted Python code on device
- Determinism: Model behavior is fixed; reproducible inference

**Alternatives**:
- Full Python — Would require Python interpreter; too large and slow
- Hybrid — Allow some dynamic Python — too complex

**Trade-offs**:
- **Pro**: Smaller, faster, more secure
- **Con**: Models must be pre-compiled to TorchScript

---

### Decision 3: JNI for Java-C++ Interface

**What**: Android apps call PyTorch via JNI (Java Native Interface), not direct C++ calls.

**Why**:
- Android Ecosystem: Android apps are primarily Java/Kotlin; JNI is standard
- Safety: JNI provides type safety and exception handling
- Portability: Same JNI code works across Android API levels

**Alternatives**:
- Direct C++ — Would require C++ Android app (less common)
- HTTP Server — Forward requests to PyTorch server (adds latency)

**Trade-offs**:
- **Pro**: Natural fit for Android ecosystem, safe
- **Con**: JNI overhead; less efficient than direct C++

---

### Decision 4: GPU Support Via NNAPI and Vulkan

**What**: Android supports GPU acceleration via NNAPI (neural network API) and Vulkan (graphics API).

**Why**:
- Performance: GPU acceleration 5-10x faster than CPU for many models
- Ubiquity: Most modern Android devices have GPU
- Standard: NNAPI is standard Android API for neural networks

**Alternatives**:
- CPU Only — Simple but slow
- Custom GPU Kernels — More control but fragmentation across devices

**Trade-offs**:
- **Pro**: Good performance on modern devices
- **Con**: Requires vendor-specific NNAPI implementations; less portable

---

## Extension Boundaries

### Extending Android Integration

**Supported Extensions:**
1. **New Backend** — Add support for new Android accelerator (e.g., custom Hexagon support)
2. **Custom Operations** — Register new operations for mobile inference
3. **Memory Optimization** — Optimize memory usage for constrained devices

**NOT Supported:**
- Training on Android — not supported, architecture doesn't support backward pass
- Full Python — would require large interpreter binary

### Integration Points

- **PyTorch Core** — Calls c10, ATen, autograd C++ libraries
- **Android SDK** — Uses Android APIs for device features
- **TorchVision** — Optional bindings for vision models
- **JNI** — Java interface for Android apps

---

## Runtime Implications

### Build and Deployment

**Build Sequence:**
1. Clone PyTorch source
2. Cross-compile PyTorch C++ for ARM architecture
3. Build Android AAR (Android Archive) with native .so libraries
4. Publish AAR to Maven repository
5. Android app adds AAR as dependency in Gradle

**At Runtime:**
- App loads .so library via JNI
- App loads .pt model file from assets
- App calls Java API to run inference
- Results returned to Java

### Device Compatibility

- **Minimum API Level** — API 21 (Android 5.0); older devices not supported
- **Architecture** — ARM64 (aarch64) primary; ARMv7 as fallback
- **Memory** — Requires at least 200MB free memory for typical model
- **GPU Support** — Optional; CPU-only operation always works

### Failure Modes

- **Model Not Found** — Inference fails if .pt file missing
- **Operation Not Supported** — Runtime error if model uses unsupported operation
- **Out of Memory** — App crashes if model too large for device
- **Device Not Compatible** — App refuses to load on unsupported device

---

## Performance Implications

### Mobile Constraints

- **CPU Speed** — ARM CPU typically 1-3 GHz (vs 3-5 GHz on desktop)
- **Memory** — 2-4 GB total; PyTorch app occupies 50-500 MB depending on model
- **Battery** — Inference drains battery; limit computation per inference
- **Thermal** — Extended inference can trigger thermal throttling

### Optimization Strategies

- **GPU Acceleration** — Use NNAPI for 5-10x speedup
- **Model Quantization** — Reduce model size and memory by 4-8x via int8 quantization
- **Pruning** — Remove unimportant weights; reduces compute
- **Selective Acceleration** — GPU for heavy layers, CPU for others

---

## Ownership Boundaries

### What Android Integration Owns

- JNI bindings to PyTorch C++ core
- Java API for model loading and inference
- Device-specific GPU handling
- Android-specific memory management

### What Android Integration Does NOT Own

- PyTorch Core — owned by c10/aten/torch
- Model Format (.pt files) — owned by torch/jit
- TorchScript Compilation — owned by torch/jit

---

## Testing and Validation

### Critical Tests

- **JNI Bindings** — Verify Java-C++ communication works
- **Model Loading** — Verify .pt models load correctly
- **Inference Correctness** — Verify results match desktop execution
- **Device Compatibility** — Verify works on multiple Android devices
- **Memory** — Verify no memory leaks; models can be loaded/unloaded

### Known Gaps

- Limited testing on edge devices (old phones, low-end devices)
- No explicit battery/power consumption profiling
- Limited testing of thermal throttling under sustained inference

---

## Related Systems

- **PyTorch Core** (c10, ATen, autograd) — Core computation
- **torch/jit/** — TorchScript compilation
- **TorchVision** — Vision model bindings (optional)
- **NNAPI** — Android neural network acceleration API

---

## References

- `android/pytorch_android/` — Main Android library
- `android/pytorch_android_torchvision/` — TorchVision bindings
- Android NDK — C++ compilation tools
- NNAPI — Android neural network API documentation
