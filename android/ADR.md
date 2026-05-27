# Architecture Decision Record: android (Mobile Platform Support)

## Architectural Role

`android/` is **PyTorch's Android platform support layer**, enabling model inference and training on Android devices. It provides:

1. **Mobile runtime**: Lightweight PyTorch runtime optimized for Android
2. **Model loading**: Serializing and loading models on mobile
3. **JNI bindings**: Java-C++ interoperability
4. **Native modules**: android-specific optimizations

Key insight: `android/` is **platform adaptation layer** that bridges PyTorch to Android's Java/Kotlin ecosystem via JNI.

## Responsibilities

### What This Subsystem Owns

1. **Mobile Runtime** (`android/pytorch_android/`)
   - PyTorch runtime optimized for Android
   - JVM wrapper for C++ runtime
   - Resource management

2. **Model Serialization**
   - Loading .pt models on mobile
   - Quantized model support
   - Memory management

3. **JNI Bindings** (`android/pytorch_android/`)
   - Java-C++ boundary
   - Type marshalling
   - Exception handling

4. **Vision Module** (`android/pytorch_android_torchvision/`)
   - TorchVision integration for Android
   - Image preprocessing
   - Model-specific utilities

### What This Subsystem Does NOT Own

- PyTorch core operations (torch, aten)
- Automatic differentiation
- Training infrastructure (only inference in most cases)

## Dependencies

### Upstream Dependencies

- Android application developers
- Mobile ML frameworks integration

### Downstream Dependencies

- PyTorch core (torch, aten, c10)
- JNI/Android SDK

## Trade-offs and Design Decisions

### JNI Overhead vs. Pure Native

**Decision**: Use JNI to provide Java API with C++ backend.

**Trade-off**:
- ✅ **Advantage**: Familiar Java API for Android developers
- ✅ **Advantage**: Leverages Java ecosystem
- ❌ **Disadvantage**: JNI overhead
- ❌ **Disadvantage**: Additional complexity

**Evidence**: JNI wrappers in pytorch_android module.

## Key Implementation Files

| File | Purpose |
|---|---|
| `android/pytorch_android/` | Main Android library |
| `android/pytorch_android_torchvision/` | Vision utilities |

Last Verified: 2026-05-27
