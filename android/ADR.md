# ADR: android — Mobile Platform Support

**Status**: ACTIVE  
**Last Updated**: 2026-05-28

## Architectural Role

The android directory provides PyTorch support for Android mobile platforms. It includes:

- **JNI bindings**: Java/C++ interface for PyTorch operations
- **Gradle build configuration**: Android-specific build system
- **Android Runtime Library** (pytorch_android): Pre-built library for app developers
- **Android SDK artifacts**: JAR/AAR packages for inclusion in Android projects

android is classified as **Platform-Specific** (non-core) because it provides deployment support for mobile, not core functionality.

## Responsibilities

### What android Owns
- **JNI bindings**: Java Native Interface bridges for torch operations
- **Android Runtime Library**: pytorch_android library (JAR/AAR artifacts)
- **Gradle build scripts**: Android-specific build configuration
- **Module metadata**: Manifest files and dependency declarations

### What android Does Not Own
- **Core functionality**: Implemented in c10, aten, torch/
- **Kernel implementations**: Implemented in aten/src/ATen/native/
- **Mobile optimization**: Partially (some operators optimized for mobile; core responsibility of ATen)

## Dependencies

### Build-Time Dependencies
- **Android NDK**: Native development kit for C++ compilation
- **Gradle**: Android build system
- **Java SDK**: Java compilation

### Runtime Dependencies
- **Android Runtime (ART)**: Executes compiled Java/native code
- **libtorch.so**: PyTorch native library (dynamically linked)

## Trade-offs and Design Decisions

### 1. JNI Overhead
**Decision**: Accept JNI marshaling overhead for Java API usability  
**Rationale**:
- **API ergonomics**: Java developers expect Java API
- **Android integration**: Integrate with Android framework naturally

**Trade-off**: ~1-5% overhead from JNI calls

## Extension Boundaries

### Public Extension Points
1. **Custom JNI bindings**: Add new Java methods wrapping C++ operations
2. **Android-specific optimization**: Implement mobile-optimized kernels

## Notes and Caveats

1. **JNI overhead is measurable**: For low-latency inference, native C++ API is faster
2. **Binary size matters on mobile**: Android APK size is limited; bundling full PyTorch is expensive
3. **Android version compatibility**: Minimum API level affects available features
