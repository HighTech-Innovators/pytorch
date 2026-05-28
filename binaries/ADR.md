# Architecture Decision Record: Binaries (Tools & Utilities)

## Architectural Role

The `binaries` directory houses command-line utilities and diagnostic tools that enable offline model compilation, optimization, benchmarking, and inspection. These tools serve as integration points that demonstrate practical usage of PyTorch's core subsystems (torch::script, ATen, c10) and provide operational capabilities for model developers and framework maintainers.

The directory represents the "user-facing operations" layer—separate from core runtime but tightly coupled to torch.jit, torch.autograd, and ATen for model manipulation and performance analysis.

## Responsibilities

- **Model compilation and optimization**: `aot_model_compiler` compiles TorchScript models using the NNC (Neural Net Compiler) backend for ahead-of-time optimization. `optimize_for_mobile` prepares models for mobile deployment by removing unused operators and data structures.
- **Performance benchmarking**: Multiple benchmarking tools (`speed_benchmark_torch.cc`, `load_benchmark_torch.cc`, `record_function_benchmark.cc`) measure model inference speed, initialization overhead, and operator-level performance characteristics.
- **Debugging and inspection**: `dump_operator_names.cc` lists all registered operators in the dispatcher; `parallel_info.cc` reports threading and parallel backend configuration; `core_overhead_benchmark.cc` measures framework overhead independent of operator kernels.
- **Model comparison and verification**: `compare_models_torch.cc` validates that model outputs remain consistent across optimizations or compilation passes by comparing reference and compiled model outputs with configurable tolerances.
- **Mobile model loading**: `lite_interpreter_model_load.cc` tests model loading in a resource-constrained environment simulating mobile deployment.
- **Launch and initialization**: `at_launch_benchmark.cc` benchmarks startup overhead for different initialization configurations.

**What it does not do**: The binaries directory does not define new tensor operations, automatic differentiation primitives, or core language features. All binaries delegate actual tensor computation to ATen/c10 subsystems and coordinate through established APIs (torch::jit::load, torch::jit::freeze_module, etc.).

## Dependencies

### Inbound Dependencies
- **torch::script** (torch/csrc/jit/): Loads, freezes, and optimizes TorchScript models. Examples: `torch::jit::load()`, `torch::jit::freeze_module()`, backend preprocessors.
- **ATen** (aten/src/ATen/): Direct tensor operations for benchmarking and comparison. Headers: `ATen/ATen.h`, `ATen/core/jit_type.h`, `ATen/Parallel.h`, `ATen/record_function.h`.
- **c10** (c10/): Fundamental types, logging, and flag parsing. Headers: `c10/core/ScalarType.h`, `c10/util/Flags.h`, `c10/mobile/CPUCachingAllocator.h`.
- **caffe2** (caffe2/): Legacy timer and utility functions for benchmarking. Headers: `caffe2/core/timer.h`, `caffe2/utils/string_utils.h`, `caffe2/core/init.h`.

### Outbound Dependencies
- **Build system consumers**: The `caffe2_binary_target` CMake macro (in cmake/) registers each tool as an executable. The build system links binaries against torch, ATen, and c10 libraries.
- **Framework integration**: No other subsystems depend on binaries. They are standalone executables, not linked into the torch library.

## Trade-offs

### Design Decisions
1. **Separate command-line tools vs. Python APIs**: Binaries provide a lightweight, dependency-free way to compile and benchmark models without importing the full torch Python ecosystem. This enables use in resource-constrained environments (CI/CD, embedded systems, offline workflows). The trade-off is duplicated logic—binaries and Python APIs may diverge.
2. **Caffe2 legacy dependencies**: Several binaries (benchmarking tools) depend on caffe2::Timer rather than std::chrono. This reflects PyTorch's dual heritage (Caffe2 and PyTorch merged in 2018). The trade-off: legacy code complexity vs. avoiding breaking changes.
3. **Minimal error handling**: Most binaries use `CAFFE_ENFORCE` for validation and exit on error rather than graceful recovery. This is appropriate for CLI tools but differs from library code that must handle errors within a process.

### Alternatives Considered
- **Python-only utilities**: Would simplify maintenance but defeat the purpose of dependency-free offline tools.
- **Unified testing framework**: Could move benchmarking logic into a centralized perf testing framework, but binaries serve both internal testing (ci/) and user workflows.

## Extension Boundaries

New binaries can be added by:
1. Creating a `.cc` file in `binaries/` with a `main()` function.
2. Adding `caffe2_binary_target("<filename>")` to `CMakeLists.txt`.
3. Linking required libraries (`torch`, `ATen`, benchmark::benchmark`) in CMakeLists.txt if dependencies exceed the default set.

Example pattern observed in `aot_model_compiler.cc`:
- Define C10_DEFINE_* flags for command-line arguments.
- Parse flags with `c10::ParseCommandLineFlags()`.
- Validate inputs with `CAFFE_ENFORCE()`.
- Call torch::jit APIs to load and manipulate models.
- Output results to stdout or files.

Extension constraints:
- Binaries must remain self-contained executables (no library linking to them).
- New binaries should not introduce new architectural dependencies (e.g., adding a GPU backend via CUDA headers would require broader framework changes).

## Runtime Implications

### Lifecycle and Initialization
- **Startup**: Each binary performs one-time initialization (`caffe2::GlobalInit` or implicit c10 setup) on entry.
- **Model loading**: Binaries that work with TorchScript (`aot_model_compiler`, `compare_models_torch`, `optimize_for_mobile`) load serialized models from disk using `torch::jit::load()`, which deserializes the model graph and operator registrations.
- **Shutdown**: Tools complete their task and exit. No persistent state or cleanup logic required.

### Concurrency
- **Benchmarking tools** (`speed_benchmark_torch.cc`, `record_function_benchmark.cc`) explicitly manage threading via ATen's parallel backend (controlled by `OMP_NUM_THREADS` and `NNPACK_NUM_THREADS`).
- `parallel_info.cc` queries and reports the active threading configuration but does not control it.
- Single-threaded tools (`aot_model_compiler.cc`, `optimize_for_mobile.cc`) work sequentially without synchronization.

### Failure Behavior
- **Invalid input**: Tools validate required flags and exit with error code 1 and usage message.
- **Model loading failure**: Tools catch exceptions from `torch::jit::load()` and abort with logged error.
- **Numeric mismatches** (in `compare_models_torch.cc`): Outputs are compared with tolerance thresholds; mismatches are logged but do not halt—the tool reports the statistics and exits with code 0.

## Performance Implications

### Known Hotspots
1. **Model loading latency**: `torch::jit::load()` deserializes the entire model into memory. For large models (>1GB), this can take 10+ seconds. Tools like `lite_interpreter_model_load.cc` explicitly measure this overhead.
2. **Operator dispatch overhead**: `dump_operator_names.cc` iterates the full dispatcher registry, which can be slow for frameworks with thousands of registered operators.
3. **Benchmarking variance**: Micro-benchmarks (`speed_benchmark_torch.cc`, `record_function_benchmark.cc`) are sensitive to CPU caching and thermal throttling. Multiple runs and statistical analysis (mean, median, stddev) are standard practice.

### Allocation Patterns
- **Temporary model copies**: `compare_models_torch.cc` clones the model (`m.clone()`) to create a reference and optimized version for comparison. For large models, this doubles memory usage temporarily.
- **Tensor buffering**: Benchmarking tools allocate input tensors once and reuse them across iterations to isolate operator performance from allocation overhead.
- **String accumulation**: Some tools (e.g., `dump_operator_names.cc`) accumulate operator names in string vectors, which is negligible for typical registries (<10,000 operators).

### Synchronization Costs
- **Threading overhead**: Multi-threaded benchmarks incur OpenMP or thread pool synchronization costs. Tools measure framework overhead by running single-threaded variants and comparing.
- **RecordFunction profiling**: `record_function_benchmark.cc` measures the overhead of PyTorch's performance profiling system (RecordFunction callbacks) by instrumenting operator calls. Overhead is typically 5-20% per callback depending on CPU and memory pressure.

## Ownership Boundaries

### State Ownership
- **Models**: Binaries load models from disk (not created within the tool). `aot_model_compiler` and `optimize_for_mobile` own the in-memory model graph during processing but do not persist changes back to the original file—they write optimized versions to new files.
- **Tensors**: Benchmark tools allocate input tensors at startup and reuse them. The tensors are owned by the tool and released on shutdown.
- **Configuration**: Binaries own command-line flags parsed at startup. No global configuration state is modified; each binary is independent.

### Borrowed State
- **Dispatcher registry**: `dump_operator_names.cc` borrows read-only access to the ATen dispatcher's operator registry.
- **Parallel backend**: Benchmarking tools borrow the active parallel backend (OpenMP, pthreads) from ATen's environment but do not reconfigure it.
- **Allocator**: Tools use the default torch memory allocator (CPU or mobile) without customization.

## Evidence & Cross-References

### Source Files
- **Model compilation**: `aot_model_compiler.cc` (141 lines) references `torch/csrc/jit/mobile/nnc/aot_compiler.h` and demonstrates NNC backend integration.
- **Model optimization**: `optimize_for_mobile.cc` (107 lines) uses mobile-specific freezing and optimization APIs.
- **Benchmarking**: `speed_benchmark_torch.cc` (340 lines), `load_benchmark_torch.cc` (93 lines), `record_function_benchmark.cc` (127 lines) measure different performance aspects.
- **Comparison**: `compare_models_torch.cc` (326 lines) loads two models and compares outputs with configurable tolerance.
- **Inspection**: `dump_operator_names.cc` (85 lines), `parallel_info.cc` (41 lines) query framework state.

### Build Configuration
- **CMakeLists.txt** conditionally builds subsets for mobile builds (`INTERN_BUILD_MOBILE`, `BUILD_LITE_INTERPRETER`, `BUILD_TEST`).
- Uses `caffe2_binary_target()` helper, indicating integration with the Caffe2 legacy build system.
- Explicit target_link_libraries for `aot_model_compiler` (links `torch`).

### Book References
- **01-architecture-overview.md**: Identifies 1,690 main() functions in scripts/binaries as execution entrypoints during framework analysis.
