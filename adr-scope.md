# ADR Scope

| Directory | Source files present | Status | Reason (if EXCLUDED) |
|---|---|---|---|
| ./android | yes | PENDING |  |
| ./android/gradle | no | EXCLUDED | Empty or stub |
| ./android/gradle/wrapper | no | EXCLUDED | Empty or stub |
| ./android/libs | no | EXCLUDED | Empty or stub |
| ./android/libs/fbjni | no | EXCLUDED | Empty or stub |
| ./android/pytorch_android | yes | PENDING |  |
| ./android/pytorch_android/host | no | EXCLUDED | Empty or stub |
| ./android/pytorch_android/src | yes | PENDING |  |
| ./android/pytorch_android/src/androidTest | yes | PENDING |  |
| ./android/pytorch_android/src/androidTest/assets | no | EXCLUDED | Empty or stub |
| ./android/pytorch_android/src/androidTest/cpp | yes | PENDING |  |
| ./android/pytorch_android/src/androidTest/java | yes | PENDING |  |
| ./android/pytorch_android/src/androidTest/java/org | yes | PENDING |  |
| ./android/pytorch_android/src/androidTest/java/org/pytorch | yes | PENDING |  |
| ./android/pytorch_android/src/androidTest/java/org/pytorch/suite | yes | PENDING |  |
| ./android/pytorch_android/src/main | yes | PENDING |  |
| ./android/pytorch_android/src/main/cpp | yes | PENDING |  |
| ./android/pytorch_android/src/main/java | yes | PENDING |  |
| ./android/pytorch_android/src/main/java/org | yes | PENDING |  |
| ./android/pytorch_android/src/main/java/org/pytorch | yes | PENDING |  |
| ./android/pytorch_android/src/main/res | no | EXCLUDED | Empty or stub |
| ./android/pytorch_android/src/main/res/values | no | EXCLUDED | Empty or stub |
| ./android/pytorch_android_torchvision | yes | PENDING |  |
| ./android/pytorch_android_torchvision/src | yes | PENDING |  |
| ./android/pytorch_android_torchvision/src/androidTest | yes | PENDING |  |
| ./android/pytorch_android_torchvision/src/androidTest/java | yes | PENDING |  |
| ./android/pytorch_android_torchvision/src/androidTest/java/org | yes | PENDING |  |
| ./android/pytorch_android_torchvision/src/androidTest/java/org/pytorch | yes | PENDING |  |
| ./android/pytorch_android_torchvision/src/androidTest/java/org/pytorch/torchvision | yes | PENDING |  |
| ./android/pytorch_android_torchvision/src/androidTest/java/org/pytorch/torchvision/suite | yes | PENDING |  |
| ./android/pytorch_android_torchvision/src/main | yes | PENDING |  |
| ./android/pytorch_android_torchvision/src/main/cpp | yes | PENDING |  |
| ./android/pytorch_android_torchvision/src/main/java | yes | PENDING |  |
| ./android/pytorch_android_torchvision/src/main/java/org | yes | PENDING |  |
| ./android/pytorch_android_torchvision/src/main/java/org/pytorch | yes | PENDING |  |
| ./android/pytorch_android_torchvision/src/main/java/org/pytorch/torchvision | yes | PENDING |  |
| ./android/pytorch_android_torchvision/src/main/res | no | EXCLUDED | Empty or stub |
| ./android/pytorch_android_torchvision/src/main/res/values | no | EXCLUDED | Empty or stub |
| ./aten | yes | PENDING |  |
| ./aten/src | yes | PENDING |  |
| ./aten/src/ATen | yes | PENDING |  |
| ./aten/src/ATen/accelerator | yes | PENDING |  |
| ./aten/src/ATen/benchmarks | yes | PENDING |  |
| ./aten/src/ATen/core | yes | COVERED |  |
| ./aten/src/ATen/core/boxing | yes | PENDING |  |
| ./aten/src/ATen/core/boxing/impl | yes | PENDING |  |
| ./aten/src/ATen/core/dispatch | yes | COVERED |  |
| ./aten/src/ATen/core/op_registration | yes | PENDING |  |
| ./aten/src/ATen/cpu | yes | PENDING |  |
| ./aten/src/ATen/cpu/vec | yes | PENDING |  |
| ./aten/src/ATen/cpu/vec/sve | yes | PENDING |  |
| ./aten/src/ATen/cpu/vec/vec128 | yes | PENDING |  |
| ./aten/src/ATen/cpu/vec/vec256 | yes | PENDING |  |
| ./aten/src/ATen/cpu/vec/vec256/vsx | yes | PENDING |  |
| ./aten/src/ATen/cpu/vec/vec256/zarch | yes | PENDING |  |
| ./aten/src/ATen/cpu/vec/vec512 | yes | PENDING |  |
| ./aten/src/ATen/cuda | yes | COVERED |  |
| ./aten/src/ATen/cuda/detail | yes | PENDING |  |
| ./aten/src/ATen/cuda/nvrtc_stub | yes | PENDING |  |
| ./aten/src/ATen/cuda/tunable | yes | PENDING |  |
| ./aten/src/ATen/cudnn | yes | PENDING |  |
| ./aten/src/ATen/detail | yes | COVERED |  |
| ./aten/src/ATen/functorch | yes | COVERED |  |
| ./aten/src/ATen/hip | yes | PENDING |  |
| ./aten/src/ATen/hip/impl | yes | PENDING |  |
| ./aten/src/ATen/metal | yes | PENDING |  |
| ./aten/src/ATen/miopen | yes | PENDING |  |
| ./aten/src/ATen/mkl | yes | PENDING |  |
| ./aten/src/ATen/mps | yes | PENDING |  |
| ./aten/src/ATen/native | yes | COVERED |  |
| ./aten/src/ATen/native/ao_sparse | yes | PENDING |  |
| ./aten/src/ATen/native/ao_sparse/quantized | yes | PENDING |  |
| ./aten/src/ATen/native/ao_sparse/quantized/cpu | yes | PENDING |  |
| ./aten/src/ATen/native/cpu | yes | COVERED |  |
| ./aten/src/ATen/native/cuda | yes | COVERED |  |
| ./aten/src/ATen/native/cuda/cutlass_extensions | yes | PENDING |  |
| ./aten/src/ATen/native/cuda/cutlass_extensions/arch | yes | PENDING |  |
| ./aten/src/ATen/native/cuda/cutlass_extensions/epilogue | yes | PENDING |  |
| ./aten/src/ATen/native/cuda/cutlass_extensions/epilogue/thread | yes | PENDING |  |
| ./aten/src/ATen/native/cuda/cutlass_extensions/gemm | yes | PENDING |  |
| ./aten/src/ATen/native/cuda/cutlass_extensions/gemm/kernel | yes | PENDING |  |
| ./aten/src/ATen/native/cuda/cutlass_extensions/gemm/threadblock | yes | PENDING |  |
| ./aten/src/ATen/native/cuda/cutlass_extensions/gemm/warp | yes | PENDING |  |
| ./aten/src/ATen/native/cuda/linalg | yes | PENDING |  |
| ./aten/src/ATen/native/cudnn | yes | PENDING |  |
| ./aten/src/ATen/native/hip | yes | PENDING |  |
| ./aten/src/ATen/native/hip/bgemm_kernels | yes | PENDING |  |
| ./aten/src/ATen/native/kleidiai | yes | PENDING |  |
| ./aten/src/ATen/native/metal | yes | PENDING |  |
| ./aten/src/ATen/native/metal/mpscnn | yes | PENDING |  |
| ./aten/src/ATen/native/metal/mpscnn/tests | yes | PENDING |  |
| ./aten/src/ATen/native/metal/ops | yes | PENDING |  |
| ./aten/src/ATen/native/miopen | yes | PENDING |  |
| ./aten/src/ATen/native/mkl | yes | PENDING |  |
| ./aten/src/ATen/native/mkldnn | yes | PENDING |  |
| ./aten/src/ATen/native/mkldnn/xpu | yes | PENDING |  |
| ./aten/src/ATen/native/mkldnn/xpu/detail | yes | PENDING |  |
| ./aten/src/ATen/native/mps | yes | PENDING |  |
| ./aten/src/ATen/native/mps/kernels | yes | PENDING |  |
| ./aten/src/ATen/native/mps/operations | yes | PENDING |  |
| ./aten/src/ATen/native/nested | yes | PENDING |  |
| ./aten/src/ATen/native/nested/cuda | yes | PENDING |  |
| ./aten/src/ATen/native/quantized | yes | PENDING |  |
| ./aten/src/ATen/native/quantized/cpu | yes | PENDING |  |
| ./aten/src/ATen/native/quantized/cpu/kernels | yes | PENDING |  |
| ./aten/src/ATen/native/quantized/cpu/qnnpack | yes | PENDING |  |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/bench | yes | PENDING |  |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/cmake | no | EXCLUDED | Build/config only |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/deps | yes | PENDING |  |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/deps/clog | yes | PENDING |  |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/deps/clog/cmake | no | EXCLUDED | Build/config only |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/deps/clog/include | yes | PENDING |  |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/deps/clog/src | no | EXCLUDED | Empty or stub |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/deps/clog/test | yes | PENDING |  |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/include | yes | PENDING |  |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/scripts | no | EXCLUDED | Empty or stub |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/src | yes | PENDING |  |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/src/hgemm | no | EXCLUDED | Empty or stub |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/src/q8avgpool | no | EXCLUDED | Empty or stub |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/src/q8conv | no | EXCLUDED | Empty or stub |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/src/q8dwconv | no | EXCLUDED | Empty or stub |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/src/q8gavgpool | no | EXCLUDED | Empty or stub |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/src/q8gemm | no | EXCLUDED | Empty or stub |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/src/q8gemm_sparse | yes | PENDING |  |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/src/q8vadd | no | EXCLUDED | Empty or stub |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack | yes | PENDING |  |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/src/requantization | yes | PENDING |  |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/src/sconv | no | EXCLUDED | Empty or stub |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/src/sdwconv | no | EXCLUDED | Empty or stub |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/src/sgemm | no | EXCLUDED | Empty or stub |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/src/u8clamp | no | EXCLUDED | Empty or stub |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/src/u8lut32norm | no | EXCLUDED | Empty or stub |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/src/u8maxpool | no | EXCLUDED | Empty or stub |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/src/u8rmax | no | EXCLUDED | Empty or stub |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/src/x8lut | no | EXCLUDED | Empty or stub |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/src/x8zip | no | EXCLUDED | Empty or stub |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/test | yes | PENDING |  |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/wrappers | no | EXCLUDED | Empty or stub |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/wrappers/hgemm | no | EXCLUDED | Empty or stub |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/wrappers/q8avgpool | no | EXCLUDED | Empty or stub |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/wrappers/q8conv | no | EXCLUDED | Empty or stub |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/wrappers/q8dwconv | no | EXCLUDED | Empty or stub |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/wrappers/q8gavgpool | no | EXCLUDED | Empty or stub |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/wrappers/q8gemm | no | EXCLUDED | Empty or stub |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/wrappers/q8gemm_sparse | no | EXCLUDED | Empty or stub |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/wrappers/q8vadd | no | EXCLUDED | Empty or stub |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/wrappers/requantization | no | EXCLUDED | Empty or stub |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/wrappers/sgemm | no | EXCLUDED | Empty or stub |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/wrappers/u8clamp | no | EXCLUDED | Empty or stub |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/wrappers/u8lut32norm | no | EXCLUDED | Empty or stub |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/wrappers/u8maxpool | no | EXCLUDED | Empty or stub |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/wrappers/u8rmax | no | EXCLUDED | Empty or stub |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/wrappers/x8lut | no | EXCLUDED | Empty or stub |
| ./aten/src/ATen/native/quantized/cpu/qnnpack/wrappers/x8zip | no | EXCLUDED | Empty or stub |
| ./aten/src/ATen/native/quantized/cuda | yes | PENDING |  |
| ./aten/src/ATen/native/quantized/cudnn | yes | PENDING |  |
| ./aten/src/ATen/native/sparse | yes | PENDING |  |
| ./aten/src/ATen/native/sparse/cuda | yes | PENDING |  |
| ./aten/src/ATen/native/sparse/eigen | yes | PENDING |  |
| ./aten/src/ATen/native/sparse/mps | no | EXCLUDED | Empty or stub |
| ./aten/src/ATen/native/sparse/mps/kernels | no | EXCLUDED | Empty or stub |
| ./aten/src/ATen/native/transformers | yes | PENDING |  |
| ./aten/src/ATen/native/transformers/cuda | yes | PENDING |  |
| ./aten/src/ATen/native/transformers/cuda/flash_attn | yes | PENDING |  |
| ./aten/src/ATen/native/transformers/cuda/mem_eff_attention | yes | PENDING |  |
| ./aten/src/ATen/native/transformers/cuda/mem_eff_attention/epilogue | yes | PENDING |  |
| ./aten/src/ATen/native/transformers/cuda/mem_eff_attention/gemm | yes | PENDING |  |
| ./aten/src/ATen/native/transformers/cuda/mem_eff_attention/iterators | yes | PENDING |  |
| ./aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernels | yes | PENDING |  |
| ./aten/src/ATen/native/transformers/cuda/mem_eff_attention/transform | yes | PENDING |  |
| ./aten/src/ATen/native/transformers/hip | yes | PENDING |  |
| ./aten/src/ATen/native/transformers/hip/flash_attn | yes | PENDING |  |
| ./aten/src/ATen/native/transformers/hip/flash_attn/aot | no | EXCLUDED | Empty or stub |
| ./aten/src/ATen/native/transformers/hip/flash_attn/ck | yes | PENDING |  |
| ./aten/src/ATen/native/transformers/hip/flash_attn/ck/fav_v3 | yes | PENDING |  |
| ./aten/src/ATen/native/transformers/xpu | yes | PENDING |  |
| ./aten/src/ATen/native/ufunc | yes | PENDING |  |
| ./aten/src/ATen/native/utils | yes | PENDING |  |
| ./aten/src/ATen/native/vulkan | yes | PENDING |  |
| ./aten/src/ATen/native/vulkan/api | yes | PENDING |  |
| ./aten/src/ATen/native/vulkan/glsl | yes | PENDING |  |
| ./aten/src/ATen/native/vulkan/impl | yes | PENDING |  |
| ./aten/src/ATen/native/vulkan/ops | yes | PENDING |  |
| ./aten/src/ATen/native/xnnpack | yes | PENDING |  |
| ./aten/src/ATen/nnapi | yes | PENDING |  |
| ./aten/src/ATen/ops | yes | PENDING |  |
| ./aten/src/ATen/quantized | yes | PENDING |  |
| ./aten/src/ATen/templates | yes | PENDING |  |
| ./aten/src/ATen/test | yes | PENDING |  |
| ./aten/src/ATen/test/test_install | yes | PENDING |  |
| ./aten/src/ATen/vulkan | yes | PENDING |  |
| ./aten/src/ATen/xpu | yes | PENDING |  |
| ./aten/src/ATen/xpu/detail | yes | PENDING |  |
| ./aten/src/ATen/xpu/level_zero_stub | yes | PENDING |  |
| ./aten/src/THC | no | EXCLUDED | Empty or stub |
| ./aten/tools | no | EXCLUDED | Empty or stub |
| ./benchmarks | yes | PENDING |  |
| ./benchmarks/data | yes | PENDING |  |
| ./benchmarks/diffusion | yes | PENDING |  |
| ./benchmarks/distributed | yes | PENDING |  |
| ./benchmarks/distributed/ddp | yes | PENDING |  |
| ./benchmarks/dynamo | yes | PENDING |  |
| ./benchmarks/dynamo/ci_expected_accuracy | yes | PENDING |  |
| ./benchmarks/dynamo/ci_expected_accuracy/rocm | no | EXCLUDED | Empty or stub |
| ./benchmarks/dynamo/genai_layers | yes | PENDING |  |
| ./benchmarks/dynamo/microbenchmarks | yes | PENDING |  |
| ./benchmarks/dynamo/microbenchmarks/operator_inp_logs | no | EXCLUDED | Empty or stub |
| ./benchmarks/dynamo/microbenchmarks/operator_inp_logs/hf_train | no | EXCLUDED | Empty or stub |
| ./benchmarks/dynamo/microbenchmarks/operator_inp_logs/timm_train | no | EXCLUDED | Empty or stub |
| ./benchmarks/dynamo/microbenchmarks/operator_inp_logs/torchbench_train | no | EXCLUDED | Empty or stub |
| ./benchmarks/dynamo/pr_time_benchmarks | yes | PENDING |  |
| ./benchmarks/dynamo/pr_time_benchmarks/benchmarks | yes | PENDING |  |
| ./benchmarks/dynamo/pr_time_benchmarks/test_check_result | no | EXCLUDED | Empty or stub |
| ./benchmarks/fastrnns | yes | PENDING |  |
| ./benchmarks/framework_overhead_benchmark | yes | PENDING |  |
| ./benchmarks/functional_autograd_benchmark | yes | PENDING |  |
| ./benchmarks/fuser | yes | PENDING |  |
| ./benchmarks/gpt_fast | yes | PENDING |  |
| ./benchmarks/inductor_backends | yes | PENDING |  |
| ./benchmarks/inference | yes | PENDING |  |
| ./benchmarks/inference/results | no | EXCLUDED | Empty or stub |
| ./benchmarks/inference/src | no | EXCLUDED | Empty or stub |
| ./benchmarks/instruction_counts | yes | PENDING |  |
| ./benchmarks/instruction_counts/applications | yes | PENDING |  |
| ./benchmarks/instruction_counts/core | yes | PENDING |  |
| ./benchmarks/instruction_counts/definitions | yes | PENDING |  |
| ./benchmarks/instruction_counts/execution | yes | PENDING |  |
| ./benchmarks/instruction_counts/worker | yes | PENDING |  |
| ./benchmarks/nested | yes | PENDING |  |
| ./benchmarks/operator_benchmark | yes | PENDING |  |
| ./benchmarks/operator_benchmark/common | yes | PENDING |  |
| ./benchmarks/operator_benchmark/common/tests | yes | PENDING |  |
| ./benchmarks/operator_benchmark/pt | yes | PENDING |  |
| ./benchmarks/operator_benchmark/pt_extension | yes | PENDING |  |
| ./benchmarks/overrides_benchmark | yes | PENDING |  |
| ./benchmarks/profiler_benchmark | yes | PENDING |  |
| ./benchmarks/record_function_benchmark | yes | PENDING |  |
| ./benchmarks/serialization | yes | PENDING |  |
| ./benchmarks/sparse | yes | PENDING |  |
| ./benchmarks/sparse/dlmc | yes | PENDING |  |
| ./benchmarks/static_runtime | yes | PENDING |  |
| ./benchmarks/tensorexpr | yes | PENDING |  |
| ./benchmarks/transformer | yes | PENDING |  |
| ./benchmarks/transformer/configs | no | EXCLUDED | Empty or stub |
| ./binaries | yes | PENDING |  |
| ./c10 | yes | PENDING |  |
| ./c10/benchmark | yes | PENDING |  |
| ./c10/core | yes | COVERED |  |
| ./c10/core/impl | yes | PENDING |  |
| ./c10/cuda | yes | COVERED |  |
| ./c10/cuda/impl | yes | PENDING |  |
| ./c10/cuda/test | yes | PENDING |  |
| ./c10/cuda/test/impl | yes | PENDING |  |
| ./c10/hip | no | EXCLUDED | Empty or stub |
| ./c10/macros | yes | PENDING |  |
| ./c10/metal | yes | PENDING |  |
| ./c10/mobile | yes | PENDING |  |
| ./c10/test | yes | PENDING |  |
| ./c10/test/core | yes | PENDING |  |
| ./c10/test/core/impl | yes | PENDING |  |
| ./c10/test/util | yes | PENDING |  |
| ./c10/util | yes | COVERED |  |
| ./c10/xpu | yes | PENDING |  |
| ./c10/xpu/impl | yes | PENDING |  |
| ./c10/xpu/test | yes | PENDING |  |
| ./c10/xpu/test/impl | yes | PENDING |  |
| ./caffe2 | yes | COVERED |  |
| ./caffe2/core | yes | PENDING |  |
| ./caffe2/perfkernels | yes | PENDING |  |
| ./caffe2/serialize | yes | PENDING |  |
| ./caffe2/utils | yes | PENDING |  |
| ./caffe2/utils/threadpool | yes | PENDING |  |
| ./cmake | no | EXCLUDED | Build/config only |
| ./cmake/External | no | EXCLUDED | Build/config only |
| ./cmake/Modules | no | EXCLUDED | Build/config only |
| ./cmake/Modules_CUDA_fix | no | EXCLUDED | Build/config only |
| ./cmake/Modules_CUDA_fix/upstream | no | EXCLUDED | Build/config only |
| ./cmake/Modules_CUDA_fix/upstream/FindCUDA | no | EXCLUDED | Build/config only |
| ./cmake/public | no | EXCLUDED | Build/config only |
| ./docs | yes | PENDING |  |
| ./docs/cpp | yes | PENDING |  |
| ./docs/cpp/source | yes | PENDING |  |
| ./docs/cpp/source/api | no | EXCLUDED | Empty or stub |
| ./docs/cpp/source/api/aten | no | EXCLUDED | Empty or stub |
| ./docs/cpp/source/api/autograd | no | EXCLUDED | Empty or stub |
| ./docs/cpp/source/api/c10 | no | EXCLUDED | Empty or stub |
| ./docs/cpp/source/api/cuda | no | EXCLUDED | Empty or stub |
| ./docs/cpp/source/api/data | no | EXCLUDED | Empty or stub |
| ./docs/cpp/source/api/library | no | EXCLUDED | Empty or stub |
| ./docs/cpp/source/api/nn | no | EXCLUDED | Empty or stub |
| ./docs/cpp/source/api/optim | no | EXCLUDED | Empty or stub |
| ./docs/cpp/source/api/serialize | no | EXCLUDED | Empty or stub |
| ./docs/cpp/source/api/stable | no | EXCLUDED | Empty or stub |
| ./docs/cpp/source/api/xpu | no | EXCLUDED | Empty or stub |
| ./docs/source | yes | PENDING |  |
| ./docs/source/_static | no | EXCLUDED | Empty or stub |
| ./docs/source/_static/css | no | EXCLUDED | Empty or stub |
| ./docs/source/_static/img | no | EXCLUDED | Empty or stub |
| ./docs/source/_static/img/aoti_debug_printer | no | EXCLUDED | Empty or stub |
| ./docs/source/_static/img/aoti_debugging_guide | no | EXCLUDED | Empty or stub |
| ./docs/source/_static/img/distributed_autograd | no | EXCLUDED | Empty or stub |
| ./docs/source/_static/img/dynamic_shapes | no | EXCLUDED | Empty or stub |
| ./docs/source/_static/img/dynamo | no | EXCLUDED | Empty or stub |
| ./docs/source/_static/img/export | no | EXCLUDED | Empty or stub |
| ./docs/source/_static/img/fine_grained_apis | no | EXCLUDED | Empty or stub |
| ./docs/source/_static/img/inductor_profiling | no | EXCLUDED | Empty or stub |
| ./docs/source/_static/img/inductor_provenance | no | EXCLUDED | Empty or stub |
| ./docs/source/_static/img/masked | no | EXCLUDED | Empty or stub |
| ./docs/source/_static/img/nested | no | EXCLUDED | Empty or stub |
| ./docs/source/_static/img/nn | no | EXCLUDED | Empty or stub |
| ./docs/source/_static/img/onnx | no | EXCLUDED | Empty or stub |
| ./docs/source/_static/img/pipeline_parallelism | no | EXCLUDED | Empty or stub |
| ./docs/source/_static/img/profiling_torch_compile | no | EXCLUDED | Empty or stub |
| ./docs/source/_static/img/tensorboard | no | EXCLUDED | Empty or stub |
| ./docs/source/_static/img/torch_cuda_memory | no | EXCLUDED | Empty or stub |
| ./docs/source/_templates | no | EXCLUDED | Empty or stub |
| ./docs/source/_templates/autosummary | no | EXCLUDED | Empty or stub |
| ./docs/source/accelerator | no | EXCLUDED | Empty or stub |
| ./docs/source/community | no | EXCLUDED | Empty or stub |
| ./docs/source/elastic | no | EXCLUDED | Empty or stub |
| ./docs/source/higher_order_ops | no | EXCLUDED | Empty or stub |
| ./docs/source/notes | no | EXCLUDED | Empty or stub |
| ./docs/source/rpc | no | EXCLUDED | Empty or stub |
| ./docs/source/scripts | yes | PENDING |  |
| ./docs/source/scripts/exportdb | yes | PENDING |  |
| ./docs/source/user_guide | yes | PENDING |  |
| ./docs/source/user_guide/torch_compiler | yes | PENDING |  |
| ./docs/source/user_guide/torch_compiler/compile | yes | PENDING |  |
| ./docs/source/user_guide/torch_compiler/compile/_static | no | EXCLUDED | Empty or stub |
| ./docs/source/user_guide/torch_compiler/export | no | EXCLUDED | Empty or stub |
| ./functorch | yes | PENDING |  |
| ./functorch/_src | yes | PENDING |  |
| ./functorch/_src/aot_autograd | yes | PENDING |  |
| ./functorch/_src/eager_transforms | yes | PENDING |  |
| ./functorch/_src/make_functional | yes | PENDING |  |
| ./functorch/_src/vmap | yes | PENDING |  |
| ./functorch/benchmarks | yes | PENDING |  |
| ./functorch/compile | yes | PENDING |  |
| ./functorch/dim | yes | PENDING |  |
| ./functorch/docs | yes | PENDING |  |
| ./functorch/docs/source | yes | PENDING |  |
| ./functorch/docs/source/_static | no | EXCLUDED | Empty or stub |
| ./functorch/docs/source/_static/css | no | EXCLUDED | Empty or stub |
| ./functorch/docs/source/_templates | no | EXCLUDED | Empty or stub |
| ./functorch/docs/source/_templates/autosummary | no | EXCLUDED | Empty or stub |
| ./functorch/docs/source/tutorials | yes | PENDING |  |
| ./functorch/docs/source/tutorials/_src | yes | PENDING |  |
| ./functorch/einops | yes | PENDING |  |
| ./functorch/examples | yes | PENDING |  |
| ./functorch/examples/compilation | yes | PENDING |  |
| ./functorch/examples/dp_cifar10 | yes | PENDING |  |
| ./functorch/examples/ensembling | yes | PENDING |  |
| ./functorch/examples/lennard_jones | yes | PENDING |  |
| ./functorch/examples/maml_omniglot | yes | PENDING |  |
| ./functorch/examples/maml_omniglot/support | yes | PENDING |  |
| ./functorch/examples/maml_regression | yes | PENDING |  |
| ./functorch/experimental | yes | PENDING |  |
| ./functorch/op_analysis | yes | PENDING |  |
| ./mypy_plugins | yes | PENDING |  |
| ./scripts | yes | PENDING |  |
| ./scripts/analysis | yes | PENDING |  |
| ./scripts/compile_tests | yes | PENDING |  |
| ./scripts/export | yes | PENDING |  |
| ./scripts/jit | yes | PENDING |  |
| ./scripts/onnx | no | EXCLUDED | Empty or stub |
| ./scripts/release | no | EXCLUDED | Empty or stub |
| ./scripts/release_notes | yes | PENDING |  |
| ./test | yes | PENDING |  |
| ./test/ao | yes | PENDING |  |
| ./test/ao/sparsity | yes | PENDING |  |
| ./test/autograd | yes | PENDING |  |
| ./test/backends | yes | PENDING |  |
| ./test/backends/xeon | yes | PENDING |  |
| ./test/benchmark_utils | yes | PENDING |  |
| ./test/compiled_autograd_skips | no | EXCLUDED | Empty or stub |
| ./test/complex_tensor | yes | PENDING |  |
| ./test/cpp | yes | PENDING |  |
| ./test/cpp/aoti_abi_check | yes | PENDING |  |
| ./test/cpp/aoti_eager | yes | PENDING |  |
| ./test/cpp/aoti_inference | yes | PENDING |  |
| ./test/cpp/api | yes | PENDING |  |
| ./test/cpp/c10d | yes | PENDING |  |
| ./test/cpp/c10d/example | yes | PENDING |  |
| ./test/cpp/common | yes | PENDING |  |
| ./test/cpp/dist_autograd | yes | PENDING |  |
| ./test/cpp/jit | yes | PENDING |  |
| ./test/cpp/jit/upgrader_models | no | EXCLUDED | Empty or stub |
| ./test/cpp/lazy | yes | PENDING |  |
| ./test/cpp/lite_interpreter_runtime | yes | PENDING |  |
| ./test/cpp/monitor | yes | PENDING |  |
| ./test/cpp/nativert | yes | PENDING |  |
| ./test/cpp/profiler | yes | PENDING |  |
| ./test/cpp/rpc | yes | PENDING |  |
| ./test/cpp/shim | yes | PENDING |  |
| ./test/cpp_api_parity | yes | PENDING |  |
| ./test/cpp_extensions | yes | PENDING |  |
| ./test/cpp_extensions/libtorch_agn_2_10_extension | yes | PENDING |  |
| ./test/cpp_extensions/libtorch_agn_2_10_extension/csrc | yes | PENDING |  |
| ./test/cpp_extensions/libtorch_agn_2_10_extension/libtorch_agn_2_10 | yes | PENDING |  |
| ./test/cpp_extensions/libtorch_agn_2_11_extension | yes | PENDING |  |
| ./test/cpp_extensions/libtorch_agn_2_11_extension/csrc | yes | PENDING |  |
| ./test/cpp_extensions/libtorch_agn_2_11_extension/libtorch_agn_2_11 | yes | PENDING |  |
| ./test/cpp_extensions/libtorch_agn_2_12_extension | yes | PENDING |  |
| ./test/cpp_extensions/libtorch_agn_2_12_extension/csrc | yes | PENDING |  |
| ./test/cpp_extensions/libtorch_agn_2_12_extension/libtorch_agn_2_12 | yes | PENDING |  |
| ./test/cpp_extensions/libtorch_agn_2_13_extension | yes | PENDING |  |
| ./test/cpp_extensions/libtorch_agn_2_13_extension/csrc | yes | PENDING |  |
| ./test/cpp_extensions/libtorch_agn_2_13_extension/libtorch_agn_2_13 | yes | PENDING |  |
| ./test/cpp_extensions/libtorch_agn_2_9_extension | yes | PENDING |  |
| ./test/cpp_extensions/libtorch_agn_2_9_extension/csrc | yes | PENDING |  |
| ./test/cpp_extensions/libtorch_agn_2_9_extension/libtorch_agn_2_9 | yes | PENDING |  |
| ./test/cpp_extensions/no_python_abi_suffix_test | yes | PENDING |  |
| ./test/cpp_extensions/open_registration_extension | yes | PENDING |  |
| ./test/cpp_extensions/open_registration_extension/torch_openreg | yes | PENDING |  |
| ./test/cpp_extensions/open_registration_extension/torch_openreg/cmake | no | EXCLUDED | Build/config only |
| ./test/cpp_extensions/open_registration_extension/torch_openreg/csrc | yes | PENDING |  |
| ./test/cpp_extensions/open_registration_extension/torch_openreg/csrc/amp | yes | PENDING |  |
| ./test/cpp_extensions/open_registration_extension/torch_openreg/csrc/aten | yes | PENDING |  |
| ./test/cpp_extensions/open_registration_extension/torch_openreg/csrc/aten/native | yes | PENDING |  |
| ./test/cpp_extensions/open_registration_extension/torch_openreg/csrc/distributed | yes | PENDING |  |
| ./test/cpp_extensions/open_registration_extension/torch_openreg/csrc/distributed/c10d | yes | PENDING |  |
| ./test/cpp_extensions/open_registration_extension/torch_openreg/csrc/profiler | yes | PENDING |  |
| ./test/cpp_extensions/open_registration_extension/torch_openreg/csrc/profiler/stubs | yes | PENDING |  |
| ./test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime | yes | PENDING |  |
| ./test/cpp_extensions/open_registration_extension/torch_openreg/include | yes | PENDING |  |
| ./test/cpp_extensions/open_registration_extension/torch_openreg/tests | yes | PENDING |  |
| ./test/cpp_extensions/open_registration_extension/torch_openreg/third_party | yes | EXCLUDED | Vendored/third-party |
| ./test/cpp_extensions/open_registration_extension/torch_openreg/third_party/openreg | yes | EXCLUDED | Vendored/third-party |
| ./test/cpp_extensions/open_registration_extension/torch_openreg/third_party/openreg/cmake | no | EXCLUDED | Build/config only |
| ./test/cpp_extensions/open_registration_extension/torch_openreg/third_party/openreg/csrc | yes | EXCLUDED | Vendored/third-party |
| ./test/cpp_extensions/open_registration_extension/torch_openreg/third_party/openreg/example | yes | EXCLUDED | Vendored/third-party |
| ./test/cpp_extensions/open_registration_extension/torch_openreg/third_party/openreg/include | yes | EXCLUDED | Vendored/third-party |
| ./test/cpp_extensions/open_registration_extension/torch_openreg/third_party/openreg/tests | yes | EXCLUDED | Vendored/third-party |
| ./test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg | yes | PENDING |  |
| ./test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/csrc | yes | PENDING |  |
| ./test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/csrc/distributed | yes | PENDING |  |
| ./test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/openreg | yes | PENDING |  |
| ./test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/openreg/amp | yes | PENDING |  |
| ./test/cpp_extensions/python_agnostic_extension | yes | PENDING |  |
| ./test/cpp_extensions/python_agnostic_extension/python_agnostic | yes | PENDING |  |
| ./test/cpp_extensions/python_agnostic_extension/python_agnostic/csrc | yes | PENDING |  |
| ./test/cpp_extensions/python_agnostic_extension/test | yes | PENDING |  |
| ./test/cpp_extensions/self_compiler_include_dirs_test | yes | PENDING |  |
| ./test/cpp_extensions/torch_test_cpp_extension | yes | PENDING |  |
| ./test/cpython | yes | PENDING |  |
| ./test/cpython/v3_13 | yes | PENDING |  |
| ./test/cpython/v3_13/data | no | EXCLUDED | Test data only |
| ./test/cpython/v3_13/mathdata | no | EXCLUDED | Test data only |
| ./test/cpython/v3_13/test_unittest | yes | PENDING |  |
| ./test/cpython/v3_13/typinganndata | yes | EXCLUDED | Test data only |
| ./test/custom_backend | yes | PENDING |  |
| ./test/custom_operator | yes | PENDING |  |
| ./test/distributed | yes | PENDING |  |
| ./test/distributed/_composable | yes | PENDING |  |
| ./test/distributed/_composable/fsdp | yes | PENDING |  |
| ./test/distributed/_composable/test_composability | yes | PENDING |  |
| ./test/distributed/_pycute | yes | PENDING |  |
| ./test/distributed/_shard | yes | PENDING |  |
| ./test/distributed/_shard/sharded_optim | yes | PENDING |  |
| ./test/distributed/_shard/sharded_tensor | yes | PENDING |  |
| ./test/distributed/_shard/sharded_tensor/ops | yes | PENDING |  |
| ./test/distributed/_shard/sharding_plan | yes | PENDING |  |
| ./test/distributed/_shard/sharding_spec | yes | PENDING |  |
| ./test/distributed/_tools | yes | PENDING |  |
| ./test/distributed/algorithms | yes | PENDING |  |
| ./test/distributed/algorithms/ddp_comm_hooks | yes | PENDING |  |
| ./test/distributed/algorithms/quantization | yes | PENDING |  |
| ./test/distributed/bin | yes | PENDING |  |
| ./test/distributed/checkpoint | yes | PENDING |  |
| ./test/distributed/checkpoint/_experimental | yes | PENDING |  |
| ./test/distributed/checkpoint/e2e | yes | PENDING |  |
| ./test/distributed/checkpoint/fsdp | yes | PENDING |  |
| ./test/distributed/elastic | yes | PENDING |  |
| ./test/distributed/elastic/agent | yes | PENDING |  |
| ./test/distributed/elastic/agent/server | yes | PENDING |  |
| ./test/distributed/elastic/agent/server/test | yes | PENDING |  |
| ./test/distributed/elastic/events | yes | PENDING |  |
| ./test/distributed/elastic/metrics | yes | PENDING |  |
| ./test/distributed/elastic/multiprocessing | yes | PENDING |  |
| ./test/distributed/elastic/multiprocessing/bin | yes | PENDING |  |
| ./test/distributed/elastic/multiprocessing/errors | yes | PENDING |  |
| ./test/distributed/elastic/rendezvous | yes | PENDING |  |
| ./test/distributed/elastic/rendezvous/out_of_tree_test_package | yes | PENDING |  |
| ./test/distributed/elastic/rendezvous/out_of_tree_test_package/src | yes | PENDING |  |
| ./test/distributed/elastic/rendezvous/out_of_tree_test_package/src/testbackend | yes | PENDING |  |
| ./test/distributed/elastic/timer | yes | PENDING |  |
| ./test/distributed/elastic/utils | yes | PENDING |  |
| ./test/distributed/elastic/utils/data | yes | EXCLUDED | Test data only |
| ./test/distributed/flight_recorder | yes | PENDING |  |
| ./test/distributed/fsdp | yes | PENDING |  |
| ./test/distributed/launcher | yes | PENDING |  |
| ./test/distributed/launcher/bin | yes | PENDING |  |
| ./test/distributed/local_tensor_tutorial_examples | yes | PENDING |  |
| ./test/distributed/nn | yes | PENDING |  |
| ./test/distributed/nn/jit | yes | PENDING |  |
| ./test/distributed/optim | yes | PENDING |  |
| ./test/distributed/pipelining | yes | PENDING |  |
| ./test/distributed/pipelining/artifacts | no | EXCLUDED | Empty or stub |
| ./test/distributed/rpc | yes | PENDING |  |
| ./test/distributed/rpc/cuda | yes | PENDING |  |
| ./test/distributed/tensor | yes | PENDING |  |
| ./test/distributed/tensor/debug | yes | PENDING |  |
| ./test/distributed/tensor/experimental | yes | PENDING |  |
| ./test/distributed/tensor/parallel | yes | PENDING |  |
| ./test/distributions | yes | PENDING |  |
| ./test/dynamo | yes | PENDING |  |
| ./test/dynamo/mock_modules | yes | PENDING |  |
| ./test/dynamo_expected_failures | no | EXCLUDED | Empty or stub |
| ./test/dynamo_skips | no | EXCLUDED | Empty or stub |
| ./test/error_messages | yes | PENDING |  |
| ./test/expect | yes | PENDING |  |
| ./test/export | yes | PENDING |  |
| ./test/forward_backward_compatibility | yes | PENDING |  |
| ./test/functorch | yes | PENDING |  |
| ./test/functorch/dim | yes | PENDING |  |
| ./test/fx | yes | PENDING |  |
| ./test/higher_order_ops | yes | PENDING |  |
| ./test/inductor | yes | PENDING |  |
| ./test/inductor/cpp | yes | PENDING |  |
| ./test/inductor/extension_backends | yes | PENDING |  |
| ./test/inductor/extension_backends/cpp | yes | PENDING |  |
| ./test/inductor/extension_backends/triton | yes | PENDING |  |
| ./test/inductor/pallas_expected_failures | no | EXCLUDED | Empty or stub |
| ./test/inductor/pallas_skip_tests | no | EXCLUDED | Empty or stub |
| ./test/inductor_expected_failures | no | EXCLUDED | Empty or stub |
| ./test/inductor_skips | no | EXCLUDED | Empty or stub |
| ./test/jit | yes | PENDING |  |
| ./test/jit/_imported_class_test | yes | PENDING |  |
| ./test/jit/_imported_class_test/very | yes | PENDING |  |
| ./test/jit/_imported_class_test/very/very | yes | PENDING |  |
| ./test/jit/fixtures | no | EXCLUDED | Test data only |
| ./test/jit/fixtures_srcs | yes | EXCLUDED | Test data only |
| ./test/jit/xnnpack | yes | PENDING |  |
| ./test/jit_hooks | yes | PENDING |  |
| ./test/lazy | yes | PENDING |  |
| ./test/metal | no | EXCLUDED | Empty or stub |
| ./test/mobile | yes | PENDING |  |
| ./test/mobile/custom_build | yes | PENDING |  |
| ./test/mobile/lightweight_dispatch | yes | PENDING |  |
| ./test/mobile/model_test | yes | PENDING |  |
| ./test/mobile/nnc | yes | PENDING |  |
| ./test/nn | yes | PENDING |  |
| ./test/nn/attention | yes | PENDING |  |
| ./test/onnx | yes | PENDING |  |
| ./test/onnx/assets | no | EXCLUDED | Empty or stub |
| ./test/onnx/exporter | yes | PENDING |  |
| ./test/onnx/internal | yes | PENDING |  |
| ./test/onnx/model_defs | yes | PENDING |  |
| ./test/onnx/ops | yes | PENDING |  |
| ./test/onnx/torchlib | yes | PENDING |  |
| ./test/optim | yes | PENDING |  |
| ./test/package | yes | PENDING |  |
| ./test/package/package_a | yes | PENDING |  |
| ./test/package/package_b | yes | PENDING |  |
| ./test/package/package_b/subpackage_0 | yes | PENDING |  |
| ./test/package/package_b/subpackage_0/subsubpackage_0 | yes | PENDING |  |
| ./test/package/package_bc | no | EXCLUDED | Empty or stub |
| ./test/package/package_c | yes | PENDING |  |
| ./test/package/package_d | yes | PENDING |  |
| ./test/package/package_d/subpackage_0 | yes | PENDING |  |
| ./test/package/package_d/subpackage_0/subsubpackage_0 | yes | PENDING |  |
| ./test/package/package_e | no | EXCLUDED | Empty or stub |
| ./test/package/test_trace_dep | yes | PENDING |  |
| ./test/profiler | yes | PENDING |  |
| ./test/python_native | yes | PENDING |  |
| ./test/quantization | yes | PENDING |  |
| ./test/quantization/ao_migration | yes | PENDING |  |
| ./test/quantization/bc | yes | PENDING |  |
| ./test/quantization/core | yes | PENDING |  |
| ./test/quantization/core/experimental | yes | PENDING |  |
| ./test/quantization/eager | yes | PENDING |  |
| ./test/quantization/fx | yes | PENDING |  |
| ./test/quantization/jit | yes | PENDING |  |
| ./test/quantization/serialized | no | EXCLUDED | Empty or stub |
| ./test/scripts | yes | PENDING |  |
| ./test/strobelight | yes | PENDING |  |
| ./test/strobelight/examples | yes | PENDING |  |
| ./test/test_img | no | EXCLUDED | Empty or stub |
| ./test/torch_np | yes | PENDING |  |
| ./test/torch_np/numpy_tests | yes | PENDING |  |
| ./test/torch_np/numpy_tests/core | yes | PENDING |  |
| ./test/torch_np/numpy_tests/fft | yes | PENDING |  |
| ./test/torch_np/numpy_tests/lib | yes | PENDING |  |
| ./test/torch_np/numpy_tests/linalg | yes | PENDING |  |
| ./test/typing | yes | PENDING |  |
| ./test/typing/fail | yes | PENDING |  |
| ./test/typing/pass | yes | PENDING |  |
| ./test/typing/reveal | yes | PENDING |  |
| ./test/xpu | yes | PENDING |  |
| ./third_party | yes | EXCLUDED | Vendored/third-party |
| ./third_party/FP16 | no | EXCLUDED | Vendored/third-party |
| ./third_party/FXdiv | no | EXCLUDED | Vendored/third-party |
| ./third_party/NNPACK | no | EXCLUDED | Vendored/third-party |
| ./third_party/NVTX | no | EXCLUDED | Vendored/third-party |
| ./third_party/VulkanMemoryAllocator | no | EXCLUDED | Vendored/third-party |
| ./third_party/XNNPACK | no | EXCLUDED | Vendored/third-party |
| ./third_party/aiter | no | EXCLUDED | Vendored/third-party |
| ./third_party/benchmark | no | EXCLUDED | Vendored/third-party |
| ./third_party/composable_kernel | no | EXCLUDED | Vendored/third-party |
| ./third_party/concurrentqueue | yes | EXCLUDED | Vendored/third-party |
| ./third_party/concurrentqueue/moodycamel | yes | EXCLUDED | Vendored/third-party |
| ./third_party/cpp-httplib | no | EXCLUDED | Vendored/third-party |
| ./third_party/cpuinfo | no | EXCLUDED | Vendored/third-party |
| ./third_party/cudnn_frontend | no | EXCLUDED | Vendored/third-party |
| ./third_party/cutlass | no | EXCLUDED | Vendored/third-party |
| ./third_party/fbgemm | no | EXCLUDED | Vendored/third-party |
| ./third_party/flash-attention | no | EXCLUDED | Vendored/third-party |
| ./third_party/flatbuffers | no | EXCLUDED | Vendored/third-party |
| ./third_party/fmt | no | EXCLUDED | Vendored/third-party |
| ./third_party/gemmlowp | no | EXCLUDED | Vendored/third-party |
| ./third_party/gemmlowp/gemmlowp | no | EXCLUDED | Vendored/third-party |
| ./third_party/gloo | no | EXCLUDED | Vendored/third-party |
| ./third_party/googletest | no | EXCLUDED | Vendored/third-party |
| ./third_party/ideep | no | EXCLUDED | Vendored/third-party |
| ./third_party/ittapi | no | EXCLUDED | Vendored/third-party |
| ./third_party/kineto | no | EXCLUDED | Vendored/third-party |
| ./third_party/kleidiai | no | EXCLUDED | Vendored/third-party |
| ./third_party/llvm-openmp | no | EXCLUDED | Vendored/third-party |
| ./third_party/mimalloc | no | EXCLUDED | Vendored/third-party |
| ./third_party/miniz-3.0.2 | yes | EXCLUDED | Vendored/third-party |
| ./third_party/miniz-3.0.2/examples | no | EXCLUDED | Vendored/third-party |
| ./third_party/mslk | no | EXCLUDED | Vendored/third-party |
| ./third_party/nlohmann | no | EXCLUDED | Vendored/third-party |
| ./third_party/onnx | no | EXCLUDED | Vendored/third-party |
| ./third_party/pocketfft | no | EXCLUDED | Vendored/third-party |
| ./third_party/protobuf | no | EXCLUDED | Vendored/third-party |
| ./third_party/psimd | no | EXCLUDED | Vendored/third-party |
| ./third_party/pthreadpool | no | EXCLUDED | Vendored/third-party |
| ./third_party/pybind11 | no | EXCLUDED | Vendored/third-party |
| ./third_party/python-peachpy | no | EXCLUDED | Vendored/third-party |
| ./third_party/sleef | no | EXCLUDED | Vendored/third-party |
| ./third_party/tensorpipe | no | EXCLUDED | Vendored/third-party |
| ./third_party/valgrind-headers | yes | EXCLUDED | Vendored/third-party |
| ./tools | yes | COVERED |  |
| ./tools/alerts | yes | PENDING |  |
| ./tools/amd_build | yes | PENDING |  |
| ./tools/autograd | yes | PENDING |  |
| ./tools/autograd/templates | yes | PENDING |  |
| ./tools/build_defs | no | EXCLUDED | Empty or stub |
| ./tools/build_defs/android | no | EXCLUDED | Empty or stub |
| ./tools/build_defs/apple | no | EXCLUDED | Empty or stub |
| ./tools/build_defs/windows | no | EXCLUDED | Empty or stub |
| ./tools/code_analyzer | yes | PENDING |  |
| ./tools/code_coverage | yes | PENDING |  |
| ./tools/code_coverage/package | yes | PENDING |  |
| ./tools/code_coverage/package/oss | yes | PENDING |  |
| ./tools/code_coverage/package/tool | yes | PENDING |  |
| ./tools/code_coverage/package/tool/parser | yes | PENDING |  |
| ./tools/code_coverage/package/util | yes | PENDING |  |
| ./tools/coverage_plugins_package | yes | PENDING |  |
| ./tools/coverage_plugins_package/src | yes | PENDING |  |
| ./tools/coverage_plugins_package/src/coverage_plugins | yes | PENDING |  |
| ./tools/dynamo | yes | PENDING |  |
| ./tools/experimental | yes | PENDING |  |
| ./tools/experimental/torchfuzz | yes | PENDING |  |
| ./tools/experimental/torchfuzz/cuda | yes | PENDING |  |
| ./tools/experimental/torchfuzz/operators | yes | PENDING |  |
| ./tools/gdb | yes | PENDING |  |
| ./tools/github | yes | PENDING |  |
| ./tools/iwyu | yes | PENDING |  |
| ./tools/jit | yes | PENDING |  |
| ./tools/jit/templates | yes | PENDING |  |
| ./tools/jit/test | yes | PENDING |  |
| ./tools/linter | yes | PENDING |  |
| ./tools/linter/adapters | yes | PENDING |  |
| ./tools/linter/adapters/_linter | yes | PENDING |  |
| ./tools/linter/clang_tidy | yes | PENDING |  |
| ./tools/lite_interpreter | yes | PENDING |  |
| ./tools/lldb | yes | PENDING |  |
| ./tools/packaging | yes | PENDING |  |
| ./tools/pyi | yes | PENDING |  |
| ./tools/setup_helpers | yes | PENDING |  |
| ./tools/shared | yes | PENDING |  |
| ./tools/stats | yes | PENDING |  |
| ./tools/stats/upload_utilization_stats | yes | PENDING |  |
| ./tools/test | yes | PENDING |  |
| ./tools/test/docstring_linter_testdata | no | EXCLUDED | Test data only |
| ./tools/test/header_only_linter_testdata | yes | EXCLUDED | Test data only |
| ./tools/test/heuristics | yes | PENDING |  |
| ./tools/test/set_linter_testdata | no | EXCLUDED | Test data only |
| ./tools/test/stable_shim_usage_linter_data | yes | EXCLUDED | Test data only |
| ./tools/testing | yes | PENDING |  |
| ./tools/testing/target_determination | yes | PENDING |  |
| ./tools/testing/target_determination/heuristics | yes | PENDING |  |
| ./tools/vendoring | no | EXCLUDED | Vendored/third-party |
| ./tools/vendoring/quack | no | EXCLUDED | Vendored/third-party |
| ./tools/vendoring/quack/patches | no | EXCLUDED | Vendored/third-party |
| ./torch | yes | PENDING |  |
| ./torch/_C | no | EXCLUDED | Empty or stub |
| ./torch/_C/_acc | no | EXCLUDED | Empty or stub |
| ./torch/_C/_dynamo | no | EXCLUDED | Empty or stub |
| ./torch/_C/_export | no | EXCLUDED | Empty or stub |
| ./torch/_C_flatbuffer | no | EXCLUDED | Empty or stub |
| ./torch/_awaits | yes | PENDING |  |
| ./torch/_custom_op | yes | PENDING |  |
| ./torch/_decomp | yes | PENDING |  |
| ./torch/_dispatch | yes | PENDING |  |
| ./torch/_dynamo | yes | COVERED |  |
| ./torch/_dynamo/backends | yes | PENDING |  |
| ./torch/_dynamo/polyfills | yes | PENDING |  |
| ./torch/_dynamo/repro | yes | PENDING |  |
| ./torch/_dynamo/variables | yes | PENDING |  |
| ./torch/_export | yes | COVERED |  |
| ./torch/_export/db | yes | PENDING |  |
| ./torch/_export/db/examples | yes | PENDING |  |
| ./torch/_export/pass_infra | yes | PENDING |  |
| ./torch/_export/passes | yes | PENDING |  |
| ./torch/_export/serde | yes | PENDING |  |
| ./torch/_export/serde/gen-cpp2 | yes | PENDING |  |
| ./torch/_functorch | yes | COVERED |  |
| ./torch/_functorch/_activation_checkpointing | yes | PENDING |  |
| ./torch/_functorch/_activation_offloading | yes | PENDING |  |
| ./torch/_functorch/_aot_autograd | yes | PENDING |  |
| ./torch/_higher_order_ops | yes | PENDING |  |
| ./torch/_higher_order_ops/passes | yes | PENDING |  |
| ./torch/_inductor | yes | COVERED |  |
| ./torch/_inductor/analysis | yes | PENDING |  |
| ./torch/_inductor/autoheuristic | yes | PENDING |  |
| ./torch/_inductor/autoheuristic/artifacts | yes | PENDING |  |
| ./torch/_inductor/codegen | yes | PENDING |  |
| ./torch/_inductor/codegen/aoti_runtime | yes | PENDING |  |
| ./torch/_inductor/codegen/cuda | yes | PENDING |  |
| ./torch/_inductor/codegen/cutedsl | yes | PENDING |  |
| ./torch/_inductor/codegen/cutlass | yes | PENDING |  |
| ./torch/_inductor/codegen/cutlass/lib_extensions | yes | PENDING |  |
| ./torch/_inductor/codegen/cutlass/lib_extensions/cutlass_mock_imports | yes | PENDING |  |
| ./torch/_inductor/codegen/cutlass/lib_extensions/cutlass_mock_imports/cuda | yes | PENDING |  |
| ./torch/_inductor/codegen/cutlass/lib_extensions/cutlass_mock_imports/pydot | yes | PENDING |  |
| ./torch/_inductor/codegen/cutlass/lib_extensions/cutlass_mock_imports/scipy | yes | PENDING |  |
| ./torch/_inductor/codegen/mtia | yes | PENDING |  |
| ./torch/_inductor/codegen/nv_universal_gemm | yes | PENDING |  |
| ./torch/_inductor/codegen/rocm | yes | PENDING |  |
| ./torch/_inductor/codegen/xpu | yes | PENDING |  |
| ./torch/_inductor/compile_worker | yes | PENDING |  |
| ./torch/_inductor/fx_passes | yes | PENDING |  |
| ./torch/_inductor/fx_passes/auto_chunker | yes | PENDING |  |
| ./torch/_inductor/fx_passes/serialized_patterns | yes | PENDING |  |
| ./torch/_inductor/kernel | yes | PENDING |  |
| ./torch/_inductor/kernel/flex | yes | PENDING |  |
| ./torch/_inductor/kernel/flex/templates | no | EXCLUDED | Empty or stub |
| ./torch/_inductor/kernel/templates | no | EXCLUDED | Empty or stub |
| ./torch/_inductor/kernel/vendored_templates | yes | EXCLUDED | Vendored/third-party |
| ./torch/_inductor/kernel/vendored_templates/cutedsl | yes | EXCLUDED | Vendored/third-party |
| ./torch/_inductor/kernel/vendored_templates/cutedsl/wrappers | yes | EXCLUDED | Vendored/third-party |
| ./torch/_inductor/lookup_table | yes | PENDING |  |
| ./torch/_inductor/package | yes | PENDING |  |
| ./torch/_inductor/runtime | yes | PENDING |  |
| ./torch/_inductor/runtime/caching | yes | PENDING |  |
| ./torch/_inductor/template_heuristics | yes | PENDING |  |
| ./torch/_lazy | yes | PENDING |  |
| ./torch/_library | yes | PENDING |  |
| ./torch/_logging | yes | PENDING |  |
| ./torch/_native | yes | PENDING |  |
| ./torch/_native/ops | yes | PENDING |  |
| ./torch/_native/ops/bmm_outer_product | yes | PENDING |  |
| ./torch/_native/ops/norm | yes | PENDING |  |
| ./torch/_native/ops/scatter_add | yes | PENDING |  |
| ./torch/_numpy | yes | PENDING |  |
| ./torch/_numpy/testing | yes | PENDING |  |
| ./torch/_prims | yes | PENDING |  |
| ./torch/_prims_common | yes | PENDING |  |
| ./torch/_refs | yes | PENDING |  |
| ./torch/_refs/linalg | yes | PENDING |  |
| ./torch/_refs/nn | yes | PENDING |  |
| ./torch/_refs/nn/functional | yes | PENDING |  |
| ./torch/_refs/special | yes | PENDING |  |
| ./torch/_strobelight | yes | PENDING |  |
| ./torch/_subclasses | yes | PENDING |  |
| ./torch/_subclasses/complex_tensor | yes | PENDING |  |
| ./torch/_subclasses/complex_tensor/_ops | yes | PENDING |  |
| ./torch/_vendor | yes | EXCLUDED | Vendored/third-party |
| ./torch/_vendor/packaging | yes | EXCLUDED | Vendored/third-party |
| ./torch/_vendor/quack | yes | EXCLUDED | Vendored/third-party |
| ./torch/accelerator | yes | PENDING |  |
| ./torch/amp | yes | COVERED |  |
| ./torch/ao | yes | PENDING |  |
| ./torch/ao/nn | yes | PENDING |  |
| ./torch/ao/nn/intrinsic | yes | PENDING |  |
| ./torch/ao/nn/intrinsic/modules | yes | PENDING |  |
| ./torch/ao/nn/intrinsic/qat | yes | PENDING |  |
| ./torch/ao/nn/intrinsic/qat/modules | yes | PENDING |  |
| ./torch/ao/nn/intrinsic/quantized | yes | PENDING |  |
| ./torch/ao/nn/intrinsic/quantized/dynamic | yes | PENDING |  |
| ./torch/ao/nn/intrinsic/quantized/dynamic/modules | yes | PENDING |  |
| ./torch/ao/nn/intrinsic/quantized/modules | yes | PENDING |  |
| ./torch/ao/nn/qat | yes | PENDING |  |
| ./torch/ao/nn/qat/dynamic | yes | PENDING |  |
| ./torch/ao/nn/qat/dynamic/modules | yes | PENDING |  |
| ./torch/ao/nn/qat/modules | yes | PENDING |  |
| ./torch/ao/nn/quantizable | yes | PENDING |  |
| ./torch/ao/nn/quantizable/modules | yes | PENDING |  |
| ./torch/ao/nn/quantized | yes | PENDING |  |
| ./torch/ao/nn/quantized/dynamic | yes | PENDING |  |
| ./torch/ao/nn/quantized/dynamic/modules | yes | PENDING |  |
| ./torch/ao/nn/quantized/modules | yes | PENDING |  |
| ./torch/ao/nn/quantized/reference | yes | PENDING |  |
| ./torch/ao/nn/quantized/reference/modules | yes | PENDING |  |
| ./torch/ao/nn/sparse | yes | PENDING |  |
| ./torch/ao/nn/sparse/quantized | yes | PENDING |  |
| ./torch/ao/nn/sparse/quantized/dynamic | yes | PENDING |  |
| ./torch/ao/ns | yes | PENDING |  |
| ./torch/ao/ns/fx | yes | PENDING |  |
| ./torch/ao/pruning | yes | PENDING |  |
| ./torch/ao/pruning/_experimental | yes | PENDING |  |
| ./torch/ao/pruning/_experimental/activation_sparsifier | yes | PENDING |  |
| ./torch/ao/pruning/_experimental/data_scheduler | yes | PENDING |  |
| ./torch/ao/pruning/_experimental/data_sparsifier | yes | PENDING |  |
| ./torch/ao/pruning/_experimental/data_sparsifier/benchmarks | yes | PENDING |  |
| ./torch/ao/pruning/_experimental/data_sparsifier/benchmarks/images | no | EXCLUDED | Empty or stub |
| ./torch/ao/pruning/_experimental/data_sparsifier/lightning | yes | PENDING |  |
| ./torch/ao/pruning/_experimental/data_sparsifier/lightning/callbacks | yes | PENDING |  |
| ./torch/ao/pruning/_experimental/data_sparsifier/lightning/tests | yes | EXCLUDED | Test data only |
| ./torch/ao/pruning/_experimental/pruner | yes | PENDING |  |
| ./torch/ao/pruning/_experimental/pruner/images | no | EXCLUDED | Empty or stub |
| ./torch/ao/pruning/scheduler | yes | PENDING |  |
| ./torch/ao/pruning/sparsifier | yes | PENDING |  |
| ./torch/ao/quantization | yes | PENDING |  |
| ./torch/ao/quantization/backend_config | yes | PENDING |  |
| ./torch/ao/quantization/experimental | yes | PENDING |  |
| ./torch/ao/quantization/fx | yes | PENDING |  |
| ./torch/ao/quantization/fx/_model_report | yes | PENDING |  |
| ./torch/autograd | yes | COVERED |  |
| ./torch/autograd/_functions | yes | PENDING |  |
| ./torch/backends | yes | PENDING |  |
| ./torch/backends/_coreml | yes | PENDING |  |
| ./torch/backends/_nnapi | yes | PENDING |  |
| ./torch/backends/cpu | yes | PENDING |  |
| ./torch/backends/cuda | yes | PENDING |  |
| ./torch/backends/cudnn | yes | PENDING |  |
| ./torch/backends/cusparselt | yes | PENDING |  |
| ./torch/backends/kleidiai | yes | PENDING |  |
| ./torch/backends/mha | yes | PENDING |  |
| ./torch/backends/miopen | yes | PENDING |  |
| ./torch/backends/mkl | yes | PENDING |  |
| ./torch/backends/mkldnn | yes | PENDING |  |
| ./torch/backends/mps | yes | PENDING |  |
| ./torch/backends/nnpack | yes | PENDING |  |
| ./torch/backends/openmp | yes | PENDING |  |
| ./torch/backends/opt_einsum | yes | PENDING |  |
| ./torch/backends/python_native | yes | PENDING |  |
| ./torch/backends/quantized | yes | PENDING |  |
| ./torch/backends/xeon | yes | PENDING |  |
| ./torch/backends/xnnpack | yes | PENDING |  |
| ./torch/compiler | yes | COVERED |  |
| ./torch/contrib | yes | PENDING |  |
| ./torch/cpu | yes | PENDING |  |
| ./torch/cpu/amp | yes | PENDING |  |
| ./torch/csrc | yes | PENDING |  |
| ./torch/csrc/acc | yes | PENDING |  |
| ./torch/csrc/api | yes | PENDING |  |
| ./torch/csrc/api/include | yes | PENDING |  |
| ./torch/csrc/api/include/torch | yes | PENDING |  |
| ./torch/csrc/api/include/torch/data | yes | PENDING |  |
| ./torch/csrc/api/include/torch/data/dataloader | yes | PENDING |  |
| ./torch/csrc/api/include/torch/data/datasets | yes | PENDING |  |
| ./torch/csrc/api/include/torch/data/detail | yes | PENDING |  |
| ./torch/csrc/api/include/torch/data/samplers | yes | PENDING |  |
| ./torch/csrc/api/include/torch/data/transforms | yes | PENDING |  |
| ./torch/csrc/api/include/torch/detail | yes | PENDING |  |
| ./torch/csrc/api/include/torch/nativert | yes | PENDING |  |
| ./torch/csrc/api/include/torch/nn | yes | PENDING |  |
| ./torch/csrc/api/include/torch/nn/functional | yes | PENDING |  |
| ./torch/csrc/api/include/torch/nn/modules | yes | PENDING |  |
| ./torch/csrc/api/include/torch/nn/modules/container | yes | PENDING |  |
| ./torch/csrc/api/include/torch/nn/options | yes | PENDING |  |
| ./torch/csrc/api/include/torch/nn/parallel | yes | PENDING |  |
| ./torch/csrc/api/include/torch/nn/utils | yes | PENDING |  |
| ./torch/csrc/api/include/torch/optim | yes | PENDING |  |
| ./torch/csrc/api/include/torch/optim/schedulers | yes | PENDING |  |
| ./torch/csrc/api/include/torch/python | yes | PENDING |  |
| ./torch/csrc/api/include/torch/serialize | yes | PENDING |  |
| ./torch/csrc/api/src | yes | PENDING |  |
| ./torch/csrc/api/src/data | yes | PENDING |  |
| ./torch/csrc/api/src/data/datasets | yes | PENDING |  |
| ./torch/csrc/api/src/data/samplers | yes | PENDING |  |
| ./torch/csrc/api/src/nn | yes | PENDING |  |
| ./torch/csrc/api/src/nn/modules | yes | PENDING |  |
| ./torch/csrc/api/src/nn/modules/container | yes | PENDING |  |
| ./torch/csrc/api/src/nn/options | yes | PENDING |  |
| ./torch/csrc/api/src/optim | yes | PENDING |  |
| ./torch/csrc/api/src/optim/schedulers | yes | PENDING |  |
| ./torch/csrc/api/src/python | yes | PENDING |  |
| ./torch/csrc/api/src/serialize | yes | PENDING |  |
| ./torch/csrc/autograd | yes | COVERED |  |
| ./torch/csrc/autograd/functions | yes | PENDING |  |
| ./torch/csrc/autograd/utils | yes | PENDING |  |
| ./torch/csrc/cpu | yes | PENDING |  |
| ./torch/csrc/cuda | yes | PENDING |  |
| ./torch/csrc/cuda/shared | yes | PENDING |  |
| ./torch/csrc/distributed | yes | PENDING |  |
| ./torch/csrc/distributed/autograd | yes | PENDING |  |
| ./torch/csrc/distributed/autograd/context | yes | PENDING |  |
| ./torch/csrc/distributed/autograd/engine | yes | PENDING |  |
| ./torch/csrc/distributed/autograd/functions | yes | PENDING |  |
| ./torch/csrc/distributed/autograd/rpc_messages | yes | PENDING |  |
| ./torch/csrc/distributed/c10d | yes | PENDING |  |
| ./torch/csrc/distributed/c10d/control_collectives | yes | PENDING |  |
| ./torch/csrc/distributed/c10d/control_plane | yes | PENDING |  |
| ./torch/csrc/distributed/c10d/cuda | yes | PENDING |  |
| ./torch/csrc/distributed/c10d/cuda/cutlass | no | EXCLUDED | Empty or stub |
| ./torch/csrc/distributed/c10d/cuda/cutlass/gemm | no | EXCLUDED | Empty or stub |
| ./torch/csrc/distributed/c10d/cuda/cutlass/gemm/kernel | no | EXCLUDED | Empty or stub |
| ./torch/csrc/distributed/c10d/quantization | yes | PENDING |  |
| ./torch/csrc/distributed/c10d/symm_mem | yes | PENDING |  |
| ./torch/csrc/distributed/c10d/symm_mem/ops | yes | PENDING |  |
| ./torch/csrc/distributed/rpc | yes | PENDING |  |
| ./torch/csrc/distributed/rpc/metrics | yes | PENDING |  |
| ./torch/csrc/distributed/rpc/profiler | yes | PENDING |  |
| ./torch/csrc/distributed/rpc/testing | yes | PENDING |  |
| ./torch/csrc/dynamo | yes | PENDING |  |
| ./torch/csrc/export | yes | PENDING |  |
| ./torch/csrc/functionalization | yes | PENDING |  |
| ./torch/csrc/functorch | yes | PENDING |  |
| ./torch/csrc/fx | yes | PENDING |  |
| ./torch/csrc/inductor | yes | PENDING |  |
| ./torch/csrc/inductor/aoti_eager | yes | PENDING |  |
| ./torch/csrc/inductor/aoti_include | yes | PENDING |  |
| ./torch/csrc/inductor/aoti_package | yes | PENDING |  |
| ./torch/csrc/inductor/aoti_runner | yes | PENDING |  |
| ./torch/csrc/inductor/aoti_runtime | yes | PENDING |  |
| ./torch/csrc/inductor/aoti_torch | yes | PENDING |  |
| ./torch/csrc/inductor/aoti_torch/c | yes | PENDING |  |
| ./torch/csrc/inductor/aoti_torch/generated | yes | EXCLUDED | Auto-generated code |
| ./torch/csrc/inductor/cpp_wrapper | yes | PENDING |  |
| ./torch/csrc/inductor/cpp_wrapper/device_internal | yes | PENDING |  |
| ./torch/csrc/inductor/static_launcher | yes | PENDING |  |
| ./torch/csrc/instruction_counter | yes | PENDING |  |
| ./torch/csrc/jit | yes | PENDING |  |
| ./torch/csrc/jit/api | yes | PENDING |  |
| ./torch/csrc/jit/backends | yes | PENDING |  |
| ./torch/csrc/jit/backends/coreml | yes | PENDING |  |
| ./torch/csrc/jit/backends/coreml/cpp | yes | PENDING |  |
| ./torch/csrc/jit/backends/coreml/objc | yes | PENDING |  |
| ./torch/csrc/jit/backends/nnapi | yes | PENDING |  |
| ./torch/csrc/jit/backends/xnnpack | yes | PENDING |  |
| ./torch/csrc/jit/backends/xnnpack/compiler | yes | PENDING |  |
| ./torch/csrc/jit/backends/xnnpack/executor | yes | PENDING |  |
| ./torch/csrc/jit/backends/xnnpack/serialization | yes | PENDING |  |
| ./torch/csrc/jit/codegen | yes | PENDING |  |
| ./torch/csrc/jit/codegen/cuda | yes | PENDING |  |
| ./torch/csrc/jit/codegen/fuser | yes | PENDING |  |
| ./torch/csrc/jit/codegen/fuser/cpu | yes | PENDING |  |
| ./torch/csrc/jit/codegen/fuser/cuda | yes | PENDING |  |
| ./torch/csrc/jit/codegen/onednn | yes | PENDING |  |
| ./torch/csrc/jit/cuda | yes | PENDING |  |
| ./torch/csrc/jit/docs | no | EXCLUDED | Empty or stub |
| ./torch/csrc/jit/frontend | yes | PENDING |  |
| ./torch/csrc/jit/ir | yes | PENDING |  |
| ./torch/csrc/jit/mobile | yes | PENDING |  |
| ./torch/csrc/jit/mobile/compatibility | yes | PENDING |  |
| ./torch/csrc/jit/mobile/model_tracer | yes | PENDING |  |
| ./torch/csrc/jit/mobile/nnc | yes | PENDING |  |
| ./torch/csrc/jit/mobile/train | yes | PENDING |  |
| ./torch/csrc/jit/mobile/train/optim | yes | PENDING |  |
| ./torch/csrc/jit/operator_upgraders | yes | PENDING |  |
| ./torch/csrc/jit/passes | yes | PENDING |  |
| ./torch/csrc/jit/passes/dbr_quantization | yes | PENDING |  |
| ./torch/csrc/jit/passes/onnx | yes | PENDING |  |
| ./torch/csrc/jit/passes/onnx/pattern_conversion | yes | PENDING |  |
| ./torch/csrc/jit/passes/quantization | yes | PENDING |  |
| ./torch/csrc/jit/passes/utils | yes | PENDING |  |
| ./torch/csrc/jit/python | yes | PENDING |  |
| ./torch/csrc/jit/runtime | yes | PENDING |  |
| ./torch/csrc/jit/runtime/interpreter | yes | PENDING |  |
| ./torch/csrc/jit/runtime/static | yes | PENDING |  |
| ./torch/csrc/jit/serialization | yes | PENDING |  |
| ./torch/csrc/jit/tensorexpr | yes | PENDING |  |
| ./torch/csrc/jit/tensorexpr/operators | yes | PENDING |  |
| ./torch/csrc/jit/tensorexpr/scripts | yes | PENDING |  |
| ./torch/csrc/jit/testing | yes | PENDING |  |
| ./torch/csrc/lazy | yes | PENDING |  |
| ./torch/csrc/lazy/backend | yes | PENDING |  |
| ./torch/csrc/lazy/core | yes | PENDING |  |
| ./torch/csrc/lazy/core/internal_ops | yes | PENDING |  |
| ./torch/csrc/lazy/core/ops | yes | PENDING |  |
| ./torch/csrc/lazy/generated | no | EXCLUDED | Auto-generated code |
| ./torch/csrc/lazy/python | yes | PENDING |  |
| ./torch/csrc/lazy/ts_backend | yes | PENDING |  |
| ./torch/csrc/lazy/ts_backend/ops | yes | PENDING |  |
| ./torch/csrc/monitor | yes | PENDING |  |
| ./torch/csrc/mps | yes | PENDING |  |
| ./torch/csrc/mtia | yes | PENDING |  |
| ./torch/csrc/mtia/profiler | yes | PENDING |  |
| ./torch/csrc/multiprocessing | yes | PENDING |  |
| ./torch/csrc/onnx | yes | PENDING |  |
| ./torch/csrc/profiler | yes | PENDING |  |
| ./torch/csrc/profiler/orchestration | yes | PENDING |  |
| ./torch/csrc/profiler/python | yes | PENDING |  |
| ./torch/csrc/profiler/standalone | yes | PENDING |  |
| ./torch/csrc/profiler/stubs | yes | PENDING |  |
| ./torch/csrc/profiler/unwind | yes | PENDING |  |
| ./torch/csrc/stable | yes | PENDING |  |
| ./torch/csrc/stable/c | yes | PENDING |  |
| ./torch/csrc/tensor | yes | PENDING |  |
| ./torch/csrc/utils | yes | PENDING |  |
| ./torch/csrc/xpu | yes | PENDING |  |
| ./torch/cuda | yes | COVERED |  |
| ./torch/cuda/amp | yes | PENDING |  |
| ./torch/distributed | yes | COVERED |  |
| ./torch/distributed/_composable | yes | PENDING |  |
| ./torch/distributed/_composable/fsdp | yes | PENDING |  |
| ./torch/distributed/_local_tensor | yes | PENDING |  |
| ./torch/distributed/_ops | yes | PENDING |  |
| ./torch/distributed/_pycute | yes | PENDING |  |
| ./torch/distributed/_shard | yes | PENDING |  |
| ./torch/distributed/_shard/checkpoint | yes | PENDING |  |
| ./torch/distributed/_shard/sharded_optim | yes | PENDING |  |
| ./torch/distributed/_shard/sharded_tensor | yes | PENDING |  |
| ./torch/distributed/_shard/sharded_tensor/_ops | yes | PENDING |  |
| ./torch/distributed/_shard/sharding_plan | yes | PENDING |  |
| ./torch/distributed/_shard/sharding_spec | yes | PENDING |  |
| ./torch/distributed/_shard/sharding_spec/chunk_sharding_spec_ops | yes | PENDING |  |
| ./torch/distributed/_sharded_tensor | yes | PENDING |  |
| ./torch/distributed/_sharding_spec | yes | PENDING |  |
| ./torch/distributed/_symmetric_memory | yes | PENDING |  |
| ./torch/distributed/_tensor | yes | COVERED |  |
| ./torch/distributed/_tools | yes | PENDING |  |
| ./torch/distributed/algorithms | yes | PENDING |  |
| ./torch/distributed/algorithms/_checkpoint | yes | PENDING |  |
| ./torch/distributed/algorithms/_comm_hooks | yes | PENDING |  |
| ./torch/distributed/algorithms/_optimizer_overlap | yes | PENDING |  |
| ./torch/distributed/algorithms/_quantization | yes | PENDING |  |
| ./torch/distributed/algorithms/ddp_comm_hooks | yes | PENDING |  |
| ./torch/distributed/algorithms/model_averaging | yes | PENDING |  |
| ./torch/distributed/autograd | yes | PENDING |  |
| ./torch/distributed/benchmarks | yes | PENDING |  |
| ./torch/distributed/checkpoint | yes | PENDING |  |
| ./torch/distributed/checkpoint/_experimental | yes | PENDING |  |
| ./torch/distributed/checkpoint/examples | yes | PENDING |  |
| ./torch/distributed/debug | yes | PENDING |  |
| ./torch/distributed/elastic | yes | PENDING |  |
| ./torch/distributed/elastic/agent | yes | PENDING |  |
| ./torch/distributed/elastic/agent/server | yes | PENDING |  |
| ./torch/distributed/elastic/events | yes | PENDING |  |
| ./torch/distributed/elastic/metrics | yes | PENDING |  |
| ./torch/distributed/elastic/multiprocessing | yes | PENDING |  |
| ./torch/distributed/elastic/multiprocessing/errors | yes | PENDING |  |
| ./torch/distributed/elastic/multiprocessing/subprocess_handler | yes | PENDING |  |
| ./torch/distributed/elastic/rendezvous | yes | PENDING |  |
| ./torch/distributed/elastic/timer | yes | PENDING |  |
| ./torch/distributed/elastic/utils | yes | PENDING |  |
| ./torch/distributed/elastic/utils/data | yes | PENDING |  |
| ./torch/distributed/examples | yes | PENDING |  |
| ./torch/distributed/flight_recorder | yes | PENDING |  |
| ./torch/distributed/flight_recorder/components | yes | PENDING |  |
| ./torch/distributed/fsdp | yes | COVERED |  |
| ./torch/distributed/fsdp/_fully_shard | yes | PENDING |  |
| ./torch/distributed/launcher | yes | PENDING |  |
| ./torch/distributed/nn | yes | PENDING |  |
| ./torch/distributed/nn/api | yes | PENDING |  |
| ./torch/distributed/nn/jit | yes | PENDING |  |
| ./torch/distributed/nn/jit/templates | yes | PENDING |  |
| ./torch/distributed/optim | yes | PENDING |  |
| ./torch/distributed/pipelining | yes | PENDING |  |
| ./torch/distributed/rpc | yes | PENDING |  |
| ./torch/distributed/rpc/_testing | yes | PENDING |  |
| ./torch/distributed/tensor | yes | PENDING |  |
| ./torch/distributed/tensor/_ops | yes | PENDING |  |
| ./torch/distributed/tensor/debug | yes | PENDING |  |
| ./torch/distributed/tensor/examples | yes | PENDING |  |
| ./torch/distributed/tensor/experimental | yes | PENDING |  |
| ./torch/distributed/tensor/experimental/_context_parallel | yes | PENDING |  |
| ./torch/distributed/tensor/parallel | yes | PENDING |  |
| ./torch/distributions | yes | COVERED |  |
| ./torch/export | yes | COVERED |  |
| ./torch/export/experimental | yes | PENDING |  |
| ./torch/export/passes | yes | PENDING |  |
| ./torch/export/pt2_archive | yes | PENDING |  |
| ./torch/fft | yes | COVERED |  |
| ./torch/func | yes | PENDING |  |
| ./torch/futures | yes | PENDING |  |
| ./torch/fx | yes | COVERED |  |
| ./torch/fx/experimental | yes | PENDING |  |
| ./torch/fx/experimental/migrate_gradual_types | yes | PENDING |  |
| ./torch/fx/experimental/shape_inference | yes | PENDING |  |
| ./torch/fx/experimental/unification | yes | PENDING |  |
| ./torch/fx/experimental/unification/multipledispatch | yes | PENDING |  |
| ./torch/fx/passes | yes | PENDING |  |
| ./torch/fx/passes/backends | yes | PENDING |  |
| ./torch/fx/passes/dialect | yes | PENDING |  |
| ./torch/fx/passes/dialect/common | yes | PENDING |  |
| ./torch/fx/passes/infra | yes | PENDING |  |
| ./torch/fx/passes/tests | yes | PENDING |  |
| ./torch/fx/passes/utils | yes | PENDING |  |
| ./torch/headeronly | yes | PENDING |  |
| ./torch/headeronly/core | yes | PENDING |  |
| ./torch/headeronly/cpu | yes | PENDING |  |
| ./torch/headeronly/cpu/vec | yes | PENDING |  |
| ./torch/headeronly/macros | yes | PENDING |  |
| ./torch/headeronly/util | yes | PENDING |  |
| ./torch/jit | yes | COVERED |  |
| ./torch/jit/_passes | yes | PENDING |  |
| ./torch/jit/mobile | yes | PENDING |  |
| ./torch/legacy | no | EXCLUDED | Empty or stub |
| ./torch/lib | yes | PENDING |  |
| ./torch/lib/libshm | yes | PENDING |  |
| ./torch/lib/libshm_windows | yes | PENDING |  |
| ./torch/linalg | yes | COVERED |  |
| ./torch/masked | yes | PENDING |  |
| ./torch/masked/maskedtensor | yes | PENDING |  |
| ./torch/monitor | yes | PENDING |  |
| ./torch/mps | yes | PENDING |  |
| ./torch/mtia | yes | PENDING |  |
| ./torch/multiprocessing | yes | COVERED |  |
| ./torch/nativert | yes | PENDING |  |
| ./torch/nativert/backends | yes | PENDING |  |
| ./torch/nativert/common | yes | PENDING |  |
| ./torch/nativert/detail | yes | PENDING |  |
| ./torch/nativert/executor | yes | PENDING |  |
| ./torch/nativert/executor/memory | yes | PENDING |  |
| ./torch/nativert/executor/triton | yes | PENDING |  |
| ./torch/nativert/graph | yes | PENDING |  |
| ./torch/nativert/graph/passes | yes | PENDING |  |
| ./torch/nativert/graph/passes/pass_manager | yes | PENDING |  |
| ./torch/nativert/kernels | yes | PENDING |  |
| ./torch/nativert/python | yes | PENDING |  |
| ./torch/nested | yes | PENDING |  |
| ./torch/nested/_internal | yes | PENDING |  |
| ./torch/nn | yes | PENDING |  |
| ./torch/nn/attention | yes | PENDING |  |
| ./torch/nn/attention/experimental | yes | PENDING |  |
| ./torch/nn/backends | yes | PENDING |  |
| ./torch/nn/intrinsic | yes | PENDING |  |
| ./torch/nn/intrinsic/modules | yes | PENDING |  |
| ./torch/nn/intrinsic/qat | yes | PENDING |  |
| ./torch/nn/intrinsic/qat/modules | yes | PENDING |  |
| ./torch/nn/intrinsic/quantized | yes | PENDING |  |
| ./torch/nn/intrinsic/quantized/dynamic | yes | PENDING |  |
| ./torch/nn/intrinsic/quantized/dynamic/modules | yes | PENDING |  |
| ./torch/nn/intrinsic/quantized/modules | yes | PENDING |  |
| ./torch/nn/modules | yes | COVERED |  |
| ./torch/nn/parallel | yes | PENDING |  |
| ./torch/nn/qat | yes | PENDING |  |
| ./torch/nn/qat/dynamic | yes | PENDING |  |
| ./torch/nn/qat/dynamic/modules | yes | PENDING |  |
| ./torch/nn/qat/modules | yes | PENDING |  |
| ./torch/nn/quantizable | yes | PENDING |  |
| ./torch/nn/quantizable/modules | yes | PENDING |  |
| ./torch/nn/quantized | yes | PENDING |  |
| ./torch/nn/quantized/_reference | yes | PENDING |  |
| ./torch/nn/quantized/_reference/modules | yes | PENDING |  |
| ./torch/nn/quantized/dynamic | yes | PENDING |  |
| ./torch/nn/quantized/dynamic/modules | yes | PENDING |  |
| ./torch/nn/quantized/modules | yes | PENDING |  |
| ./torch/nn/utils | yes | COVERED |  |
| ./torch/nn/utils/_expanded_weights | yes | PENDING |  |
| ./torch/numa | yes | PENDING |  |
| ./torch/onnx | yes | PENDING |  |
| ./torch/onnx/_internal | yes | PENDING |  |
| ./torch/onnx/_internal/exporter | yes | PENDING |  |
| ./torch/onnx/_internal/exporter/_torchlib | yes | PENDING |  |
| ./torch/onnx/_internal/exporter/_torchlib/ops | yes | PENDING |  |
| ./torch/onnx/_internal/fx | yes | PENDING |  |
| ./torch/onnx/_internal/fx/passes | yes | PENDING |  |
| ./torch/onnx/_internal/torchscript_exporter | yes | PENDING |  |
| ./torch/onnx/ops | yes | PENDING |  |
| ./torch/optim | yes | COVERED |  |
| ./torch/optim/_multi_tensor | yes | PENDING |  |
| ./torch/package | yes | COVERED |  |
| ./torch/package/analyze | yes | PENDING |  |
| ./torch/profiler | yes | COVERED |  |
| ./torch/quantization | yes | PENDING |  |
| ./torch/quantization/fx | yes | PENDING |  |
| ./torch/signal | yes | PENDING |  |
| ./torch/signal/windows | yes | PENDING |  |
| ./torch/sparse | yes | COVERED |  |
| ./torch/special | yes | PENDING |  |
| ./torch/testing | yes | PENDING |  |
| ./torch/testing/_internal | yes | PENDING |  |
| ./torch/testing/_internal/codegen | yes | PENDING |  |
| ./torch/testing/_internal/data | yes | EXCLUDED | Test data only |
| ./torch/testing/_internal/distributed | yes | PENDING |  |
| ./torch/testing/_internal/distributed/_shard | yes | PENDING |  |
| ./torch/testing/_internal/distributed/_shard/sharded_tensor | yes | PENDING |  |
| ./torch/testing/_internal/distributed/_tensor | yes | PENDING |  |
| ./torch/testing/_internal/distributed/nn | yes | PENDING |  |
| ./torch/testing/_internal/distributed/nn/api | yes | PENDING |  |
| ./torch/testing/_internal/distributed/rpc | yes | PENDING |  |
| ./torch/testing/_internal/distributed/rpc/examples | yes | PENDING |  |
| ./torch/testing/_internal/distributed/rpc/jit | yes | PENDING |  |
| ./torch/testing/_internal/generated | yes | EXCLUDED | Auto-generated code |
| ./torch/testing/_internal/opinfo | yes | PENDING |  |
| ./torch/testing/_internal/opinfo/definitions | yes | PENDING |  |
| ./torch/testing/_internal/optests | yes | PENDING |  |
| ./torch/testing/_internal/test_module | yes | PENDING |  |
| ./torch/utils | yes | COVERED |  |
| ./torch/utils/_debug_mode | yes | PENDING |  |
| ./torch/utils/_strobelight | yes | PENDING |  |
| ./torch/utils/_strobelight/examples | yes | PENDING |  |
| ./torch/utils/_sympy | yes | PENDING |  |
| ./torch/utils/backcompat | yes | PENDING |  |
| ./torch/utils/benchmark | yes | PENDING |  |
| ./torch/utils/benchmark/examples | yes | PENDING |  |
| ./torch/utils/benchmark/examples/sparse | yes | PENDING |  |
| ./torch/utils/benchmark/op_fuzzers | yes | PENDING |  |
| ./torch/utils/benchmark/utils | yes | PENDING |  |
| ./torch/utils/benchmark/utils/valgrind_wrapper | yes | PENDING |  |
| ./torch/utils/data | yes | PENDING |  |
| ./torch/utils/data/_utils | yes | PENDING |  |
| ./torch/utils/data/datapipes | yes | PENDING |  |
| ./torch/utils/data/datapipes/dataframe | yes | PENDING |  |
| ./torch/utils/data/datapipes/iter | yes | PENDING |  |
| ./torch/utils/data/datapipes/map | yes | PENDING |  |
| ./torch/utils/data/datapipes/utils | yes | PENDING |  |
| ./torch/utils/hipify | yes | PENDING |  |
| ./torch/utils/jit | yes | PENDING |  |
| ./torch/utils/model_dump | yes | PENDING |  |
| ./torch/utils/serialization | yes | PENDING |  |
| ./torch/utils/tensorboard | yes | PENDING |  |
| ./torch/utils/viz | yes | PENDING |  |
| ./torch/xpu | yes | PENDING |  |
| ./torchgen | yes | COVERED |  |
| ./torchgen/_autoheuristic | yes | PENDING |  |
| ./torchgen/_autoheuristic/mixed_mm | yes | PENDING |  |
| ./torchgen/_autoheuristic/mm | yes | PENDING |  |
| ./torchgen/_autoheuristic/pad_mm | yes | PENDING |  |
| ./torchgen/aoti | yes | PENDING |  |
| ./torchgen/api | yes | PENDING |  |
| ./torchgen/api/types | yes | PENDING |  |
| ./torchgen/decompositions | yes | PENDING |  |
| ./torchgen/dest | yes | PENDING |  |
| ./torchgen/fuse | yes | PENDING |  |
| ./torchgen/operator_versions | yes | PENDING |  |
| ./torchgen/selective_build | yes | PENDING |  |
| ./torchgen/shape_functions | yes | PENDING |  |
| ./torchgen/static_runtime | yes | PENDING |  |
