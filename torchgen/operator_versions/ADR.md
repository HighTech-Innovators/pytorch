# `torchgen/operator_versions`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torchgen/operator_versions` generates the mobile operator-upgrader translation unit that preserves bytecode backward compatibility. It owns the transformation from JIT upgrader metadata into the generated `torch/csrc/jit/mobile/upgrader_mobile.cpp` source file.

## Key Files

| File | Purpose |
|---|---|
| `gen_mobile_upgraders.py` | Pulls upgrader bytecode and operator version metadata from JIT, renders C++ tables with `CodeTemplate`, and writes `upgrader_mobile.cpp` |
| `gen_mobile_upgraders_constant.py` | Holds the generated-file banner used at the top of the emitted C++ source |

## Public Interface

The main entry point is the CLI `main()` function in `gen_mobile_upgraders.py`. Other generator code can reuse `write_cpp()`, `construct_version_maps()`, `get_upgrader_bytecode_function_to_index_map()`, `construct_instruction()`, `construct_constants()`, `construct_types()`, and `construct_operators()`. The emitted output is named by `UPGRADER_MOBILE_FILE_NAME` and built from `UPGRADER_CPP_SRC`, `OPERATOR_VERSION_MAP`, and the `ByteCode` enum.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/jit](torch/jit/ADR.md) | depends-on | Imports `generate_upgraders_bytecode()` and reads `torch._C._get_operator_version_map()` plus `torch._C._get_upgrader_ranges()` to discover upgrader metadata |
| [torch/csrc/jit/mobile](torch/csrc/jit/mobile/ADR.md) | depended-on-by | Writes `upgrader_mobile.cpp`, which mobile import and flatbuffer loading code consume through `getUpgraderBytecodeList()` |
| [torchgen](torchgen/ADR.md) | depends-on | Uses `CodeTemplate` and follows torchgen's deterministic code-emission model |

## Runtime Behaviour

`main()` calls `generate_upgraders_bytecode()`, sorts the returned upgrader dictionaries with `sort_upgrader()`, computes the output path under `torch/csrc/jit/mobile`, and passes everything to `write_cpp()`. `construct_version_maps()` walks the operator version map returned by `torch._C._get_operator_version_map()`, matches each entry with `torch._C._get_upgrader_ranges()`, skips the excluded `aten::full*` schemas, and emits a vector of `Upgrader({min_version, max_version, name, bytecode_index})` records per operator. `write_cpp()` renders instruction, constant, type, register-size, and operator-string tables for every upgrader and then writes one generated `upgrader_mobile.cpp` file. The generated C++ initializes `getUpgraderBytecodeList()` by constructing `ByteCodeFunctionWithOperator` entries and appending every referenced operator string to the underlying `mobile::Function`.

## Performance Profile

- **Allocation sites** - The generator builds large Python string fragments for `instruction_list_str`, `constant_list_str`, `type_list_str`, and the final `upgrader_file_content`, but it only emits one translation unit. The generated C++ then materializes `std::vector<Instruction>`, `std::vector<c10::IValue>`, `std::vector<c10::TypePtr>`, and `std::vector<OperatorString>` at runtime.
- **Synchronization costs** - The Python generator is single-threaded and performs no explicit locking. Mobile runtime lookup cost is shifted to static initialization in the generated `getOperatorVersionMapForMobile()` and `getUpgraderBytecodeList()` functions.
- **Data movement** - `construct_instruction()`, `construct_constants()`, `construct_types()`, and `construct_operators()` copy structured Python metadata into serialized C++ initializer lists. `write_cpp()` writes the entire generated translation unit in one `out_file.write()` call.
- **Redundant or repeated work** - Each run walks every operator and every upgrader entry again, because the authoritative metadata comes from live JIT helpers rather than a cached snapshot. `sort_upgrader()` and `get_upgrader_bytecode_function_to_index_map()` keep the output order deterministic and prevent duplicate index assignment for the excluded `full_*` upgraders.

## Design Rationale

PyTorch keeps mobile upgrader generation separate from the main ATen codegen path because the source of truth is JIT bytecode compatibility data, not `native_functions.yaml`. The generator uses runtime JIT metadata APIs so the emitted upgrader tables stay aligned with the operator version map that mobile import code actually consults.
