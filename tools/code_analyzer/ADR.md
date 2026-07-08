# `tools/code_analyzer`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`tools/code_analyzer` generates selective-build operator lists for mobile and custom builds. It consumes root-operator lists, traced model YAML, training operator lists, kernel metadata, model metadata, and an operator dependency graph, then emits YAML and headers that tell torchgen and mobile registration code which operators and kernel dtypes to include. It directly supports the selective-dispatch path referenced by generated registration code: only selected native functions receive generated registrations when a `SelectiveBuilder` filters them.

## Key Files

| File | Purpose |
|---|---|
| `gen_op_registration_allowlist.py` | Computes canonical operator names and transitive dependency closure from root ops and an operator dependency graph |
| `gen_operators_yaml.py` | Merges static roots, traced operators, training roots, model metadata, kernel metadata, custom classes, and build features into selected-operators YAML |
| `gen_oplist.py` | Combines selected-operator YAML files, enforces overload-inclusion rules, writes `selected_operators.yaml`, writes selected mobile ops headers, and emits supported-model registration data |

## Public Interface

`gen_op_registration_allowlist.py` accepts `--root-ops` and optional `--op-dependency`, then prints a space-separated transitive closure. `gen_operators_yaml.py` accepts root/training roots, model metadata filters, model YAML paths, dependency graph path, overload policy, output path, and `--include-all-operators`, then writes a selective-build YAML document. `gen_oplist.py` accepts an output directory, a model-file-list path, and overload policy, then writes `selected_operators.yaml`, `selected_mobile_ops.h`, and `SupportedMobileModelsRegistration.cpp`. Programmatic helpers include `canonical_name()`, `load_op_dep_graph()`, `gen_transitive_closure()`, `fill_output()`, `extract_all_operators()`, and `extract_training_operators()`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torchgen](torchgen/ADR.md) | depends-on | Uses `torchgen.selective_build.operator` and `torchgen.selective_build.selector` to represent and merge selected operators and kernel metadata |
| [c10/mobile](c10/mobile/ADR.md) | generates-for | Produces mobile selected-operator artifacts that reduce registered operator and kernel sets for custom mobile builds |
| [aten/src/ATen/core](aten/src/ATen/core/ADR.md) | generates-for | Feeds generated registration filters so dispatcher tables contain only selected operator schemas and kernels in selective builds |
| [tools/jit](tools/jit/ADR.md) | related-to | Produces operator-selection YAML consumed by generated JIT unboxing and mobile operator registration flows |
| `tools/lite_interpreter` | depends-on | Calls `write_selected_mobile_ops()` to emit selected mobile operator headers |

## Runtime Behaviour

At generation time, `gen_op_registration_allowlist.py` canonicalizes names by dropping overload suffixes, loads dependency YAML, and performs a queue-based transitive closure that always considers `__BASE__` and adds `__ROOT__` for training builds. `gen_operators_yaml.py` loads model YAML files, filters them by model name, version, asset, and backend, verifies requested metadata exists, buckets operators by root/traced/training/static/closure semantics, merges duplicate operator entries, and writes a YAML document with `operators`, `custom_classes`, `build_features`, and optional `kernel_metadata`. `gen_oplist.py` loads one or more selective YAML documents, combines them into a single `SelectiveBuilder`, rejects `include_all_overloads` unless explicitly allowed, and writes the artifacts that downstream builds consume.

## Performance Profile

The scripts operate on YAML and sets, so their build-time cost scales with the number of model metadata files, listed operators, and graph edges in the dependency YAML. The closure algorithm visits each reachable operator once and emits sorted names, which keeps custom-build allowlists deterministic. Selective output reduces later compilation, link size, mobile binary size, and dispatcher registration work by filtering codegen to only required schemas and kernels. The analyzer trades a small Python preprocessing step for lower generated-code volume in ATen and JIT registration paths.

## Design Rationale

Mobile and custom deployments need a smaller operator surface than full PyTorch, but the dispatcher still requires consistent schema and kernel registration for every included operator. `tools/code_analyzer` keeps that selection policy in data files instead of hardcoding it in `torchgen/gen.py`. Static analysis uses dependency closure because root operators can call other ATen operators internally; traced builds can preserve overload precision because traces list concrete operators. The output format matches `torchgen.selective_build.SelectiveBuilder`, so the same selection object filters ATen registrations, training operators, kernel dtype metadata, and generated JIT unboxing code.
