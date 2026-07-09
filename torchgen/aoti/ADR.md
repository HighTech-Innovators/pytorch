# `torchgen/aoti`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torchgen/aoti` owns the operator allowlists and ABI metadata that drive AOTInductor C shim generation. It records which ATen operators receive generated stable C wrappers and how versioned fallback signatures evolve without breaking already-compiled AOTInductor models.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Marks `torchgen.aoti` as an importable package |
| `fallback_ops.py` | Defines `inductor_fallback_ops` and `aten_shimified_ops`, the metadata consumed by `torchgen/gen_aoti_c_shim.py` |

## Public Interface

| Symbol | Description |
|---|---|
| `inductor_fallback_ops` | Dictionary keyed by overload names such as `aten._flash_attention_forward.default`; entries describe fallback shim availability, versioned `new_args`, and `since` feature guards |
| `aten_shimified_ops` | Dictionary keyed by ATen overload names such as `aten.fill_.Scalar`; entries drive `c_shim_aten.{h/cpp}` generation for `torch/csrc/stable/ops.h` |

`fallback_ops.py` intentionally exposes data rather than functions. `torchgen/gen_aoti_c_shim.py` imports `aten_shimified_ops` and `inductor_fallback_ops`, then `gen_aoti_c_shim_files()` selects `aten_shimified_ops` for the ATen shim backend and `inductor_fallback_ops` for device-specific AOTInductor fallback shims.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torchgen](torchgen/ADR.md) | depends-on | Supplies `torchgen/gen.py` and `torchgen/gen_aoti_c_shim.py`, which import and interpret the dictionaries in `fallback_ops.py` |
| [torch/_inductor](torch/_inductor/ADR.md) | depends-on | Mirrors the fallback operator set from `torch/_inductor/lowering.py` for AOTInductor-compiled models |
| [torch/csrc](torch/csrc/ADR.md) | depended-on-by | Generated shim declarations and implementations are installed under `torch/csrc/inductor/aoti_torch/generated` and used by stable C APIs |
| [aten/src/ATen](aten/src/ATen/ADR.md) | depends-on | Dictionary keys name concrete ATen overloads from the operator registry, including `aten.add.Tensor`, `aten.mm.out`, and `aten.pad.default` |

## Runtime Behaviour

At code-generation time, `torchgen/gen_aoti_c_shim.py` imports `torchgen.aoti.fallback_ops` and reads `inductor_fallback_ops` or `aten_shimified_ops` according to the requested backend. Version metadata such as `v2`, `new_args`, and `since` controls whether generated declarations gain additional arguments and whether the C shim wraps them in `#if TORCH_FEATURE_VERSION >= ...` guards. The source comments require maintainers to run `python torchgen/gen.py --update-aoti-c-shim` after adding fallback entries so the generated headers stay synchronized with the dictionaries. The module performs no runtime dispatch itself; generated C shim code later calls through the dispatcher rather than bypassing it.

## Performance Profile

`fallback_ops.py` executes as a static dictionary definition during torchgen import, so its direct runtime cost is proportional to constructing the `inductor_fallback_ops` and `aten_shimified_ops` metadata once per generator process. The performance-sensitive path sits in generated AOTInductor shims, not in this Python module; the dictionaries choose which operators receive C wrappers and preserve ABI shape across releases. Keeping fallback metadata as literal dictionaries gives `gen_aoti_c_shim.py` O(1) lookup by overload name when it checks version information for an operator. The `since` guards avoid emitting unavailable declarations for older feature versions, which prevents runtime compatibility checks from moving into compiled model execution.

## Design Rationale

`torchgen/aoti` isolates AOTInductor ABI policy from the larger torchgen driver so fallback additions and signature-version bumps have one review location. The comments in `fallback_ops.py` explicitly allow adding new fallback operators while forbidding removal because compiled models can depend on existing C shim declarations. The `vN` and `since` metadata format encodes binary-compatibility decisions in source control instead of relying on ad hoc generator conditionals. An empty `__init__.py` keeps the package lightweight and lets generator modules import only the metadata they need.
