# `functorch/einops`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`functorch/einops` provides an einops-style rearrangement frontend on top of first-class dimensions. It owns pattern parsing and the cached lowering from human-readable rearrangement strings to `functorch.dim` indexing and ordering operations.

## Key Files

| File | Purpose |
|---|---|
| `rearrange.py` | Parses rearrangement patterns, memoizes compiled callables, and implements `rearrange()` |
| `_parsing.py` | Tokenizes and validates einops-style expressions into `ParsedExpression` objects |
| `__init__.py` | Reexports the canonical `rearrange` API from `torch._functorch.einops` |

## Public Interface

The public API is `rearrange()`, reexported through `__init__.py`. Internal entry points that other functorch code uses are `_create_rearrange_callable()`, `parse_pattern()`, `validate_rearrange_expressions()`, `ParsedExpression`, `AnonymousAxis`, and the `_ellipsis` marker used during parsing.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [functorch/dim](functorch/dim/ADR.md) | depends-on | `rearrange.py` imports `dims` and lowers patterns to `tensor[...]` and `.order(...)` over first-class dimensions |
| [torch/_functorch](torch/_functorch/ADR.md) | mutual | `__init__.py` reexports `torch._functorch.einops.rearrange`, while `torch/_functorch/einops.py` lazy-imports `functorch.einops.rearrange._create_rearrange_callable()` |

## Runtime Behaviour

`_create_rearrange_callable()` parses the left and right sides of the pattern with `parse_pattern()` and `validate_rearrange_expressions()`, computes how many named, anonymous, and ellipsis dimensions are needed, and caches the resulting callable with `@functools.lru_cache(256)`. For each cache miss it synthesizes Python source for `do_rearrange`, calls `exec()` to build that function, and makes it create `dims(n)`, bind any explicit axis lengths, index `tensor[left_dims]`, reorder with `.order(right_dims)`, and optionally reduce anonymous unit axes with `.sum(..., keepdim=False)`. `_parsing.py` turns identifiers, parentheses, numbers, and ellipses into `ParsedExpression.composition`, rejects malformed axis names or duplicate identifiers, and preserves composite-axis structure so `rearrange.py` can flatten and unflatten dimensions correctly.

## Performance Profile

- **Allocation sites** - Every new `(tensor_ndim, pattern, axes_lengths)` combination allocates `ParsedExpression` objects and one generated callable in `_create_rearrange_callable()`. `rearrange()` also allocates a stacked tensor with `torch.stack()` when callers pass a list or tuple instead of a single tensor.
- **Synchronization costs** - The implementation is pure Python and uses no explicit locking beyond the function-cache path managed by CPython. Runtime overhead comes from parsing and callable generation on cache misses, not from cross-thread coordination.
- **Data movement** - Rearrangements themselves are expressed as `functorch.dim` indexing and `order()` calls, so they follow the view-oriented movement rules implemented there. Anonymous unit axes add an explicit `.sum(...)` after reordering, which performs a real reduction over those dimensions.
- **Redundant or repeated work** - `@lru_cache(256)` avoids reparsing and regenerating callables for repeated patterns on tensors with the same rank. Cache misses still pay the full `parse_pattern()` plus `exec()` cost because the module compiles each distinct rearrangement into a new specialized Python function.

## Design Rationale

This layer lowers readable einops syntax into existing first-class-dimension primitives instead of inventing a second tensor-transformation engine. The split between `_parsing.py` and `rearrange.py` keeps grammar validation separate from execution lowering, and the reexport through `torch._functorch.einops` lets `torch.func.rearrange` share the same lowering logic.
