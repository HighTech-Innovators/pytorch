# `torch/distributions`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/distributions` owns PyTorch's probability-distribution framework. It defines the abstract `Distribution` contract, concrete distributions, constraint and transform systems, and pairwise KL-divergence registration used by probabilistic and variational code.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Re-exports concrete distributions, transform utilities, and KL helpers into the package namespace |
| `distribution.py` | Defines the abstract `Distribution` base class, validation behavior, and sampling protocol |
| `constraints.py` | Defines `Constraint` subclasses such as `_Simplex`, `_PositiveDefinite`, and `_CorrCholesky` used for parameter and support validation |
| `transforms.py` | Defines `Transform`, `_InverseTransform`, `ComposeTransform`, `AffineTransform`, `SigmoidTransform`, and other invertible mappings |
| `kl.py` | Maintains the KL registry and dispatches `kl_divergence` to registered implementations such as `_kl_normal_normal` and `_kl_multivariatenormal_multivariatenormal` |

## Public Interface

The package exports `Distribution`, `ExponentialFamily`, concrete classes such as `Bernoulli`, `Categorical`, `Normal`, `MultivariateNormal`, `Independent`, `MixtureSameFamily`, and `TransformedDistribution`, plus `constraints`, `transforms`, `register_kl`, `kl_divergence`, `biject_to`, and `transform_to`. Important protocol methods include `Distribution.sample`, `Distribution.rsample`, `Distribution.log_prob`, `Distribution.expand`, `Transform.__call__`, `Transform.inv`, and `Transform.log_abs_det_jacobian`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/special](torch/special/ADR.md) | depends-on | `kl.py` uses numerically stable helpers such as `torch.special.xlogy`, and many distributions rely on special functions through `torch` |
| [torch/linalg](torch/linalg/ADR.md) | depends-on | matrix-valued distributions and transform constraints depend on Cholesky-style linear algebra operations exposed through `torch.linalg` |
| [torch/nn](torch/nn/ADR.md) | depends-on | `transforms.py` imports `torch.nn.functional.pad` and `softplus` for transform implementations |

## Runtime Behaviour

`Distribution.__init__()` stores `_batch_shape` and `_event_shape`, then walks `arg_constraints` and calls each constraint's `check()` method unless validation is disabled. `Distribution.sample()` wraps `rsample()` in `torch.no_grad()`, which preserves the same sampling code path while suppressing autograd history for non-reparameterized use. `Transform.__call__()` and `_inv_call()` optionally memoize one `(x, y)` pair when `cache_size=1`, and `Transform.inv` lazily builds a paired `_InverseTransform` that mirrors the original object's domain and codomain. `kl_divergence()` first checks `_KL_MEMOIZE`, falls back to `_dispatch_kl()` when it sees a new `(type(p), type(q))` pair, and then calls the registered implementation such as `_kl_beta_beta` or `_kl_categorical_categorical`.

## Performance Profile

Argument validation can be expensive because `Distribution.__init__()` touches every constrained parameter tensor and executes per-element `Constraint.check()` logic. `Distribution.expand()` is defined to reuse `Tensor.expand()` semantics, so subclasses can change batch shape without allocating new parameter storage when broadcasting is enough. `Transform` caching avoids repeated forward or inverse computation for numerically sensitive bijections, but only remembers a single pair to cap memory growth. KL lookup is amortized by `_KL_MEMOIZE`, so repeated divergence calls between the same distribution types pay the type-resolution cost once per process.

## Design Rationale

The package splits the distribution problem into orthogonal pieces: `Distribution` models sampling and density evaluation, `Constraint` models validity, `Transform` models invertible support changes, and `kl.py` models pairwise information geometry. That separation lets concrete classes reuse shared validation, broadcasting, and transformation machinery instead of reimplementing probabilistic boilerplate in every distribution file.
