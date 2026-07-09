# `torch/distributions`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/distributions` implements parameterized probability distributions, constraints, transforms, and KL divergence utilities on top of PyTorch tensors. It supports stochastic computation graphs through `sample()`, differentiable reparameterized samples through `rsample()`, and differentiable log-density computations through `log_prob()`.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Imports and exports distribution classes such as `Normal`, `Categorical`, `MultivariateNormal`, `TransformedDistribution`, `Wishart`, and KL helpers |
| `distribution.py` | Defines the abstract `Distribution` base class, validation logic, shape properties, `sample()`, and `expand()` contract |
| `transforms.py` | Defines `Transform`, `_InverseTransform`, `ComposeTransform`, `AffineTransform`, `TanhTransform`, `StackTransform`, and related bijections |
| `kl.py` | Maintains `_KL_REGISTRY`, `_KL_MEMOIZE`, `register_kl()`, `_dispatch_kl()`, and concrete KL formulas |
| `constraints.py` | Defines `Constraint`, `_Dependent`, interval, simplex, Cholesky, positive-definite, cat, and stack constraints |

## Public Interface

| Symbol | Description |
|---|---|
| `Distribution` | Base class with `batch_shape`, `event_shape`, `arg_constraints`, `support`, `sample()`, `rsample()`, `log_prob()`, and `expand()` conventions |
| `Normal`, `Categorical`, `MultivariateNormal`, `LowRankMultivariateNormal`, `Wishart` | Concrete distribution classes exported from `__init__.py` |
| `Transform`, `ComposeTransform`, `AffineTransform`, `TanhTransform`, `SoftmaxTransform` | Transform classes used directly and by `TransformedDistribution` |
| `constraints.Constraint` and module-level constraints | Validation objects used by distribution constructors and transform domains |
| `register_kl()` / `kl_divergence()` | Decorator and dispatcher for pairwise KL divergence implementations |
| `biject_to` / `transform_to` | Constraint registry helpers exported by `__init__.py` for parameter transformations |

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch](torch/ADR.md) | depends-on | Uses tensors, broadcasting, random sampling, linear algebra, special functions, and module exports |
| [torch/autograd](torch/autograd/ADR.md) | depends-on | `log_prob()` and `rsample()` expressions participate in gradient computation for REINFORCE and pathwise estimators |
| [torch/nn](torch/nn/ADR.md) | depended-on-by | Neural network modules commonly construct distributions from model outputs for losses, policies, and variational objectives |
| [aten/src/ATen](aten/src/ATen/ADR.md) | depends-on | Tensor operations inside distributions and transforms execute through ATen kernels |

## Runtime Behaviour

`Distribution.__init__()` records `_batch_shape` and `_event_shape`, then validates constructor arguments against `arg_constraints` unless validation is disabled. The validation path skips dependent constraints through `constraints.is_dependent()` and raises a `ValueError` when `constraint.check(value)` contains invalid entries. `Transform.__call__()` and `_inv_call()` optionally cache the most recent `(x, y)` pair when `cache_size == 1`, and `kl_divergence()` finds the most specific registered formula through `_dispatch_kl()`.

## Performance Profile

Distribution classes rely on vectorized tensor math, so sampling and `log_prob()` scale with the underlying ATen operations rather than Python loops. `Distribution.expand()` is specified to call `Tensor.expand()` on parameters, which creates views and avoids allocating repeated parameter storage. Validation can add full-tensor constraint checks to construction, and `Transform` caching speeds numerically expensive inverses at the cost of keeping one tensor pair alive. `kl.py` memoizes dispatch results in `_KL_MEMOIZE`, reducing repeated subclass-resolution overhead after the first KL lookup for a type pair.

## Design Rationale

The package separates mathematical interfaces from tensor execution: distribution classes express probability formulas while `torch` and ATen perform numerical work. Constraints encode argument validity and support in reusable objects, so transforms, distributions, and parameter registries share one validation vocabulary. The KL registry uses decorators instead of methods so new distribution pairs can define cross-type formulas without changing either distribution class.
