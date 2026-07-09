# `torch/_higher_order_ops`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torch/_higher_order_ops` defines compiler-aware higher-order operators that carry Python callables, FX graph regions, structured control flow, Triton kernels, and custom execution regions through PyTorch dispatch. These operators let Dynamo, FX, AOTAutograd, functionalization, fake tensor mode, and Inductor preserve program structure that ordinary ATen operator calls cannot represent.

## Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Re-exports higher-order operators such as `cond`, `while_loop`, `invoke_subgraph`, `flex_attention`, `scan`, and `map` |
| `triton_kernel_wrap.py` | Wraps Triton kernels in higher-order operators with side tables, functionalization, fake, proxy, and dense implementations |
| `invoke_subgraph.py` | Represents nested compile regions and subgraph invocation with forward/backward pairing metadata |
| `flex_attention.py` | Implements `flex_attention` and `flex_attention_backward` HOPs, including autocast, autograd, fake, proxy, and functionalization paths |
| `utils.py` | Provides shared graph tracing, alias/mutation checks, fake registration, and saved-value utilities for HOP implementations |
| `cond.py` | Implements the structured `cond` operator and validates branch graphs, aliases, mutation, fake, proxy, autograd, and vmap paths |
| `while_loop.py` | Implements the structured `while_loop` operator and stacked-output variant across dispatch modes |

## Public Interface

`__init__.py` exports `cond`, `switch`, `while_loop`, `invoke_subgraph`, `scan`, `map`, `flex_attention`, `flex_attention_backward`, `BaseHOP`, `flat_apply`, `foreach_map`, `_foreach_map`, `flex_gemm`, `with_effects`, `auto_functionalized`, `auto_functionalized_v2`, `associative_scan`, `out_dtype`, `executorch_call_delegate`, `call_torchbind`, `run_const_graph`, `InvokeQuant`, `invoke_leaf_function`, `invoke_quant`, `invoke_quant_packed`, `wrap_with_set_grad_enabled`, `wrap_with_autocast`, `wrap_activation_checkpoint`, `strict_mode`, `aoti_call_delegate`, `local_map_hop`, `print`, `inductor_compiled_code`, and `inline_asm_elementwise`. Important concrete operator objects include `cond_op`, `while_loop_op`, `invoke_subgraph`, `flex_attention`, `flex_attention_backward`, `triton_kernel_wrapper_mutation`, and `triton_kernel_wrapper_functional`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch](torch/ADR.md) | depends-on | HOP classes inherit from `torch._ops.HigherOrderOperator` and register Python implementations for dispatcher keys |
| [torch/fx](torch/fx/ADR.md) | depends-on | Operators materialize callables as `torch.fx.GraphModule` instances and integrate with proxy tensor tracing |
| [torch/_dynamo](torch/_dynamo/ADR.md) | depended-on-by | Dynamo captures HOPs such as `cond`, `while_loop`, Triton wrappers, and nested compile regions instead of inlining Python control flow |
| [torch/_functorch](torch/_functorch/ADR.md) | depends-on | Autograd and vmap implementations use AOTAutograd, functionalization, batch-dim utilities, and partitioning helpers |
| [torch/_inductor](torch/_inductor/ADR.md) | depended-on-by | Inductor consumes HOP metadata for Triton kernels, flex attention lowering, and nested compiled subgraphs |
| [torch/utils](torch/utils/ADR.md) | depends-on | Implementations rely on pytree helpers, checkpoint dispatch modes, debug mode, and ordered-set utilities |

## Runtime Behaviour

Each operator subclasses or instantiates `HigherOrderOperator`, validates callable operands and tensor arguments, and then routes execution through dispatch-key-specific Python implementations. `cond.py` and `while_loop.py` trace branch or loop bodies for `ProxyTorchDispatchMode`, run fake-mode shape checks, and register autograd and functionalization handlers. `invoke_subgraph.py` stores `NestedCompileRegionOptions`, assigns per-call IDs through thread-local state, uses `InvokeSubgraphAutogradOp` for backward handling, and stamps FX node metadata that downstream passes can pair. `triton_kernel_wrap.py` stores kernels and constant args in `KernelSideTable` because FX nodes cannot directly hold arbitrary Triton kernel objects.

## Performance Profile

Higher-order operators introduce compile-time cost because they materialize Python callables as FX graphs, check aliasing and mutation, and install fake, proxy, autograd, vmap, and functionalization handlers. They preserve performance opportunities by keeping control flow, nested regions, and Triton kernel launches explicit for Inductor rather than erasing them into eager Python execution. `KernelSideTable` keeps Triton kernel lookup and ID lookup O(1) behind a lock for insertion, while read access avoids locking. `flex_attention.py` enforces stride properties such as last-dimension unit stride and carries `kernel_options` so generated attention kernels can choose efficient memory layouts.

## Design Rationale

The directory isolates higher-order semantics from both ATen primitive operators and end-user `torch` namespace definitions. This separation lets each HOP define precise behavior for eager execution, fake tensor propagation, proxy tracing, functionalization, autograd, and compiler lowering without changing the generic dispatcher contract. The design also gives experimental structured control flow and compiler-only constructs a shared implementation pattern before they become stable public APIs.
