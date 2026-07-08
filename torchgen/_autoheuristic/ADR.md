# `torchgen/_autoheuristic`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`torchgen/_autoheuristic` trains and emits learned heuristics for compiler-time choice selection. It owns benchmark runners, log parsing, feature engineering, decision and regression tree training, evaluation, and per-optimization recipes such as `mm`, `mixed_mm`, and `pad_mm`.

## Key Files

| File | Purpose |
|---|---|
| `README.md` | Documents the end-to-end AutoHeuristic workflow for collecting data, training a model, and generating an artifact |
| `train.py` | Defines the `AHTrain` base class, argument parsing, log deserialization, categorical encoding, and artifact writing |
| `train_decision.py` | Implements `AHTrainDecisionTree`, grid search, safety-threshold selection, evaluation, and decision-tree codegen |
| `train_regression.py` | Implements `AHTrainRegressionTree`, grouped timing reduction, thresholding, and regression-tree codegen |
| `ah_tree.py` | Wraps sklearn trees in a custom `DecisionTree` that supports pruning, DOT export, and Python code generation |
| `benchmark_runner.py` | Defines the benchmark collection harness that toggles autoheuristic environment variables and repeatedly runs benchmarks |
| `benchmark_utils.py` | Supplies matrix-shape and tensor helpers used by benchmark collection scripts |

## Public Interface

This subtree exposes `BenchmarkRunner`, `AHTrain`, `AHTrainDecisionTree`, `AHTrainRegressionTree`, and `DecisionTree` as the main programmatic entry points. Users run `generate_heuristic.sh`, `collect_data.sh`, `merge_data.py`, `train_decision.py`, `train_regression.py`, and the per-optimization scripts under `mm/`, `mixed_mm/`, and `pad_mm/`. Generated heuristics are emitted through `write_heuristic_to_file()` into `torch/_inductor/autoheuristic/artifacts/_<heuristic_name>.py`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch/_inductor](torch/_inductor/ADR.md) | mutual | `benchmark_runner.py` sets `torch._inductor.config.autoheuristic_log_path`, `train.py` and `train_regression.py` import `torch._inductor.autoheuristic.autoheuristic_utils`, and generated heuristics target `torch/_inductor/autoheuristic/artifacts/` |
| [torchgen](torchgen/ADR.md) | depended-on-by | This subtree extends torchgen with heuristic-training and artifact-generation workflows for code-generated compiler choices |

## Runtime Behaviour

`BenchmarkRunner.run()` sets `TORCHINDUCTOR_AUTOHEURISTIC_USE` or `TORCHINDUCTOR_AUTOHEURISTIC_COLLECT`, updates `torch._inductor.config.autoheuristic_log_path`, optionally calls `torch.cuda.set_device()`, and then drives `create_input()` plus `run_benchmark()` inside `main()`. `AHTrain.parse_log()` and `deserialize_data()` read the JSON metadata header with `get_metadata_str_from_log()`, load the CSV rows with pandas, and one-hot encode categorical features through `handle_categorical_features()`. `AHTrainDecisionTree.train_and_evaluate_models()` sweeps `max_depth`, `min_samples_leaf`, and `criterion`, prunes ranking trees with `DecisionTree.prune()`, computes safe probability thresholds with `DecisionEvaluator`, and emits Python heuristics through `codegen()` and `write_heuristic_to_file()`. `AHTrainRegressionTree.get_df()` groups measurements by feature set and choice, filters unstable samples with `relative_std`, derives `winner`, `speedup`, and `target`, and then writes a regression-style artifact through `dt_to_python()`.

## Performance Profile

- **Allocation sites** - Training loads full log files into pandas DataFrames, expands categorical columns with `pd.get_dummies()`, and creates per-model result tables in `train_and_evaluate_models()`. Artifact generation allocates one Python source buffer per heuristic in `write_heuristic_to_file()`.
- **Synchronization costs** - The training path is single-process Python around pandas and sklearn, so it pays CPU compute cost instead of lock contention. Benchmark collection in `BenchmarkRunner.main()` is intentionally serialized, because it repeats the same input `num_reps` times to collect stable timing feedback.
- **Data movement** - `deserialize_data()` reads CSV logs from disk into DataFrames, the trainers repeatedly split them into train, validation, and test subsets, and generated heuristics are written back out under `torch/_inductor/autoheuristic/artifacts/`. The benchmark helpers create random tensors with `torch.randn()` and feed them directly into Inductor autotuning paths.
- **Redundant or repeated work** - `train_and_evaluate_models()` reruns predictions and evaluation across every dataset for each hyperparameter combination. `BenchmarkRunner.main()` also replays each sampled input multiple times, which increases collection time but reduces noise in the logged feedback that later drives `winner` and `target` selection.

## Design Rationale

PyTorch keeps AutoHeuristic training next to torchgen because the output is generated source code, not just offline analysis. The code separates reusable training infrastructure in `train.py` and `ah_tree.py` from optimization-specific recipes under `mm/`, `mixed_mm/`, and `pad_mm/`, so each optimization can customize features and safety logic without forking the full pipeline.
