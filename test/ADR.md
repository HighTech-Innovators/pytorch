# `test`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`test` owns the top-level Python and C++ validation suites for PyTorch. It defines the test entry points, platform-specific selection logic, and the repository conventions for running targeted tests instead of the whole suite.

## Key Files

| File | Purpose |
|---|---|
| `run_test.py` | Main test runner that selects tests, applies platform blocklists, shards work, and launches subprocesses |
| `conftest.py` | Shared pytest configuration for tests that rely on pytest collection hooks |
| `pytest_shard_custom.py` | Custom sharding helpers used by pytest-based workflows |
| `run_doctests.sh` | Shell entry point for documentation doctest execution |
| `test_torch.py` | Large umbrella regression suite for public tensor and operator behaviour |

## Public Interface

`run_test.py` exposes the repository's main test CLI and the `TestChoices` container used for argument validation. The broader test surface also depends on `torch.testing._internal.common_utils.TestCase`, `run_tests()`, and `TestEnvironment`, which supply the standard base class, runner, and environment-flag registry used by most files under `test/`.

## Dependencies

| Component | Direction | Nature |
|---|---|---|
| [torch](torch/ADR.md) | depends-on | `run_test.py` imports `torch`, `torch.distributed`, and feature flags such as `torch._C.has_lapack` to decide which tests can run |
| [tools](tools/ADR.md) | depends-on | `run_test.py` imports `tools.stats.*`, `tools.testing.discover_tests`, `tools.testing.test_run`, and target-determination helpers |
| [functorch](functorch/ADR.md) | depended-on-by | Selection logic in `run_test.py` has an explicit `--functorch` path and the suite contains `functorch/` test groups |

## Runtime Behaviour

`run_test.py` prepends `REPO_ROOT` to `sys.path`, imports test-discovery and statistics helpers from `tools/`, then removes the temporary path entry after those imports complete. When the runner processes CLI options, it filters `selected_tests` by focused groups such as `--functorch`, `--onnx`, `--cpp`, `--mps`, and `--xpu`, extends `WINDOWS_BLOCKLIST`, `ROCM_BLOCKLIST`, and `S390X_BLOCKLIST`, and excludes tests when backends like distributed or LAPACK are unavailable.

The runner also handles backend-specific execution policy. `maybe_set_hip_visible_devies()` assigns a distinct `HIP_VISIBLE_DEVICES` index to ROCm worker processes, and `torch.testing._internal.common_utils.TestEnvironment.def_flag()` and `def_setting()` register environment-driven test flags while `torch.backends.disable_global_flags()` freezes backend-global mutability on import.

## Performance Profile

- **Allocation sites** - Test selection builds large Python lists such as `selected_tests`, blocklists, and shard descriptions, and XML or JSON artifact upload paths allocate report payloads only when CI helpers are enabled.
- **Synchronization costs** - `run_test.py` coordinates subprocess pools, distributed availability checks, and ROCm worker GPU assignment; `maybe_set_hip_visible_devies()` exists specifically to avoid GPU oversubscription across parallel workers.
- **Data movement** - The runner moves results through subprocess boundaries, uploads XML-derived JSON when CI helpers are present, and passes environment-variable state into child test processes for reproduction.
- **Redundant or repeated work** - The selection pipeline intentionally prunes large parts of the tree before launch, because the repository policy is to run focused files or classes rather than the entire suite.

## Design Rationale

PyTorch centralizes test selection in one runner because backend availability, platform quirks, and CI heuristics must stay consistent across thousands of files. The split between `test/` entry points and `torch.testing._internal.common_utils` lets individual tests stay small while one shared runner owns sharding, blocklists, and reproducible environment handling.
