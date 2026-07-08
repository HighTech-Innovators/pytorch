# `tools/stats`

- [Role](#role)
- [Key Files](#key-files)
- [Public Interface](#public-interface)
- [Dependencies](#dependencies)
- [Runtime Behaviour](#runtime-behaviour)
- [Performance Profile](#performance-profile)
- [Design Rationale](#design-rationale)

## Role

`tools/stats` turns CI artifacts and live host telemetry into normalized JSON records for dashboards and backend storage. It owns artifact download, XML parsing, disabled-test analysis, test-time caching, and utilization logging.

## Key Files

| File | Purpose |
|---|---|
| `upload_stats_lib.py` | Shared helpers for downloading artifacts, unzipping archives, normalizing JSON payloads, and uploading to S3 or DynamoDB |
| `upload_test_stats.py` | Parses JUnit XML reports into test-case records and aggregated summaries for workflow uploads |
| `import_test_stats.py` | Downloads and caches generated test-time, test-rating, and disabled-test metadata used by CI selection logic |
| `monitor.py` | Continuously samples CPU, memory, process, and GPU utilization and prints structured JSON records |
| `check_disabled_tests.py` | Analyzes rerun-disabled-tests reports and emits re-enable or still-flaky decisions |
| `utilization_stats_lib.py` | Defines the dataclasses and schema helpers used by `monitor.py` and downstream storage |

## Public Interface

This directory exposes CLI entry points in `upload_test_stats.py`, `check_disabled_tests.py`, `export_test_times.py`, and `monitor.py`. Other scripts import `download_s3_artifacts()`, `download_gha_artifacts()`, `upload_to_s3()`, `upload_workflow_stats_to_s3()`, `process_xml_element()`, `get_test_times()`, `get_test_class_times()`, and `get_disabled_tests()`. `utilization_stats_lib.py` exposes `UtilizationMetadata`, `UtilizationRecord`, `RecordData`, `GpuUsage`, `UtilizationStats`, and `WorkflowInfo` as the structured schema for telemetry.

## Dependencies

There are no notable src-local ADR-tracked dependencies. The scripts compose one another inside `tools/stats/` and talk directly to GitHub Actions, S3, DynamoDB, NVML, AMD SMI, and XML test reports.

## Runtime Behaviour

`upload_stats_lib.py` paginates GitHub Actions artifacts in `_get_artifact_urls()`, downloads them with `_download_artifact()`, expands zip archives with `unzip()`, and strips NaN or Inf values out of nested payloads in `remove_nan_inf()`. `upload_test_stats.py` walks every `<testcase>` element in `parse_xml_report()`, preserves the report directory as `invoking_file`, aggregates class-level counts in `summarize_test_cases()`, and uploads serialized results through `upload_workflow_stats_to_s3()` or `upload_to_s3()`. `check_disabled_tests.py` parses rerun-disabled-tests XML in `process_report()`, counts `num_green` and `num_red` by `name;classname;filename`, and writes one record per disabled test in `save_results()`. `monitor.py` runs `_collect_data()` in a background thread, reduces each logging window in `_output_data()`, and prints `UtilizationMetadata` plus `UtilizationRecord` JSON built from `UtilizationStats` and `GpuUsage`.

## Performance Profile

- **Allocation sites** - `upload_test_stats.py` creates one Python dict per XML testcase in `process_xml_element()` and stores flattened lists before upload. `monitor.py` appends one `UsageData` object per collection interval and then materializes `GpuUsage` and `UtilizationRecord` objects for each output window.
- **Synchronization costs** - `SharedResource` protects producer and consumer state with a `threading.Lock`, and `upload_test_stats.py` waits for `Pool.join()` before flattening all parsed XML results. Network-bound uploads in `upload_to_s3()` and `upload_to_dynamodb()` dominate wall time after parsing completes.
- **Data movement** - `upload_to_s3()` serializes newline-delimited JSON and gzips it before upload, while `unzip()` expands entire artifacts onto disk before later passes glob XML files. `monitor.py` copies sampled process and GPU lists out of `SharedResource` with `deepcopy()` before aggregation.
- **Redundant or repeated work** - `fetch_and_cache()` reuses downloaded metadata files for three hours and `get_job_name()` memoizes workflow job lookups, which cuts repeated API traffic. `get_tests()` still reparses every XML report for each workflow run, because the pipeline rebuilds JSON from raw reports instead of storing a reusable intermediate form.

## Design Rationale

PyTorch keeps this logic in standalone Python scripts so GitHub workflows can normalize CI data without embedding storage logic in every job. The design in `README.md` deliberately routes raw artifacts through an intermediate store first, because `upload-test-stats.yml` can hold backend credentials while untrusted PR jobs only need permission to upload artifacts.
