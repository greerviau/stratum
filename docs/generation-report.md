# GenerationReport

`GenerationReport` is returned by `Pipeline.generate()` and `Pipeline.agenerate()`.
It summarises every entity that was processed: succeeded, failed, or skipped.

```python
report = pipeline.generate(entity_ids=ids)
print(report)
# GenerationReport(entities=980, records=4200, failed=18, skipped=2, duration=3.41s)
```

---

## Fields

| Field | Type | Description |
|-------|------|-------------|
| `success_count` | `int` | Entities that completed successfully. Always populated. |
| `record_count` | `int` | Total records written across all succeeded entities. Equals `success_count` for standard 1-to-1 features; larger for fan-out features. |
| `duration_s` | `float` | Wall-clock seconds for the entire `generate()` call. |
| `succeeded` | `dict[str, ExtractionResult]` | Extracted results keyed by entity ID. Only populated when `store_results=True` (the default). |
| `failed` | `dict[str, list[str]]` | Error messages keyed by entity ID, for every entity that failed. |
| `exceptions` | `dict[str, BaseException]` | Raw exceptions keyed by entity ID, for failures caused by unhandled exceptions. Empty for schema violations (which produce `failed` entries but no exception). |
| `skipped` | `set[str]` | Entity IDs skipped because a stored value already existed and `overwrite=False` was passed. |

## Properties

| Property | Description |
|----------|-------------|
| `failure_count` | `len(failed)` |
| `skip_count` | `len(skipped)` |
| `total_count` | `success_count + failure_count + skip_count` — every entity touched |
| `throughput` | `success_count / duration_s`. `0.0` when `duration_s` is zero. |

`len(report)` returns `total_count`.

---

## Checking outcomes

```python
report = pipeline.generate(entity_ids=ids)

print(f"{report.success_count} succeeded, {report.failure_count} failed")
print(f"{report.record_count} total records at {report.throughput:.1f} entities/s")

if report.failed:
    print("Failed entities:", list(report.failed.keys()))
```

---

## Inspecting failures

`failed` maps each entity to its list of error strings:

```python
for entity_id, errors in report.failed.items():
    print(entity_id, errors)
# user_42  ["Field 'score': Value is NaN"]
# user_99  ["Field 'label': Expected str, got NoneType"]
```

For unhandled exceptions (bugs, network errors, etc.) use `exceptions` to get
the raw traceback:

```python
import traceback

for entity_id, exc in report.exceptions.items():
    print(f"--- {entity_id} ---")
    print("".join(traceback.format_exception(exc)))
```

### error_summary()

When processing thousands of entities, the same error often repeats.
`error_summary()` groups `failed` by error message and sorts by frequency:

```python
for message, entity_ids in report.error_summary().items():
    print(f"{len(entity_ids):4d}x  {message}")
# 487x  Field 'score': Value is None but field is not nullable
#   3x  Field 'label': Expected str, got NoneType
#   1x  Unhandled exception in pipeline for feature 'MyFeature', entity 'u_7': ...
```

This makes systematic failures (a schema regression, a bad data segment) immediately
visible without scrolling through thousands of individual entries.

---

## Exporting to a DataFrame

`to_dataframe()` returns a pandas DataFrame with one row per entity,
covering succeeded, failed, and skipped outcomes.

```python
df = report.to_dataframe()
#    entity_id    status  record_count   error
# 0     user_1  succeeded           1    None
# 1     user_2     failed        None   Field 'score': ...
# 2     user_3    skipped        None    None
```

Columns:

| Column | Type | Notes |
|--------|------|-------|
| `entity_id` | `str` | The entity ID |
| `status` | `str` | `"succeeded"`, `"failed"`, or `"skipped"` |
| `record_count` | `int \| None` | Records written; `None` for failed/skipped |
| `error` | `str \| None` | Errors joined by `"; "`; `None` for succeeded/skipped |

Requires pandas (`pip install calcine[parquet]`).

> **Note:** Succeeded rows only appear when `store_results=True` was used during
> `generate()`. If `store_results=False`, `report.succeeded` is empty and
> `to_dataframe()` will not include succeeded rows — use `success_count` for
> the count instead.

---

## Controlling memory usage with store_results

By default, `generate()` stores the full `ExtractionResult` for every
succeeded entity in `report.succeeded`. For large runs this can consume
significant memory.

Pass `store_results=False` to suppress this:

```python
report = pipeline.generate(entity_ids=ids, store_results=False)

# Counts and failures still work:
print(report.success_count, report.failure_count)

# But succeeded is empty — retrieve from the store instead:
value = pipeline.retrieve("user_42")
```

`success_count`, `record_count`, `failed`, `exceptions`, and `skipped` are
always populated regardless of `store_results`.

---

## Per-phase timing

`timing_summary()` returns p50, p95, max, mean, and total wall-clock seconds
for each pipeline phase — source read, feature extract, and store write.
Only succeeded entities contribute.

```python
report = pipeline.generate(entity_ids=ids)

summary = report.timing_summary()
# {"read": {"p50": 0.003, "p95": 0.018, "max": 0.12, "mean": 0.005, "total": 5.1},
#  "extract": {...},
#  "write": {...}}

print(f"p95 read:    {summary['read']['p95']*1000:.1f} ms")
print(f"p95 extract: {summary['extract']['p95']*1000:.1f} ms")
print(f"p95 write:   {summary['write']['p95']*1000:.1f} ms")
```

This makes it easy to identify the bottleneck in a pipeline — if `read` dominates,
look at the data source; if `extract` dominates, the feature computation is the
bottleneck; if `write` dominates, look at store I/O.

Raw per-entity timings are available in `report.phase_timings` if you need custom
aggregation:

```python
import statistics
report.phase_timings["extract"]          # list of floats (one per succeeded entity)
statistics.median(report.phase_timings["read"])
```

### Batch mode note

When using `batch_size > 1`, the `"extract"` time is the total `extract_batch()`
duration divided by the batch size — a per-entity average, not an individually
measured value.  `"read"` and `"write"` are always measured per entity.

---

## Re-running failed entities

`report.failed` is a plain dict, so retrying failures is straightforward:

```python
report = pipeline.generate(entity_ids=ids)

if report.failed:
    retry_report = pipeline.generate(entity_ids=list(report.failed.keys()))
```
