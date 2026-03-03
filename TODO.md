# calcine — TODO

Priority tiers: **P0** = blocking real use, **P1** = significant gap, **P2** = quality of life, **P3** = stretch.

---

## Core value proposition

calcine's actual differentiation is in **pipeline orchestration** (concurrency, per-entity error isolation, incremental generation, reporting) and the **schema validation system**. The built-in sources and file-based stores are conveniences, not the product. Design decisions should reinforce this.

---

## P0 — Core functional gaps

- [x] **Partitioned concurrency in `generate()`** — `generate()` currently iterates entities serially. Need a flexible model that supports three modes without changing the rest of the API:

  1. **Flat concurrency** — `concurrency: int` acts as a semaphore cap; up to N entities processed concurrently from a shared pool.
  2. **Partition function** — `partition_by: Callable[[str], Hashable]` groups entity IDs by the return value. Each group is a serial queue; up to `concurrency` groups run concurrently. Handles rate-limit-per-account, shared-shard writes, ordered-per-user processing, etc.
  3. **Explicit partitions** — `partitions: dict[str, list[str]]` lets the caller supply pre-built groups directly (useful when partition structure comes from external metadata).

  Target API:
  ```python
  # flat
  await pipeline.generate(entity_ids=ids, concurrency=16)
  # partitioned by function
  await pipeline.generate(entity_ids=ids, partition_by=lambda eid: eid.split("_")[0], concurrency=8)
  # partitioned explicitly
  await pipeline.generate(partitions={"shard_0": [...], "shard_1": [...]}, concurrency=4)
  ```
  Default (`concurrency=1`, no partition args) stays fully serial so existing behaviour is unchanged.

- [x] **Incremental generation (`only_missing` mode)** — Add an `overwrite: bool = True` parameter to `generate()`. When `False`, skip entities where `store.exists()` returns True. Makes re-running pipelines cheap.

- [x] **Progress reporting** — Add a `on_progress` callback parameter to `generate()` that receives `(completed: int, total: int, report_so_far: GenerationReport)` after each entity. Lets callers wire in tqdm, logging, or custom telemetry without coupling calcine to any specific library.

- [ ] **Multi-feature pipeline** — Support running multiple `Feature` instances against a single `DataSource` in one `Pipeline`. The source is read once; each feature gets the same raw data. Avoids re-reading the source N times for N features from the same origin. This is the most common real-world pattern and a significant gap in the current API.

---

## P1 — Significant feature gaps

- [x] **Batch extract** — Add an optional `async extract_batch(raws: list[Any], context: dict) -> list[Any]` method to `Feature`. When defined, `generate()` should collect a batch of raw reads and call it once instead of N times. Critical for ML model inference and bulk DB queries.

- [ ] **Feature versioning** — Allow a `version: str` class attribute on `Feature` that is included in the store key, so `MeanPurchaseValue_v2` coexists with `MeanPurchaseValue_v1` without collision. This is what separates a pipeline runner from an actual feature store; without it, re-extracting a feature silently overwrites prior results.

- [ ] **Schema statistical constraints** — Extend the schema type system with value-level validation: range bounds (`min`/`max` for numeric types), `allow_inf: bool` on Float types, and finite-only enforcement. The NaN fix was a correctness patch; this makes the schema genuinely useful as a data quality gate rather than just a shape/type checker.

- [x] **Fix NaN in Float32/Float64 validation** — `float("nan")` currently passes. Add `allow_nan: bool = False` parameter (default False for new code, keep True as opt-in). This is a correctness bug, not just a limitation.

- [ ] **SQLite store** — A `SQLiteStore(path)` that persists features to a single SQLite file keyed by `(feature_name, entity_id)`. Persistent, zero-config, no directory explosion, portable. The right default persistent store; `FileStore` should be de-emphasized in docs in favour of this once it exists.

- [ ] **`GenerationReport` per-phase timing** — Track wall time for source read, extract, and store write separately per entity. Surface aggregates (p50/p95/max per phase) in `GenerationReport`. This turns calcine from a pipeline runner into a bottleneck analysis tool — a real reason to adopt it over a hand-rolled loop.

- [ ] **Fault-tolerant SourceBundle** — Add `SourceBundle(..., fault_tolerant: bool = False)`. When enabled, a failing sub-source returns `None` for its key rather than propagating the exception. Lets features degrade gracefully when optional sources are unavailable.

- [ ] **ParquetStore append optimization** — Current implementation loads the entire Parquet file, modifies a row, and rewrites. Replace with an append-and-deduplicate strategy (or partition by entity) to make writes O(1) rather than O(n).

---

## P2 — Quality of life

- [ ] **Cross-field schema validation** — Add an optional `validate_record(self, record: dict) -> list[str]` hook to `FeatureSchema` (or `FeatureType`) for constraints that span multiple fields (e.g., `end_time > start_time`).

- [ ] **Simplify Feature API — deprecate `pre_extract` / `post_extract`** — These lifecycle hooks add cognitive overhead for minimal gain; users put this logic in `extract`. Deprecate them and document `extract` as the single transformation point. Reduces the framework's apparent surface area and the feeling that calcine is imposing structure for its own sake.

- [ ] **Demote built-in sources to examples/extras** — `FileSource`, `DirectorySource`, `HTTPSource`, and `DataFrameSource` are thin wrappers that create a false impression of completeness. Move them to a `calcine.contrib` subpackage or clearly document them as reference implementations, not production components. The `DataSource` ABC is the product; the built-ins are scaffolding.

- [ ] **`retrieve_batch` concurrency** — `retrieve_batch()` currently iterates stores serially. Fan it out with `asyncio.gather`.

- [ ] **`Pipeline` async context manager** — Support `async with Pipeline(...) as p:` so stores that need setup/teardown (e.g., connection pools) can manage their lifecycle cleanly.

- [ ] **Store inspection without a Feature reference** — Add a `FeatureStore.list_features() -> list[str]` and `FeatureStore.list_entities(feature_name: str) -> list[str]` to make stores introspectable without needing a live `Feature` instance.

- [ ] **HTTPSource retry/backoff** — Add `retries: int = 0` and `backoff: float = 0.5` parameters to `HTTPSource`. Retry on transient HTTP errors (5xx, timeouts) with exponential backoff.

---

## P3 — Stretch / future

- [ ] **S3 / object storage source and store** — `S3Source` and `S3Store` via `boto3`/`aiobotocore` as an optional extra `calcine[s3]`.

- [ ] **Redis store** — `RedisStore` for low-latency feature serving as an optional extra `calcine[redis]`.

- [ ] **Feature registry** — A lightweight `FeatureRegistry` that maps feature names to classes, enabling pipeline construction from config files or CLI invocations.

- [ ] **CLI** — `calcine generate`, `calcine inspect`, `calcine delete` commands for operating on stores from the terminal without writing Python.

- [ ] **Stream-mode generate** — Instead of collecting all `entity_ids` upfront, accept an `AsyncIterator[str]` so pipelines can process infinite or externally-driven entity streams (Kafka, webhooks, etc.).
