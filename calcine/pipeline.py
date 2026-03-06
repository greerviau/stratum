"""Pipeline: ties DataSource → Feature → FeatureStore together."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Hashable
from concurrent.futures import Executor
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .extraction import ExtractionResult
from .features.base import Feature
from .sources.base import DataSource
from .stores.base import FeatureStore

# ---------------------------------------------------------------------------
# Module-level executor worker functions
# ---------------------------------------------------------------------------
# These must be defined at module level (not as closures) so that
# ProcessPoolExecutor can pickle them for inter-process dispatch.


async def _validate_extraction(feature: Feature, result: ExtractionResult) -> list[str]:
    """Validate metadata and all records in an ``ExtractionResult``.

    Returns a non-empty list of error strings on failure, empty on success.
    For results with more than one record, errors are prefixed with the
    sub-entity ID so callers can identify which record failed.
    """
    if feature.metadata_schema is not None and result.metadata is not None:
        errors = feature.metadata_schema.validate(result.metadata)
        if errors:
            return errors
    multi = len(result.records) > 1
    all_errors: list[str] = []
    for sub_id, record in result.records.items():
        for e in await feature.validate(record):
            all_errors.append(f"sub-entity '{sub_id}': {e}" if multi else e)
    return all_errors


def _run_entity_in_executor(
    source: DataSource,
    feature: Feature,
    entity_id: str,
    context: dict[str, Any],
) -> tuple[ExtractionResult, list[str], dict[str, float]]:
    """Run one entity's extract pipeline stages in a thread or process.

    Executes: ``source.read`` → ``feature.extract`` → validate.

    Store writes are intentionally excluded so they remain in the main
    process (preserving correct behaviour for in-memory and async stores).

    Returns:
        ``(result, errors, phase_times)`` where *errors* is non-empty on
        validation failure and empty on success, and *phase_times* is a
        dict with keys ``"read"`` and ``"extract"`` (wall-clock seconds).

    Raises:
        Any exception propagated from the pipeline stages.  The caller
        records it as an unhandled pipeline error.
    """

    async def _work() -> tuple[ExtractionResult, list[str], dict[str, float]]:
        t0 = time.perf_counter()
        raw = await source.read(entity_id=entity_id, context=context)
        t1 = time.perf_counter()
        result = await feature.extract(raw, context, entity_id=entity_id)
        t2 = time.perf_counter()
        errors = await _validate_extraction(feature, result)
        return result, errors, {"read": t1 - t0, "extract": t2 - t1}

    return asyncio.run(_work())


def _run_batch_in_executor(
    source: DataSource,
    feature: Feature,
    entity_ids: list[str],
    context: dict[str, Any],
    entity_contexts: list[dict[str, Any]] | None,
) -> list[tuple[ExtractionResult, list[str], dict[str, float]] | BaseException]:
    """Run one batch's extract pipeline stages in a thread or process.

    Mirrors the logic of ``Pipeline._process_batch`` steps 2–4
    (concurrent reads, extract_batch, validate).
    Store writes and report updates are handled by the caller in the main
    process.

    Returns:
        One entry per item in *entity_ids*, in order:

        - ``(result, [], phase_times)`` — success
        - ``(result, errors_list, phase_times)`` — validation failure
        - ``BaseException`` — unhandled error for that entity

        *phase_times* contains ``"read"`` and ``"extract"`` seconds.
        The ``"extract"`` value is the total ``extract_batch`` duration
        divided by the number of valid entities (per-entity average).

    A whole-batch ``extract_batch`` failure is caught and stored as a
    ``BaseException`` in every valid slot, preserving the same per-entity
    failure isolation as the non-executor path.
    """

    async def _work() -> list[tuple[ExtractionResult, list[str], dict[str, float]] | BaseException]:
        # --- Concurrent reads (timed individually) ---
        async def _timed_read(eid: str, ctx: dict) -> tuple[Any, float]:
            t0 = time.perf_counter()
            raw = await source.read(entity_id=eid, context=ctx)
            return raw, time.perf_counter() - t0

        raw_timed: list[tuple[Any, float] | BaseException] = await asyncio.gather(
            *[
                _timed_read(
                    eid,
                    entity_contexts[i] if entity_contexts is not None else context,
                )
                for i, eid in enumerate(entity_ids)
            ],
            return_exceptions=True,
        )

        results: dict[
            int, tuple[ExtractionResult, list[str], dict[str, float]] | BaseException
        ] = {}
        valid_indices: list[int] = []
        valid_raws: list[Any] = []
        read_times: dict[int, float] = {}

        # --- Filter read failures ---
        for idx, res in enumerate(raw_timed):
            if isinstance(res, BaseException):
                results[idx] = res
                continue
            raw, read_time = res
            read_times[idx] = read_time
            valid_indices.append(idx)
            valid_raws.append(raw)

        if not valid_indices:
            return [results[i] for i in range(len(entity_ids))]

        valid_ids = [entity_ids[i] for i in valid_indices]
        valid_entity_ctxs = (
            [entity_contexts[i] for i in valid_indices] if entity_contexts is not None else None
        )

        # --- Batch extract (timed; divide by batch size for per-entity average) ---
        t_extract_0 = time.perf_counter()
        try:
            batch_results: list[ExtractionResult | BaseException] = await feature.extract_batch(
                valid_raws, context, entity_ids=valid_ids, entity_contexts=valid_entity_ctxs
            )
        except Exception as exc:
            for idx in valid_indices:
                results[idx] = exc
            return [results[i] for i in range(len(entity_ids))]
        extract_time_each = (time.perf_counter() - t_extract_0) / len(valid_indices)

        # --- Validate ---
        for idx, result in zip(valid_indices, batch_results, strict=True):
            if isinstance(result, BaseException):
                results[idx] = result
                continue
            try:
                errors = await _validate_extraction(feature, result)
                results[idx] = (
                    result,
                    errors,
                    {"read": read_times.get(idx, 0.0), "extract": extract_time_each},
                )
            except Exception as exc:
                results[idx] = exc

        return [results[i] for i in range(len(entity_ids))]

    return asyncio.run(_work())


@dataclass
class GenerationReport:
    """Summary of a ``Pipeline.generate()`` run.

    Attributes:
        succeeded: Mapping of ``entity_id`` to :class:`~calcine.ExtractionResult`
            for every entity that was processed without error.  Only populated
            when ``generate()`` is called with ``store_results=True`` (default).
            Use ``success_count`` to count successes regardless of *store_results*.
        failed: Mapping of ``entity_id`` to a list of error strings for
            every entity whose processing failed (source read error,
            extraction error, or schema violation).
        exceptions: Mapping of ``entity_id`` to the raw ``BaseException`` for
            failures caused by unhandled exceptions (not schema violations).
            Useful for accessing the full traceback via
            ``traceback.format_exception(report.exceptions[eid])``.
        skipped: Set of entity IDs that were skipped because a stored value
            already existed and ``overwrite=False`` was passed to
            ``generate()``.
        success_count: Number of entities that succeeded.  Always populated
            regardless of *store_results*.
        record_count: Total records produced across all succeeded entities.
            Equals *success_count* for single-record features; larger for
            fan-out features.  Always populated.
        duration_s: Wall-clock seconds for the entire ``generate()`` call.
        phase_timings: Raw per-entity wall-clock seconds for each pipeline
            phase, keyed by ``"read"``, ``"extract"``, and ``"write"``.
            Only populated for succeeded entities.  Use
            :meth:`timing_summary` for aggregated statistics rather than
            reading this directly.
    """

    succeeded: dict[str, ExtractionResult] = field(default_factory=dict)
    failed: dict[str, list[str]] = field(default_factory=dict)
    exceptions: dict[str, BaseException] = field(default_factory=dict)
    skipped: set[str] = field(default_factory=set)
    success_count: int = 0
    record_count: int = 0
    duration_s: float = 0.0
    phase_timings: dict[str, list[float]] = field(
        default_factory=lambda: {"read": [], "extract": [], "write": []},
        repr=False,
    )

    @property
    def failure_count(self) -> int:
        """Number of entities that failed processing."""
        return len(self.failed)

    @property
    def skip_count(self) -> int:
        """Number of entities skipped because a value already existed in the store."""
        return len(self.skipped)

    @property
    def total_count(self) -> int:
        """Total entities touched: succeeded + failed + skipped."""
        return self.success_count + self.failure_count + self.skip_count

    @property
    def throughput(self) -> float:
        """Successfully processed entities per second. ``0.0`` if duration is zero."""
        return self.success_count / self.duration_s if self.duration_s > 0 else 0.0

    def __len__(self) -> int:
        return self.total_count

    def timing_summary(self) -> dict[str, dict[str, float]]:
        """Aggregate timing statistics per pipeline phase.

        Returns p50, p95, max, mean, and total wall-clock seconds for each
        phase (``"read"``, ``"extract"``, ``"write"``).  Phases with no
        data are omitted.  Only succeeded entities contribute to timings.

        For batch extraction (``batch_size > 1``), the ``"extract"`` time
        is the total ``extract_batch`` duration divided by the batch size
        — a per-entity average, not an individually measured time.

        Returns:
            Dict mapping phase name to a statistics dict with keys
            ``p50``, ``p95``, ``max``, ``mean``, ``total`` (all in
            seconds).  Empty dict if no timing data has been collected.

        Example::

            summary = report.timing_summary()
            print(summary["read"]["p95"])   # 95th-percentile read time
            print(summary["write"]["max"])  # slowest single write
        """
        result: dict[str, dict[str, float]] = {}
        for phase, times in self.phase_timings.items():
            if not times:
                continue
            n = len(times)
            sorted_t = sorted(times)
            result[phase] = {
                "p50": sorted_t[int(n * 0.50)],
                "p95": sorted_t[min(int(n * 0.95), n - 1)],
                "max": sorted_t[-1],
                "mean": sum(sorted_t) / n,
                "total": sum(sorted_t),
            }
        return result

    def error_summary(self) -> dict[str, list[str]]:
        """Group failed entities by their error message.

        Returns a dict mapping each unique error string to the list of
        entity IDs that produced it.  Useful for diagnosing systematic
        failures — e.g. 500 entities all failing with the same schema
        violation shows up as one entry rather than 500.

        Returns:
            ``{error_message: [entity_id, ...]}``, ordered by descending
            count (most common error first).
        """
        groups: dict[str, list[str]] = {}
        for eid, errors in self.failed.items():
            key = "; ".join(errors)
            groups.setdefault(key, []).append(eid)
        return dict(sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True))

    def __repr__(self) -> str:
        return (
            f"GenerationReport(entities={self.success_count}, "
            f"records={self.record_count}, "
            f"failed={self.failure_count}, skipped={self.skip_count}, "
            f"duration={self.duration_s:.2f}s)"
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Export a pandas DataFrame with one row per entity.

        Columns: ``entity_id``, ``status`` ("succeeded" / "failed" / "skipped"),
        ``record_count`` (int or ``None``), ``error`` (str or ``None``).

        Note: succeeded rows are only present when ``store_results=True`` was
        used during ``generate()``.
        """
        rows: list[dict[str, Any]] = []
        for eid, result in self.succeeded.items():
            rows.append(
                {
                    "entity_id": eid,
                    "status": "succeeded",
                    "record_count": len(result.records),
                    "error": None,
                }
            )
        for eid, errors in self.failed.items():
            rows.append(
                {
                    "entity_id": eid,
                    "status": "failed",
                    "record_count": None,
                    "error": "; ".join(errors),
                }
            )
        for eid in self.skipped:
            rows.append(
                {
                    "entity_id": eid,
                    "status": "skipped",
                    "record_count": None,
                    "error": None,
                }
            )
        return pd.DataFrame(rows)


class Pipeline:
    """Orchestrate a featurization workflow.

    Ties together a ``DataSource``, a ``Feature``, and a ``FeatureStore``
    into a ``generate`` / ``retrieve`` interface.

    Args:
        source: Where to read raw data for each entity.
        feature: How to extract the feature from raw data.
        store: Where to persist and retrieve extracted feature values.

    Example::

        pipeline = Pipeline(
            source=DataFrameSource(df),
            feature=MeanPurchaseValue(),
            store=MemoryStore(),
        )

        report = pipeline.generate(entity_ids=["u1", "u2"])
        print(report)  # GenerationReport(succeeded=2, failed=0, skipped=0)

        value = pipeline.retrieve("u1")
    """

    def __init__(
        self,
        source: DataSource,
        feature: Feature,
        store: FeatureStore,
    ) -> None:
        self.source = source
        self.feature = feature
        self.store = store

    async def _process_entity(
        self,
        entity_id: str,
        context: dict[str, Any],
        context_fn: Callable[[str], dict[str, Any]] | None,
        overwrite: bool,
        report: GenerationReport,
        feature_name: str,
        store_results: bool,
        executor: Executor | None,
    ) -> None:
        """Process a single entity and update *report* in place."""
        try:
            if not overwrite and await self.store.aexists(self.feature, entity_id):
                report.skipped.add(entity_id)
                return

            entity_ctx = {**context, **context_fn(entity_id)} if context_fn else context

            if executor is not None:
                loop = asyncio.get_running_loop()
                result, errors, _phase_times = await loop.run_in_executor(
                    executor,
                    _run_entity_in_executor,
                    self.source,
                    self.feature,
                    entity_id,
                    entity_ctx,
                )
            else:
                _t0 = time.perf_counter()
                raw = await self.source.read(entity_id=entity_id, context=entity_ctx)
                _t1 = time.perf_counter()
                result = await self.feature.extract(raw, entity_ctx, entity_id=entity_id)
                _t2 = time.perf_counter()
                errors = await _validate_extraction(self.feature, result)
                _phase_times = {"read": _t1 - _t0, "extract": _t2 - _t1}

            if errors:
                report.failed[entity_id] = errors
                return

            _t_write = time.perf_counter()
            await self.store.awrite(self.feature, entity_id, result, context=entity_ctx)
            _phase_times["write"] = time.perf_counter() - _t_write
            for _phase, _t in _phase_times.items():
                report.phase_timings[_phase].append(_t)
            report.success_count += 1
            report.record_count += len(result.records)
            if store_results:
                report.succeeded[entity_id] = result

        except Exception as exc:
            report.failed[entity_id] = [
                f"Unhandled exception in pipeline for feature '{feature_name}', "
                f"entity '{entity_id}': {type(exc).__name__}: {exc}"
            ]
            report.exceptions[entity_id] = exc

    async def _process_batch(
        self,
        entity_ids: list[str],
        context: dict[str, Any],
        context_fn: Callable[[str], dict[str, Any]] | None,
        overwrite: bool,
        report: GenerationReport,
        feature_name: str,
        on_entity_done: Callable[[], None] | None,
        store_results: bool,
        executor: Executor | None,
    ) -> None:
        """Process a batch of entities with a single ``extract_batch`` call.

        Reads all entity raws concurrently, then delegates to
        ``Feature.extract_batch`` for vectorised computation.  Per-entity
        failure isolation is preserved: a ``BaseException`` returned for an
        individual slot is recorded as that entity's failure without
        affecting the rest of the batch.
        """
        # --- 1. Overwrite check: separate skips from entities to process ---
        to_process: list[str] = []
        for entity_id in entity_ids:
            try:
                if not overwrite and await self.store.aexists(self.feature, entity_id):
                    report.skipped.add(entity_id)
                    if on_entity_done is not None:
                        on_entity_done()
                    continue
            except Exception as exc:
                report.failed[entity_id] = [
                    f"Unhandled exception in pipeline for feature '{feature_name}', "
                    f"entity '{entity_id}': {type(exc).__name__}: {exc}"
                ]
                report.exceptions[entity_id] = exc
                if on_entity_done is not None:
                    on_entity_done()
                continue
            to_process.append(entity_id)

        if not to_process:
            return

        # --- Executor path: steps 2-4 run in thread/process; store.write stays here ---
        if executor is not None:
            batch_entity_contexts: list[dict[str, Any]] | None = (
                [{**context, **context_fn(eid)} for eid in to_process] if context_fn else None
            )
            loop = asyncio.get_running_loop()
            try:
                slot_results: list[
                    tuple[ExtractionResult, list[str]] | BaseException
                ] = await loop.run_in_executor(
                    executor,
                    _run_batch_in_executor,
                    self.source,
                    self.feature,
                    to_process,
                    context,
                    batch_entity_contexts,
                )
            except Exception as exc:
                for entity_id in to_process:
                    report.failed[entity_id] = [
                        f"Unhandled exception in pipeline for feature '{feature_name}', "
                        f"entity '{entity_id}': {type(exc).__name__}: {exc}"
                    ]
                    report.exceptions[entity_id] = exc
                    if on_entity_done is not None:
                        on_entity_done()
                return

            for i, (entity_id, slot) in enumerate(zip(to_process, slot_results, strict=True)):
                entity_ctx = (
                    batch_entity_contexts[i] if batch_entity_contexts is not None else context
                )
                if isinstance(slot, BaseException):
                    report.failed[entity_id] = [
                        f"Unhandled exception in pipeline for feature '{feature_name}', "
                        f"entity '{entity_id}': {type(slot).__name__}: {slot}"
                    ]
                    report.exceptions[entity_id] = slot
                else:
                    result, errors, _phase_times = slot
                    if errors:
                        report.failed[entity_id] = errors
                    else:
                        try:
                            _t_write = time.perf_counter()
                            await self.store.awrite(
                                self.feature, entity_id, result, context=entity_ctx
                            )
                            _phase_times["write"] = time.perf_counter() - _t_write
                            for _phase, _t in _phase_times.items():
                                report.phase_timings[_phase].append(_t)
                            report.success_count += 1
                            report.record_count += len(result.records)
                            if store_results:
                                report.succeeded[entity_id] = result
                        except Exception as exc:
                            report.failed[entity_id] = [
                                f"Unhandled exception in pipeline for feature '{feature_name}', "
                                f"entity '{entity_id}': {type(exc).__name__}: {exc}"
                            ]
                            report.exceptions[entity_id] = exc
                if on_entity_done is not None:
                    on_entity_done()
            return

        # --- 2. Per-entity contexts (needed by both read and extract_batch) ---
        entity_ctxs: dict[str, dict[str, Any]] = {
            eid: {**context, **context_fn(eid)} if context_fn else context for eid in to_process
        }

        # --- 3. Concurrent reads (timed individually) ---
        async def _timed_read(eid: str) -> tuple[Any, float]:
            t0 = time.perf_counter()
            raw = await self.source.read(entity_id=eid, context=entity_ctxs[eid])
            return raw, time.perf_counter() - t0

        raw_timed: list[tuple[Any, float] | BaseException] = await asyncio.gather(
            *[_timed_read(eid) for eid in to_process],
            return_exceptions=True,
        )

        # --- 4. Filter read failures ---
        valid_ids: list[str] = []
        valid_raws: list[Any] = []
        read_times: dict[str, float] = {}

        for entity_id, res in zip(to_process, raw_timed, strict=True):
            if isinstance(res, BaseException):
                report.failed[entity_id] = [
                    f"Unhandled exception in pipeline for feature '{feature_name}', "
                    f"entity '{entity_id}': {type(res).__name__}: {res}"
                ]
                report.exceptions[entity_id] = res
                if on_entity_done is not None:
                    on_entity_done()
                continue
            raw, read_time = res
            read_times[entity_id] = read_time
            valid_ids.append(entity_id)
            valid_raws.append(raw)

        if not valid_ids:
            return

        # --- 5. Batch extract (timed; divide by batch size for per-entity average) ---
        entity_contexts: list[dict[str, Any]] | None = (
            [entity_ctxs[eid] for eid in valid_ids] if context_fn else None
        )
        _t_extract_0 = time.perf_counter()
        try:
            batch_results: list[
                ExtractionResult | BaseException
            ] = await self.feature.extract_batch(
                valid_raws,
                context,
                entity_ids=valid_ids,
                entity_contexts=entity_contexts,
            )
        except Exception as exc:
            # Whole-batch failure: attribute to every entity in this batch.
            for entity_id in valid_ids:
                report.failed[entity_id] = [
                    f"Unhandled exception in pipeline for feature '{feature_name}', "
                    f"entity '{entity_id}': {type(exc).__name__}: {exc}"
                ]
                report.exceptions[entity_id] = exc
                if on_entity_done is not None:
                    on_entity_done()
            return
        _extract_time_each = (
            (time.perf_counter() - _t_extract_0) / len(valid_ids) if valid_ids else 0.0
        )

        # --- 6. Validate and write — per entity ---
        for entity_id, result in zip(valid_ids, batch_results, strict=True):
            if isinstance(result, BaseException):
                report.failed[entity_id] = [
                    f"Unhandled exception in pipeline for feature '{feature_name}', "
                    f"entity '{entity_id}': {type(result).__name__}: {result}"
                ]
                report.exceptions[entity_id] = result
            else:
                try:
                    errors = await _validate_extraction(self.feature, result)
                    if errors:
                        report.failed[entity_id] = errors
                    else:
                        _t_write = time.perf_counter()
                        await self.store.awrite(
                            self.feature, entity_id, result, context=entity_ctxs[entity_id]
                        )
                        report.phase_timings["read"].append(read_times.get(entity_id, 0.0))
                        report.phase_timings["extract"].append(_extract_time_each)
                        report.phase_timings["write"].append(time.perf_counter() - _t_write)
                        report.success_count += 1
                        report.record_count += len(result.records)
                        if store_results:
                            report.succeeded[entity_id] = result
                except Exception as exc:
                    report.failed[entity_id] = [
                        f"Unhandled exception in pipeline for feature '{feature_name}', "
                        f"entity '{entity_id}': {type(exc).__name__}: {exc}"
                    ]
                    report.exceptions[entity_id] = exc

            if on_entity_done is not None:
                on_entity_done()

    async def agenerate(
        self,
        entity_ids: list[str] | None = None,
        context: dict[str, Any] | None = None,
        *,
        partitions: dict[str, list[str]] | None = None,
        partition_by: Callable[[str], Hashable] | None = None,
        context_fn: Callable[[str], dict[str, Any]] | None = None,
        partition_context_fn: Callable[[Hashable], dict[str, Any]] | None = None,
        concurrency: int = 1,
        batch_size: int = 1,
        overwrite: bool = True,
        store_results: bool = True,
        on_progress: Callable[[int, int, GenerationReport], None] | None = None,
        executor: Executor | None = None,
    ) -> GenerationReport:
        """Extract and store features for a collection of entities.

        Within each partition, entities are always processed **serially** and
        in order.  Across partitions, up to *concurrency* partitions run
        concurrently.

        When *batch_size* > 1, entities are grouped into sub-batches and
        ``Feature.extract_batch`` is called once per sub-batch instead of
        once per entity.  This enables vectorised computation (ML inference,
        batch API calls, bulk DB queries) that is dramatically faster than
        N individual calls.

        Three concurrency modes:

        1. **Flat** (default) — pass ``entity_ids``; ``concurrency`` caps how
           many entities (or batches, when *batch_size* > 1) run at once::

               await pipeline.agenerate(entity_ids=ids, concurrency=16)

        2. **Partition function** — pass ``entity_ids`` and ``partition_by``;
           entities are grouped by the function's return value and processed
           serially within each group::

               await pipeline.agenerate(
                   entity_ids=ids,
                   partition_by=lambda eid: eid.split("_")[0],
                   concurrency=8,
               )

        3. **Explicit partitions** — pass ``partitions`` directly::

               await pipeline.agenerate(
                   partitions={"shard_0": [...], "shard_1": [...]},
                   concurrency=4,
               )

        Args:
            entity_ids: Flat list of entity IDs to process.  Mutually
                exclusive with *partitions*.
            context: Arbitrary dict forwarded to ``Feature.extract`` /
                ``Feature.extract_batch``.  Defaults to an empty dict.
            context_fn: Optional callable ``(entity_id) -> dict`` that
                returns per-entity context additions.  The returned dict is
                merged on top of *context* (and any *partition_context_fn*
                result), so entity-specific values shadow shared ones.  For
                batch extraction, the merged dicts are passed as
                *entity_contexts*.
            partition_context_fn: Optional callable ``(partition_key) -> dict``
                that returns context additions for every entity in a
                partition.  The returned dict is merged on top of *context*
                and ``_partition_key`` but below *context_fn* (i.e.
                ``{**context, "_partition_key": key, **partition_context_fn(key), **context_fn(eid)}``).
                Useful for injecting partition-specific metadata such as a
                DB connection handle or a partition-level model into every
                extraction call in that partition.  Note: the partition key
                is always available as ``context["_partition_key"]`` without
                needing this callback — use it only when you need to derive
                *additional* values from the key.
            partitions: Pre-built partition mapping
                ``{partition_key: [entity_ids]}``.  Mutually exclusive with
                *entity_ids*.
            partition_by: Callable that maps an entity ID to a partition key.
                Requires *entity_ids*; cannot be combined with *partitions*.
            concurrency: Maximum number of partitions (or batches in flat
                mode) that run concurrently.  Defaults to ``1`` (fully
                serial).
            batch_size: Number of entities to collect before calling
                ``Feature.extract_batch`` once.  ``1`` (default) uses the
                per-entity ``extract`` path unchanged.  In flat mode,
                *concurrency* controls how many batches run in parallel.
            overwrite: When ``False``, entities that already have a stored
                value are skipped without re-extracting.  Skipped entities
                appear in ``GenerationReport.skipped``.  Defaults to ``True``.
            store_results: When ``True`` (default), extracted values are kept
                in ``GenerationReport.succeeded`` so callers can inspect them
                without a separate ``retrieve`` call.  Set to ``False`` for
                large runs where holding all results in memory is undesirable;
                succeeded entity IDs are still tracked (so ``success_count``
                and membership checks work) but the values are ``None``.
            on_progress: Optional sync callback invoked after each entity
                resolves (succeeds, fails, or is skipped).  Receives
                ``(completed: int, total: int, report: GenerationReport)``.
            executor: Optional ``concurrent.futures.Executor`` (e.g.
                ``ThreadPoolExecutor`` or ``ProcessPoolExecutor``) to run
                the extract pipeline stages (read → extract → validate)
                outside the asyncio event loop.
                Use ``ProcessPoolExecutor`` to bypass the GIL for CPU-bound
                feature extraction.  ``store.write`` always runs in the
                main process, so all store implementations (including
                ``MemoryStore``) work correctly.  When using
                ``ProcessPoolExecutor``, ``source`` and ``feature`` must be
                picklable.  Set ``concurrency`` to match the executor's
                ``max_workers`` so all workers can run concurrently.

        Returns:
            ``GenerationReport`` collecting every success, failure, and skip.

        Raises:
            ValueError: If arguments are mutually exclusive or invalid.
        """
        # --- Validate arguments ---
        if entity_ids is not None and partitions is not None:
            raise ValueError("Pass either 'entity_ids' or 'partitions', not both.")
        if entity_ids is None and partitions is None:
            raise ValueError("One of 'entity_ids' or 'partitions' is required.")
        if partition_by is not None and partitions is not None:
            raise ValueError("'partition_by' cannot be combined with explicit 'partitions'.")
        if concurrency < 1:
            raise ValueError(f"'concurrency' must be >= 1, got {concurrency}.")
        if batch_size < 1:
            raise ValueError(f"'batch_size' must be >= 1, got {batch_size}.")

        if context is None:
            context = {}

        # --- Build partition map ---
        partition_map: dict[Any, list[str]]
        if partitions is not None:
            partition_map = dict(partitions)
        elif partition_by is not None:
            partition_map = {}
            for eid in entity_ids:  # type: ignore[union-attr]
                key = partition_by(eid)
                partition_map.setdefault(key, []).append(eid)
        elif batch_size > 1:
            # Flat batch mode: each chunk of batch_size entities becomes a
            # partition so concurrency controls how many batches run at once.
            ids = entity_ids  # type: ignore[union-attr]
            partition_map = {i: ids[i : i + batch_size] for i in range(0, len(ids), batch_size)}
        else:
            # Flat entity mode: one partition per entity so the semaphore
            # directly controls max concurrency at the individual-entity level.
            partition_map = {eid: [eid] for eid in entity_ids}  # type: ignore[union-attr]

        total = sum(len(v) for v in partition_map.values())
        completed = 0
        report = GenerationReport()
        _t0 = time.perf_counter()
        feature_name = type(self.feature).__name__
        semaphore = asyncio.Semaphore(concurrency)
        # Only inject _partition_key when the caller explicitly grouped entities
        # into partitions.  In flat entity/batch mode the key is the entity ID
        # or a batch index, which is not meaningful partition information.
        has_explicit_partitions = partitions is not None or partition_by is not None

        async def run_partition(partition_key: Any, entities: list[str]) -> None:
            nonlocal completed
            async with semaphore:
                partition_ctx = {
                    **context,
                    **({"_partition_key": partition_key} if has_explicit_partitions else {}),
                    **(partition_context_fn(partition_key) if partition_context_fn else {}),
                }
                if batch_size == 1:
                    for entity_id in entities:
                        await self._process_entity(
                            entity_id,
                            partition_ctx,
                            context_fn,
                            overwrite,
                            report,
                            feature_name,
                            store_results,
                            executor,
                        )
                        completed += 1
                        if on_progress is not None:
                            on_progress(completed, total, report)
                else:

                    def on_entity_done() -> None:
                        nonlocal completed
                        completed += 1
                        if on_progress is not None:
                            on_progress(completed, total, report)

                    for i in range(0, len(entities), batch_size):
                        sub_batch = entities[i : i + batch_size]
                        await self._process_batch(
                            sub_batch,
                            partition_ctx,
                            context_fn,
                            overwrite,
                            report,
                            feature_name,
                            on_entity_done,
                            store_results,
                            executor,
                        )

        await asyncio.gather(
            *[run_partition(key, entities) for key, entities in partition_map.items()]
        )
        report.duration_s = time.perf_counter() - _t0
        return report

    async def aretrieve(self, entity_id: str) -> Any:
        """Read the stored feature value for one entity.

        Args:
            entity_id: The entity identifier.

        Returns:
            The stored feature value.

        Raises:
            KeyError: If no feature has been generated for ``entity_id``.
            StoreError: If the underlying store read fails.
        """
        return await self.store.aread(self.feature, entity_id)

    async def aretrieve_batch(self, entity_ids: list[str]) -> dict[str, Any]:
        """Read stored feature values for multiple entities concurrently.

        Entities with no stored value are silently omitted from the result.

        Args:
            entity_ids: List of entity identifiers.

        Returns:
            Dict mapping ``entity_id`` to feature value for every entity
            that has a stored value.
        """

        async def _try_read(entity_id: str) -> tuple[str, Any] | None:
            try:
                return (entity_id, await self.store.aread(self.feature, entity_id))
            except KeyError:
                return None

        pairs = await asyncio.gather(*[_try_read(eid) for eid in entity_ids])
        return dict(p for p in pairs if p is not None)

    # ------------------------------------------------------------------
    # Synchronous public interface (default)
    # ------------------------------------------------------------------

    def generate(self, *args: Any, **kwargs: Any) -> GenerationReport:
        """Extract and store features for a collection of entities.

        Blocking version of :meth:`agenerate` — the default interface for
        batch jobs, training scripts, and Airflow DAGs.  All keyword arguments
        are forwarded to :meth:`agenerate` unchanged.

        Use :meth:`agenerate` directly when already inside an async context
        (FastAPI handlers, async task workers).
        """
        return asyncio.run(self.agenerate(*args, **kwargs))

    def retrieve(self, entity_id: str) -> Any:
        """Read the stored feature value for one entity.

        Blocking version of :meth:`aretrieve`.  Use :meth:`aretrieve` when
        already inside an async context.
        """
        return asyncio.run(self.aretrieve(entity_id))

    def retrieve_batch(self, entity_ids: list[str]) -> dict[str, Any]:
        """Read stored feature values for multiple entities.

        Blocking version of :meth:`aretrieve_batch`.  Use
        :meth:`aretrieve_batch` when already inside an async context.
        """
        return asyncio.run(self.aretrieve_batch(entity_ids))
