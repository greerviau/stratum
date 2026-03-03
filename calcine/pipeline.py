"""Pipeline: ties DataSource → Feature → FeatureStore together."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Hashable
from concurrent.futures import Executor
from dataclasses import dataclass, field
from typing import Any

from .features.base import Feature
from .sources.base import DataSource
from .stores.base import FeatureStore

# ---------------------------------------------------------------------------
# Module-level executor worker functions
# ---------------------------------------------------------------------------
# These must be defined at module level (not as closures) so that
# ProcessPoolExecutor can pickle them for inter-process dispatch.


def _run_entity_in_executor(
    source: DataSource,
    feature: Feature,
    entity_id: str,
    context: dict[str, Any],
) -> tuple[Any, list[str]]:
    """Run one entity's extract pipeline stages in a thread or process.

    Executes: ``source.read`` → ``feature.pre_extract`` → ``feature.extract``
    → ``feature.post_extract`` → ``feature.validate``.

    Store writes are intentionally excluded so they remain in the main
    process (preserving correct behaviour for in-memory and async stores).

    Returns:
        ``(result, errors)`` where *errors* is non-empty on validation
        failure and empty on success.

    Raises:
        Any exception propagated from the pipeline stages.  The caller
        records it as an unhandled pipeline error.
    """

    async def _work() -> tuple[Any, list[str]]:
        raw = await source.read(entity_id=entity_id)
        raw = await feature.pre_extract(raw)
        result = await feature.extract(raw, context, entity_id=entity_id)
        result = await feature.post_extract(result)
        errors = await feature.validate(result)
        return result, errors

    return asyncio.run(_work())


def _run_batch_in_executor(
    source: DataSource,
    feature: Feature,
    entity_ids: list[str],
    context: dict[str, Any],
    entity_contexts: list[dict[str, Any]] | None,
) -> list[tuple[Any, list[str]] | BaseException]:
    """Run one batch's extract pipeline stages in a thread or process.

    Mirrors the logic of ``Pipeline._process_batch`` steps 2–4
    (concurrent reads, pre_extract, extract_batch, post_extract, validate).
    Store writes and report updates are handled by the caller in the main
    process.

    Returns:
        One entry per item in *entity_ids*, in order:

        - ``(result, [])`` — success
        - ``(None, errors_list)`` — validation failure
        - ``BaseException`` — unhandled error for that entity

    A whole-batch ``extract_batch`` failure is caught and stored as a
    ``BaseException`` in every valid slot, preserving the same per-entity
    failure isolation as the non-executor path.
    """

    async def _work() -> list[tuple[Any, list[str]] | BaseException]:
        # --- Concurrent reads ---
        raw_results: list[Any] = await asyncio.gather(
            *[source.read(entity_id=eid) for eid in entity_ids],
            return_exceptions=True,
        )

        results: dict[int, tuple[Any, list[str]] | BaseException] = {}
        valid_indices: list[int] = []
        valid_raws: list[Any] = []

        # --- Pre-extract; filter read failures ---
        for idx, (_eid, raw) in enumerate(zip(entity_ids, raw_results, strict=True)):
            if isinstance(raw, BaseException):
                results[idx] = raw
                continue
            try:
                pre = await feature.pre_extract(raw)
                valid_indices.append(idx)
                valid_raws.append(pre)
            except Exception as exc:
                results[idx] = exc

        if not valid_indices:
            return [results[i] for i in range(len(entity_ids))]

        valid_ids = [entity_ids[i] for i in valid_indices]
        valid_entity_ctxs = (
            [entity_contexts[i] for i in valid_indices] if entity_contexts is not None else None
        )

        # --- Batch extract ---
        try:
            batch_results: list[Any] = await feature.extract_batch(
                valid_raws, context, entity_ids=valid_ids, entity_contexts=valid_entity_ctxs
            )
        except Exception as exc:
            for idx in valid_indices:
                results[idx] = exc
            return [results[i] for i in range(len(entity_ids))]

        # --- Post-extract and validate ---
        for idx, result in zip(valid_indices, batch_results, strict=True):
            if isinstance(result, BaseException):
                results[idx] = result
                continue
            try:
                result = await feature.post_extract(result)
                errors = await feature.validate(result)
                results[idx] = (result, errors)
            except Exception as exc:
                results[idx] = exc

        return [results[i] for i in range(len(entity_ids))]

    return asyncio.run(_work())


@dataclass
class GenerationReport:
    """Summary of a ``Pipeline.generate()`` run.

    Attributes:
        succeeded: Mapping of ``entity_id`` to extracted feature value for
            every entity that was processed without error.  Values are
            ``None`` when ``generate()`` is called with ``store_results=False``.
        failed: Mapping of ``entity_id`` to a list of error strings for
            every entity whose processing failed (source read error,
            extraction error, or schema violation).
        skipped: Set of entity IDs that were skipped because a stored value
            already existed and ``overwrite=False`` was passed to
            ``generate()``.
    """

    succeeded: dict[str, Any] = field(default_factory=dict)
    failed: dict[str, list[str]] = field(default_factory=dict)
    skipped: set[str] = field(default_factory=set)

    @property
    def success_count(self) -> int:
        """Number of successfully processed entities."""
        return len(self.succeeded)

    @property
    def failure_count(self) -> int:
        """Number of entities that failed processing."""
        return len(self.failed)

    @property
    def skip_count(self) -> int:
        """Number of entities skipped because a value already existed in the store."""
        return len(self.skipped)

    def __repr__(self) -> str:
        return (
            f"GenerationReport(succeeded={self.success_count}, "
            f"failed={self.failure_count}, skipped={self.skip_count})"
        )


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

        report = await pipeline.generate(entity_ids=["u1", "u2"])
        print(report)  # GenerationReport(succeeded=2, failed=0, skipped=0)

        value = await pipeline.retrieve("u1")
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
            if not overwrite and await self.store.exists(self.feature, entity_id):
                report.skipped.add(entity_id)
                return

            entity_ctx = {**context, **context_fn(entity_id)} if context_fn else context

            if executor is not None:
                loop = asyncio.get_running_loop()
                result, errors = await loop.run_in_executor(
                    executor,
                    _run_entity_in_executor,
                    self.source,
                    self.feature,
                    entity_id,
                    entity_ctx,
                )
            else:
                raw = await self.source.read(entity_id=entity_id)
                raw = await self.feature.pre_extract(raw)
                result = await self.feature.extract(raw, entity_ctx, entity_id=entity_id)
                result = await self.feature.post_extract(result)
                errors = await self.feature.validate(result)

            if errors:
                report.failed[entity_id] = errors
                return

            await self.store.write(self.feature, entity_id, result)
            report.succeeded[entity_id] = result if store_results else None

        except Exception as exc:
            report.failed[entity_id] = [
                f"Unhandled exception in pipeline for feature '{feature_name}', "
                f"entity '{entity_id}': {type(exc).__name__}: {exc}"
            ]

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
                if not overwrite and await self.store.exists(self.feature, entity_id):
                    report.skipped.add(entity_id)
                    if on_entity_done is not None:
                        on_entity_done()
                    continue
            except Exception as exc:
                report.failed[entity_id] = [
                    f"Unhandled exception in pipeline for feature '{feature_name}', "
                    f"entity '{entity_id}': {type(exc).__name__}: {exc}"
                ]
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
                    tuple[Any, list[str]] | BaseException
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
                    if on_entity_done is not None:
                        on_entity_done()
                return

            for entity_id, slot in zip(to_process, slot_results, strict=True):
                if isinstance(slot, BaseException):
                    report.failed[entity_id] = [
                        f"Unhandled exception in pipeline for feature '{feature_name}', "
                        f"entity '{entity_id}': {type(slot).__name__}: {slot}"
                    ]
                else:
                    result, errors = slot
                    if errors:
                        report.failed[entity_id] = errors
                    else:
                        try:
                            await self.store.write(self.feature, entity_id, result)
                            report.succeeded[entity_id] = result if store_results else None
                        except Exception as exc:
                            report.failed[entity_id] = [
                                f"Unhandled exception in pipeline for feature '{feature_name}', "
                                f"entity '{entity_id}': {type(exc).__name__}: {exc}"
                            ]
                if on_entity_done is not None:
                    on_entity_done()
            return

        # --- 2. Concurrent reads (return_exceptions preserves per-entity errors) ---
        raw_results: list[Any] = await asyncio.gather(
            *[self.source.read(entity_id=eid) for eid in to_process],
            return_exceptions=True,
        )

        # --- 3. Pre-extract per entity; exclude read failures ---
        valid_ids: list[str] = []
        valid_raws: list[Any] = []

        for entity_id, raw in zip(to_process, raw_results, strict=True):
            if isinstance(raw, BaseException):
                report.failed[entity_id] = [
                    f"Unhandled exception in pipeline for feature '{feature_name}', "
                    f"entity '{entity_id}': {type(raw).__name__}: {raw}"
                ]
                if on_entity_done is not None:
                    on_entity_done()
                continue
            try:
                pre = await self.feature.pre_extract(raw)
                valid_ids.append(entity_id)
                valid_raws.append(pre)
            except Exception as exc:
                report.failed[entity_id] = [
                    f"Unhandled exception in pipeline for feature '{feature_name}', "
                    f"entity '{entity_id}': {type(exc).__name__}: {exc}"
                ]
                if on_entity_done is not None:
                    on_entity_done()

        if not valid_ids:
            return

        # --- 4. Batch extract ---
        entity_contexts: list[dict[str, Any]] | None = (
            [{**context, **context_fn(eid)} for eid in valid_ids] if context_fn else None
        )
        try:
            batch_results: list[Any] = await self.feature.extract_batch(
                valid_raws, context, entity_ids=valid_ids, entity_contexts=entity_contexts
            )
        except Exception as exc:
            # Whole-batch failure: attribute to every entity in this batch.
            for entity_id in valid_ids:
                report.failed[entity_id] = [
                    f"Unhandled exception in pipeline for feature '{feature_name}', "
                    f"entity '{entity_id}': {type(exc).__name__}: {exc}"
                ]
                if on_entity_done is not None:
                    on_entity_done()
            return

        # --- 5. Post-extract, validate, write — per entity ---
        for entity_id, result in zip(valid_ids, batch_results, strict=True):
            if isinstance(result, BaseException):
                report.failed[entity_id] = [
                    f"Unhandled exception in pipeline for feature '{feature_name}', "
                    f"entity '{entity_id}': {type(result).__name__}: {result}"
                ]
            else:
                try:
                    result = await self.feature.post_extract(result)
                    errors = await self.feature.validate(result)
                    if errors:
                        report.failed[entity_id] = errors
                    else:
                        await self.store.write(self.feature, entity_id, result)
                        report.succeeded[entity_id] = result if store_results else None
                except Exception as exc:
                    report.failed[entity_id] = [
                        f"Unhandled exception in pipeline for feature '{feature_name}', "
                        f"entity '{entity_id}': {type(exc).__name__}: {exc}"
                    ]

            if on_entity_done is not None:
                on_entity_done()

    async def generate(
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

               await pipeline.generate(entity_ids=ids, concurrency=16)

        2. **Partition function** — pass ``entity_ids`` and ``partition_by``;
           entities are grouped by the function's return value and processed
           serially within each group::

               await pipeline.generate(
                   entity_ids=ids,
                   partition_by=lambda eid: eid.split("_")[0],
                   concurrency=8,
               )

        3. **Explicit partitions** — pass ``partitions`` directly::

               await pipeline.generate(
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
                but below *context_fn* (i.e.
                ``{**context, **partition_context_fn(key), **context_fn(eid)}``).
                Useful for injecting partition-specific metadata such as the
                region name, a shard index, or a partition-level model
                handle into every extraction call in that partition.  The
                *partition_key* is whatever value ``partition_by`` returns
                (or the explicit dict key when *partitions* is used).  In
                flat entity/batch mode the key is the entity ID or batch
                start index — typically not useful there.
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
                the extract pipeline stages (read → pre_extract → extract →
                post_extract → validate) outside the asyncio event loop.
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
        feature_name = type(self.feature).__name__
        semaphore = asyncio.Semaphore(concurrency)

        async def run_partition(partition_key: Any, entities: list[str]) -> None:
            nonlocal completed
            async with semaphore:
                partition_ctx = (
                    {**context, **partition_context_fn(partition_key)}
                    if partition_context_fn
                    else context
                )
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
        return report

    async def retrieve(self, entity_id: str) -> Any:
        """Read the stored feature value for one entity.

        Args:
            entity_id: The entity identifier.

        Returns:
            The stored feature value.

        Raises:
            KeyError: If no feature has been generated for ``entity_id``.
            StoreError: If the underlying store read fails.
        """
        return await self.store.read(self.feature, entity_id)

    async def retrieve_batch(self, entity_ids: list[str]) -> dict[str, Any]:
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
                return (entity_id, await self.store.read(self.feature, entity_id))
            except KeyError:
                return None

        pairs = await asyncio.gather(*[_try_read(eid) for eid in entity_ids])
        return dict(p for p in pairs if p is not None)

    # ------------------------------------------------------------------
    # Synchronous convenience wrappers
    # ------------------------------------------------------------------

    def generate_sync(self, *args: Any, **kwargs: Any) -> GenerationReport:
        """Blocking version of :meth:`generate` for use outside an async context.

        Equivalent to ``asyncio.run(pipeline.generate(...))``.  All keyword
        arguments are forwarded to :meth:`generate` unchanged.

        .. note::
            Raises ``RuntimeError`` if called from inside a running event loop
            (e.g. Jupyter notebooks or FastAPI handlers).  Use the async
            :meth:`generate` directly in those contexts.
        """
        return asyncio.run(self.generate(*args, **kwargs))

    def retrieve_sync(self, entity_id: str) -> Any:
        """Blocking version of :meth:`retrieve` for use outside an async context.

        .. note::
            Raises ``RuntimeError`` if called from inside a running event loop.
        """
        return asyncio.run(self.retrieve(entity_id))

    def retrieve_batch_sync(self, entity_ids: list[str]) -> dict[str, Any]:
        """Blocking version of :meth:`retrieve_batch` for use outside an async context.

        .. note::
            Raises ``RuntimeError`` if called from inside a running event loop.
        """
        return asyncio.run(self.retrieve_batch(entity_ids))
