"""Pipeline: ties DataSource → Feature → FeatureStore together."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from typing import Any

from .features.base import Feature
from .sources.base import DataSource
from .stores.base import FeatureStore


@dataclass
class GenerationReport:
    """Summary of a ``Pipeline.generate()`` run.

    Attributes:
        succeeded: Mapping of ``entity_id`` to extracted feature value for
            every entity that was processed without error.
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
        overwrite: bool,
        report: GenerationReport,
        feature_name: str,
    ) -> None:
        """Process a single entity and update *report* in place."""
        try:
            if not overwrite and await self.store.exists(self.feature, entity_id):
                report.skipped.add(entity_id)
                return

            raw = await self.source.read(entity_id=entity_id)
            raw = await self.feature.pre_extract(raw)
            result = await self.feature.extract(raw, context, entity_id=entity_id)
            result = await self.feature.post_extract(result)
            errors = await self.feature.validate(result)

            if errors:
                report.failed[entity_id] = errors
                return

            await self.store.write(self.feature, entity_id, result)
            report.succeeded[entity_id] = result

        except Exception as exc:
            report.failed[entity_id] = [
                f"Unhandled exception in pipeline for feature '{feature_name}', "
                f"entity '{entity_id}': {type(exc).__name__}: {exc}"
            ]

    async def _process_batch(
        self,
        entity_ids: list[str],
        context: dict[str, Any],
        overwrite: bool,
        report: GenerationReport,
        feature_name: str,
        on_entity_done: Callable[[], None] | None,
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
        try:
            batch_results: list[Any] = await self.feature.extract_batch(
                valid_raws, context, entity_ids=valid_ids
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
                        report.succeeded[entity_id] = result
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
        concurrency: int = 1,
        batch_size: int = 1,
        overwrite: bool = True,
        on_progress: Callable[[int, int, GenerationReport], None] | None = None,
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
            on_progress: Optional sync callback invoked after each entity
                resolves (succeeds, fails, or is skipped).  Receives
                ``(completed: int, total: int, report: GenerationReport)``.

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

        async def run_partition(entities: list[str]) -> None:
            nonlocal completed
            async with semaphore:
                if batch_size == 1:
                    for entity_id in entities:
                        await self._process_entity(
                            entity_id, context, overwrite, report, feature_name
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
                            sub_batch, context, overwrite, report, feature_name, on_entity_done
                        )

        await asyncio.gather(*[run_partition(entities) for entities in partition_map.values()])
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
        """Read stored feature values for multiple entities.

        Entities with no stored value are silently omitted from the result.

        Args:
            entity_ids: List of entity identifiers.

        Returns:
            Dict mapping ``entity_id`` to feature value for every entity
            that has a stored value.
        """
        results: dict[str, Any] = {}
        for entity_id in entity_ids:
            try:
                results[entity_id] = await self.store.read(self.feature, entity_id)
            except KeyError:
                pass
        return results
