"""Tests for Pipeline and GenerationReport."""

from __future__ import annotations

import asyncio
from typing import Any

import pandas as pd
import pytest

from calcine import ExtractionResult, GenerationReport, Pipeline
from calcine.features.base import Feature
from calcine.schema import FeatureSchema, types
from calcine.sources import DataFrameSource
from calcine.sources.base import DataSource
from calcine.stores import MemoryStore

# ---------------------------------------------------------------------------
# Helper feature implementations
# ---------------------------------------------------------------------------


class MeanFeature(Feature):
    schema = FeatureSchema({"mean_value": types.Float64(nullable=False)})

    async def extract(
        self, raw: pd.DataFrame, context: dict, entity_id: str | None = None
    ) -> ExtractionResult:
        if raw.empty:
            raise ValueError("No rows for entity")
        return ExtractionResult.of(entity_id, {"mean_value": float(raw["amount"].mean())})


class FailingFeature(Feature):
    async def extract(self, raw, context, entity_id=None):
        raise RuntimeError("Intentional extraction failure")


class SlowFeature(Feature):
    """Yields once during extract so other coroutines can interleave."""

    def __init__(self, order_log: list[str]) -> None:
        self.order_log = order_log

    async def extract(
        self, raw: pd.DataFrame, context: dict, entity_id: str | None = None
    ) -> ExtractionResult:
        self.order_log.append(f"start:{entity_id or '?'}")
        await asyncio.sleep(0)  # yield to event loop
        self.order_log.append(f"end:{entity_id or '?'}")
        return ExtractionResult.of(entity_id, {"value": 1.0})


@pytest.fixture
def df():
    return pd.DataFrame({"entity_id": ["u1", "u1", "u2"], "amount": [10.0, 20.0, 15.0]})


@pytest.fixture
def wide_df():
    """DataFrame with 6 entities for concurrency / partition tests."""
    rows = [(f"u{i}", float(i)) for i in range(1, 7)]
    return pd.DataFrame(rows, columns=["entity_id", "amount"])


# ---------------------------------------------------------------------------
# generate() — existing behaviour (serial, no new args)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_returns_report(df):
    pipeline = Pipeline(
        source=DataFrameSource(df),
        feature=MeanFeature(),
        store=MemoryStore(),
    )
    report = await pipeline.agenerate(entity_ids=["u1", "u2"])

    assert isinstance(report, GenerationReport)
    assert report.success_count == 2
    assert report.failure_count == 0


@pytest.mark.asyncio
async def test_generate_correct_values(df):
    pipeline = Pipeline(
        source=DataFrameSource(df),
        feature=MeanFeature(),
        store=MemoryStore(),
    )
    report = await pipeline.agenerate(entity_ids=["u1", "u2"])

    # u1 has rows [10, 20] → mean 15; u2 has [15] → mean 15
    assert report.succeeded["u1"].records["u1"]["mean_value"] == 15.0
    assert report.succeeded["u2"].records["u2"]["mean_value"] == 15.0


@pytest.mark.asyncio
async def test_generate_never_raises_on_entity_failure(df):
    """A failing feature should not crash generate() — it collects errors."""
    pipeline = Pipeline(
        source=DataFrameSource(df),
        feature=FailingFeature(),
        store=MemoryStore(),
    )
    report = await pipeline.agenerate(entity_ids=["u1", "u2"])

    assert report.failure_count == 2
    assert report.success_count == 0
    assert "u1" in report.failed
    assert "u2" in report.failed


@pytest.mark.asyncio
async def test_generate_missing_entity_does_not_crash(df):
    """An entity not in the source is captured as a failure, not a crash."""
    pipeline = Pipeline(
        source=DataFrameSource(df),
        feature=MeanFeature(),
        store=MemoryStore(),
    )
    report = await pipeline.agenerate(entity_ids=["u1", "u_missing"])

    assert "u1" in report.succeeded
    assert "u_missing" in report.failed


@pytest.mark.asyncio
async def test_generate_schema_violation_captured_as_failure(df):
    """Schema violations should be recorded in failed, not raise."""

    class BadFeature(Feature):
        schema = FeatureSchema({"score": types.Float64(nullable=False)})

        async def extract(self, raw, context, entity_id=None):
            return ExtractionResult.of(entity_id, {"score": "not_a_float"})  # wrong type

    pipeline = Pipeline(
        source=DataFrameSource(df),
        feature=BadFeature(),
        store=MemoryStore(),
    )
    report = await pipeline.agenerate(entity_ids=["u1"])

    assert "u1" in report.failed
    assert len(report.failed["u1"]) > 0


@pytest.mark.asyncio
async def test_generate_default_context_is_empty(df):
    """generate() should work without an explicit context argument."""
    pipeline = Pipeline(
        source=DataFrameSource(df),
        feature=MeanFeature(),
        store=MemoryStore(),
    )
    report = await pipeline.agenerate(entity_ids=["u1"])
    assert "u1" in report.succeeded


@pytest.mark.asyncio
async def test_generate_context_forwarded_to_extract(df):
    received: dict = {}

    class ContextCapture(Feature):
        async def extract(self, raw, context, entity_id=None):
            received.update(context)
            return ExtractionResult.of(entity_id, {"v": 1.0})

    pipeline = Pipeline(
        source=DataFrameSource(df),
        feature=ContextCapture(),
        store=MemoryStore(),
    )
    ctx = {"version": "v2", "ts": 999}
    await pipeline.agenerate(entity_ids=["u1"], context=ctx)

    assert received == ctx


# ---------------------------------------------------------------------------
# generate() — flat concurrency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_flat_concurrency_correct_results(wide_df):
    """Flat concurrency with concurrency>1 should produce the same results."""
    pipeline = Pipeline(
        source=DataFrameSource(wide_df),
        feature=MeanFeature(),
        store=MemoryStore(),
    )
    ids = [f"u{i}" for i in range(1, 7)]
    report = await pipeline.agenerate(entity_ids=ids, concurrency=3)

    assert report.success_count == 6
    assert report.failure_count == 0
    for i in range(1, 7):
        assert report.succeeded[f"u{i}"].records[f"u{i}"]["mean_value"] == float(i)


@pytest.mark.asyncio
async def test_generate_flat_concurrency_full_parallelism(wide_df):
    """concurrency >= len(ids) should process everything concurrently."""
    pipeline = Pipeline(
        source=DataFrameSource(wide_df),
        feature=MeanFeature(),
        store=MemoryStore(),
    )
    ids = [f"u{i}" for i in range(1, 7)]
    report = await pipeline.agenerate(entity_ids=ids, concurrency=100)

    assert report.success_count == 6


# ---------------------------------------------------------------------------
# generate() — partition_by
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_partition_by_correct_results(wide_df):
    """partition_by should produce the same results as serial processing."""
    pipeline = Pipeline(
        source=DataFrameSource(wide_df),
        feature=MeanFeature(),
        store=MemoryStore(),
    )
    ids = [f"u{i}" for i in range(1, 7)]
    # Partition into two groups: odd vs even entity number
    report = await pipeline.agenerate(
        entity_ids=ids,
        partition_by=lambda eid: int(eid[1:]) % 2,
        concurrency=2,
    )

    assert report.success_count == 6
    assert report.failure_count == 0


@pytest.mark.asyncio
async def test_generate_partition_by_groups_entities():
    """Verify entities are routed to the correct partition."""
    df = pd.DataFrame(
        {
            "entity_id": [f"a_{i}" for i in range(3)] + [f"b_{i}" for i in range(3)],
            "amount": [1.0] * 6,
        }
    )
    pipeline = Pipeline(
        source=DataFrameSource(df),
        feature=MeanFeature(),
        store=MemoryStore(),
    )
    ids = [f"a_{i}" for i in range(3)] + [f"b_{i}" for i in range(3)]
    report = await pipeline.agenerate(
        entity_ids=ids,
        partition_by=lambda eid: eid.split("_")[0],
        concurrency=2,
    )
    assert report.success_count == 6


@pytest.mark.asyncio
async def test_generate_serial_within_partition():
    """Entities within a partition must be processed one at a time, in order."""
    order: list[str] = []

    class OrderedFeature(Feature):
        async def extract(self, raw, context, entity_id=None):
            order.append(raw)
            await asyncio.sleep(0)  # yield — would allow interleaving if not serial
            return ExtractionResult.of(entity_id, {"v": 1.0})

    # Two partitions with 3 entities each; concurrency=2 lets both run at once.
    # Entities within each partition must still appear in order.
    source_map = {f"p0_e{i}": f"p0_e{i}" for i in range(3)}
    source_map.update({f"p1_e{i}": f"p1_e{i}" for i in range(3)})

    class MappedSource(DataFrameSource):
        async def read(self, entity_id=None, **kwargs):
            return entity_id  # raw is just the entity_id string

    from calcine.sources.base import DataSource

    class EchoSource(DataSource):
        async def read(self, entity_id=None, **kwargs):
            return entity_id

    pipeline = Pipeline(
        source=EchoSource(),
        feature=OrderedFeature(),
        store=MemoryStore(),
    )
    ids = [f"p0_e{i}" for i in range(3)] + [f"p1_e{i}" for i in range(3)]
    await pipeline.agenerate(
        entity_ids=ids,
        partition_by=lambda eid: eid.split("_")[0],
        concurrency=2,
    )

    p0 = [e for e in order if e.startswith("p0")]
    p1 = [e for e in order if e.startswith("p1")]
    assert p0 == ["p0_e0", "p0_e1", "p0_e2"]
    assert p1 == ["p1_e0", "p1_e1", "p1_e2"]


# ---------------------------------------------------------------------------
# generate() — explicit partitions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_explicit_partitions_correct_results(wide_df):
    pipeline = Pipeline(
        source=DataFrameSource(wide_df),
        feature=MeanFeature(),
        store=MemoryStore(),
    )
    report = await pipeline.agenerate(
        partitions={
            "group_a": ["u1", "u2", "u3"],
            "group_b": ["u4", "u5", "u6"],
        },
        concurrency=2,
    )
    assert report.success_count == 6
    assert report.failure_count == 0


@pytest.mark.asyncio
async def test_generate_explicit_partitions_serial_within_group():
    """Entities within an explicit partition are always processed in the given order."""
    from calcine.sources.base import DataSource

    class EchoSource(DataSource):
        async def read(self, entity_id=None, **kwargs):
            return entity_id

    processed: list[str] = []

    class RecordFeature(Feature):
        async def extract(self, raw, context, entity_id=None):
            processed.append(raw)
            return ExtractionResult.of(entity_id, {"v": 1.0})

    pipeline = Pipeline(
        source=EchoSource(),
        feature=RecordFeature(),
        store=MemoryStore(),
    )
    await pipeline.agenerate(
        partitions={"only": ["u1", "u2", "u3", "u4", "u5", "u6"]},
        concurrency=1,
    )
    # Single partition, serial — order must be preserved
    assert processed == ["u1", "u2", "u3", "u4", "u5", "u6"]


# ---------------------------------------------------------------------------
# generate() — overwrite=False (incremental mode)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_overwrite_false_skips_existing(df):
    """Entities already in the store should be skipped when overwrite=False."""
    store = MemoryStore()
    pipeline = Pipeline(source=DataFrameSource(df), feature=MeanFeature(), store=store)

    # First run: populate u1
    await pipeline.agenerate(entity_ids=["u1"])

    # Second run: u1 should be skipped, u2 should be processed
    report = await pipeline.agenerate(entity_ids=["u1", "u2"], overwrite=False)

    assert "u1" in report.skipped
    assert "u2" in report.succeeded
    assert report.skip_count == 1
    assert report.success_count == 1
    assert report.failure_count == 0


@pytest.mark.asyncio
async def test_generate_overwrite_true_reprocesses(df):
    """overwrite=True (default) should reprocess even if the entity is stored."""
    store = MemoryStore()
    pipeline = Pipeline(source=DataFrameSource(df), feature=MeanFeature(), store=store)

    await pipeline.agenerate(entity_ids=["u1"])

    # Overwrite with same data — should succeed, not skip
    report = await pipeline.agenerate(entity_ids=["u1"], overwrite=True)

    assert report.success_count == 1
    assert report.skip_count == 0


@pytest.mark.asyncio
async def test_generate_overwrite_false_progress_includes_skips(df):
    """on_progress should be called for skipped entities too."""
    store = MemoryStore()
    pipeline = Pipeline(source=DataFrameSource(df), feature=MeanFeature(), store=store)
    await pipeline.agenerate(entity_ids=["u1", "u2"])

    calls: list[tuple[int, int]] = []
    report = await pipeline.agenerate(
        entity_ids=["u1", "u2"],
        overwrite=False,
        on_progress=lambda c, t, _: calls.append((c, t)),
    )

    assert report.skip_count == 2
    assert len(calls) == 2
    assert calls[-1] == (2, 2)


# ---------------------------------------------------------------------------
# generate() — store_results=False
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_store_results_false_values_are_none(df):
    """store_results=False: success_count is correct but succeeded is empty."""
    pipeline = Pipeline(source=DataFrameSource(df), feature=MeanFeature(), store=MemoryStore())
    report = await pipeline.agenerate(entity_ids=["u1", "u2"], store_results=False)

    assert report.success_count == 2
    assert len(report.succeeded) == 0


@pytest.mark.asyncio
async def test_generate_store_results_false_values_still_written_to_store(df):
    """store_results=False does not affect what gets written to the store."""
    store = MemoryStore()
    pipeline = Pipeline(source=DataFrameSource(df), feature=MeanFeature(), store=store)
    await pipeline.agenerate(entity_ids=["u1"], store_results=False)

    value = await pipeline.aretrieve("u1")
    assert value["mean_value"] == 15.0


@pytest.mark.asyncio
async def test_generate_store_results_false_failures_still_recorded(df):
    """Failures are recorded normally even when store_results=False."""
    pipeline = Pipeline(source=DataFrameSource(df), feature=FailingFeature(), store=MemoryStore())
    report = await pipeline.agenerate(entity_ids=["u1"], store_results=False)

    assert report.failure_count == 1
    assert "u1" in report.failed


@pytest.mark.asyncio
async def test_generate_store_results_false_with_batch_size(wide_df):
    """store_results=False works with batch_size > 1."""
    pipeline = Pipeline(
        source=DataFrameSource(wide_df),
        feature=BatchTrackingFeature(),
        store=MemoryStore(),
    )
    ids = [f"u{i}" for i in range(1, 7)]
    report = await pipeline.agenerate(entity_ids=ids, batch_size=3, store_results=False)

    assert report.success_count == 6
    assert len(report.succeeded) == 0


# ---------------------------------------------------------------------------
# generate() — on_progress callback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_on_progress_call_count(df):
    """on_progress is called exactly once per entity."""
    calls: list[tuple[int, int]] = []
    pipeline = Pipeline(source=DataFrameSource(df), feature=MeanFeature(), store=MemoryStore())

    await pipeline.agenerate(
        entity_ids=["u1", "u2"],
        on_progress=lambda c, t, _: calls.append((c, t)),
    )

    assert len(calls) == 2
    assert calls[-1] == (2, 2)


@pytest.mark.asyncio
async def test_generate_on_progress_total_matches_success_count(wide_df):
    """total reported to on_progress matches the number of entities."""
    totals: list[int] = []
    pipeline = Pipeline(source=DataFrameSource(wide_df), feature=MeanFeature(), store=MemoryStore())
    ids = [f"u{i}" for i in range(1, 7)]

    await pipeline.agenerate(
        entity_ids=ids,
        on_progress=lambda c, t, _: totals.append(t),
    )

    assert all(t == 6 for t in totals)


@pytest.mark.asyncio
async def test_generate_on_progress_receives_live_report(df):
    """The report passed to on_progress reflects state at time of call."""
    snapshots: list[int] = []
    pipeline = Pipeline(source=DataFrameSource(df), feature=MeanFeature(), store=MemoryStore())

    await pipeline.agenerate(
        entity_ids=["u1", "u2"],
        on_progress=lambda c, t, r: snapshots.append(
            r.success_count + r.failure_count + r.skip_count
        ),
    )

    # After each call, the report should have exactly `completed` resolved entities
    assert snapshots == [1, 2]


@pytest.mark.asyncio
async def test_generate_on_progress_with_partitions(wide_df):
    """on_progress fires for every entity across all partitions."""
    calls: list[int] = []
    pipeline = Pipeline(source=DataFrameSource(wide_df), feature=MeanFeature(), store=MemoryStore())

    await pipeline.agenerate(
        partitions={"g0": ["u1", "u2", "u3"], "g1": ["u4", "u5", "u6"]},
        concurrency=2,
        on_progress=lambda c, t, _: calls.append(c),
    )

    assert len(calls) == 6
    assert max(calls) == 6


# ---------------------------------------------------------------------------
# generate() — argument validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_raises_if_both_entity_ids_and_partitions(df):
    pipeline = Pipeline(source=DataFrameSource(df), feature=MeanFeature(), store=MemoryStore())
    with pytest.raises(ValueError, match="entity_ids.*partitions|partitions.*entity_ids"):
        await pipeline.agenerate(entity_ids=["u1"], partitions={"p": ["u1"]})


@pytest.mark.asyncio
async def test_generate_raises_if_neither_entity_ids_nor_partitions(df):
    pipeline = Pipeline(source=DataFrameSource(df), feature=MeanFeature(), store=MemoryStore())
    with pytest.raises(ValueError):
        await pipeline.agenerate()


@pytest.mark.asyncio
async def test_generate_raises_if_partition_by_with_partitions(df):
    pipeline = Pipeline(source=DataFrameSource(df), feature=MeanFeature(), store=MemoryStore())
    with pytest.raises(ValueError, match="partition_by"):
        await pipeline.agenerate(
            partitions={"p": ["u1"]},
            partition_by=lambda e: e,
        )


@pytest.mark.asyncio
async def test_generate_raises_if_concurrency_less_than_one(df):
    pipeline = Pipeline(source=DataFrameSource(df), feature=MeanFeature(), store=MemoryStore())
    with pytest.raises(ValueError, match="concurrency"):
        await pipeline.agenerate(entity_ids=["u1"], concurrency=0)


@pytest.mark.asyncio
async def test_generate_raises_if_batch_size_less_than_one(df):
    pipeline = Pipeline(source=DataFrameSource(df), feature=MeanFeature(), store=MemoryStore())
    with pytest.raises(ValueError, match="batch_size"):
        await pipeline.agenerate(entity_ids=["u1"], batch_size=0)


# ---------------------------------------------------------------------------
# batch_size / extract_batch tests
# ---------------------------------------------------------------------------


class BatchTrackingFeature(Feature):
    """Records how extract_batch is called so tests can inspect batch sizes."""

    def __init__(self) -> None:
        self.calls: list[int] = []  # sizes of each extract_batch call

    async def extract(self, raw, context, entity_id=None):
        return ExtractionResult.of(entity_id, {"v": 1.0})

    async def extract_batch(self, raws, context, entity_ids=None, entity_contexts=None):
        self.calls.append(len(raws))
        return [
            ExtractionResult.of(eid, {"v": float(i)})
            for i, eid in enumerate(entity_ids or range(len(raws)))
        ]


class PartialFailBatchFeature(Feature):
    """extract_batch returns an exception for every other entity."""

    async def extract(self, raw, context, entity_id=None):
        return ExtractionResult.of(entity_id, {"v": 1.0})

    async def extract_batch(self, raws, context, entity_ids=None, entity_contexts=None):
        results = []
        eids = entity_ids or [None] * len(raws)
        for i, (_raw, eid) in enumerate(zip(raws, eids, strict=True)):
            if i % 2 == 1:
                results.append(ValueError(f"Simulated failure for item {i}"))
            else:
                results.append(ExtractionResult.of(eid, {"v": float(i)}))
        return results


class WholeBatchFailFeature(Feature):
    """extract_batch always raises, failing the entire batch."""

    async def extract(self, raw, context, entity_id=None):
        return ExtractionResult.of(entity_id, {"v": 1.0})

    async def extract_batch(self, raws, context, entity_ids=None, entity_contexts=None):
        raise RuntimeError("Whole batch exploded")


@pytest.mark.asyncio
async def test_batch_extract_called_once_per_batch(wide_df):
    """extract_batch should be called once per batch, not once per entity."""
    feature = BatchTrackingFeature()
    pipeline = Pipeline(source=DataFrameSource(wide_df), feature=feature, store=MemoryStore())

    ids = [f"u{i}" for i in range(1, 7)]
    report = await pipeline.agenerate(entity_ids=ids, batch_size=3)

    assert report.success_count == 6
    # 6 entities / batch_size 3 → 2 calls
    assert len(feature.calls) == 2
    assert all(n == 3 for n in feature.calls)


@pytest.mark.asyncio
async def test_batch_extract_correct_results(wide_df):
    """Results from extract_batch should be stored per entity correctly."""
    pipeline = Pipeline(
        source=DataFrameSource(wide_df),
        feature=BatchTrackingFeature(),
        store=MemoryStore(),
    )
    ids = [f"u{i}" for i in range(1, 7)]
    report = await pipeline.agenerate(entity_ids=ids, batch_size=6)

    assert report.success_count == 6
    assert report.failure_count == 0


@pytest.mark.asyncio
async def test_batch_extract_partial_failure_isolated(wide_df):
    """A per-slot BaseException in extract_batch should fail only that entity."""
    pipeline = Pipeline(
        source=DataFrameSource(wide_df),
        feature=PartialFailBatchFeature(),
        store=MemoryStore(),
    )
    ids = [f"u{i}" for i in range(1, 7)]
    report = await pipeline.agenerate(entity_ids=ids, batch_size=6)

    # Items at index 0, 2, 4 succeed; 1, 3, 5 fail
    assert report.success_count == 3
    assert report.failure_count == 3


@pytest.mark.asyncio
async def test_batch_extract_whole_batch_failure(wide_df):
    """If extract_batch raises, all entities in that batch are marked failed."""
    pipeline = Pipeline(
        source=DataFrameSource(wide_df),
        feature=WholeBatchFailFeature(),
        store=MemoryStore(),
    )
    ids = [f"u{i}" for i in range(1, 7)]
    report = await pipeline.agenerate(entity_ids=ids, batch_size=6)

    assert report.success_count == 0
    assert report.failure_count == 6


@pytest.mark.asyncio
async def test_batch_extract_default_falls_back_to_extract(df):
    """The default extract_batch calls extract() per item — same results."""
    # MeanFeature only implements extract(), so extract_batch uses the default
    pipeline = Pipeline(source=DataFrameSource(df), feature=MeanFeature(), store=MemoryStore())

    r_individual = await pipeline.agenerate(entity_ids=["u1", "u2"])
    store2 = MemoryStore()
    pipeline2 = Pipeline(source=DataFrameSource(df), feature=MeanFeature(), store=store2)
    r_batch = await pipeline2.agenerate(entity_ids=["u1", "u2"], batch_size=2)

    assert r_individual.succeeded["u1"] == r_batch.succeeded["u1"]
    assert r_individual.succeeded["u2"] == r_batch.succeeded["u2"]


@pytest.mark.asyncio
async def test_batch_extract_default_per_item_failure_isolated(df):
    """Default extract_batch wraps individual failures — other items still succeed."""
    pipeline = Pipeline(source=DataFrameSource(df), feature=MeanFeature(), store=MemoryStore())
    # u_missing will fail; u1 should still succeed
    report = await pipeline.agenerate(entity_ids=["u1", "u_missing"], batch_size=2)

    assert "u1" in report.succeeded
    assert "u_missing" in report.failed


@pytest.mark.asyncio
async def test_batch_size_with_concurrency(wide_df):
    """batch_size and concurrency can be combined: concurrent batches of N."""
    feature = BatchTrackingFeature()
    pipeline = Pipeline(source=DataFrameSource(wide_df), feature=feature, store=MemoryStore())

    ids = [f"u{i}" for i in range(1, 7)]
    report = await pipeline.agenerate(entity_ids=ids, batch_size=2, concurrency=3)

    assert report.success_count == 6
    # 6 entities / batch_size 2 → 3 batches, each called once
    assert len(feature.calls) == 3


@pytest.mark.asyncio
async def test_batch_size_with_overwrite_false(wide_df):
    """overwrite=False skips in batch mode; remaining entities still processed."""
    store = MemoryStore()
    pipeline = Pipeline(source=DataFrameSource(wide_df), feature=MeanFeature(), store=store)

    # Pre-populate u1, u2
    await pipeline.agenerate(entity_ids=["u1", "u2"])

    ids = [f"u{i}" for i in range(1, 7)]
    report = await pipeline.agenerate(entity_ids=ids, batch_size=3, overwrite=False)

    assert report.skip_count == 2
    assert report.success_count == 4


@pytest.mark.asyncio
async def test_batch_size_on_progress_fires_per_entity(wide_df):
    """on_progress should fire once per entity, not once per batch."""
    calls: list[int] = []
    pipeline = Pipeline(
        source=DataFrameSource(wide_df), feature=BatchTrackingFeature(), store=MemoryStore()
    )
    ids = [f"u{i}" for i in range(1, 7)]
    await pipeline.agenerate(
        entity_ids=ids,
        batch_size=3,
        on_progress=lambda c, t, _: calls.append(c),
    )

    assert len(calls) == 6
    assert calls[-1] == 6


@pytest.mark.asyncio
async def test_batch_size_with_partition_by(wide_df):
    """batch_size should work within partitioned mode."""
    feature = BatchTrackingFeature()
    pipeline = Pipeline(source=DataFrameSource(wide_df), feature=feature, store=MemoryStore())

    ids = [f"u{i}" for i in range(1, 7)]
    # 2 partitions (odd/even), 3 entities each, batch_size=2 → 2 sub-batches per partition
    report = await pipeline.agenerate(
        entity_ids=ids,
        partition_by=lambda eid: int(eid[1:]) % 2,
        concurrency=2,
        batch_size=2,
    )

    assert report.success_count == 6
    # Each partition has 3 entities, batched by 2 → ceil(3/2)=2 calls per partition × 2 partitions
    assert len(feature.calls) == 4


# ---------------------------------------------------------------------------
# retrieve() / retrieve_batch() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_after_generate(df):
    store = MemoryStore()
    pipeline = Pipeline(
        source=DataFrameSource(df),
        feature=MeanFeature(),
        store=store,
    )
    await pipeline.agenerate(entity_ids=["u1"])

    result = await pipeline.aretrieve("u1")
    assert result["mean_value"] == 15.0


@pytest.mark.asyncio
async def test_retrieve_missing_raises_key_error(df):
    pipeline = Pipeline(
        source=DataFrameSource(df),
        feature=MeanFeature(),
        store=MemoryStore(),
    )
    with pytest.raises(KeyError):
        await pipeline.aretrieve("nonexistent")


@pytest.mark.asyncio
async def test_retrieve_batch(df):
    pipeline = Pipeline(
        source=DataFrameSource(df),
        feature=MeanFeature(),
        store=MemoryStore(),
    )
    await pipeline.agenerate(entity_ids=["u1", "u2"])

    results = await pipeline.aretrieve_batch(["u1", "u2", "u_none"])
    assert "u1" in results
    assert "u2" in results
    assert "u_none" not in results  # missing silently omitted


# ---------------------------------------------------------------------------
# GenerationReport tests
# ---------------------------------------------------------------------------


def test_report_counts():
    report = GenerationReport(
        failed={"c": ["err"]},
        skipped={"d"},
        success_count=2,
        record_count=3,
    )
    assert report.success_count == 2
    assert report.record_count == 3
    assert report.failure_count == 1
    assert report.skip_count == 1


def test_report_repr():
    report = GenerationReport(success_count=1, record_count=2, skipped={"b"})
    r = repr(report)
    assert "entities=1" in r
    assert "records=2" in r
    assert "skipped=1" in r
    assert "duration=" in r


def test_report_skipped_default_empty():
    report = GenerationReport()
    assert report.skipped == set()


def test_report_total_count():
    report = GenerationReport(
        failed={"c": ["err"]},
        skipped={"d"},
        success_count=2,
        record_count=2,
    )
    assert report.total_count == 4
    assert len(report) == 4


def test_report_throughput():
    report = GenerationReport(success_count=100, duration_s=2.0)
    assert report.throughput == 50.0


def test_report_throughput_zero_duration():
    report = GenerationReport(success_count=10, duration_s=0.0)
    assert report.throughput == 0.0


def test_report_error_summary_groups_by_message():
    report = GenerationReport(
        failed={
            "u1": ["Field 'score': Value is NaN"],
            "u2": ["Field 'score': Value is NaN"],
            "u3": ["KeyError: 'missing'"],
        }
    )
    summary = report.error_summary()
    assert summary["Field 'score': Value is NaN"] == ["u1", "u2"]
    assert summary["KeyError: 'missing'"] == ["u3"]


def test_report_error_summary_ordered_by_count():
    report = GenerationReport(
        failed={
            "u1": ["rare error"],
            "u2": ["common error"],
            "u3": ["common error"],
            "u4": ["common error"],
        }
    )
    keys = list(report.error_summary().keys())
    assert keys[0] == "common error"


def test_report_error_summary_empty():
    assert GenerationReport().error_summary() == {}


# ---------------------------------------------------------------------------
# context_fn tests
# ---------------------------------------------------------------------------


class ContextCaptureFeature(Feature):
    """Stores the context it received so tests can inspect it."""

    def __init__(self) -> None:
        self.received: dict[str, dict] = {}

    async def extract(self, raw, context: dict, entity_id: str | None = None) -> ExtractionResult:
        self.received[entity_id or "?"] = dict(context)
        return ExtractionResult.of(entity_id, {"v": 1.0})


class ContextCaptureBatchFeature(Feature):
    """Stores entity_contexts received by extract_batch."""

    def __init__(self) -> None:
        self.received_contexts: list[dict] | None = None

    async def extract(self, raw, context: dict, entity_id: str | None = None) -> ExtractionResult:
        return ExtractionResult.of(entity_id, {"v": 1.0})

    async def extract_batch(self, raws, context, entity_ids=None, entity_contexts=None):
        self.received_contexts = list(entity_contexts) if entity_contexts is not None else None
        eids = entity_ids or [None] * len(raws)
        return [ExtractionResult.of(eid, {"v": 1.0}) for eid in eids]


@pytest.mark.asyncio
async def test_context_fn_per_entity_context_received(df):
    """context_fn result should be visible in Feature.extract for each entity."""
    feature = ContextCaptureFeature()
    pipeline = Pipeline(source=DataFrameSource(df), feature=feature, store=MemoryStore())

    labels = {"u1": "premium", "u2": "standard"}
    await pipeline.agenerate(
        entity_ids=["u1", "u2"],
        context={"shared": True},
        context_fn=lambda eid: {"tier": labels[eid]},
    )

    assert feature.received["u1"] == {"shared": True, "tier": "premium"}
    assert feature.received["u2"] == {"shared": True, "tier": "standard"}


@pytest.mark.asyncio
async def test_context_fn_overrides_shared_context(df):
    """Entity-specific values from context_fn should shadow shared context keys."""
    feature = ContextCaptureFeature()
    pipeline = Pipeline(source=DataFrameSource(df), feature=feature, store=MemoryStore())

    await pipeline.agenerate(
        entity_ids=["u1", "u2"],
        context={"key": "shared_value"},
        context_fn=lambda eid: {"key": f"entity_{eid}"},
    )

    assert feature.received["u1"]["key"] == "entity_u1"
    assert feature.received["u2"]["key"] == "entity_u2"


@pytest.mark.asyncio
async def test_context_fn_without_shared_context(df):
    """context_fn should work even when no shared context is provided."""
    feature = ContextCaptureFeature()
    pipeline = Pipeline(source=DataFrameSource(df), feature=feature, store=MemoryStore())

    await pipeline.agenerate(
        entity_ids=["u1"],
        context_fn=lambda eid: {"entity_label": eid.upper()},
    )

    assert feature.received["u1"] == {"entity_label": "U1"}


@pytest.mark.asyncio
async def test_context_fn_with_batch_size_passes_entity_contexts(wide_df):
    """With batch_size > 1, context_fn results should reach extract_batch as entity_contexts."""
    feature = ContextCaptureBatchFeature()
    entity_ids = [f"e{i}" for i in range(4)]
    pipeline = Pipeline(source=DataFrameSource(wide_df), feature=feature, store=MemoryStore())

    await pipeline.agenerate(
        entity_ids=entity_ids,
        context={"shared": 1},
        context_fn=lambda eid: {"label": eid},
        batch_size=4,
    )

    assert feature.received_contexts is not None
    assert len(feature.received_contexts) == 4
    for i, ctx in enumerate(feature.received_contexts):
        assert ctx["shared"] == 1
        assert ctx["label"] == entity_ids[i]


@pytest.mark.asyncio
async def test_context_fn_none_passes_shared_context_to_batch(wide_df):
    """Without context_fn, entity_contexts should be None in extract_batch."""
    feature = ContextCaptureBatchFeature()
    entity_ids = [f"e{i}" for i in range(3)]
    pipeline = Pipeline(source=DataFrameSource(wide_df), feature=feature, store=MemoryStore())

    await pipeline.agenerate(entity_ids=entity_ids, context={"shared": 1}, batch_size=3)

    assert feature.received_contexts is None


# ---------------------------------------------------------------------------
# executor tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_executor_per_entity_success(df):
    """executor path should succeed and write values to the store."""
    from concurrent.futures import ThreadPoolExecutor

    pipeline = Pipeline(source=DataFrameSource(df), feature=MeanFeature(), store=MemoryStore())
    with ThreadPoolExecutor(max_workers=2) as pool:
        report = await pipeline.agenerate(entity_ids=["u1", "u2"], executor=pool, concurrency=2)

    assert report.success_count == 2
    assert report.failure_count == 0
    # Values must be persisted to the store (write happens in main process)
    assert await pipeline.aretrieve("u1") == {"mean_value": 15.0}
    assert await pipeline.aretrieve("u2") == {"mean_value": 15.0}


@pytest.mark.asyncio
async def test_executor_exception_recorded_as_failure(df):
    """Exceptions raised inside the executor should appear in report.failed."""
    from concurrent.futures import ThreadPoolExecutor

    pipeline = Pipeline(source=DataFrameSource(df), feature=FailingFeature(), store=MemoryStore())
    with ThreadPoolExecutor(max_workers=2) as pool:
        report = await pipeline.agenerate(entity_ids=["u1"], executor=pool)

    assert report.failure_count == 1
    assert "u1" in report.failed
    assert "RuntimeError" in report.failed["u1"][0]


@pytest.mark.asyncio
async def test_executor_validation_failure_recorded(df):
    """Schema validation failures from the executor path appear in report.failed."""
    from concurrent.futures import ThreadPoolExecutor

    class WrongTypeFeature(Feature):
        schema = FeatureSchema({"mean_value": types.Float64(nullable=False)})

        async def extract(self, raw, context, entity_id=None):
            return ExtractionResult.of(entity_id, {"mean_value": "not_a_float"})

    pipeline = Pipeline(source=DataFrameSource(df), feature=WrongTypeFeature(), store=MemoryStore())
    with ThreadPoolExecutor(max_workers=1) as pool:
        report = await pipeline.agenerate(entity_ids=["u1"], executor=pool)

    assert report.failure_count == 1
    assert "u1" in report.failed


@pytest.mark.asyncio
async def test_executor_store_write_in_main_process(df):
    """store.write must occur in the main process so MemoryStore is populated."""
    from concurrent.futures import ThreadPoolExecutor

    store = MemoryStore()
    pipeline = Pipeline(source=DataFrameSource(df), feature=MeanFeature(), store=store)
    with ThreadPoolExecutor(max_workers=2) as pool:
        await pipeline.agenerate(entity_ids=["u1", "u2"], executor=pool, concurrency=2)

    # Both reads must be visible in the same MemoryStore instance
    result = await store.aread(MeanFeature(), "u1")
    assert "mean_value" in result


@pytest.mark.asyncio
async def test_executor_with_batch_size(wide_df):
    """With batch_size > 1, each batch is dispatched as a single executor call."""
    from concurrent.futures import ThreadPoolExecutor

    feature = BatchTrackingFeature()
    entity_ids = [f"u{i}" for i in range(1, 7)]
    pipeline = Pipeline(source=DataFrameSource(wide_df), feature=feature, store=MemoryStore())

    with ThreadPoolExecutor(max_workers=2) as pool:
        report = await pipeline.agenerate(
            entity_ids=entity_ids, batch_size=3, concurrency=2, executor=pool
        )

    assert report.success_count == 6
    # Two batches of 3 → extract_batch called twice (once per batch)
    assert feature.calls == [3, 3]


@pytest.mark.asyncio
async def test_executor_with_concurrency(wide_df):
    """concurrency > 1 with an executor should process all entities correctly."""
    from concurrent.futures import ThreadPoolExecutor

    entity_ids = [f"u{i}" for i in range(1, 7)]
    pipeline = Pipeline(source=DataFrameSource(wide_df), feature=MeanFeature(), store=MemoryStore())

    with ThreadPoolExecutor(max_workers=3) as pool:
        report = await pipeline.agenerate(entity_ids=entity_ids, executor=pool, concurrency=3)

    assert report.success_count == 6
    assert report.failure_count == 0


# ---------------------------------------------------------------------------
# sync wrapper tests
# ---------------------------------------------------------------------------


def test_generate_sync_basic(df):
    """generate_sync should return a GenerationReport and values should be stored."""
    pipeline = Pipeline(source=DataFrameSource(df), feature=MeanFeature(), store=MemoryStore())
    report = pipeline.generate(entity_ids=["u1", "u2"])

    assert isinstance(report, GenerationReport)
    assert report.success_count == 2
    assert report.failure_count == 0


def test_retrieve_sync(df):
    """retrieve_sync should return the value previously stored."""
    pipeline = Pipeline(source=DataFrameSource(df), feature=MeanFeature(), store=MemoryStore())
    pipeline.generate(entity_ids=["u1"])

    result = pipeline.retrieve("u1")
    assert result == {"mean_value": 15.0}


def test_retrieve_batch_sync(df):
    """retrieve_batch_sync should return a dict of stored values."""
    pipeline = Pipeline(source=DataFrameSource(df), feature=MeanFeature(), store=MemoryStore())
    pipeline.generate(entity_ids=["u1", "u2"])

    results = pipeline.retrieve_batch(["u1", "u2", "u_missing"])
    assert set(results.keys()) == {"u1", "u2"}
    assert "u_missing" not in results


# ---------------------------------------------------------------------------
# partition_context_fn tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_partition_context_fn_visible_in_extract(wide_df):
    """partition_context_fn result should be visible in Feature.extract."""
    feature = ContextCaptureFeature()
    entity_ids = ["u1", "u2", "u3", "u4", "u5", "u6"]
    pipeline = Pipeline(source=DataFrameSource(wide_df), feature=feature, store=MemoryStore())

    # Group odd/even entities into two partitions
    await pipeline.agenerate(
        entity_ids=entity_ids,
        partition_by=lambda eid: int(eid[1:]) % 2,  # 0 = even, 1 = odd
        partition_context_fn=lambda key: {"parity": "even" if key == 0 else "odd"},
    )

    # Odd entity IDs: u1, u3, u5
    for eid in ["u1", "u3", "u5"]:
        assert feature.received[eid]["parity"] == "odd"
    # Even entity IDs: u2, u4, u6
    for eid in ["u2", "u4", "u6"]:
        assert feature.received[eid]["parity"] == "even"


@pytest.mark.asyncio
async def test_partition_context_fn_with_explicit_partitions(wide_df):
    """partition_context_fn should work with explicitly passed partitions."""
    feature = ContextCaptureFeature()
    pipeline = Pipeline(source=DataFrameSource(wide_df), feature=feature, store=MemoryStore())

    await pipeline.agenerate(
        partitions={"alpha": ["u1", "u2"], "beta": ["u3", "u4"]},
        partition_context_fn=lambda key: {"group": key},
    )

    assert feature.received["u1"]["group"] == "alpha"
    assert feature.received["u2"]["group"] == "alpha"
    assert feature.received["u3"]["group"] == "beta"
    assert feature.received["u4"]["group"] == "beta"


@pytest.mark.asyncio
async def test_partition_context_fn_shadowed_by_context_fn(wide_df):
    """context_fn should override partition_context_fn for the same key."""
    feature = ContextCaptureFeature()
    pipeline = Pipeline(source=DataFrameSource(wide_df), feature=feature, store=MemoryStore())

    await pipeline.agenerate(
        partitions={"group_a": ["u1", "u2"]},
        partition_context_fn=lambda key: {"label": "partition_value"},
        context_fn=lambda eid: {"label": f"entity_{eid}"},
    )

    # entity-level context_fn wins over partition_context_fn
    assert feature.received["u1"]["label"] == "entity_u1"
    assert feature.received["u2"]["label"] == "entity_u2"


@pytest.mark.asyncio
async def test_partition_context_fn_shared_context_is_base(wide_df):
    """Shared context is the base; partition_context_fn layers on top."""
    feature = ContextCaptureFeature()
    pipeline = Pipeline(source=DataFrameSource(wide_df), feature=feature, store=MemoryStore())

    await pipeline.agenerate(
        partitions={"p": ["u1"]},
        context={"shared_key": "shared", "partition_key": "will_be_overridden"},
        partition_context_fn=lambda key: {"partition_key": key},
    )

    ctx = feature.received["u1"]
    assert ctx["shared_key"] == "shared"  # shared context preserved
    assert ctx["partition_key"] == "p"  # partition_context_fn shadows


@pytest.mark.asyncio
async def test_partition_key_injected_with_explicit_partitions(wide_df):
    """_partition_key should be in context when using explicit partitions."""
    feature = ContextCaptureFeature()
    pipeline = Pipeline(source=DataFrameSource(wide_df), feature=feature, store=MemoryStore())

    await pipeline.agenerate(partitions={"shard_a": ["u1", "u2"], "shard_b": ["u3"]})

    assert feature.received["u1"]["_partition_key"] == "shard_a"
    assert feature.received["u2"]["_partition_key"] == "shard_a"
    assert feature.received["u3"]["_partition_key"] == "shard_b"


@pytest.mark.asyncio
async def test_partition_key_injected_with_partition_by(wide_df):
    """_partition_key should be in context when using partition_by."""
    feature = ContextCaptureFeature()
    pipeline = Pipeline(source=DataFrameSource(wide_df), feature=feature, store=MemoryStore())

    await pipeline.agenerate(
        entity_ids=["u1", "u2", "u3"],
        partition_by=lambda eid: "even" if int(eid[1]) % 2 == 0 else "odd",
    )

    assert feature.received["u1"]["_partition_key"] == "odd"
    assert feature.received["u2"]["_partition_key"] == "even"
    assert feature.received["u3"]["_partition_key"] == "odd"


@pytest.mark.asyncio
async def test_partition_context_fn_can_shadow_partition_key(wide_df):
    """partition_context_fn values layer on top of _partition_key."""
    feature = ContextCaptureFeature()
    pipeline = Pipeline(source=DataFrameSource(wide_df), feature=feature, store=MemoryStore())

    await pipeline.agenerate(
        partitions={"p": ["u1"]},
        partition_context_fn=lambda key: {"extra": f"from_{key}"},
    )

    ctx = feature.received["u1"]
    assert ctx["_partition_key"] == "p"
    assert ctx["extra"] == "from_p"


# ---------------------------------------------------------------------------
# Context forwarding to source.read and store.write
# ---------------------------------------------------------------------------


class ContextCapturingSource(DataSource):
    """Source that records the context kwarg it receives on each read."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        self.read_contexts: dict[str, Any] = {}

    async def read(
        self, entity_id: str | None = None, context: dict | None = None, **kwargs: Any
    ) -> pd.DataFrame:
        self.read_contexts[entity_id] = context
        return self._df[self._df["entity_id"] == entity_id]


class ContextCapturingStore(MemoryStore):
    """Store that records the context kwarg it receives on each write."""

    def __init__(self) -> None:
        super().__init__()
        self.write_contexts: dict[str, Any] = {}

    async def awrite(
        self, feature: Any, entity_id: str, data: Any, context: dict | None = None
    ) -> None:
        self.write_contexts[entity_id] = context
        await super().awrite(feature, entity_id, data)


@pytest.mark.asyncio
async def test_context_forwarded_to_source_read(df):
    source = ContextCapturingSource(df)
    pipeline = Pipeline(source=source, feature=MeanFeature(), store=MemoryStore())

    ctx = {"region": "us-east", "env": "prod"}
    await pipeline.agenerate(entity_ids=["u1", "u2"], context=ctx)

    assert source.read_contexts["u1"] == ctx
    assert source.read_contexts["u2"] == ctx


@pytest.mark.asyncio
async def test_context_forwarded_to_store_write(df):
    store = ContextCapturingStore()
    pipeline = Pipeline(source=DataFrameSource(df), feature=MeanFeature(), store=store)

    ctx = {"region": "us-east"}
    await pipeline.agenerate(entity_ids=["u1", "u2"], context=ctx)

    assert store.write_contexts["u1"] == ctx
    assert store.write_contexts["u2"] == ctx


@pytest.mark.asyncio
async def test_partition_context_forwarded_to_source(df):
    """Source should receive the fully merged (partition + entity) context."""
    source = ContextCapturingSource(df)
    pipeline = Pipeline(source=source, feature=MeanFeature(), store=MemoryStore())

    await pipeline.agenerate(
        entity_ids=["u1", "u2"],
        context={"base": True},
        partition_by=lambda eid: "group",
        partition_context_fn=lambda key: {"partition": key},
        context_fn=lambda eid: {"entity": eid},
    )

    assert source.read_contexts["u1"] == {
        "base": True,
        "_partition_key": "group",
        "partition": "group",
        "entity": "u1",
    }
    assert source.read_contexts["u2"] == {
        "base": True,
        "_partition_key": "group",
        "partition": "group",
        "entity": "u2",
    }


@pytest.mark.asyncio
async def test_context_forwarded_to_source_and_store_in_batch_mode(df):
    """Context should reach source.read and store.write in batch (extract_batch) mode."""
    source = ContextCapturingSource(df)
    store = ContextCapturingStore()
    pipeline = Pipeline(source=source, feature=MeanFeature(), store=store)

    ctx = {"batch": True}
    await pipeline.agenerate(entity_ids=["u1", "u2"], context=ctx, batch_size=2)

    assert source.read_contexts["u1"] == ctx
    assert source.read_contexts["u2"] == ctx
    assert store.write_contexts["u1"] == ctx
    assert store.write_contexts["u2"] == ctx
