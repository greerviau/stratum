"""Tests for Pipeline and GenerationReport."""

from __future__ import annotations

import asyncio

import pandas as pd
import pytest

from stratum import GenerationReport, Pipeline
from stratum.features.base import Feature
from stratum.schema import FeatureSchema, types
from stratum.sources import DataFrameSource
from stratum.stores import MemoryStore

# ---------------------------------------------------------------------------
# Helper feature implementations
# ---------------------------------------------------------------------------


class MeanFeature(Feature):
    schema = FeatureSchema({"mean_value": types.Float64(nullable=False)})

    async def extract(self, raw: pd.DataFrame, context: dict, entity_id: str | None = None) -> dict:
        if raw.empty:
            raise ValueError("No rows for entity")
        return {"mean_value": float(raw["amount"].mean())}


class FailingFeature(Feature):
    async def extract(self, raw, context, entity_id=None):
        raise RuntimeError("Intentional extraction failure")


class HookTrackingFeature(Feature):
    def __init__(self) -> None:
        self.pre_called = False
        self.post_called = False

    async def pre_extract(self, raw):
        self.pre_called = True
        return raw

    async def extract(self, raw, context, entity_id=None):
        return {"value": 42.0}

    async def post_extract(self, result):
        self.post_called = True
        return result


class SlowFeature(Feature):
    """Yields once during extract so other coroutines can interleave."""

    def __init__(self, order_log: list[str]) -> None:
        self.order_log = order_log

    async def extract(self, raw: pd.DataFrame, context: dict, entity_id: str | None = None) -> dict:
        self.order_log.append(f"start:{entity_id or '?'}")
        await asyncio.sleep(0)  # yield to event loop
        self.order_log.append(f"end:{entity_id or '?'}")
        return {"value": 1.0}


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
    report = await pipeline.generate(entity_ids=["u1", "u2"])

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
    report = await pipeline.generate(entity_ids=["u1", "u2"])

    # u1 has rows [10, 20] → mean 15; u2 has [15] → mean 15
    assert report.succeeded["u1"]["mean_value"] == 15.0
    assert report.succeeded["u2"]["mean_value"] == 15.0


@pytest.mark.asyncio
async def test_generate_never_raises_on_entity_failure(df):
    """A failing feature should not crash generate() — it collects errors."""
    pipeline = Pipeline(
        source=DataFrameSource(df),
        feature=FailingFeature(),
        store=MemoryStore(),
    )
    report = await pipeline.generate(entity_ids=["u1", "u2"])

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
    report = await pipeline.generate(entity_ids=["u1", "u_missing"])

    assert "u1" in report.succeeded
    assert "u_missing" in report.failed


@pytest.mark.asyncio
async def test_generate_schema_violation_captured_as_failure(df):
    """Schema violations should be recorded in failed, not raise."""

    class BadFeature(Feature):
        schema = FeatureSchema({"score": types.Float64(nullable=False)})

        async def extract(self, raw, context, entity_id=None):
            return {"score": "not_a_float"}  # wrong type

    pipeline = Pipeline(
        source=DataFrameSource(df),
        feature=BadFeature(),
        store=MemoryStore(),
    )
    report = await pipeline.generate(entity_ids=["u1"])

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
    report = await pipeline.generate(entity_ids=["u1"])
    assert "u1" in report.succeeded


@pytest.mark.asyncio
async def test_generate_context_forwarded_to_extract(df):
    received: dict = {}

    class ContextCapture(Feature):
        async def extract(self, raw, context, entity_id=None):
            received.update(context)
            return {"v": 1.0}

    pipeline = Pipeline(
        source=DataFrameSource(df),
        feature=ContextCapture(),
        store=MemoryStore(),
    )
    ctx = {"version": "v2", "ts": 999}
    await pipeline.generate(entity_ids=["u1"], context=ctx)

    assert received == ctx


@pytest.mark.asyncio
async def test_generate_hooks_called(df):
    feature = HookTrackingFeature()
    pipeline = Pipeline(
        source=DataFrameSource(df),
        feature=feature,
        store=MemoryStore(),
    )
    await pipeline.generate(entity_ids=["u1"])

    assert feature.pre_called
    assert feature.post_called


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
    report = await pipeline.generate(entity_ids=ids, concurrency=3)

    assert report.success_count == 6
    assert report.failure_count == 0
    for i in range(1, 7):
        assert report.succeeded[f"u{i}"]["mean_value"] == float(i)


@pytest.mark.asyncio
async def test_generate_flat_concurrency_full_parallelism(wide_df):
    """concurrency >= len(ids) should process everything concurrently."""
    pipeline = Pipeline(
        source=DataFrameSource(wide_df),
        feature=MeanFeature(),
        store=MemoryStore(),
    )
    ids = [f"u{i}" for i in range(1, 7)]
    report = await pipeline.generate(entity_ids=ids, concurrency=100)

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
    report = await pipeline.generate(
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
    report = await pipeline.generate(
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
            return {"v": 1.0}

    # Two partitions with 3 entities each; concurrency=2 lets both run at once.
    # Entities within each partition must still appear in order.
    source_map = {f"p0_e{i}": f"p0_e{i}" for i in range(3)}
    source_map.update({f"p1_e{i}": f"p1_e{i}" for i in range(3)})

    class MappedSource(DataFrameSource):
        async def read(self, entity_id=None, **kwargs):
            return entity_id  # raw is just the entity_id string

    from stratum.sources.base import DataSource

    class EchoSource(DataSource):
        async def read(self, entity_id=None, **kwargs):
            return entity_id

    pipeline = Pipeline(
        source=EchoSource(),
        feature=OrderedFeature(),
        store=MemoryStore(),
    )
    ids = [f"p0_e{i}" for i in range(3)] + [f"p1_e{i}" for i in range(3)]
    await pipeline.generate(
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
    report = await pipeline.generate(
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
    from stratum.sources.base import DataSource

    class EchoSource(DataSource):
        async def read(self, entity_id=None, **kwargs):
            return entity_id

    processed: list[str] = []

    class RecordFeature(Feature):
        async def extract(self, raw, context, entity_id=None):
            processed.append(raw)
            return {"v": 1.0}

    pipeline = Pipeline(
        source=EchoSource(),
        feature=RecordFeature(),
        store=MemoryStore(),
    )
    await pipeline.generate(
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
    await pipeline.generate(entity_ids=["u1"])

    # Second run: u1 should be skipped, u2 should be processed
    report = await pipeline.generate(entity_ids=["u1", "u2"], overwrite=False)

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

    await pipeline.generate(entity_ids=["u1"])

    # Overwrite with same data — should succeed, not skip
    report = await pipeline.generate(entity_ids=["u1"], overwrite=True)

    assert report.success_count == 1
    assert report.skip_count == 0


@pytest.mark.asyncio
async def test_generate_overwrite_false_progress_includes_skips(df):
    """on_progress should be called for skipped entities too."""
    store = MemoryStore()
    pipeline = Pipeline(source=DataFrameSource(df), feature=MeanFeature(), store=store)
    await pipeline.generate(entity_ids=["u1", "u2"])

    calls: list[tuple[int, int]] = []
    report = await pipeline.generate(
        entity_ids=["u1", "u2"],
        overwrite=False,
        on_progress=lambda c, t, _: calls.append((c, t)),
    )

    assert report.skip_count == 2
    assert len(calls) == 2
    assert calls[-1] == (2, 2)


# ---------------------------------------------------------------------------
# generate() — on_progress callback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_on_progress_call_count(df):
    """on_progress is called exactly once per entity."""
    calls: list[tuple[int, int]] = []
    pipeline = Pipeline(source=DataFrameSource(df), feature=MeanFeature(), store=MemoryStore())

    await pipeline.generate(
        entity_ids=["u1", "u2"],
        on_progress=lambda c, t, _: calls.append((c, t)),
    )

    assert len(calls) == 2
    assert calls[-1] == (2, 2)


@pytest.mark.asyncio
async def test_generate_on_progress_total_matches_entity_count(wide_df):
    """total reported to on_progress matches the number of entities."""
    totals: list[int] = []
    pipeline = Pipeline(source=DataFrameSource(wide_df), feature=MeanFeature(), store=MemoryStore())
    ids = [f"u{i}" for i in range(1, 7)]

    await pipeline.generate(
        entity_ids=ids,
        on_progress=lambda c, t, _: totals.append(t),
    )

    assert all(t == 6 for t in totals)


@pytest.mark.asyncio
async def test_generate_on_progress_receives_live_report(df):
    """The report passed to on_progress reflects state at time of call."""
    snapshots: list[int] = []
    pipeline = Pipeline(source=DataFrameSource(df), feature=MeanFeature(), store=MemoryStore())

    await pipeline.generate(
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

    await pipeline.generate(
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
        await pipeline.generate(entity_ids=["u1"], partitions={"p": ["u1"]})


@pytest.mark.asyncio
async def test_generate_raises_if_neither_entity_ids_nor_partitions(df):
    pipeline = Pipeline(source=DataFrameSource(df), feature=MeanFeature(), store=MemoryStore())
    with pytest.raises(ValueError):
        await pipeline.generate()


@pytest.mark.asyncio
async def test_generate_raises_if_partition_by_with_partitions(df):
    pipeline = Pipeline(source=DataFrameSource(df), feature=MeanFeature(), store=MemoryStore())
    with pytest.raises(ValueError, match="partition_by"):
        await pipeline.generate(
            partitions={"p": ["u1"]},
            partition_by=lambda e: e,
        )


@pytest.mark.asyncio
async def test_generate_raises_if_concurrency_less_than_one(df):
    pipeline = Pipeline(source=DataFrameSource(df), feature=MeanFeature(), store=MemoryStore())
    with pytest.raises(ValueError, match="concurrency"):
        await pipeline.generate(entity_ids=["u1"], concurrency=0)


@pytest.mark.asyncio
async def test_generate_raises_if_batch_size_less_than_one(df):
    pipeline = Pipeline(source=DataFrameSource(df), feature=MeanFeature(), store=MemoryStore())
    with pytest.raises(ValueError, match="batch_size"):
        await pipeline.generate(entity_ids=["u1"], batch_size=0)


# ---------------------------------------------------------------------------
# batch_size / extract_batch tests
# ---------------------------------------------------------------------------


class BatchTrackingFeature(Feature):
    """Records how extract_batch is called so tests can inspect batch sizes."""

    def __init__(self) -> None:
        self.calls: list[int] = []  # sizes of each extract_batch call

    async def extract(self, raw, context, entity_id=None):
        return {"v": 1.0}

    async def extract_batch(self, raws, context, entity_ids=None):
        self.calls.append(len(raws))
        return [{"v": float(i)} for i in range(len(raws))]


class PartialFailBatchFeature(Feature):
    """extract_batch returns an exception for every other entity."""

    async def extract(self, raw, context, entity_id=None):
        return {"v": 1.0}

    async def extract_batch(self, raws, context, entity_ids=None):
        results = []
        for i, _raw in enumerate(raws):
            if i % 2 == 1:
                results.append(ValueError(f"Simulated failure for item {i}"))
            else:
                results.append({"v": float(i)})
        return results


class WholeBatchFailFeature(Feature):
    """extract_batch always raises, failing the entire batch."""

    async def extract(self, raw, context, entity_id=None):
        return {"v": 1.0}

    async def extract_batch(self, raws, context, entity_ids=None):
        raise RuntimeError("Whole batch exploded")


@pytest.mark.asyncio
async def test_batch_extract_called_once_per_batch(wide_df):
    """extract_batch should be called once per batch, not once per entity."""
    feature = BatchTrackingFeature()
    pipeline = Pipeline(source=DataFrameSource(wide_df), feature=feature, store=MemoryStore())

    ids = [f"u{i}" for i in range(1, 7)]
    report = await pipeline.generate(entity_ids=ids, batch_size=3)

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
    report = await pipeline.generate(entity_ids=ids, batch_size=6)

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
    report = await pipeline.generate(entity_ids=ids, batch_size=6)

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
    report = await pipeline.generate(entity_ids=ids, batch_size=6)

    assert report.success_count == 0
    assert report.failure_count == 6


@pytest.mark.asyncio
async def test_batch_extract_default_falls_back_to_extract(df):
    """The default extract_batch calls extract() per item — same results."""
    # MeanFeature only implements extract(), so extract_batch uses the default
    pipeline = Pipeline(source=DataFrameSource(df), feature=MeanFeature(), store=MemoryStore())

    r_individual = await pipeline.generate(entity_ids=["u1", "u2"])
    store2 = MemoryStore()
    pipeline2 = Pipeline(source=DataFrameSource(df), feature=MeanFeature(), store=store2)
    r_batch = await pipeline2.generate(entity_ids=["u1", "u2"], batch_size=2)

    assert r_individual.succeeded["u1"] == r_batch.succeeded["u1"]
    assert r_individual.succeeded["u2"] == r_batch.succeeded["u2"]


@pytest.mark.asyncio
async def test_batch_extract_default_per_item_failure_isolated(df):
    """Default extract_batch wraps individual failures — other items still succeed."""
    pipeline = Pipeline(source=DataFrameSource(df), feature=MeanFeature(), store=MemoryStore())
    # u_missing will fail; u1 should still succeed
    report = await pipeline.generate(entity_ids=["u1", "u_missing"], batch_size=2)

    assert "u1" in report.succeeded
    assert "u_missing" in report.failed


@pytest.mark.asyncio
async def test_batch_size_with_concurrency(wide_df):
    """batch_size and concurrency can be combined: concurrent batches of N."""
    feature = BatchTrackingFeature()
    pipeline = Pipeline(source=DataFrameSource(wide_df), feature=feature, store=MemoryStore())

    ids = [f"u{i}" for i in range(1, 7)]
    report = await pipeline.generate(entity_ids=ids, batch_size=2, concurrency=3)

    assert report.success_count == 6
    # 6 entities / batch_size 2 → 3 batches, each called once
    assert len(feature.calls) == 3


@pytest.mark.asyncio
async def test_batch_size_with_overwrite_false(wide_df):
    """overwrite=False skips in batch mode; remaining entities still processed."""
    store = MemoryStore()
    pipeline = Pipeline(source=DataFrameSource(wide_df), feature=MeanFeature(), store=store)

    # Pre-populate u1, u2
    await pipeline.generate(entity_ids=["u1", "u2"])

    ids = [f"u{i}" for i in range(1, 7)]
    report = await pipeline.generate(entity_ids=ids, batch_size=3, overwrite=False)

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
    await pipeline.generate(
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
    report = await pipeline.generate(
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
    await pipeline.generate(entity_ids=["u1"])

    result = await pipeline.retrieve("u1")
    assert result["mean_value"] == 15.0


@pytest.mark.asyncio
async def test_retrieve_missing_raises_key_error(df):
    pipeline = Pipeline(
        source=DataFrameSource(df),
        feature=MeanFeature(),
        store=MemoryStore(),
    )
    with pytest.raises(KeyError):
        await pipeline.retrieve("nonexistent")


@pytest.mark.asyncio
async def test_retrieve_batch(df):
    pipeline = Pipeline(
        source=DataFrameSource(df),
        feature=MeanFeature(),
        store=MemoryStore(),
    )
    await pipeline.generate(entity_ids=["u1", "u2"])

    results = await pipeline.retrieve_batch(["u1", "u2", "u_none"])
    assert "u1" in results
    assert "u2" in results
    assert "u_none" not in results  # missing silently omitted


# ---------------------------------------------------------------------------
# GenerationReport tests
# ---------------------------------------------------------------------------


def test_report_counts():
    report = GenerationReport(
        succeeded={"a": 1, "b": 2},
        failed={"c": ["err"]},
        skipped={"d"},
    )
    assert report.success_count == 2
    assert report.failure_count == 1
    assert report.skip_count == 1


def test_report_repr():
    report = GenerationReport(succeeded={"a": 1}, failed={}, skipped={"b"})
    assert "succeeded=1" in repr(report)
    assert "failed=0" in repr(report)
    assert "skipped=1" in repr(report)


def test_report_skipped_default_empty():
    report = GenerationReport()
    assert report.skipped == set()
    assert report.skip_count == 0
