"""Tests for fan-out extraction via ExtractionResult."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from calcine import ExtractionResult, Pipeline
from calcine.features.base import Feature
from calcine.schema import FeatureSchema, types
from calcine.sources.base import DataSource
from calcine.stores.memory import MemoryStore

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class SimpleSource(DataSource):
    """Returns a dict keyed by entity_id."""

    def __init__(self, data: dict):
        self._data = data

    async def read(self, entity_id: str, context: dict) -> dict:
        if entity_id not in self._data:
            raise KeyError(entity_id)
        return self._data[entity_id]


class SegmentFeature(Feature):
    """Fan-out: one recording → N segment records with parent metadata."""

    metadata_schema = FeatureSchema(
        {
            "sample_rate": types.Int64(nullable=False),
            "speaker_id": types.String(nullable=True),
        }
    )
    schema = FeatureSchema(
        {
            "rms": types.Float64(nullable=False),
            "duration_ms": types.Float64(nullable=False),
        }
    )

    async def extract(
        self, raw: dict, context: dict, entity_id: str | None = None
    ) -> ExtractionResult:
        return ExtractionResult(
            metadata={"sample_rate": raw["sample_rate"], "speaker_id": raw.get("speaker_id")},
            records={
                f"{entity_id}/{i}": {"rms": seg["rms"], "duration_ms": seg["duration_ms"]}
                for i, seg in enumerate(raw["segments"])
            },
        )


class NoMetaSegmentFeature(Feature):
    """Fan-out without parent metadata."""

    schema = FeatureSchema({"value": types.Float64(nullable=False)})

    async def extract(
        self, raw: dict, context: dict, entity_id: str | None = None
    ) -> ExtractionResult:
        return ExtractionResult(
            records={f"{entity_id}/{i}": {"value": v} for i, v in enumerate(raw["values"])}
        )


class BadMetaFeature(Feature):
    """Fan-out that returns invalid parent metadata."""

    metadata_schema = FeatureSchema({"count": types.Int64(nullable=False)})
    schema = FeatureSchema({"v": types.Float64(nullable=False)})

    async def extract(
        self, raw: dict, context: dict, entity_id: str | None = None
    ) -> ExtractionResult:
        return ExtractionResult(
            metadata={"count": "not-an-int"},  # should fail validation
            records={f"{entity_id}/0": {"v": 1.0}},
        )


class BadRecordFeature(Feature):
    """Fan-out that returns an invalid sub-entity record."""

    schema = FeatureSchema({"v": types.Float64(nullable=False)})

    async def extract(
        self, raw: dict, context: dict, entity_id: str | None = None
    ) -> ExtractionResult:
        return ExtractionResult(
            records={
                f"{entity_id}/0": {"v": 1.0},
                f"{entity_id}/1": {"v": "not-a-float"},  # invalid
            }
        )


class RaisingFeature(Feature):
    """Feature that raises during extract."""

    async def extract(
        self, raw: dict, context: dict, entity_id: str | None = None
    ) -> ExtractionResult:
        raise ValueError("boom")


# ---------------------------------------------------------------------------
# ExtractionResult dataclass
# ---------------------------------------------------------------------------


def test_extraction_result_with_metadata():
    r = ExtractionResult(records={"a/0": {"x": 1}}, metadata={"key": "val"})
    assert r.records == {"a/0": {"x": 1}}
    assert r.metadata == {"key": "val"}


def test_extraction_result_without_metadata():
    r = ExtractionResult(records={"a/0": {"x": 1}})
    assert r.metadata is None


def test_extraction_result_of_convenience():
    r = ExtractionResult.of("u1", {"score": 0.9})
    assert r.records == {"u1": {"score": 0.9}}
    assert r.metadata is None


# ---------------------------------------------------------------------------
# Pipeline: fan-out generate
# ---------------------------------------------------------------------------


SOURCE_DATA = {
    "rec1": {
        "sample_rate": 16000,
        "speaker_id": "alice",
        "segments": [
            {"rms": 0.1, "duration_ms": 100.0},
            {"rms": 0.2, "duration_ms": 200.0},
        ],
    },
    "rec2": {
        "sample_rate": 44100,
        "speaker_id": None,
        "segments": [
            {"rms": 0.5, "duration_ms": 50.0},
        ],
    },
}


@pytest.mark.asyncio
async def test_fanout_basic_generate():
    store = MemoryStore()
    pipeline = Pipeline(
        source=SimpleSource(SOURCE_DATA),
        feature=SegmentFeature(),
        store=store,
    )
    report = await pipeline.agenerate(entity_ids=["rec1", "rec2"])

    assert report.success_count == 2
    assert report.failure_count == 0

    # Parent metadata stored under the source entity_id
    meta1 = await store.aread(SegmentFeature(), "rec1")
    assert meta1 == {"sample_rate": 16000, "speaker_id": "alice"}

    meta2 = await store.aread(SegmentFeature(), "rec2")
    assert meta2 == {"sample_rate": 44100, "speaker_id": None}

    # Sub-entity records
    seg0 = await store.aread(SegmentFeature(), "rec1/0")
    assert seg0 == {"rms": 0.1, "duration_ms": 100.0}

    seg1 = await store.aread(SegmentFeature(), "rec1/1")
    assert seg1 == {"rms": 0.2, "duration_ms": 200.0}

    seg_r2 = await store.aread(SegmentFeature(), "rec2/0")
    assert seg_r2 == {"rms": 0.5, "duration_ms": 50.0}


@pytest.mark.asyncio
async def test_fanout_report_stores_extraction_result():
    store = MemoryStore()
    pipeline = Pipeline(
        source=SimpleSource(SOURCE_DATA),
        feature=SegmentFeature(),
        store=store,
    )
    report = await pipeline.agenerate(entity_ids=["rec1"])
    assert isinstance(report.succeeded["rec1"], ExtractionResult)
    assert "rec1/0" in report.succeeded["rec1"].records


@pytest.mark.asyncio
async def test_fanout_store_results_false():
    store = MemoryStore()
    pipeline = Pipeline(
        source=SimpleSource(SOURCE_DATA),
        feature=SegmentFeature(),
        store=store,
    )
    report = await pipeline.agenerate(entity_ids=["rec1"], store_results=False)
    assert report.success_count == 1
    assert len(report.succeeded) == 0
    # Data is still written to the store
    assert await store.aexists(SegmentFeature(), "rec1")


@pytest.mark.asyncio
async def test_fanout_no_metadata():
    data = {"e1": {"values": [1.0, 2.0, 3.0]}}
    store = MemoryStore()
    pipeline = Pipeline(
        source=SimpleSource(data),
        feature=NoMetaSegmentFeature(),
        store=store,
    )
    report = await pipeline.agenerate(entity_ids=["e1"])
    assert report.success_count == 1

    # Tombstone written under parent key for overwrite=False support
    parent_val = await store.aread(NoMetaSegmentFeature(), "e1")
    assert parent_val == {}

    sub0 = await store.aread(NoMetaSegmentFeature(), "e1/0")
    assert sub0 == {"value": 1.0}


# ---------------------------------------------------------------------------
# Pipeline: overwrite=False
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fanout_overwrite_false_skips_existing():
    store = MemoryStore()
    feature = SegmentFeature()
    pipeline = Pipeline(source=SimpleSource(SOURCE_DATA), feature=feature, store=store)

    # First run
    report1 = await pipeline.agenerate(entity_ids=["rec1"], overwrite=False)
    assert report1.success_count == 1
    assert report1.skip_count == 0

    # Second run — parent entity exists; should skip
    report2 = await pipeline.agenerate(entity_ids=["rec1"], overwrite=False)
    assert report2.skip_count == 1
    assert report2.success_count == 0


@pytest.mark.asyncio
async def test_fanout_overwrite_true_rewrites():
    store = MemoryStore()
    feature = SegmentFeature()
    pipeline = Pipeline(source=SimpleSource(SOURCE_DATA), feature=feature, store=store)

    await pipeline.agenerate(entity_ids=["rec1"])
    report2 = await pipeline.agenerate(entity_ids=["rec1"], overwrite=True)
    assert report2.success_count == 1
    assert report2.skip_count == 0


# ---------------------------------------------------------------------------
# Pipeline: validation failures
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fanout_bad_parent_metadata_fails_entity():
    data = {"e1": {}}
    store = MemoryStore()
    pipeline = Pipeline(
        source=SimpleSource(data),
        feature=BadMetaFeature(),
        store=store,
    )
    report = await pipeline.agenerate(entity_ids=["e1"])
    assert report.failure_count == 1
    assert report.success_count == 0
    assert "e1" in report.failed


@pytest.mark.asyncio
async def test_fanout_bad_record_fails_entity():
    data = {"e1": {}}
    store = MemoryStore()
    pipeline = Pipeline(
        source=SimpleSource(data),
        feature=BadRecordFeature(),
        store=store,
    )
    report = await pipeline.agenerate(entity_ids=["e1"])
    assert report.failure_count == 1
    errors = report.failed["e1"]
    # Error messages should reference the bad sub-entity
    assert any("e1/1" in e for e in errors)


@pytest.mark.asyncio
async def test_fanout_extract_raises_is_caught():
    data = {"e1": {}}
    store = MemoryStore()
    pipeline = Pipeline(
        source=SimpleSource(data),
        feature=RaisingFeature(),
        store=store,
    )
    report = await pipeline.agenerate(entity_ids=["e1"])
    assert report.failure_count == 1
    assert "boom" in report.failed["e1"][0]


# ---------------------------------------------------------------------------
# MemoryStore: alist_entities / list_entities
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_alist_entities_all():
    store = MemoryStore()
    feature = SegmentFeature()
    pipeline = Pipeline(source=SimpleSource(SOURCE_DATA), feature=feature, store=store)
    await pipeline.agenerate(entity_ids=["rec1", "rec2"])

    entities = await store.alist_entities(feature)
    assert set(entities) == {"rec1", "rec1/0", "rec1/1", "rec2", "rec2/0"}


@pytest.mark.asyncio
async def test_alist_entities_with_prefix():
    store = MemoryStore()
    feature = SegmentFeature()
    pipeline = Pipeline(source=SimpleSource(SOURCE_DATA), feature=feature, store=store)
    await pipeline.agenerate(entity_ids=["rec1", "rec2"])

    sub_ids = await store.alist_entities(feature, prefix="rec1/")
    assert set(sub_ids) == {"rec1/0", "rec1/1"}


@pytest.mark.asyncio
async def test_alist_entities_empty_store():
    store = MemoryStore()
    feature = SegmentFeature()
    assert await store.alist_entities(feature) == []


def test_list_entities_sync():
    store = MemoryStore()
    feature = NoMetaSegmentFeature()
    # Write via ExtractionResult so the store sees the correct structure
    store.write(
        feature,
        "e1",
        ExtractionResult(records={"e1/0": {"value": 1.0}, "e1/1": {"value": 2.0}}),
    )

    ids = store.list_entities(feature, prefix="e1/")
    assert set(ids) == {"e1/0", "e1/1"}


# ---------------------------------------------------------------------------
# awrite with ExtractionResult directly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_awrite_writes_metadata_and_records():
    store = MemoryStore()
    feature = SegmentFeature()
    result = ExtractionResult(
        metadata={"sample_rate": 8000, "speaker_id": "bob"},
        records={
            "r/0": {"rms": 0.3, "duration_ms": 10.0},
            "r/1": {"rms": 0.4, "duration_ms": 20.0},
        },
    )
    await store.awrite(feature, "r", result)

    assert await store.aread(feature, "r") == {"sample_rate": 8000, "speaker_id": "bob"}
    assert await store.aread(feature, "r/0") == {"rms": 0.3, "duration_ms": 10.0}
    assert await store.aread(feature, "r/1") == {"rms": 0.4, "duration_ms": 20.0}


@pytest.mark.asyncio
async def test_awrite_none_metadata_writes_tombstone():
    store = MemoryStore()
    feature = NoMetaSegmentFeature()
    result = ExtractionResult(records={"e/0": {"value": 5.0}})
    await store.awrite(feature, "e", result)

    assert await store.aread(feature, "e") == {}
    assert await store.aexists(feature, "e")


@pytest.mark.asyncio
async def test_awrite_single_record_no_tombstone():
    """For single-record features, entity_id is the record key — no extra tombstone write."""
    store = MemoryStore()

    class SingleFeature(Feature):
        async def extract(self, raw, context, entity_id=None):
            return ExtractionResult.of(entity_id, {"v": raw})

    feature = SingleFeature()
    result = ExtractionResult.of("u1", {"v": 42})
    await store.awrite(feature, "u1", result)

    assert await store.aread(feature, "u1") == {"v": 42}
    # Only one entry — no spurious tombstone
    assert await store.alist_entities(feature) == ["u1"]


# ---------------------------------------------------------------------------
# Concurrency and batch_size
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fanout_concurrent_generate():
    data = {
        f"r{i}": {
            "sample_rate": 100,
            "speaker_id": None,
            "segments": [{"rms": float(i), "duration_ms": 1.0}],
        }
        for i in range(10)
    }
    store = MemoryStore()
    pipeline = Pipeline(
        source=SimpleSource(data),
        feature=SegmentFeature(),
        store=store,
    )
    report = await pipeline.agenerate(entity_ids=list(data.keys()), concurrency=5)
    assert report.success_count == 10
    assert report.failure_count == 0


@pytest.mark.asyncio
async def test_fanout_with_batch_size():
    """Fan-out features work correctly when batch_size > 1."""
    store = MemoryStore()
    pipeline = Pipeline(
        source=SimpleSource(SOURCE_DATA),
        feature=SegmentFeature(),
        store=store,
    )
    report = await pipeline.agenerate(entity_ids=["rec1", "rec2"], batch_size=4)
    assert report.success_count == 2
    assert report.failure_count == 0

    # Sub-entities must still be written correctly
    assert await store.aexists(SegmentFeature(), "rec1/0")
    assert await store.aexists(SegmentFeature(), "rec2/0")


@pytest.mark.asyncio
async def test_fanout_batch_size_with_concurrency():
    data = {
        f"r{i}": {
            "sample_rate": 100,
            "speaker_id": None,
            "segments": [{"rms": float(i), "duration_ms": 1.0}],
        }
        for i in range(8)
    }
    store = MemoryStore()
    pipeline = Pipeline(
        source=SimpleSource(data),
        feature=SegmentFeature(),
        store=store,
    )
    report = await pipeline.agenerate(entity_ids=list(data.keys()), batch_size=4, concurrency=2)
    assert report.success_count == 8
    assert report.failure_count == 0


# ---------------------------------------------------------------------------
# Executor: fan-out with ThreadPoolExecutor
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fanout_executor_success():
    store = MemoryStore()
    pipeline = Pipeline(
        source=SimpleSource(SOURCE_DATA),
        feature=SegmentFeature(),
        store=store,
    )
    with ThreadPoolExecutor(max_workers=2) as ex:
        report = await pipeline.agenerate(entity_ids=["rec1", "rec2"], executor=ex, concurrency=2)

    assert report.success_count == 2
    assert report.failure_count == 0
    assert await store.aexists(SegmentFeature(), "rec1/0")
    assert await store.aexists(SegmentFeature(), "rec2/0")


@pytest.mark.asyncio
async def test_fanout_executor_validation_failure():
    data = {"e1": {}}
    store = MemoryStore()
    pipeline = Pipeline(
        source=SimpleSource(data),
        feature=BadRecordFeature(),
        store=store,
    )
    with ThreadPoolExecutor(max_workers=1) as ex:
        report = await pipeline.agenerate(entity_ids=["e1"], executor=ex)

    assert report.failure_count == 1
    assert any("e1/1" in e for e in report.failed["e1"])


@pytest.mark.asyncio
async def test_fanout_executor_exception_caught():
    data = {"e1": {}}
    store = MemoryStore()
    pipeline = Pipeline(
        source=SimpleSource(data),
        feature=RaisingFeature(),
        store=store,
    )
    with ThreadPoolExecutor(max_workers=1) as ex:
        report = await pipeline.agenerate(entity_ids=["e1"], executor=ex)

    assert report.failure_count == 1
    assert "boom" in report.failed["e1"][0]
