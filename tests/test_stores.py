"""Tests for built-in FeatureStore implementations."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from stratum.features.base import Feature
from stratum.serializers import JSONSerializer, NumpySerializer
from stratum.stores import FileStore, MemoryStore
from stratum.stores.base import FeatureStore

# Optional parquet deps
try:
    import pandas  # noqa: F401
    import pyarrow  # noqa: F401

    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False


# ---------------------------------------------------------------------------
# Shared fixture feature
# ---------------------------------------------------------------------------


class DummyFeature(Feature):
    async def extract(self, raw, context, entity_id=None):
        return raw


class AnotherFeature(Feature):
    async def extract(self, raw, context, entity_id=None):
        return raw


@pytest.fixture
def feature() -> DummyFeature:
    return DummyFeature()


# ---------------------------------------------------------------------------
# MemoryStore
# ---------------------------------------------------------------------------


class TestMemoryStore:
    @pytest.mark.asyncio
    async def test_write_and_read(self, feature):
        store = MemoryStore()
        await store.write(feature, "e1", {"score": 0.9})
        assert await store.read(feature, "e1") == {"score": 0.9}

    @pytest.mark.asyncio
    async def test_exists_false_before_write(self, feature):
        store = MemoryStore()
        assert not await store.exists(feature, "e1")

    @pytest.mark.asyncio
    async def test_exists_true_after_write(self, feature):
        store = MemoryStore()
        await store.write(feature, "e1", 42)
        assert await store.exists(feature, "e1")

    @pytest.mark.asyncio
    async def test_delete(self, feature):
        store = MemoryStore()
        await store.write(feature, "e1", "data")
        await store.delete(feature, "e1")
        assert not await store.exists(feature, "e1")

    @pytest.mark.asyncio
    async def test_read_missing_raises_key_error(self, feature):
        store = MemoryStore()
        with pytest.raises(KeyError):
            await store.read(feature, "missing")

    @pytest.mark.asyncio
    async def test_delete_missing_raises_key_error(self, feature):
        store = MemoryStore()
        with pytest.raises(KeyError):
            await store.delete(feature, "missing")

    @pytest.mark.asyncio
    async def test_overwrite(self, feature):
        store = MemoryStore()
        await store.write(feature, "e1", "first")
        await store.write(feature, "e1", "second")
        assert await store.read(feature, "e1") == "second"

    @pytest.mark.asyncio
    async def test_feature_isolation(self):
        """Different feature classes should not share namespace."""
        fa = DummyFeature()
        fb = AnotherFeature()
        store = MemoryStore()

        await store.write(fa, "e1", "value_a")
        await store.write(fb, "e1", "value_b")

        assert await store.read(fa, "e1") == "value_a"
        assert await store.read(fb, "e1") == "value_b"

    @pytest.mark.asyncio
    async def test_stores_arbitrary_types(self, feature):
        store = MemoryStore()
        arr = np.zeros((3, 4), dtype=np.float32)
        await store.write(feature, "e1", arr)
        result = await store.read(feature, "e1")
        np.testing.assert_array_equal(result, arr)


# ---------------------------------------------------------------------------
# FileStore
# ---------------------------------------------------------------------------


class TestFileStore:
    @pytest.mark.asyncio
    async def test_write_and_read_pickle(self, feature):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStore(tmpdir)
            await store.write(feature, "e1", {"key": "val", "num": 7})
            result = await store.read(feature, "e1")
            assert result == {"key": "val", "num": 7}

    @pytest.mark.asyncio
    async def test_exists(self, feature):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStore(tmpdir)
            assert not await store.exists(feature, "e1")
            await store.write(feature, "e1", "hello")
            assert await store.exists(feature, "e1")

    @pytest.mark.asyncio
    async def test_delete(self, feature):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStore(tmpdir)
            await store.write(feature, "e1", "data")
            await store.delete(feature, "e1")
            assert not await store.exists(feature, "e1")

    @pytest.mark.asyncio
    async def test_read_missing_raises_key_error(self, feature):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStore(tmpdir)
            with pytest.raises(KeyError):
                await store.read(feature, "missing")

    @pytest.mark.asyncio
    async def test_json_serializer(self, feature):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStore(tmpdir, serializer=JSONSerializer())
            payload = {"name": "alice", "count": 42}
            await store.write(feature, "e1", payload)
            assert await store.read(feature, "e1") == payload

    @pytest.mark.asyncio
    async def test_numpy_serializer(self, feature):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStore(tmpdir, serializer=NumpySerializer())
            arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            await store.write(feature, "e1", arr)
            result = await store.read(feature, "e1")
            np.testing.assert_array_equal(result, arr)

    @pytest.mark.asyncio
    async def test_creates_directories_automatically(self, feature):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = f"{tmpdir}/a/b/c"
            store = FileStore(nested)
            await store.write(feature, "e1", "nested")
            assert await store.read(feature, "e1") == "nested"

    @pytest.mark.asyncio
    async def test_overwrite(self, feature):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStore(tmpdir)
            await store.write(feature, "e1", "v1")
            await store.write(feature, "e1", "v2")
            assert await store.read(feature, "e1") == "v2"


# ---------------------------------------------------------------------------
# ParquetStore (skipped if deps unavailable)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_PARQUET, reason="pandas/pyarrow not installed")
class TestParquetStore:
    @pytest.mark.asyncio
    async def test_write_and_read(self, feature):
        from stratum.stores import ParquetStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = ParquetStore(tmpdir)
            await store.write(feature, "e1", {"score": 0.5, "count": 3})
            result = await store.read(feature, "e1")
            assert result["score"] == pytest.approx(0.5)
            assert result["count"] == 3

    @pytest.mark.asyncio
    async def test_exists(self, feature):
        from stratum.stores import ParquetStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = ParquetStore(tmpdir)
            assert not await store.exists(feature, "e1")
            await store.write(feature, "e1", {"v": 1.0})
            assert await store.exists(feature, "e1")

    @pytest.mark.asyncio
    async def test_delete(self, feature):
        from stratum.stores import ParquetStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = ParquetStore(tmpdir)
            await store.write(feature, "e1", {"v": 1.0})
            await store.delete(feature, "e1")
            assert not await store.exists(feature, "e1")

    @pytest.mark.asyncio
    async def test_multiple_entities(self, feature):
        from stratum.stores import ParquetStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = ParquetStore(tmpdir)
            await store.write(feature, "e1", {"v": 1.0})
            await store.write(feature, "e2", {"v": 2.0})

            r1 = await store.read(feature, "e1")
            r2 = await store.read(feature, "e2")
            assert r1["v"] == pytest.approx(1.0)
            assert r2["v"] == pytest.approx(2.0)

    @pytest.mark.asyncio
    async def test_overwrite_entity(self, feature):
        from stratum.stores import ParquetStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = ParquetStore(tmpdir)
            await store.write(feature, "e1", {"v": 1.0})
            await store.write(feature, "e1", {"v": 99.0})
            result = await store.read(feature, "e1")
            assert result["v"] == pytest.approx(99.0)

    @pytest.mark.asyncio
    async def test_read_missing_raises_key_error(self, feature):
        from stratum.stores import ParquetStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = ParquetStore(tmpdir)
            with pytest.raises(KeyError):
                await store.read(feature, "missing")


# ---------------------------------------------------------------------------
# FeatureStore base class — default method behaviour
# ---------------------------------------------------------------------------


class ReadOnlyStore(FeatureStore):
    """Minimal store that only implements read — no write/exists/delete."""

    def __init__(self, data: dict):
        self._data = data

    async def read(self, feature, entity_id):
        key = (type(feature).__name__, entity_id)
        if key not in self._data:
            raise KeyError(entity_id)
        return self._data[key]


@pytest.mark.asyncio
async def test_read_only_store_can_be_instantiated():
    """A store that only overrides read() should be constructable."""
    store = ReadOnlyStore({("DummyFeature", "e1"): {"v": 42}})
    feature = DummyFeature()
    result = await store.read(feature, "e1")
    assert result == {"v": 42}


@pytest.mark.asyncio
async def test_read_only_store_write_raises_not_implemented():
    store = ReadOnlyStore({})
    feature = DummyFeature()
    with pytest.raises(NotImplementedError, match="write"):
        await store.write(feature, "e1", {"v": 1})


@pytest.mark.asyncio
async def test_read_only_store_exists_raises_not_implemented():
    store = ReadOnlyStore({})
    feature = DummyFeature()
    with pytest.raises(NotImplementedError, match="exists"):
        await store.exists(feature, "e1")


@pytest.mark.asyncio
async def test_read_only_store_delete_raises_not_implemented():
    store = ReadOnlyStore({})
    feature = DummyFeature()
    with pytest.raises(NotImplementedError, match="delete"):
        await store.delete(feature, "e1")
