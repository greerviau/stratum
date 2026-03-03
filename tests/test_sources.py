"""Tests for built-in DataSource implementations."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from calcine.exceptions import SourceError
from calcine.sources import DataFrameSource, DataSource, DirectorySource, FileSource, SourceBundle

# ---------------------------------------------------------------------------
# DataFrameSource
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df():
    return pd.DataFrame({"entity_id": ["u1", "u1", "u2"], "amount": [10.0, 20.0, 15.0]})


@pytest.mark.asyncio
async def test_dataframe_source_basic(sample_df):
    source = DataFrameSource(sample_df)
    result = await source.read(entity_id="u1")
    assert len(result) == 2
    assert sorted(result["amount"].tolist()) == [10.0, 20.0]


@pytest.mark.asyncio
async def test_dataframe_source_single_row(sample_df):
    source = DataFrameSource(sample_df)
    result = await source.read(entity_id="u2")
    assert len(result) == 1
    assert result.iloc[0]["amount"] == 15.0


@pytest.mark.asyncio
async def test_dataframe_source_missing_entity_returns_empty(sample_df):
    source = DataFrameSource(sample_df)
    result = await source.read(entity_id="u_missing")
    assert result.empty


@pytest.mark.asyncio
async def test_dataframe_source_no_entity_id_raises(sample_df):
    source = DataFrameSource(sample_df)
    with pytest.raises(SourceError):
        await source.read()


@pytest.mark.asyncio
async def test_dataframe_source_custom_entity_col():
    df = pd.DataFrame({"user": ["a", "b"], "val": [1, 2]})
    source = DataFrameSource(df, entity_col="user")
    result = await source.read(entity_id="a")
    assert len(result) == 1
    assert result.iloc[0]["val"] == 1


@pytest.mark.asyncio
async def test_dataframe_source_bad_entity_col_raises(sample_df):
    source = DataFrameSource(sample_df, entity_col="nonexistent")
    with pytest.raises(SourceError):
        await source.read(entity_id="u1")


# ---------------------------------------------------------------------------
# FileSource
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_file_source_reads_bytes():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        f.write(b"hello calcine")
        path = f.name
    try:
        source = FileSource(path)
        data = await source.read()
        assert data == b"hello calcine"
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_file_source_missing_file_raises():
    source = FileSource("/nonexistent/path/definitely_not_here.bin")
    with pytest.raises(SourceError):
        await source.read()


@pytest.mark.asyncio
async def test_file_source_accepts_entity_id_kwarg():
    """entity_id kwarg should be accepted (and ignored) without error."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"data")
        path = f.name
    try:
        source = FileSource(path)
        data = await source.read(entity_id="does_not_matter")
        assert data == b"data"
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# DirectorySource
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_directory_source_read_all():
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "file_a.txt").write_bytes(b"alpha")
        Path(tmpdir, "file_b.txt").write_bytes(b"beta")
        Path(tmpdir, "file_c.txt").write_bytes(b"gamma")

        source = DirectorySource(tmpdir, pattern="*.txt")
        files = await source.read()

        assert len(files) == 3
        assert set(files) == {b"alpha", b"beta", b"gamma"}


@pytest.mark.asyncio
async def test_directory_source_pattern_filters():
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "include.txt").write_bytes(b"yes")
        Path(tmpdir, "exclude.bin").write_bytes(b"no")

        source = DirectorySource(tmpdir, pattern="*.txt")
        files = await source.read()

        assert files == [b"yes"]


@pytest.mark.asyncio
async def test_directory_source_stream():
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "x.dat").write_bytes(b"x")
        Path(tmpdir, "y.dat").write_bytes(b"y")

        source = DirectorySource(tmpdir, pattern="*.dat")
        collected = []
        async for chunk in source.stream():
            collected.append(chunk)

        assert set(collected) == {b"x", b"y"}


@pytest.mark.asyncio
async def test_directory_source_empty_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        source = DirectorySource(tmpdir, pattern="*.txt")
        files = await source.read()
        assert files == []


# ---------------------------------------------------------------------------
# SourceBundle
# ---------------------------------------------------------------------------


class FixedSource(DataSource):
    """Test source that returns a fixed value regardless of kwargs."""

    def __init__(self, value):
        self.value = value

    async def read(self, **kwargs):
        return self.value


class EchoEntitySource(DataSource):
    """Test source that echoes back the entity_id it was called with."""

    async def read(self, entity_id=None, **kwargs):
        return entity_id


@pytest.mark.asyncio
async def test_bundle_returns_named_dict():
    source = SourceBundle(a=FixedSource(1), b=FixedSource(2), c=FixedSource(3))
    result = await source.read()
    assert result == {"a": 1, "b": 2, "c": 3}


@pytest.mark.asyncio
async def test_bundle_forwards_kwargs_to_all_sources():
    source = SourceBundle(x=EchoEntitySource(), y=EchoEntitySource())
    result = await source.read(entity_id="u42")
    assert result == {"x": "u42", "y": "u42"}


@pytest.mark.asyncio
async def test_bundle_mixed_source_types(sample_df):
    source = SourceBundle(
        frame=DataFrameSource(sample_df),
        constant=FixedSource({"config": True}),
    )
    result = await source.read(entity_id="u1")
    assert len(result["frame"]) == 2  # two rows for u1
    assert result["constant"] == {"config": True}


@pytest.mark.asyncio
async def test_bundle_single_source():
    source = SourceBundle(only=FixedSource("solo"))
    result = await source.read()
    assert result == {"only": "solo"}


def test_bundle_requires_at_least_one_source():
    with pytest.raises(ValueError):
        SourceBundle()


@pytest.mark.asyncio
async def test_bundle_in_pipeline(sample_df):
    """SourceBundle integrates end-to-end with Pipeline."""
    from calcine import Pipeline
    from calcine.features.base import Feature
    from calcine.stores import MemoryStore

    class CombinedFeature(Feature):
        async def extract(self, raw: dict, context: dict, entity_id=None) -> dict:
            frame = raw["purchases"]
            multiplier = raw["config"]["multiplier"]
            return {"value": float(frame["amount"].mean()) * multiplier}

    pipeline = Pipeline(
        source=SourceBundle(
            purchases=DataFrameSource(sample_df),
            config=FixedSource({"multiplier": 2.0}),
        ),
        feature=CombinedFeature(),
        store=MemoryStore(),
    )

    report = await pipeline.generate(entity_ids=["u1"])
    assert report.succeeded["u1"]["value"] == pytest.approx(30.0)  # 15.0 * 2
