"""Shared fixtures for calcine tests."""

from __future__ import annotations

import pandas as pd
import pytest

from calcine.features.base import Feature
from calcine.schema import FeatureSchema, types
from calcine.sources import DataFrameSource
from calcine.stores import MemoryStore

# ---------------------------------------------------------------------------
# Concrete feature fixtures
# ---------------------------------------------------------------------------


class MeanAmountFeature(Feature):
    """Extracts the mean of the 'amount' column from a DataFrame slice."""

    schema = FeatureSchema({"mean_value": types.Float64(nullable=False)})

    async def extract(self, raw: pd.DataFrame, context: dict) -> dict:
        return {"mean_value": float(raw["amount"].mean())}


class RawPassthroughFeature(Feature):
    """Returns the raw input unchanged (no schema)."""

    async def extract(self, raw, context):
        return raw


# ---------------------------------------------------------------------------
# Data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entity_id": ["u1", "u1", "u2", "u3"],
            "amount": [10.0, 20.0, 15.0, 5.0],
        }
    )


@pytest.fixture
def mean_feature() -> MeanAmountFeature:
    return MeanAmountFeature()


@pytest.fixture
def memory_store() -> MemoryStore:
    return MemoryStore()


@pytest.fixture
def df_source(sample_df: pd.DataFrame) -> DataFrameSource:
    return DataFrameSource(sample_df)
