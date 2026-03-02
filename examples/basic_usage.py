"""End-to-end stratum example.

Demonstrates:
  - Defining a custom Feature with a FeatureSchema
  - Using DataFrameSource and MemoryStore
  - Running Pipeline.generate() and Pipeline.retrieve()

Run with:
    python examples/basic_usage.py
"""

import asyncio

import pandas as pd

from stratum import Pipeline
from stratum.features.base import Feature
from stratum.schema import FeatureSchema, types
from stratum.sources import DataFrameSource
from stratum.stores import MemoryStore

# ---------------------------------------------------------------------------
# 1. Define a Feature
# ---------------------------------------------------------------------------


class MeanPurchaseValue(Feature):
    """Compute the mean purchase amount for each user."""

    schema = FeatureSchema({"mean_value": types.Float64(nullable=False, default=0.0)})

    async def extract(self, raw: pd.DataFrame, context: dict, entity_id: str | None = None) -> dict:
        if raw.empty:
            raise ValueError("No purchase rows found for this entity")
        return {"mean_value": float(raw["amount"].mean())}


# ---------------------------------------------------------------------------
# 2. Prepare data
# ---------------------------------------------------------------------------

df = pd.DataFrame(
    {
        "entity_id": ["u1", "u1", "u2"],
        "amount": [10.0, 20.0, 15.0],
    }
)

# ---------------------------------------------------------------------------
# 3. Build and run the pipeline
# ---------------------------------------------------------------------------

pipeline = Pipeline(
    source=DataFrameSource(df),
    feature=MeanPurchaseValue(),
    store=MemoryStore(),
)


async def main() -> None:
    # Generate features for a batch of entities
    report = await pipeline.generate(entity_ids=["u1", "u2", "u_missing"])
    print("Generation report:", report)
    print("  Succeeded:", report.succeeded)
    print("  Failed:   ", report.failed)

    # Retrieve individual features
    u1_feature = await pipeline.retrieve("u1")
    print("\nRetrieved u1:", u1_feature)
    # → {"mean_value": 15.0}

    # Retrieve a batch (missing entities are silently omitted)
    batch = await pipeline.retrieve_batch(["u1", "u2", "u_missing"])
    print("Batch retrieve:", batch)


if __name__ == "__main__":
    asyncio.run(main())
