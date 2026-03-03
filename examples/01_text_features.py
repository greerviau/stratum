"""01 — Text / Review Feature Extraction

Loads 98k user reviews from examples/data/reviews.csv and extracts
per-user statistics into a ParquetStore.

Demonstrates:
  - DataFrameSource on a realistic multi-row-per-entity dataset
  - Multi-field FeatureSchema (Int64, Float64, Category)
  - ParquetStore for tabular feature output
  - context dict threading (run_id tag)
  - Timing a moderate-scale pipeline (~5 000 entities)

Run:
    python examples/01_text_features.py
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pandas as pd

from calcine import Pipeline
from calcine.features.base import Feature
from calcine.schema import FeatureSchema, types
from calcine.sources import DataFrameSource
from calcine.stores import ParquetStore

DATA = Path(__file__).parent / "data"
STORE_PATH = Path(__file__).parent / "data" / "store_01_text"

# ---------------------------------------------------------------------------
# Feature definition
# ---------------------------------------------------------------------------

CATEGORIES = ["electronics", "clothing", "books", "home", "sports", "food", "toys"]


class ReviewStats(Feature):
    """Per-user aggregate statistics extracted from review text and metadata."""

    schema = FeatureSchema(
        {
            "review_count": types.Int64(nullable=False),
            "avg_word_count": types.Float64(nullable=False),
            "vocab_size": types.Int64(nullable=False),
            "avg_rating": types.Float64(nullable=False),
            "five_star_ratio": types.Float64(nullable=False),
            "top_category": types.Category(categories=CATEGORIES, nullable=False),
            "total_helpful_votes": types.Int64(nullable=False),
        }
    )

    async def extract(self, raw: pd.DataFrame, context: dict, entity_id: str | None = None) -> dict:
        if raw.empty:
            raise ValueError("No reviews found for this entity")

        words_per_review = raw["review"].str.split().str.len()
        all_words = " ".join(raw["review"]).split()

        return {
            "review_count": int(len(raw)),
            "avg_word_count": float(words_per_review.mean()),
            "vocab_size": int(len(set(all_words))),
            "avg_rating": float(raw["rating"].mean()),
            "five_star_ratio": float((raw["rating"] == 5).mean()),
            "top_category": str(raw["category"].mode().iloc[0]),
            "total_helpful_votes": int(raw["helpful_votes"].sum()),
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


async def main() -> None:
    print("Loading reviews.csv …")
    df = pd.read_csv(DATA / "reviews.csv")
    entity_ids = df["entity_id"].unique().tolist()
    print(f"  {len(df):,} reviews  |  {len(entity_ids):,} users")

    STORE_PATH.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline(
        source=DataFrameSource(df),
        feature=ReviewStats(),
        store=ParquetStore(str(STORE_PATH)),
    )

    print(f"\nRunning pipeline for {len(entity_ids):,} entities …")
    t0 = time.perf_counter()
    report = await pipeline.generate(
        entity_ids=entity_ids,
        context={"run_id": "example-01", "schema_version": "v1"},
    )
    elapsed = time.perf_counter() - t0

    print(f"  Done in {elapsed:.2f}s  ({elapsed / len(entity_ids) * 1000:.2f} ms/entity)")
    print(f"  {report.success_count:,} succeeded  |  {report.failure_count:,} failed")

    if report.failed:
        print("\n  Sample failures:")
        for eid, errs in list(report.failed.items())[:3]:
            print(f"    {eid}: {errs[0]}")

    # Spot-check a feature
    sample_id = entity_ids[0]
    feature = await pipeline.retrieve(sample_id)
    print(f"\nSample  ({sample_id}):")
    for k, v in feature.items():
        print(f"  {k:<22} {v}")

    # Demonstrate batch retrieval
    batch = await pipeline.retrieve_batch(entity_ids[:5])
    print(f"\nBatch retrieved {len(batch)} entities — avg ratings:")
    for eid, feat in batch.items():
        print(f"  {eid}  avg_rating={feat['avg_rating']:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
