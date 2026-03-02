"""03 — Multi-Source Feature with SourceBundle

Combines three completely independent sources to produce a single
"user_risk_score" feature per entity.

Sources:
  1. transactions (DataFrameSource) — purchase history
  2. profiles     (DataFrameSource) — demographics and plan
  3. thresholds   (inline dict)     — global config pushed in as context

No assumption is made about the relationship between sources.
The Feature simply receives a dict keyed by source name and unpacks
whatever it needs.

Demonstrates:
  - SourceBundle with mixed source types
  - context for global config that doesn't vary per entity
  - pre_extract hook to drop columns the Feature doesn't use
  - MemoryStore for prototyping, then swapping to FileStore

Run:
    python examples/03_multi_source.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stratum import Pipeline
from stratum.features.base import Feature
from stratum.schema import FeatureSchema, types
from stratum.sources import DataFrameSource, SourceBundle
from stratum.sources.base import DataSource
from stratum.stores import MemoryStore

DATA = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# A "constant" source that returns the same value for every entity.
# Useful for global config, lookup tables, or model metadata.
# ---------------------------------------------------------------------------


class ConstantSource(DataSource):
    """Returns the same value for any entity_id."""

    def __init__(self, value: Any) -> None:
        self.value = value

    async def read(self, **kwargs: Any) -> Any:
        return self.value


# ---------------------------------------------------------------------------
# Feature definition
# ---------------------------------------------------------------------------

PLAN_WEIGHT = {"free": 0.2, "basic": 0.4, "pro": 0.7, "enterprise": 1.0}
COUNTRY_RISK = {
    "US": 0.3,
    "UK": 0.3,
    "DE": 0.2,
    "FR": 0.2,
    "CA": 0.3,
    "AU": 0.3,
    "JP": 0.1,
    "BR": 0.6,
}


class UserRiskScore(Feature):
    """Combine transaction history + demographics → a [0, 1] risk score."""

    schema = FeatureSchema(
        {
            "risk_score": types.Float64(nullable=False),
            "spend_total": types.Float64(nullable=False),
            "spend_stddev": types.Float64(nullable=False),
            "txn_count": types.Int64(nullable=False),
            "plan": types.Category(
                categories=list(PLAN_WEIGHT.keys()),
                nullable=False,
            ),
            "flagged": types.Boolean(nullable=False),
        }
    )

    async def pre_extract(self, raw: dict) -> dict:
        # Drop columns we never use so extract() stays clean
        if "helpful_votes" in raw.get("transactions", pd.DataFrame()).columns:
            raw["transactions"] = raw["transactions"].drop(
                columns=["helpful_votes", "category", "review"], errors="ignore"
            )
        return raw

    async def extract(self, raw: dict, context: dict, entity_id: str | None = None) -> dict:
        txns = raw["transactions"]  # DataFrame
        profile = raw["profile"]  # Series (one row)
        limits = raw["limits"]  # dict of thresholds

        # --- Transaction signals ---
        if txns.empty:
            raise ValueError("No transaction history for this entity")

        spend = txns["rating"].astype(float) * 10  # proxy for spend
        spend_total = float(spend.sum())
        spend_std = float(spend.std(ddof=0))
        txn_count = int(len(txns))

        # --- Profile signals ---
        # DataFrameSource returns a DataFrame — take the first (and only) row
        if profile.empty:
            raise ValueError("No profile found for this entity")
        row = profile.iloc[0]
        plan = str(row["account_plan"])
        tenure_days = int(row["tenure_days"])
        country = str(row["country"])

        plan_w = PLAN_WEIGHT.get(plan, 0.5)
        country_r = COUNTRY_RISK.get(country, 0.5)
        tenure_w = min(tenure_days / 365.0, 5.0) / 5.0  # cap at 5 years

        # --- Risk formula (arbitrary, for illustration) ---
        velocity = spend_total / max(txn_count, 1)
        high_velocity = velocity > limits["velocity_threshold"]
        risk = (
            0.3 * country_r + 0.2 * (1 - plan_w) + 0.2 * (1 - tenure_w) + 0.3 * float(high_velocity)
        )

        return {
            "risk_score": round(float(np.clip(risk, 0, 1)), 4),
            "spend_total": round(spend_total, 2),
            "spend_stddev": round(spend_std, 2),
            "txn_count": txn_count,
            "plan": plan,
            "flagged": bool(high_velocity),
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


async def main() -> None:
    print("Loading data …")
    reviews_df = pd.read_csv(DATA / "reviews.csv")
    profiles_df = pd.read_csv(DATA / "user_profiles.csv")

    # Align entity sets: only users that appear in both sources
    profile_ids = set(profiles_df["entity_id"])
    review_ids = set(reviews_df["entity_id"])
    entity_ids = sorted(profile_ids & review_ids)
    print(f"  {len(entity_ids):,} entities with both reviews and profile data")

    # Global thresholds passed via context
    thresholds = {"velocity_threshold": 30.0}

    pipeline = Pipeline(
        source=SourceBundle(
            transactions=DataFrameSource(reviews_df),
            profile=DataFrameSource(profiles_df),
            limits=ConstantSource(thresholds),
        ),
        feature=UserRiskScore(),
        store=MemoryStore(),
    )

    print(f"\nGenerating risk scores for {len(entity_ids):,} entities …")
    report = await pipeline.generate(entity_ids=entity_ids[:500])  # sample 500

    print(f"  {report.success_count} OK  |  {report.failure_count} failed")

    # Show distribution
    scores = [f["risk_score"] for f in report.succeeded.values()]
    flagged = sum(1 for f in report.succeeded.values() if f["flagged"])
    print(
        f"\n  risk_score  min={min(scores):.3f}  "
        f"mean={sum(scores) / len(scores):.3f}  max={max(scores):.3f}"
    )
    print(f"  flagged (high velocity): {flagged}/{len(scores)}")

    # Spot-check one entity
    sample_id = entity_ids[0]
    feat = await pipeline.retrieve(sample_id)
    print(f"\n{sample_id}:")
    for k, v in feat.items():
        print(f"  {k:<16} {v}")

    # --- Show that swapping the store is trivial ---
    print("\n--- Swap to MemoryStore (same pipeline, new store) ---")
    pipeline2 = Pipeline(
        source=pipeline.source,
        feature=pipeline.feature,
        store=MemoryStore(),
    )
    r2 = await pipeline2.generate(entity_ids=entity_ids[:10])
    print(f"  {r2.success_count} entities re-generated into new store")


if __name__ == "__main__":
    asyncio.run(main())
