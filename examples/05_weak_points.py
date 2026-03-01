"""05 — Known Weak Points and Recommended Mitigations

This file documents five current limitations of stratum, each with:
  1. A minimal reproduction showing the exact behavior
  2. The recommended workaround pattern

Weak points covered:
  A. NaN silently passes Float64 schema validation
  B. Empty source data is the Feature's problem, not the framework's
  C. SourceBundle is all-or-nothing: one sub-source failure drops the entity
  D. Feature store namespace is keyed only on class name (collision risk)
  E. generate() is serial — resolved in current version (see example 06)

Run:
    python examples/05_weak_points.py
"""

from __future__ import annotations

import asyncio
import math
import time
from typing import Any

import numpy as np
import pandas as pd
from stratum import Pipeline
from stratum.features.base import Feature
from stratum.schema import FeatureSchema, types
from stratum.sources import DataFrameSource, SourceBundle
from stratum.sources.base import DataSource
from stratum.stores import MemoryStore

SEP = "\n" + "─" * 60 + "\n"

# ---------------------------------------------------------------------------
# Shared minimal dataframe
# ---------------------------------------------------------------------------

DF = pd.DataFrame(
    {
        "entity_id": ["u1", "u1", "u2"],
        "amount": [10.0, 20.0, 15.0],
    }
)


# ===========================================================================
# A — NaN passes Float64 validation
# ===========================================================================


async def demo_nan() -> None:
    print(SEP + "A — NaN silently passes Float64 validation\n")

    t = types.Float64(nullable=False)
    nan_errors = t.validate(float("nan"))
    print(f"  types.Float64(nullable=False).validate(float('nan')) → {nan_errors!r}")
    print("  ↳ Empty list — NaN is a valid IEEE float, so the schema passes it.")
    print()

    # In a pipeline: an entity with all-NaN data would succeed
    class NanFeature(Feature):
        schema = FeatureSchema({"score": types.Float64(nullable=False)})

        async def extract(self, raw: pd.DataFrame, context: dict) -> dict:
            # mean() on an empty group returns NaN
            return {"score": float(raw["amount"].mean())}

    pipeline = Pipeline(
        source=DataFrameSource(DF),
        feature=NanFeature(),
        store=MemoryStore(),
    )
    # u_ghost has no rows → mean() returns NaN → schema passes → stored silently
    report = await pipeline.generate(entity_ids=["u1", "u_ghost"])
    ghost_result = report.succeeded.get("u_ghost")
    print(f"  u_ghost (no rows): in succeeded={ghost_result is not None}, value={ghost_result}")
    print()

    print("  MITIGATION — reject NaN/Inf in post_extract:\n")
    print(
        "    async def post_extract(self, result):\n"
        "        for k, v in result.items():\n"
        "            if isinstance(v, float) and not math.isfinite(v):\n"
        "                raise ValueError(f'Non-finite value for {k!r}: {v}')\n"
        "        return result"
    )

    class SafeNanFeature(Feature):
        schema = FeatureSchema({"score": types.Float64(nullable=False)})

        async def extract(self, raw: pd.DataFrame, context: dict) -> dict:
            return {"score": float(raw["amount"].mean())}

        async def post_extract(self, result: dict) -> dict:
            for k, v in result.items():
                if isinstance(v, float) and not math.isfinite(v):
                    raise ValueError(f"Non-finite value for {k!r}: {v!r}")
            return result

    pipeline2 = Pipeline(
        source=DataFrameSource(DF),
        feature=SafeNanFeature(),
        store=MemoryStore(),
    )
    report2 = await pipeline2.generate(entity_ids=["u1", "u_ghost"])
    print(f"\n  After mitigation — u_ghost in failed: {'u_ghost' in report2.failed}")
    if "u_ghost" in report2.failed:
        print(f"  Error: {report2.failed['u_ghost'][0][:80]}")


# ===========================================================================
# B — Empty source data is the Feature's responsibility
# ===========================================================================


async def demo_empty_source() -> None:
    print(SEP + "B — Empty source data reaches extract() unchecked\n")

    print(
        "  DataFrameSource returns an empty DataFrame for unknown entities\n"
        "  rather than raising.  The Feature must guard against it.\n"
    )

    class UnguardedFeature(Feature):
        schema = FeatureSchema({"mean": types.Float64(nullable=False)})

        async def extract(self, raw: pd.DataFrame, context: dict) -> dict:
            # This does NOT raise on empty — it returns NaN (see weak point A)
            return {"mean": float(raw["amount"].mean())}

    r = await Pipeline(
        source=DataFrameSource(DF),
        feature=UnguardedFeature(),
        store=MemoryStore(),
    ).generate(entity_ids=["u_missing"])

    print(f"  u_missing with UnguardedFeature → succeeded: {bool(r.succeeded)}")
    print()

    print("  MITIGATION — guard at the top of extract():\n")
    print(
        "    async def extract(self, raw, context):\n"
        "        if isinstance(raw, pd.DataFrame) and raw.empty:\n"
        "            raise ValueError('No source data for this entity')\n"
        "        ..."
    )

    class GuardedFeature(Feature):
        schema = FeatureSchema({"mean": types.Float64(nullable=False)})

        async def extract(self, raw: pd.DataFrame, context: dict) -> dict:
            if raw.empty:
                raise ValueError("No source data for this entity")
            return {"mean": float(raw["amount"].mean())}

    r2 = await Pipeline(
        source=DataFrameSource(DF),
        feature=GuardedFeature(),
        store=MemoryStore(),
    ).generate(entity_ids=["u_missing"])
    print(f"\n  After mitigation — u_missing in failed: {'u_missing' in r2.failed}")


# ===========================================================================
# C — SourceBundle is all-or-nothing
# ===========================================================================


async def demo_bundle_partial_failure() -> None:
    print(SEP + "C — SourceBundle: one sub-source failure drops the whole entity\n")

    class GoodSource(DataSource):
        async def read(self, **kwargs: Any) -> str:
            return "good data"

    class FlakySource(DataSource):
        async def read(self, entity_id: str | None = None, **kwargs: Any) -> str:
            if entity_id == "u2":
                raise ConnectionError("Simulated network error")
            return "flaky data"

    pipeline = Pipeline(
        source=SourceBundle(good=GoodSource(), flaky=FlakySource()),
        feature=type(
            "F",
            (Feature,),
            {
                "extract": lambda self, raw, ctx: {"v": raw["good"]},
            },
        )(),
        store=MemoryStore(),
    )
    report = await pipeline.generate(entity_ids=["u1", "u2"])
    print(f"  u1 (both sources OK) → succeeded: {'u1' in report.succeeded}")
    print(f"  u2 (flaky failed)    → failed:    {'u2' in report.failed}")
    print(f"  Error: {report.failed.get('u2', [''])[0][:80]}")
    print()
    print(
        "  ↳ The 'good' source succeeded for u2, but the result is discarded\n"
        "    because 'flaky' raised.  There is no way to get partial bundle results."
    )
    print()
    print("  MITIGATION — wrap unreliable sources in a try/except DataSource:\n")
    print(
        "    class FaultTolerantSource(DataSource):\n"
        "        def __init__(self, source, default=None):\n"
        "            self.source, self.default = source, default\n"
        "        async def read(self, **kwargs):\n"
        "            try:\n"
        "                return await self.source.read(**kwargs)\n"
        "            except Exception:\n"
        "                return self.default\n"
    )


# ===========================================================================
# D — Feature namespace collision
# ===========================================================================


async def demo_name_collision() -> None:
    print(SEP + "D — Feature class-name collision in the store\n")

    # Simulate two feature classes from different teams / modules that both
    # happen to be named "EngagementScore".
    class EngagementScore(Feature):  # "team A" version
        async def extract(self, raw, context):
            return {"v": "team_A_value"}

    # Rebind same name — identical __name__, different implementation
    _EngagementScore_A = EngagementScore

    class EngagementScore(Feature):  # "team B" version  # noqa: F811
        async def extract(self, raw, context):
            return {"v": "team_B_value"}

    _EngagementScore_B = EngagementScore

    feat_a = _EngagementScore_A()
    feat_b = _EngagementScore_B()

    store = MemoryStore()
    await store.write(feat_a, "e1", {"v": "team_A_value"})
    await store.write(feat_b, "e1", {"v": "team_B_value"})  # silently overwrites!

    result = await store.read(feat_a, "e1")
    print(f"  Wrote team_A then team_B for entity 'e1'.")
    print(f"  Reading via feat_a → {result}  (expected team_A, got team_B)")
    print(f"\n  Both classes share __name__='{store._feature_key(feat_a)}' — same store key.")
    print()
    print("  MITIGATION — add a unique feature_name attribute and use it as the key:\n")
    print(
        "    class EngagementScore(Feature):\n"
        "        feature_name = 'team_a.engagement_score'  # explicit, unique\n"
        "        ...\n"
        "\n"
        "    class ModuleAwareStore(MemoryStore):\n"
        "        def _feature_key(self, feature):\n"
        "            name = getattr(type(feature), 'feature_name', None)\n"
        "            if name:\n"
        "                return name\n"
        "            cls = type(feature)\n"
        "            return f'{cls.__module__}.{cls.__qualname__}'\n"
    )

    class ModuleAwareStore(MemoryStore):
        def _feature_key(self, feature):
            name = getattr(type(feature), "feature_name", None)
            if name:
                return name
            cls = type(feature)
            return f"{cls.__module__}.{cls.__qualname__}"

    class TeamAScore(Feature):
        feature_name = "team_a.engagement_score"

        async def extract(self, raw, context):
            return {"v": "team_A_value"}

    class TeamBScore(Feature):
        feature_name = "team_b.engagement_score"

        async def extract(self, raw, context):
            return {"v": "team_B_value"}

    safe_store = ModuleAwareStore()
    await safe_store.write(TeamAScore(), "e1", {"v": "team_A_value"})
    await safe_store.write(TeamBScore(), "e1", {"v": "team_B_value"})
    result_a = await safe_store.read(TeamAScore(), "e1")
    result_b = await safe_store.read(TeamBScore(), "e1")
    print(f"  After mitigation → TeamA: {result_a}  TeamB: {result_b}")


# ===========================================================================
# E — generate() is serial
# ===========================================================================


async def demo_serial_processing() -> None:
    print(SEP + "E — generate() serial processing [RESOLVED]\n")

    print(
        "  This was previously a limitation: generate() processed entities\n"
        "  one at a time, so N entities × latency = N × latency total time.\n"
        "\n"
        "  generate() now supports built-in concurrency:\n"
        "\n"
        "    # Flat: up to 20 entities concurrently\n"
        "    await pipeline.generate(entity_ids=ids, concurrency=20)\n"
        "\n"
        "    # Partitioned: serial within region, concurrent across regions\n"
        "    await pipeline.generate(\n"
        "        entity_ids=ids,\n"
        "        partition_by=lambda eid: eid.split('_')[0],\n"
        "        concurrency=8,\n"
        "    )\n"
        "\n"
        "  See examples/06_partitioned_generation.py for a full walkthrough.\n"
    )

    N = 60

    class IoFeature(Feature):
        async def extract(self, raw, context):
            await asyncio.sleep(0.005)  # 5 ms simulated async I/O
            return {"v": 1.0}

    df = pd.DataFrame({"entity_id": [f"e{i}" for i in range(N)], "x": range(N)})
    entity_ids = df["entity_id"].tolist()
    pipeline = Pipeline(DataFrameSource(df), IoFeature(), MemoryStore())

    t0 = time.perf_counter()
    await pipeline.generate(entity_ids=entity_ids)
    t_seq = time.perf_counter() - t0

    t0 = time.perf_counter()
    await pipeline.generate(entity_ids=entity_ids, concurrency=20)
    t_conc = time.perf_counter() - t0

    print(f"  Serial  ({N} entities × 5ms):      {t_seq:.2f}s")
    print(f"  Concurrent (concurrency=20):     {t_conc:.2f}s   ({t_seq / t_conc:.1f}x faster)")


# ===========================================================================
# Main
# ===========================================================================


async def main() -> None:
    print("stratum — Known Weak Points & Mitigations")
    await demo_nan()
    await demo_empty_source()
    await demo_bundle_partial_failure()
    await demo_name_collision()
    await demo_serial_processing()
    print(SEP + "Done.")


if __name__ == "__main__":
    asyncio.run(main())
