"""04 — Time-Series Features + Scale

Extracts statistical features from 502k sensor readings across 1 000 sensors,
then benchmarks sequential vs. manual-concurrent pipeline execution.

Features extracted per sensor:
  mean, std, min, max, p10, p90, trend_slope, anomaly_score, spike_count

Demonstrates:
  - Variable-length time-series (each sensor has 200–800 readings)
  - Numpy-heavy extract() with linear regression for trend
  - FileStore + JSONSerializer for human-readable output
  - Historical concurrency workaround via asyncio.gather over batches
    (generate() now has built-in concurrency — see example 06)

Run:
    python examples/04_timeseries_scale.py
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import numpy as np
import pandas as pd

from calcine import Pipeline
from calcine.features.base import Feature
from calcine.schema import FeatureSchema, types
from calcine.serializers import JSONSerializer
from calcine.sources import DataFrameSource
from calcine.stores import FileStore, MemoryStore

DATA = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# Feature definition
# ---------------------------------------------------------------------------


class SensorStats(Feature):
    """Rich statistical summary of a sensor's time series."""

    schema = FeatureSchema(
        {
            "n_readings": types.Int64(nullable=False),
            "mean": types.Float64(nullable=False),
            "std": types.Float64(nullable=False),
            "p10": types.Float64(nullable=False),
            "p90": types.Float64(nullable=False),
            "min": types.Float64(nullable=False),
            "max": types.Float64(nullable=False),
            "trend_slope": types.Float64(nullable=False),  # units/reading
            "anomaly_score": types.Float64(nullable=False),  # z-score of last reading
            "spike_count": types.Int64(nullable=False),  # readings > 3σ from mean
        }
    )

    async def extract(self, raw: pd.DataFrame, context: dict, entity_id: str | None = None) -> dict:
        if raw.empty:
            raise ValueError("No readings for this sensor")

        v = raw["value"].to_numpy(dtype=np.float64)
        t = np.arange(len(v), dtype=np.float64)

        mu = float(v.mean())
        std = float(v.std(ddof=0)) or 1e-9  # guard against zero std

        # Linear trend via least-squares
        slope = float(np.polyfit(t, v, 1)[0])

        # Anomaly score: z-score of the most recent reading
        anomaly = float((v[-1] - mu) / std)

        # Spikes: readings more than 3σ from the mean
        spikes = int((np.abs(v - mu) > 3 * std).sum())

        return {
            "n_readings": int(len(v)),
            "mean": round(mu, 4),
            "std": round(float(v.std(ddof=0)), 4),
            "p10": round(float(np.percentile(v, 10)), 4),
            "p90": round(float(np.percentile(v, 90)), 4),
            "min": round(float(v.min()), 4),
            "max": round(float(v.max()), 4),
            "trend_slope": round(slope, 6),
            "anomaly_score": round(anomaly, 4),
            "spike_count": spikes,
        }


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


async def run_sequential(pipeline: Pipeline, entity_ids: list[str]) -> tuple[float, int]:
    """Standard pipeline.generate() — serial entity processing."""
    t0 = time.perf_counter()
    report = await pipeline.generate(entity_ids=entity_ids)
    return time.perf_counter() - t0, report.success_count


async def run_concurrent_batches(
    pipeline: Pipeline,
    entity_ids: list[str],
    batch_size: int = 100,
) -> tuple[float, int]:
    """
    WORKAROUND: split entity_ids into batches, run each batch with
    asyncio.gather so batches execute concurrently.

    This helps when extract() or source.read() contains real async I/O
    (HTTP calls, database queries, etc.).  For pure CPU work it won't help
    because of the GIL, but for I/O-bound features the speedup can be large.
    """
    t0 = time.perf_counter()
    total_ok = 0

    batches = [entity_ids[i : i + batch_size] for i in range(0, len(entity_ids), batch_size)]

    async def _batch(ids: list[str]) -> int:
        r = await pipeline.generate(entity_ids=ids)
        return r.success_count

    results = await asyncio.gather(*[_batch(b) for b in batches])
    total_ok = sum(results)
    return time.perf_counter() - t0, total_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    print("Loading sensor_readings.csv …")
    df = pd.read_csv(DATA / "sensor_readings.csv")
    entity_ids = sorted(df["entity_id"].unique().tolist())
    print(f"  {len(df):,} readings  |  {len(entity_ids):,} sensors")

    store_path = str(DATA / "store_04_sensors")

    pipeline = Pipeline(
        source=DataFrameSource(df),
        feature=SensorStats(),
        store=FileStore(store_path, serializer=JSONSerializer()),
    )

    # -- Full run --
    print(f"\n[Sequential]  Running for all {len(entity_ids)} sensors …")
    t_seq, n_ok = await run_sequential(pipeline, entity_ids)
    print(f"  {n_ok} OK  |  {t_seq:.2f}s  |  {t_seq / len(entity_ids) * 1000:.1f} ms/entity")

    # Spot-check
    sample = await pipeline.retrieve("sensor_0000")
    print("\nsensor_0000:")
    for k, v in sample.items():
        print(f"  {k:<16} {v}")

    # -- Serial vs. concurrent benchmark with artificial async latency --
    print("\n" + "=" * 60)
    print("WEAK POINT: serial entity processing")
    print("=" * 60)
    print(
        "\ngenerate() processes entities one at a time.  If extract()\n"
        "awaits any real I/O (API call, DB query, model inference),\n"
        "every entity waits for the previous one to finish.\n"
    )

    BENCH_N = 200  # use a subset so the demo is fast

    # Simulate a feature that does 2 ms of async I/O per entity
    class SlowFeature(Feature):
        async def extract(self, raw, context, entity_id=None):
            await asyncio.sleep(0.002)  # simulate async API call
            return {"value": 1.0}

    slow_pipeline = Pipeline(
        source=DataFrameSource(df),
        feature=SlowFeature(),
        store=MemoryStore(),
    )

    print(f"Benchmarking {BENCH_N} entities with 2 ms simulated I/O per entity:\n")

    t1, ok1 = await run_sequential(slow_pipeline, entity_ids[:BENCH_N])
    print(f"  Sequential ({BENCH_N} entities):      {t1:.2f}s")

    slow_pipeline2 = Pipeline(
        source=DataFrameSource(df),
        feature=SlowFeature(),
        store=MemoryStore(),
    )
    t2, ok2 = await run_concurrent_batches(slow_pipeline2, entity_ids[:BENCH_N], batch_size=50)
    print(f"  Concurrent batches (size=50):    {t2:.2f}s   ({t1 / t2:.1f}x faster)")

    print(
        "\nMitigation: wrap generate() with asyncio.gather over batches\n"
        "(see run_concurrent_batches() in this file).\n"
        "A future Pipeline.generate_concurrent(batch_size=N) would\n"
        "make this ergonomic out of the box."
    )


if __name__ == "__main__":
    asyncio.run(main())
