"""06 — Partitioned Generation

Demonstrates every generate() concurrency mode against a simulated IoT
sensor fleet where ordering and rate-limiting constraints vary by region.

Scenario
--------
  480 sensors are split across 6 regional clusters (80 sensors each):

      sensor_R0_000 … sensor_R0_079   ← Region 0
      sensor_R1_000 … sensor_R1_079   ← Region 1
      …
      sensor_R5_000 … sensor_R5_079   ← Region 5

  A regional API adds ~5 ms latency per sensor read (simulated with
  asyncio.sleep).  The downstream feature store is region-partitioned:
  concurrent writes within the same region would corrupt ordering, so
  sensors inside a region must be processed serially.  Across regions
  there is no constraint, so all six can run at the same time.

Sections
--------
  A. Serial baseline                 — concurrency=1 (original behaviour)
  B. Flat concurrency                — concurrency=20, no ordering guarantees
  C. Partitioned by region function  — partition_by, serial-within-region
  D. Explicit partition dict         — same grouping built ahead of time
  E. Incremental re-run              — overwrite=False skips already-stored
  F. on_progress                     — real-time feedback compatible with tqdm

Run:
    python examples/06_partitioned_generation.py
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from stratum import Pipeline
from stratum.features.base import Feature
from stratum.schema import FeatureSchema, types
from stratum.sources.base import DataSource
from stratum.stores import MemoryStore

# ---------------------------------------------------------------------------
# Simulated regional API source
# ---------------------------------------------------------------------------

# Synthetic sensor registry: each sensor has a baseline reading and noise level
SENSORS: dict[str, dict[str, float]] = {}
REGIONS = 6
SENSORS_PER_REGION = 80

for r in range(REGIONS):
    for s in range(SENSORS_PER_REGION):
        sid = f"sensor_R{r}_{s:03d}"
        SENSORS[sid] = {"baseline": float(r * 10 + s % 20), "noise": float(r + 1)}


class RegionalAPISource(DataSource):
    """Simulates a regional API that takes ~5 ms per sensor read.

    In real usage this would be an HTTP call, database query, or
    other I/O-bound operation.
    """

    LATENCY = 0.005  # seconds

    async def read(self, entity_id: str | None = None, **kwargs: Any) -> dict[str, float]:
        await asyncio.sleep(self.LATENCY)  # simulate network round-trip
        if entity_id not in SENSORS:
            raise KeyError(f"Unknown sensor: {entity_id}")
        return dict(SENSORS[entity_id])


# ---------------------------------------------------------------------------
# Feature
# ---------------------------------------------------------------------------


class SensorReading(Feature):
    """Trivial feature: store the latest reading with a derived alert flag."""

    schema = FeatureSchema(
        {
            "baseline": types.Float64(nullable=False),
            "noise": types.Float64(nullable=False),
            "alert": types.Boolean(nullable=False),
        }
    )

    async def extract(self, raw: dict, context: dict, entity_id: str | None = None) -> dict:
        threshold = context.get("alert_threshold", 15.0)
        return {
            "baseline": raw["baseline"],
            "noise": raw["noise"],
            "alert": bool(raw["baseline"] > threshold),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def region_of(entity_id: str) -> str:
    """Extract the region key from a sensor ID (e.g. 'sensor_R3_042' → 'R3')."""
    return entity_id.split("_")[1]


def divider(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def fmt(seconds: float) -> str:
    return f"{seconds:.2f}s"


ALL_IDS = sorted(SENSORS.keys())
SAMPLE = ALL_IDS[:120]  # 120 sensors (2 per region × 60 readings) for fast demos


# ---------------------------------------------------------------------------
# A. Serial baseline
# ---------------------------------------------------------------------------


async def section_a() -> None:
    divider("A. Serial baseline  (concurrency=1)")
    print(
        f"  {len(SAMPLE)} sensors × {RegionalAPISource.LATENCY * 1000:.0f} ms latency "
        f"= ~{len(SAMPLE) * RegionalAPISource.LATENCY:.1f}s expected"
    )

    pipeline = Pipeline(
        source=RegionalAPISource(),
        feature=SensorReading(),
        store=MemoryStore(),
    )

    t0 = time.perf_counter()
    report = await pipeline.generate(entity_ids=SAMPLE)
    elapsed = time.perf_counter() - t0

    print(f"  Result : {report.success_count} OK  |  {report.failure_count} failed")
    print(f"  Time   : {fmt(elapsed)}")


# ---------------------------------------------------------------------------
# B. Flat concurrency — maximum throughput, no ordering guarantees
# ---------------------------------------------------------------------------


async def section_b() -> None:
    divider("B. Flat concurrency  (concurrency=20)")
    print("  All sensors compete for 20 concurrent slots.")
    print("  Processing order within a region is NOT guaranteed.")

    pipeline = Pipeline(
        source=RegionalAPISource(),
        feature=SensorReading(),
        store=MemoryStore(),
    )

    t0 = time.perf_counter()
    report = await pipeline.generate(entity_ids=SAMPLE, concurrency=20)
    elapsed = time.perf_counter() - t0

    print(f"  Result : {report.success_count} OK  |  {report.failure_count} failed")
    print(f"  Time   : {fmt(elapsed)}")
    print(
        f"  Expected speedup vs serial: ~{len(SAMPLE) * RegionalAPISource.LATENCY / elapsed:.1f}x"
    )


# ---------------------------------------------------------------------------
# C. Partitioned by region function
# ---------------------------------------------------------------------------


async def section_c() -> None:
    divider("C. Partitioned by region  (partition_by, concurrency=6)")
    print("  partition_by groups sensors by region automatically.")
    print("  Within each region: serial (ordered).  Across regions: concurrent.")
    print()
    print("  partition_by = lambda eid: eid.split('_')[1]")
    print(f"  → {REGIONS} partitions of {SENSORS_PER_REGION} sensors each")
    print(
        f"  → expected time ≈ {SENSORS_PER_REGION * RegionalAPISource.LATENCY * 20:.1f}s / region"
    )
    print(f"    with {REGIONS} regions running concurrently")

    # Use all sensors for this section to make the partition structure clear
    all_ids = ALL_IDS

    pipeline = Pipeline(
        source=RegionalAPISource(),
        feature=SensorReading(),
        store=MemoryStore(),
    )

    t0 = time.perf_counter()
    report = await pipeline.generate(
        entity_ids=all_ids,
        partition_by=region_of,
        concurrency=REGIONS,  # all 6 regions run at once
    )
    elapsed = time.perf_counter() - t0

    print(f"\n  Result : {report.success_count} OK  |  {report.failure_count} failed")
    print(f"  Time   : {fmt(elapsed)}")

    # Verify that ordering was preserved within each region
    regions_seen: dict[str, list[str]] = {}
    for eid in report.succeeded:
        r = region_of(eid)
        regions_seen.setdefault(r, []).append(eid)

    ordering_ok = all(grp == sorted(grp) for grp in regions_seen.values())
    print(f"  Within-region order preserved: {ordering_ok}")


# ---------------------------------------------------------------------------
# D. Explicit partition dict
# ---------------------------------------------------------------------------


async def section_d() -> None:
    divider("D. Explicit partitions  (partitions=dict, concurrency=3)")
    print("  Useful when partition structure comes from external metadata")
    print("  (e.g. a shard map, a DB query, a config file).")

    # Build partitions from the first 3 regions only, in explicit order
    explicit: dict[str, list[str]] = {}
    for r in range(3):
        key = f"R{r}"
        explicit[key] = sorted(eid for eid in ALL_IDS if region_of(eid) == key)

    total = sum(len(v) for v in explicit.values())
    print(f"\n  {len(explicit)} partitions, {total} sensors total")
    for k, v in explicit.items():
        print(f"    {k}: {len(v)} sensors")

    pipeline = Pipeline(
        source=RegionalAPISource(),
        feature=SensorReading(),
        store=MemoryStore(),
    )

    t0 = time.perf_counter()
    report = await pipeline.generate(
        partitions=explicit,
        concurrency=3,
    )
    elapsed = time.perf_counter() - t0

    print(f"\n  Result : {report.success_count} OK  |  {report.failure_count} failed")
    print(f"  Time   : {fmt(elapsed)}")


# ---------------------------------------------------------------------------
# E. Incremental re-run with overwrite=False
# ---------------------------------------------------------------------------


async def section_e() -> None:
    divider("E. Incremental re-run  (overwrite=False)")
    print("  First run: populate a subset.  Second run: skip already-stored,")
    print("  process only new entities.  Useful for resuming interrupted runs.")

    store = MemoryStore()
    pipeline = Pipeline(
        source=RegionalAPISource(),
        feature=SensorReading(),
        store=store,
    )
    ids = SAMPLE

    # --- First run: process half the sensors ---
    first_half = ids[: len(ids) // 2]
    t0 = time.perf_counter()
    r1 = await pipeline.generate(entity_ids=first_half, concurrency=20)
    t1 = time.perf_counter() - t0
    print(f"\n  First run  ({len(first_half)} sensors): {r1.success_count} OK  [{fmt(t1)}]")

    # --- Second run: all sensors, but skip already-stored ones ---
    t0 = time.perf_counter()
    r2 = await pipeline.generate(entity_ids=ids, concurrency=20, overwrite=False)
    t2 = time.perf_counter() - t0
    print(
        f"  Second run ({len(ids)} sensors): "
        f"{r2.success_count} newly processed  |  "
        f"{r2.skip_count} skipped  [{fmt(t2)}]"
    )
    print(f"  Time saved by skipping: ~{fmt(t1 * r2.skip_count / max(r1.success_count, 1))}")
    print(f"  Report: {r2}")


# ---------------------------------------------------------------------------
# F. on_progress callback
# ---------------------------------------------------------------------------


async def section_f() -> None:
    divider("F. on_progress callback")
    print("  Pass on_progress to get real-time feedback during a long run.")
    print("  The callback receives (completed, total, report) after each entity.")
    print("  Compatible with tqdm, rich.Progress, or plain print.\n")

    pipeline = Pipeline(
        source=RegionalAPISource(),
        feature=SensorReading(),
        store=MemoryStore(),
    )

    bar_width = 40
    last_print: list[int] = [-1]  # mutable to allow mutation inside closure

    def on_progress(completed: int, total: int, report) -> None:
        # Only reprint when percentage changes
        pct = int(completed / total * 100)
        if pct == last_print[0]:
            return
        last_print[0] = pct
        filled = int(bar_width * completed / total)
        bar = "#" * filled + "-" * (bar_width - filled)
        print(
            f"\r  [{bar}] {pct:3d}%  "
            f"{completed}/{total}  "
            f"ok={report.success_count} fail={report.failure_count}",
            end="",
            flush=True,
        )

    await pipeline.generate(
        entity_ids=SAMPLE,
        concurrency=20,
        on_progress=on_progress,
    )
    print()  # newline after progress bar
    print("\n  Done — on_progress fired once per entity throughout the run.")

    # tqdm integration hint
    print(
        "\n  To use tqdm:\n"
        "    from tqdm import tqdm\n"
        "    bar = tqdm(total=len(ids))\n"
        "    await pipeline.generate(\n"
        "        entity_ids=ids,\n"
        "        on_progress=lambda c, t, _: bar.update(1),\n"
        "    )\n"
        "    bar.close()"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    print(f"Sensor fleet: {len(SENSORS)} sensors across {REGIONS} regions")
    print(f"Simulated API latency: {RegionalAPISource.LATENCY * 1000:.0f} ms/sensor")

    await section_a()
    await section_b()
    await section_c()
    await section_d()
    await section_e()
    await section_f()

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(
        """
  Mode                   | When to use
  -----------------------|----------------------------------------------
  concurrency=1          | Safety / debugging; guaranteed serial order
  concurrency=N (flat)   | Stateless sources; max throughput needed
  partition_by=fn        | Groups derivable from entity ID (prefix, hash)
  partitions=dict        | Groups from external metadata or shard maps
  overwrite=False        | Resume interrupted runs; cheaply refresh subset
  on_progress=cb         | Progress bars, logging, dashboards
"""
    )


if __name__ == "__main__":
    asyncio.run(main())
