"""07 — Batch Extract Benchmark

Demonstrates and benchmarks Feature.extract_batch() against individual
extract() calls, showing the performance gain for compute-bound features.

The Bottleneck Being Solved
---------------------------
generate(concurrency=N) overlaps I/O waits — it helps when your source
read() calls an API or database and you want those round-trips to happen
in parallel.

But concurrency does nothing to speed up the *compute* step.  If extract()
itself is slow because it calls an ML model, a batch embedding API, or does
expensive CPU work, every entity still waits its turn.

extract_batch() fixes this: instead of calling extract() once per entity,
the pipeline collects a batch of raw inputs and calls extract_batch() once,
enabling:

  - GPU ML inference    — one forward pass for N entities vs N forward passes
  - Batch embedding APIs — one HTTP call for N texts vs N HTTP calls
  - Bulk DB queries      — one IN (...) query vs N individual queries
  - Vectorised numpy     — one (N, D) matrix op vs N (1, D) ops

Simulation Model
----------------
Each scenario simulates a "batch embedding API" with:
  - Source read: 3 ms async latency per entity (database/API fetch)
  - Individual extract: 2 ms fixed overhead + 0.5 ms/item (per-call API cost)
  - Batch extract_batch: 5 ms fixed overhead + 0.05 ms/item (amortised API cost)

For a batch of 50 entities:
  - 50 × individual:   50 × 2.5 ms = 125 ms
  - 1 × batch call:    5 ms + 50 × 0.05 ms = 7.5 ms  (≈17× cheaper)

Scenarios Benchmarked (200 entities)
-------------------------------------
  1. Serial individual          — baseline (worst case)
  2. Concurrent individual      — concurrency=20, reads overlap, compute serial
  3. Serial batches             — batch_size=50, reads concurrent within batch
  4. Concurrent batches         — batch_size=50, concurrency=4 (best case)

Run:
    python examples/07_batch_extract_benchmark.py
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np

from stratum import Pipeline
from stratum.features.base import Feature
from stratum.schema import FeatureSchema, types
from stratum.sources.base import DataSource
from stratum.stores import MemoryStore

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

N_ENTITIES = 200
EMBEDDING_DIM = 128        # output embedding dimension
INPUT_DIM = 512            # input feature dimension

READ_LATENCY = 0.003       # 3 ms per source read (DB / API fetch)
EXTRACT_OVERHEAD = 0.002   # 2 ms per-call overhead for individual extract
EXTRACT_PER_ITEM = 0.0005  # 0.5 ms additional per item in individual extract
BATCH_OVERHEAD = 0.005     # 5 ms per-call overhead for batch extract
BATCH_PER_ITEM = 0.00005   # 0.05 ms per item in batch (amortised)

# Shared weight matrix — represents a linear projection layer
rng = np.random.default_rng(42)
WEIGHT_MATRIX = rng.standard_normal((INPUT_DIM, EMBEDDING_DIM)).astype(np.float32)

# Synthetic entity data: each entity has a random input vector
ENTITY_DATA: dict[str, np.ndarray] = {
    f"e{i:04d}": rng.standard_normal(INPUT_DIM).astype(np.float32)
    for i in range(N_ENTITIES)
}

ALL_IDS = sorted(ENTITY_DATA.keys())

# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------


class VectorSource(DataSource):
    """Returns a pre-computed input vector with simulated async read latency."""

    async def read(self, entity_id: str | None = None, **kwargs: Any) -> np.ndarray:
        await asyncio.sleep(READ_LATENCY)
        return ENTITY_DATA[entity_id]  # type: ignore[index]


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


class IndividualEmbedding(Feature):
    """Projects each entity's vector individually — one API call per entity."""

    schema = FeatureSchema({"embedding": types.NDArray(shape=(EMBEDDING_DIM,), dtype="float32")})

    async def extract(self, raw: np.ndarray, context: dict) -> dict:
        # Simulate per-call overhead (HTTP round-trip, model warm-up, etc.)
        await asyncio.sleep(EXTRACT_OVERHEAD + EXTRACT_PER_ITEM)
        embedding = raw @ WEIGHT_MATRIX
        return {"embedding": embedding}


class BatchEmbedding(Feature):
    """Projects a whole batch in one call — amortises fixed overhead across entities."""

    schema = FeatureSchema({"embedding": types.NDArray(shape=(EMBEDDING_DIM,), dtype="float32")})

    async def extract(self, raw: np.ndarray, context: dict) -> dict:
        # Fallback for the default extract_batch path (shouldn't normally be called
        # when batch_size > 1, but keeps the class fully functional on its own)
        await asyncio.sleep(EXTRACT_OVERHEAD + EXTRACT_PER_ITEM)
        return {"embedding": raw @ WEIGHT_MATRIX}

    async def extract_batch(
        self, raws: list[np.ndarray], context: dict
    ) -> list[dict | BaseException]:
        # One "API call" for the whole batch: fixed overhead + tiny per-item cost
        n = len(raws)
        await asyncio.sleep(BATCH_OVERHEAD + BATCH_PER_ITEM * n)

        # Vectorised projection: (N, INPUT_DIM) @ (INPUT_DIM, EMBEDDING_DIM) → (N, EMBEDDING_DIM)
        batch_matrix = np.stack(raws)         # (N, INPUT_DIM)
        embeddings = batch_matrix @ WEIGHT_MATRIX  # (N, EMBEDDING_DIM)

        return [{"embedding": embeddings[i]} for i in range(n)]


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def header(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


async def run(
    feature: Feature,
    label: str,
    *,
    batch_size: int = 1,
    concurrency: int = 1,
) -> float:
    pipeline = Pipeline(source=VectorSource(), feature=feature, store=MemoryStore())
    t0 = time.perf_counter()
    report = await pipeline.generate(
        entity_ids=ALL_IDS,
        batch_size=batch_size,
        concurrency=concurrency,
    )
    elapsed = time.perf_counter() - t0
    throughput = report.success_count / elapsed
    print(
        f"  {label:<42}  {elapsed:5.2f}s  "
        f"({throughput:6.0f} entities/s)  "
        f"ok={report.success_count}"
    )
    return elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    print(f"Batch Extract Benchmark — {N_ENTITIES} entities")
    print(
        f"  Source latency   : {READ_LATENCY * 1000:.0f} ms/read\n"
        f"  Individual extract: {EXTRACT_OVERHEAD * 1000:.0f} ms overhead "
        f"+ {EXTRACT_PER_ITEM * 1000:.1f} ms/item\n"
        f"  Batch extract_batch: {BATCH_OVERHEAD * 1000:.0f} ms overhead "
        f"+ {BATCH_PER_ITEM * 1000:.2f} ms/item\n"
        f"  → For batch_size=50: individual={50 * (EXTRACT_OVERHEAD + EXTRACT_PER_ITEM) * 1000:.0f} ms "
        f"vs batch={( BATCH_OVERHEAD + BATCH_PER_ITEM * 50) * 1000:.1f} ms"
    )

    # ── Scenario 1: Serial individual ─────────────────────────────────────
    header("1. Serial individual  (batch_size=1, concurrency=1)")
    print("   Every entity: read → extract() → write, one at a time.")
    print("   Bottleneck: N reads + N extract calls, all sequential.\n")
    t1 = await run(IndividualEmbedding(), "serial individual", batch_size=1, concurrency=1)

    # ── Scenario 2: Concurrent individual ─────────────────────────────────
    header("2. Concurrent individual  (batch_size=1, concurrency=20)")
    print("   Reads overlap — 20 entities in flight at once.")
    print("   BUT extract() still called once per entity.\n")
    t2 = await run(IndividualEmbedding(), "concurrent individual  (concurrency=20)",
                   batch_size=1, concurrency=20)

    # ── Scenario 3: Serial batches ─────────────────────────────────────────
    header("3. Batch extract  (batch_size=50, concurrency=1)")
    print("   Reads within each batch happen concurrently.")
    print("   extract_batch() called 4× (200/50) instead of 200×.\n")
    t3 = await run(BatchEmbedding(), "serial batches  (batch_size=50)",
                   batch_size=50, concurrency=1)

    # ── Scenario 4: Concurrent batches ────────────────────────────────────
    header("4. Batch + concurrent  (batch_size=50, concurrency=4)")
    print("   4 batches of 50 run at the same time.")
    print("   Reads AND compute are both parallelised.\n")
    t4 = await run(BatchEmbedding(), "concurrent batches (batch_size=50, concurrency=4)",
                   batch_size=50, concurrency=4)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    print(
        f"\n  {'Scenario':<44}  {'Time':>6}  {'vs serial':>9}"
        f"\n  {'─' * 44}  {'─' * 6}  {'─' * 9}"
        f"\n  {'1. Serial individual':<44}  {t1:5.2f}s  {'(baseline)':>9}"
        f"\n  {'2. Concurrent individual':<44}  {t2:5.2f}s  {t1 / t2:>8.1f}x"
        f"\n  {'3. Batch extract (serial batches)':<44}  {t3:5.2f}s  {t1 / t3:>8.1f}x"
        f"\n  {'4. Batch + concurrent':<44}  {t4:5.2f}s  {t1 / t4:>8.1f}x"
    )
    print(
        f"""
  What each improvement contributes
  ─────────────────────────────────
  Serial → Concurrent:  {t1 / t2:.1f}× — reads no longer queue behind each other.
                        Does NOT help with compute-bound extract().

  Concurrent → Batch:   {t2 / t3:.1f}× — extract_batch() amortises the fixed
                        per-call overhead across {50} entities at once.
                        Works for CPU and GPU alike.

  Batch → Batch+Conc:   {t3 / t4:.1f}× — multiple batches fly simultaneously;
                        useful when source reads dominate batch compute time.

  Combine all three:    {t1 / t4:.1f}× total speedup over the serial baseline.
"""
    )

    print("  Real-world equivalents for extract_batch():")
    print("    ML inference  — model.forward(batch) vs N × model.forward(x)")
    print("    Embeddings    — client.embed(texts[:2048]) vs N × client.embed(text)")
    print("    Database      — SELECT … WHERE id IN (…) vs N × SELECT … WHERE id = ?")


if __name__ == "__main__":
    asyncio.run(main())
