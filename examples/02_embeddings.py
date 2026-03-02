"""02 — Document Embedding Features

Reads 1 000 plain-text document files from examples/data/documents/ and
produces a 64-dimensional embedding for each entity, stored as a NumPy array.

The embedding is a random-projection of character tri-gram term frequencies —
no ML framework required, all numpy.

Demonstrates:
  - FileSource inside a custom per-entity routing DataSource
  - NDArray schema with shape and dtype enforcement
  - FileStore + NumpySerializer for compact array persistence
  - Single-field schema (the whole feature value is the array, not a dict)
  - post_extract hook: L2-normalise the vector after extraction

Run:
    python examples/02_embeddings.py
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import numpy as np

from stratum import Pipeline
from stratum.features.base import Feature
from stratum.schema import FeatureSchema, types
from stratum.serializers import NumpySerializer
from stratum.sources.base import DataSource
from stratum.stores import FileStore

DATA = Path(__file__).parent / "data"
DOC_DIR = DATA / "documents"
STORE = Path(__file__).parent / "data" / "store_02_embeddings"

EMBED_DIM = 64
VOCAB_SIZE = 4096  # number of tri-gram hash buckets

# Fixed random projection matrix (seeded so results are reproducible)
_rng = np.random.default_rng(0)
PROJECTION = (_rng.standard_normal((VOCAB_SIZE, EMBED_DIM)) / np.sqrt(EMBED_DIM)).astype(np.float32)


# ---------------------------------------------------------------------------
# Custom DataSource: routes entity_id to its file
# ---------------------------------------------------------------------------


class DocumentDirectorySource(DataSource):
    """Reads a single document file for each entity_id.

    Expects files named  {doc_dir}/{entity_id}.txt
    """

    def __init__(self, doc_dir: Path) -> None:
        self.doc_dir = doc_dir

    async def read(self, entity_id: str | None = None, **kwargs: Any) -> str:
        if entity_id is None:
            raise ValueError("entity_id required")
        path = self.doc_dir / f"{entity_id}.txt"
        if not path.exists():
            raise FileNotFoundError(f"No document for entity '{entity_id}'")
        return path.read_text()


# ---------------------------------------------------------------------------
# Embedding helper (pure numpy, no ML deps)
# ---------------------------------------------------------------------------


def _trigram_embed(text: str) -> np.ndarray:
    """Map text → EMBED_DIM float32 vector via hashed tri-gram projection."""
    text = text.lower()
    counts = np.zeros(VOCAB_SIZE, dtype=np.float32)
    for i in range(len(text) - 2):
        trigram = text[i : i + 3]
        bucket = hash(trigram) % VOCAB_SIZE
        counts[bucket] += 1.0
    # TF normalise
    total = counts.sum()
    if total > 0:
        counts /= total
    return (PROJECTION.T @ counts).astype(np.float32)


# ---------------------------------------------------------------------------
# Feature definition
# ---------------------------------------------------------------------------


class DocumentEmbedding(Feature):
    """64-dim L2-normalised document embedding.

    Returns a raw ndarray (not a dict).  The single-field schema validates
    the array shape and dtype directly, and NumpySerializer stores it as a
    compact binary blob — no dict wrapper needed.
    """

    # Single-field schema: validates a raw ndarray value directly
    schema = FeatureSchema(
        {"embedding": types.NDArray(shape=(EMBED_DIM,), dtype="float32", nullable=False)}
    )

    async def extract(self, raw: str, context: dict, entity_id: str | None = None) -> np.ndarray:
        return _trigram_embed(raw)

    async def post_extract(self, result: np.ndarray) -> np.ndarray:
        """L2-normalise the vector."""
        norm = float(np.linalg.norm(result))
        if norm > 0:
            result = (result / norm).astype(np.float32)
        return result


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # already L2-normalised


async def main() -> None:
    doc_files = sorted(DOC_DIR.glob("*.txt"))
    entity_ids = [f.stem for f in doc_files]  # "doc_0000", "doc_0001", …
    print(f"Found {len(entity_ids)} documents in {DOC_DIR}")

    STORE.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline(
        source=DocumentDirectorySource(DOC_DIR),
        feature=DocumentEmbedding(),
        store=FileStore(str(STORE), serializer=NumpySerializer()),
    )

    print(f"Generating embeddings for {len(entity_ids)} entities …")
    t0 = time.perf_counter()
    report = await pipeline.generate(entity_ids=entity_ids)
    elapsed = time.perf_counter() - t0

    print(
        f"  Done in {elapsed:.2f}s  |  {report.success_count} OK  |  {report.failure_count} failed"
    )

    # Retrieve and compute pairwise cosine similarities for a small sample
    sample_ids = entity_ids[:5]
    embeddings = await pipeline.retrieve_batch(sample_ids)

    # Stored value is a raw ndarray (not a dict)
    print("\nCosine similarity matrix (first 5 docs):")
    header = "      " + "  ".join(f"{e[-4:]}" for e in sample_ids)
    print(header)
    for id_a in sample_ids:
        row = f"{id_a[-4:]}  "
        for id_b in sample_ids:
            sim = _cosine_sim(embeddings[id_a], embeddings[id_b])
            row += f"{sim:5.3f}  "
        print(row)

    # Show that schema rejects a wrong-shaped array
    bad_vec = np.zeros(32, dtype=np.float32)  # wrong dim
    errors = DocumentEmbedding.schema.validate(bad_vec)
    print(f"\nSchema on wrong-shape array: {errors}")

    good_vec = np.zeros(EMBED_DIM, dtype=np.float32)
    errors = DocumentEmbedding.schema.validate(good_vec)
    print(f"Schema on correct array:      {errors}")


if __name__ == "__main__":
    asyncio.run(main())
