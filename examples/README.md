# Examples

End-to-end scripts that demonstrate stratum across different data formats,
storage backends, and source configurations.

## Setup

First install the library with all optional dependencies:

```bash
uv pip install -e ".[dev]"
```

Then generate the large example datasets (**required** before running examples
01, 03, and 04):

```bash
python examples/data/generate_data.py
```

This creates:

| File | Size | Used by |
|------|------|---------|
| `data/reviews.csv` | ~25 MB | examples 01, 03 |
| `data/sensor_readings.csv` | ~16 MB | example 04 |
| `data/documents/` | ~4 MB (1 000 files) | example 02 |
| `data/user_profiles.csv` | ~160 KB | example 03 |

> **Note:** The large files are in `.gitignore`.  `user_profiles.csv` is small
> enough to commit and is included in the repo.

---

## Example index

### `basic_usage.py` — Hello world

The simplest possible pipeline: DataFrame source → mean feature → memory store.

```bash
python examples/basic_usage.py
```

---

### `01_text_features.py` — Tabular features from review text

Extracts 7 per-user statistics from 98k reviews into a `ParquetStore`.
Demonstrates multi-field schema, `Category` type, and timing a moderate-scale
run (~5 000 entities).

```bash
python examples/01_text_features.py
```

**Interesting parts:** `ReviewStats.extract`, `ParquetStore` output,
`context` threading.

---

### `02_embeddings.py` — NDArray embeddings + NumpySerializer

Generates a 64-dim document embedding per entity using a hashed tri-gram
projection (no ML library required).  Stores raw numpy arrays via
`FileStore + NumpySerializer`.  Prints a cosine similarity matrix.

```bash
python examples/02_embeddings.py
```

**Interesting parts:** returning a raw `ndarray` from `extract`,
single-field schema validation, `post_extract` for L2 normalisation.

---

### `03_multi_source.py` — SourceBundle with mixed source types

Combines transaction history (`DataFrameSource`), user demographics
(`DataFrameSource`), and a static config dict (`ConstantSource`) to compute
a per-user risk score.

```bash
python examples/03_multi_source.py
```

**Interesting parts:** `SourceBundle`, `pre_extract` column trimming,
swapping `MemoryStore` ↔ `FileStore` without touching source or feature.

---

### `04_timeseries_scale.py` — Time-series stats + scale benchmark

Computes 10 statistical features (mean, std, trend slope, anomaly score,
spike count) over 502k sensor readings, then benchmarks serial vs. a manual
`asyncio.gather` workaround (historical — see example 06 for the built-in
solution).

```bash
python examples/04_timeseries_scale.py
```

**Interesting parts:** variable-length time series, `JSONSerializer`,
numpy-heavy feature extraction.

---

### `05_weak_points.py` — Known limitations and mitigations

Five documented limitations, each with a minimal reproduction and a concrete
fix:

| # | Limitation | Mitigation |
|---|-----------|-----------|
| A | `NaN` passes `Float64` | `post_extract` finite check |
| B | Empty source silently reaches `extract` | Guard `raw.empty` |
| C | `SourceBundle` is all-or-nothing | `FaultTolerantSource` wrapper |
| D | Feature class-name collision in store | `feature_name` attribute + custom `_feature_key` |
| E | `generate()` is serial | now solved — see example 06 |

```bash
python examples/05_weak_points.py
```

---

### `06_partitioned_generation.py` — Concurrency modes and incremental runs

480 simulated IoT sensors across 6 regional clusters demonstrate every
`generate()` concurrency mode.  A 5 ms async latency per sensor makes the
speedups tangible.

```bash
python examples/06_partitioned_generation.py
```

Six sections, no external data required:

| Section | What it shows |
|---------|--------------|
| A | Serial baseline (`concurrency=1`) |
| B | Flat concurrency — max throughput, no ordering guarantee |
| C | `partition_by` — serial within region, concurrent across regions |
| D | Explicit `partitions` dict — groups from external metadata |
| E | `overwrite=False` — skip already-stored entities on re-run |
| F | `on_progress` — real-time progress bar + tqdm integration hint |

**Interesting parts:** `RegionalAPISource` with simulated latency, within-region
ordering verification, incremental timing comparison, progress bar render.

---

## Adding your own example

Follow the naming convention `NN_description.py` and open a PR.  Examples
should be self-contained (or depend only on `examples/data/`) and runnable
with a single `python examples/NN_description.py` command.
