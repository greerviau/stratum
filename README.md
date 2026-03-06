# calcine

[![CI](https://github.com/greerviau/calcine/actions/workflows/ci.yml/badge.svg)](https://github.com/greerviau/calcine/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A source-agnostic, type-agnostic featurization pipeline framework for Python.

```
DataSource  ──►  Feature  ──►  FeatureStore
```

calcine gives you a clean three-part abstraction for building reproducible,
validated feature extraction pipelines — over any data source and any storage
backend, with no lock-in on format or framework.

---

## Highlights

- **Pipeline orchestration** — concurrent entity processing with a semaphore cap; per-entity error isolation so valid results are always stored even when others fail; incremental generation skips already-stored entities; partition-by support for rate-limit-per-account and ordered-per-user scenarios
- **Type-safe schemas** — validate scalars, strings, categoricals, ndarrays, bytes, lists, and dicts before anything hits the store; the same schema validates on read, making it a typed contract between feature producers and consumers
- **Detailed reporting** — `GenerationReport` tracks successes, failures, and skips with per-phase timing (read / extract / write); `timing_summary()` surfaces p50/p95/max per phase so you can pinpoint bottlenecks; `error_summary()` groups failures by message; exports to a pandas DataFrame
- **Fan-out extraction** — `extract()` returns an `ExtractionResult` with one or many records; each sub-entity is stored, validated, and retrievable independently
- **Composable sources** — `SourceBundle` reads from multiple sources concurrently and delivers a single `dict` to `extract`; combine any data origins without changing the pipeline
- **Executor support** — offload CPU-bound extraction to thread or process pools via `executor=`; store writes always remain in the main process so all store backends work correctly

---

## Installation

```bash
pip install calcine                  # core
pip install "calcine[http]"          # + async HTTP source
pip install "calcine[parquet]"       # + Parquet store
pip install "calcine[dev]"           # + test/lint tools
```

---

## Quick start

```python
from calcine import ExtractionResult, Pipeline
from calcine.features.base import Feature
from calcine.schema import FeatureSchema, types
from calcine.sources.base import DataSource
from calcine.stores import MemoryStore


# --- 1. Define your data source (any async I/O: DB, API, S3, …) ---

class UserDBSource(DataSource):
    async def read(self, entity_id: str, **kwargs) -> dict:
        return await db.fetch_user(entity_id)  # your async call here


# --- 2. Define a schema-validated feature ---

class UserEngagementFeature(Feature):
    schema = FeatureSchema({
        "spend_tier":  types.Category(categories=["low", "mid", "high", "whale"]),
        "event_rate":  types.Float64(nullable=False),
        "total_spend": types.Float64(nullable=False),
    })

    async def extract(self, raw: dict, context: dict, entity_id=None) -> ExtractionResult:
        spend = raw["total_spend"]
        if spend < 100:    tier = "low"
        elif spend < 1000: tier = "mid"
        elif spend < 3000: tier = "high"
        else:              tier = "whale"
        return ExtractionResult.of(entity_id, {
            "spend_tier":  tier,
            "event_rate":  raw["event_count"] / raw["days_active"],
            "total_spend": spend,
        })


# --- 3. Build and run ---

pipeline = Pipeline(
    source=UserDBSource(),
    feature=UserEngagementFeature(),
    store=MemoryStore(),
)

# Concurrent reads; failures are isolated per entity
report = pipeline.generate(entity_ids=user_ids, concurrency=32)
print(report)
# GenerationReport(entities=997, records=997, failed=3, skipped=0, duration=1.24s)

# Identify bottlenecks across read / extract / write phases
summary = report.timing_summary()
print(f"p95 read:    {summary['read']['p95']*1000:.1f} ms")
print(f"p95 extract: {summary['extract']['p95']*1000:.1f} ms")

# Re-run later — already-stored entities are skipped automatically
pipeline.generate(entity_ids=new_user_ids, overwrite=False)

value = pipeline.retrieve("u42")
```

See [`examples/basic_usage.py`](examples/basic_usage.py) for a fully runnable version with a simulated async source, bad-data handling, and incremental generation.

---

## Multiple sources with SourceBundle

When your feature needs data from more than one place, compose sources with
`SourceBundle`. All sources are read concurrently; `Feature.extract` receives
a plain `dict` keyed by whatever names you choose:

```python
from calcine.sources import SourceBundle

pipeline = Pipeline(
    source=SourceBundle(
        transactions=TransactionSource(),
        profile=ProfileSource(),
        embeddings=EmbeddingSource(),
    ),
    feature=MyFeature(),
    store=MemoryStore(),
)


class MyFeature(Feature):
    async def extract(self, raw: dict, context: dict, entity_id=None) -> ExtractionResult:
        txns = raw["transactions"]
        prof = raw["profile"]
        embs = raw["embeddings"]
        ...
```

No assumptions are made about what the sources represent or how they relate.

---

## Fan-out extraction

When one source entity produces multiple independently-stored sub-entity records
(audio → segments, document → chunks, session → events), return an
`ExtractionResult` with multiple records from `extract`:

```python
from calcine import ExtractionResult

class AudioSegmentFeature(Feature):
    metadata_schema = FeatureSchema({
        "sample_rate": types.Int64(nullable=False),
        "speaker_id":  types.String(nullable=True),
    })
    schema = FeatureSchema({
        "rms": types.Float64(nullable=False),
    })

    async def extract(self, raw: bytes, context: dict, entity_id: str | None = None) -> ExtractionResult:
        segments = split_audio(raw)
        return ExtractionResult(
            metadata={"sample_rate": 16000, "speaker_id": "alice"},
            records={f"{entity_id}/{i}": {"rms": rms(s)} for i, s in enumerate(segments)},
        )


report = pipeline.generate(entity_ids=recording_ids)

# Retrieve parent metadata and sub-entity records
meta     = store.read(feature, "recording_001")
sub_ids  = store.list_entities(feature, prefix="recording_001/")
segments = [store.read(feature, sid) for sid in sub_ids]
```

`ExtractionResult.of(entity_id, value)` is a convenience constructor for
single-record features. For fan-out, pass `records` directly with sub-entity IDs.
Parent metadata and sub-entity records are stored under separate keys;
`overwrite=False` skips the source entity if its parent key already exists.

---

## Schema system

```python
from calcine.schema import FeatureSchema, types

schema = FeatureSchema({
    "score":     types.Float64(nullable=False, default=0.0),
    "category":  types.Category(categories=["low", "mid", "high"]),
    "embedding": types.NDArray(shape=(None, 128), dtype="float32"),
    "label":     types.String(nullable=True),
    "active":    types.Boolean(),
    "count":     types.Int64(nullable=False),
    "tags":      types.List(item_type=types.String()),
    "scores":    types.Dict(key_type=types.String(), value_type=types.Float64()),
    "payload":   types.Bytes(),
    "anything":  types.Any(),
})
```

For non-dict features (e.g. raw arrays), use a single-field schema:

```python
schema = FeatureSchema({"_vec": types.NDArray(shape=(128,), dtype="float32")})
errors = schema.validate(arr)   # validates the array directly
```

See [`docs/schema.md`](docs/schema.md) for the full reference.

---

## Built-in components

calcine ships with reference sources, stores, and serializers for common patterns.
See [`docs/sources.md`](docs/sources.md) and [`docs/stores.md`](docs/stores.md).

---

## Documentation

See [`docs/`](docs/README.md) for the full documentation index.

---

## Running the tests

```bash
uv pip install -e ".[dev]"
pytest
```

---

## Project layout

```
calcine/
├── pipeline.py        Pipeline + GenerationReport
├── extraction.py      ExtractionResult
├── schema.py          FeatureSchema + type system
├── serializers.py     Serializer ABC + Pickle / JSON / Numpy
├── exceptions.py      SourceError, StoreError, SchemaViolationError
├── sources/           DataSource ABC + built-in sources
├── features/          Feature ABC
└── stores/            FeatureStore ABC + built-in stores

tests/                 test suite mirroring the calcine structure
examples/              runnable end-to-end scripts + generated datasets
docs/                  architecture, extension guide, schema reference
```

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). All contributions are welcome —
new sources, stores, schema types, bug fixes, and documentation improvements.

## License

[MIT](LICENSE)
