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

- **Fully async** — all I/O is `async`; plays nicely with any async application
- **Type-safe schemas** — validate floats, ints, strings, booleans, categoricals,
  ndarrays, and bytes before anything hits the store
- **Composable sources** — `SourceBundle` fans out to multiple sources concurrently;
  combine DataFrames, files, HTTP, or anything custom
- **Batteries included** — file, DataFrame, and HTTP sources out of the box;
  memory, file, and Parquet stores; pluggable serializers
- **Never crashes on partial failure** — `generate()` captures every per-entity
  error in a `GenerationReport`; valid entities are always written
- **No required pandas** — the core only needs `numpy`; pandas is optional

---

## Installation

```bash
pip install calcine                  # core (numpy only)
pip install "calcine[http]"          # + async HTTP source
pip install "calcine[parquet]"       # + Parquet store
pip install "calcine[dev]"           # + test/lint tools
```

---

## Quick start

```python
import asyncio
import pandas as pd
from calcine import Pipeline
from calcine.features.base import Feature
from calcine.schema import FeatureSchema, types
from calcine.sources import DataFrameSource
from calcine.stores import MemoryStore


class MeanPurchaseValue(Feature):
    schema = FeatureSchema({"mean_value": types.Float64(nullable=False, default=0.0)})

    async def extract(self, raw, context):
        if raw.empty:
            raise ValueError("No data for this entity")
        return {"mean_value": float(raw["amount"].mean())}


df = pd.DataFrame({
    "entity_id": ["u1", "u1", "u2"],
    "amount":    [10.0, 20.0, 15.0],
})

pipeline = Pipeline(
    source=DataFrameSource(df),
    feature=MeanPurchaseValue(),
    store=MemoryStore(),
)


async def main():
    report = await pipeline.generate(entity_ids=["u1", "u2"])
    print(report)               # GenerationReport(succeeded=2, failed=0)

    value = await pipeline.retrieve("u1")
    print(value)                # {"mean_value": 15.0}

asyncio.run(main())
```

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
    async def extract(self, raw: dict, context: dict) -> dict:
        txns = raw["transactions"]
        prof = raw["profile"]
        embs = raw["embeddings"]
        ...
```

No assumptions are made about what the sources represent or how they relate.

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

## Available components

### Sources

| Class | Description |
|-------|-------------|
| `FileSource(path)` | Read bytes from a single file |
| `DirectorySource(path, pattern)` | Stream bytes from files matching a glob |
| `DataFrameSource(df, entity_col)` | Filter a pandas DataFrame by entity |
| `HTTPSource(url_template, ...)` | Async HTTP GET (requires `[http]`) |
| `SourceBundle(**sources)` | Read multiple sources concurrently |

### Stores

| Class | Description |
|-------|-------------|
| `MemoryStore` | Dict-backed, no I/O — for tests and prototyping |
| `FileStore(path, serializer)` | One file per entity per feature |
| `ParquetStore(path)` | Parquet files partitioned by feature (requires `[parquet]`) |

### Serializers (for FileStore)

| Class | Best for |
|-------|----------|
| `PickleSerializer` | Any Python object (default) |
| `JSONSerializer` | Dict / list / primitive features |
| `NumpySerializer` | `numpy` arrays |

---

## Documentation

| Document | Description |
|----------|-------------|
| [`docs/architecture.md`](docs/architecture.md) | Design rationale and core decisions |
| [`docs/extending.md`](docs/extending.md) | Adding custom sources, stores, and schema types |
| [`docs/schema.md`](docs/schema.md) | Full schema type reference |
| [`examples/README.md`](examples/README.md) | Guide to the runnable examples |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Development setup and PR process |

---

## Running the tests

```bash
uv pip install -e ".[dev]"
pytest
```

99 tests, covering the pipeline, schema, all built-in sources and stores.

---

## Project layout

```
calcine/
├── pipeline.py        Pipeline + GenerationReport
├── schema.py          FeatureSchema + type system
├── serializers.py     Serializer ABC + Pickle / JSON / Numpy
├── exceptions.py      SourceError, StoreError, SchemaViolationError
├── sources/           DataSource ABC + FileSource, DataFrameSource, HTTPSource, SourceBundle
├── features/          Feature ABC
└── stores/            FeatureStore ABC + MemoryStore, FileStore, ParquetStore

tests/                 99 tests mirroring the calcine structure
examples/              5 runnable end-to-end scripts + generated datasets
docs/                  Architecture, extension guide, schema reference
```

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). All contributions are welcome —
new sources, stores, schema types, bug fixes, and documentation improvements.

## License

[MIT](LICENSE)
