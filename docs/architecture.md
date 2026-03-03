# Architecture

This document explains the core design decisions in calcine and the
reasoning behind them.

## The three-part abstraction

```
DataSource ──► Feature ──► FeatureStore
                 │
             GenerationReport
```

Each component has a single, narrow responsibility:

| Component | Responsibility |
|-----------|---------------|
| `DataSource` | Fetching raw data for a given entity |
| `Feature` | Transforming raw data into a typed value |
| `FeatureStore` | Persisting and retrieving feature values |

`Pipeline` is a thin coordinator — it calls these three components in sequence
and collects results. It contains no domain logic of its own.

---

## Why everything is async

All I/O methods (`DataSource.read`, `FeatureStore.write`, `FeatureStore.read`)
are `async` even when a concrete implementation does nothing I/O-bound (e.g.
`MemoryStore`).

**Rationale:** Sources and stores often do I/O in practice — reading from
object storage, querying a database, calling an inference API. Making the
interface async from the start means:

- Implementations never need to block the event loop
- The `Pipeline` can be run inside any async application without adapter wrappers
- Users don't face a breaking API change when they switch from `MemoryStore`
  to a network-backed store

---

## Why `raw` is `Any`

`Feature.extract(raw, context)` takes `raw: Any`. This is intentional.

The type of `raw` is entirely determined by the `DataSource` — a
`DataFrameSource` produces a `pd.DataFrame`, a `FileSource` produces `bytes`,
and a `SourceBundle` produces a `dict`. There is no single type that covers all
of these, and constraining it would either force every Feature to accept `Any`
anyway or require awkward generic parameterisation.

The schema system (`FeatureSchema`) handles output typing rigorously. Input
typing is left to the Feature author, who always knows what source they pair
with.

---

## SourceBundle design

`SourceBundle` is itself a `DataSource` — it composes at the source layer
rather than at the pipeline layer. This keeps `Pipeline.__init__` unchanged
(still `source: DataSource`) and means bundles can be nested or mixed freely.

Sources in a bundle are read **concurrently** via `asyncio.gather`. This
matters for sources that do real I/O (HTTP, database) where waiting for each
source serially would be wasteful.

The downside is that a failure in any sub-source fails the whole bundle for
that entity. See [Weak Point C in the examples](../examples/05_weak_points.py)
for the mitigation pattern.

---

## Schema design

The schema system is deliberately simple:

- `FeatureSchema` holds a `dict[str, FeatureType]`
- Validation returns `list[str]` — never raises
- Schema failures produce entries in `report.failed`, not exceptions

This means the pipeline never aborts due to a schema violation. Partial output
(valid entities) is always captured even when some entities fail validation.

**Single-field schema for non-dict features:**
When a `Feature` returns a raw value (e.g. a numpy array), the schema can
validate it directly using a single-field schema. The field name is arbitrary;
only its type validator matters:

```python
schema = FeatureSchema({"_vec": types.NDArray(shape=(128,), dtype="float32")})
errors = schema.validate(arr)   # validates arr directly, not a dict
```

---

## Error handling philosophy

calcine distinguishes three categories of failure:

| Error type | Cause | Pipeline behaviour |
|-----------|-------|--------------------|
| `SourceError` | I/O or missing data in the source | Entity goes to `report.failed` |
| Schema violation | `Feature.validate` returns non-empty list | Entity goes to `report.failed` |
| `StoreError` | Persistent storage failure | Entity goes to `report.failed` |
| Unhandled exception | Bug in Feature or unexpected error | Entity goes to `report.failed` |

`Pipeline.generate()` **never raises**. The caller can inspect
`report.failed` to decide whether to retry, alert, or discard.

---

## Store key design

By default, stores use `type(feature).__name__` as the namespace key.  This is
simple and works well for most use cases.

For multi-team or multi-module codebases where class name collisions are
possible, override `_feature_key` in a store subclass (see
[Weak Point D in the examples](../examples/05_weak_points.py)).

---

## Serializers

Serializers are an implementation detail of `FileStore` — they are not part of
the `FeatureStore` interface. This means:

- `MemoryStore` and `ParquetStore` have no serializer concept
- `FileStore` can accept any `Serializer` without changing the `FeatureStore` API
- Users can add custom serializers (e.g. MessagePack, Arrow IPC) without
  touching any framework code

---

## Known limitations

See [`examples/05_weak_points.py`](../examples/05_weak_points.py) for
executable demonstrations and workarounds for:

- `float('nan')` passing `Float64` validation
- Empty `DataFrameSource` results silently reaching `extract()`
- `SourceBundle` all-or-nothing failure semantics
- Feature class-name collisions in stores
- Serial entity processing in `generate()`
