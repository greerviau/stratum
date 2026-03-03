# Extending calcine

Every component in calcine is designed to be subclassed.  This guide shows
the minimum required implementation for each extension point.

## Custom DataSource

```python
from typing import Any
from calcine.sources.base import DataSource
from calcine.exceptions import SourceError


class BigQuerySource(DataSource):
    """Read rows from a BigQuery table filtered by entity_id."""

    def __init__(self, client, table: str, entity_col: str = "entity_id"):
        self.client = client
        self.table = table
        self.entity_col = entity_col

    async def read(self, entity_id: str | None = None, **kwargs: Any):
        if entity_id is None:
            raise ValueError("entity_id is required")
        try:
            query = f"""
                SELECT * FROM `{self.table}`
                WHERE {self.entity_col} = @entity_id
            """
            job = self.client.query(query, job_config=...)
            return await asyncio.wrap_future(job.result())
        except Exception as exc:
            raise SourceError(
                source_name=type(self).__name__,
                entity_id=entity_id,
                cause=exc,
            ) from exc
```

**Rules:**

- Accept `entity_id` as a keyword argument; scope results to it
- Wrap all I/O failures in `SourceError` with the three required fields
- Return whatever type your `Feature` expects — there's no constraint on format
- Use `asyncio.get_running_loop().run_in_executor(None, ...)` to offload
  synchronous blocking calls

---

## Custom Feature

```python
from calcine.features.base import Feature
from calcine.schema import FeatureSchema, types


class SentimentFeature(Feature):
    schema = FeatureSchema({
        "score":     types.Float64(nullable=False),
        "label":     types.Category(categories=["positive", "neutral", "negative"]),
        "confident": types.Boolean(nullable=False),
    })

    def __init__(self, model):
        self.model = model

    async def pre_extract(self, raw):
        # Strip HTML, normalise whitespace, etc.
        return raw.strip()

    async def extract(self, raw: str, context: dict) -> dict:
        score = await self.model.predict(raw)
        label = "positive" if score > 0.6 else "negative" if score < 0.4 else "neutral"
        return {
            "score":     float(score),
            "label":     label,
            "confident": abs(score - 0.5) > 0.3,
        }

    async def post_extract(self, result: dict) -> dict:
        # Round to 4 dp for storage efficiency
        result["score"] = round(result["score"], 4)
        return result
```

**Lifecycle reminder:**

```
source.read() → pre_extract(raw) → extract(raw, context)
              → post_extract(result) → validate(result) → store.write()
```

All hooks have pass-through defaults — only override what you need.

---

## Custom FeatureStore

```python
import pickle
from calcine.stores.base import FeatureStore
from calcine.exceptions import StoreError


class RedisStore(FeatureStore):
    def __init__(self, redis):
        self.redis = redis

    def _key(self, feature, entity_id: str) -> str:
        return f"calcine:{self._feature_key(feature)}:{entity_id}"

    async def write(self, feature, entity_id, data):
        try:
            await self.redis.set(self._key(feature, entity_id), pickle.dumps(data))
        except Exception as exc:
            raise StoreError(
                store_name=type(self).__name__,
                feature_name=self._feature_key(feature),
                entity_id=entity_id,
                cause=exc,
            ) from exc

    async def read(self, feature, entity_id):
        try:
            raw = await self.redis.get(self._key(feature, entity_id))
        except Exception as exc:
            raise StoreError(
                store_name=type(self).__name__,
                feature_name=self._feature_key(feature),
                entity_id=entity_id,
                cause=exc,
            ) from exc
        if raw is None:
            raise KeyError(f"No data for feature '{self._feature_key(feature)}', entity '{entity_id}'")
        return pickle.loads(raw)

    async def exists(self, feature, entity_id) -> bool:
        return bool(await self.redis.exists(self._key(feature, entity_id)))

    async def delete(self, feature, entity_id):
        deleted = await self.redis.delete(self._key(feature, entity_id))
        if not deleted:
            raise KeyError(f"No data for feature '{self._feature_key(feature)}', entity '{entity_id}'")
```

**Rules:**

- `read` and `delete` must raise `KeyError` (not `StoreError`) when the entity
  simply doesn't exist — this is how `retrieve_batch` knows to silently skip it
- Wrap all other I/O failures in `StoreError` with the four required fields
- Use `_feature_key(feature)` for namespacing, not `id(feature)`

---

## Custom Serializer (for FileStore)

```python
import msgpack
from calcine.serializers import Serializer


class MsgPackSerializer(Serializer):
    def serialize(self, data) -> bytes:
        return msgpack.packb(data, use_bin_type=True)

    def deserialize(self, raw: bytes):
        return msgpack.unpackb(raw, raw=False)
```

```python
store = FileStore("/data/features", serializer=MsgPackSerializer())
```

---

## Custom schema type

```python
from calcine.schema import FeatureType, types
from typing import Any


class PositiveFloat(FeatureType):
    """A float that must be strictly greater than zero."""

    def _validate_value(self, value: Any) -> list[str]:
        try:
            f = float(value)
        except (TypeError, ValueError):
            return [f"Expected numeric value, got {type(value).__name__}"]
        if f <= 0:
            return [f"Expected positive float, got {f}"]
        return []


# Use it directly
schema = FeatureSchema({"price": PositiveFloat(nullable=False)})
```

---

## Combining extensions

Extensions compose naturally:

```python
Pipeline(
    source=SourceBundle(
        events=BigQuerySource(bq_client, "project.dataset.events"),
        profile=PostgresSource(pg_pool, "users"),
    ),
    feature=SentimentFeature(model=my_model),
    store=RedisStore(redis_client),
)
```
