"""In-memory feature store."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import FeatureStore

if TYPE_CHECKING:
    from ..features.base import Feature


class MemoryStore(FeatureStore):
    """In-memory feature store backed by a nested dict.

    Fully functional with no I/O dependencies — ideal for tests, notebooks,
    and rapid prototyping.  Data does not persist across ``MemoryStore``
    instances or process restarts.

    Example::

        store = MemoryStore()
        pipeline = Pipeline(source, feature, store)
        await pipeline.generate(["u1", "u2"])

        value = await store.read(feature, "u1")
    """

    def __init__(self) -> None:
        # Structure: {feature_name: {entity_id: data}}
        self._data: dict[str, dict[str, Any]] = {}

    async def write(self, feature: Feature, entity_id: str, data: Any, context: dict | None = None) -> None:
        key = self._feature_key(feature)
        if key not in self._data:
            self._data[key] = {}
        self._data[key][entity_id] = data

    async def read(self, feature: Feature, entity_id: str) -> Any:
        key = self._feature_key(feature)
        try:
            return self._data[key][entity_id]
        except KeyError:
            raise KeyError(f"No data for feature '{key}', entity '{entity_id}'") from None

    async def exists(self, feature: Feature, entity_id: str) -> bool:
        key = self._feature_key(feature)
        return key in self._data and entity_id in self._data[key]

    async def delete(self, feature: Feature, entity_id: str) -> None:
        key = self._feature_key(feature)
        try:
            del self._data[key][entity_id]
        except KeyError:
            raise KeyError(f"No data for feature '{key}', entity '{entity_id}'") from None

    def __repr__(self) -> str:  # pragma: no cover
        sizes = {k: len(v) for k, v in self._data.items()}
        return f"MemoryStore({sizes})"
