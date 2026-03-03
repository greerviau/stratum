"""File-based feature store."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..exceptions import StoreError
from ..serializers import PickleSerializer, Serializer
from .base import FeatureStore

if TYPE_CHECKING:
    from ..features.base import Feature


class FileStore(FeatureStore):
    """Persist feature values as individual files on the local filesystem.

    Directory layout::

        {base_path}/
            {FeatureClassName}/
                {entity_id}.bin

    Args:
        path: Base directory where feature files are stored.  Created
            automatically on first write.
        serializer: Serializer to use for encoding values.  Defaults to
            ``PickleSerializer``.

    Example::

        store = FileStore("/tmp/features", serializer=JSONSerializer())
        await store.write(feature, "u1", {"score": 0.92})
    """

    def __init__(self, path: str, serializer: Serializer | None = None) -> None:
        self.path = Path(path)
        self.serializer: Serializer = serializer if serializer is not None else PickleSerializer()

    def _entity_path(self, feature: Feature, entity_id: str) -> Path:
        return self.path / self._feature_key(feature) / f"{entity_id}.bin"

    async def write(self, feature: Feature, entity_id: str, data: Any) -> None:
        path = self._entity_path(feature, entity_id)
        loop = asyncio.get_running_loop()

        def _write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.serializer.serialize(data))

        try:
            await loop.run_in_executor(None, _write)
        except Exception as exc:
            raise StoreError(
                store_name=type(self).__name__,
                feature_name=self._feature_key(feature),
                entity_id=entity_id,
                cause=exc,
            ) from exc

    async def read(self, feature: Feature, entity_id: str) -> Any:
        path = self._entity_path(feature, entity_id)
        if not path.exists():
            raise KeyError(
                f"No data for feature '{self._feature_key(feature)}', entity '{entity_id}'"
            )
        loop = asyncio.get_running_loop()
        try:
            raw = await loop.run_in_executor(None, path.read_bytes)
            return self.serializer.deserialize(raw)
        except KeyError:
            raise
        except Exception as exc:
            raise StoreError(
                store_name=type(self).__name__,
                feature_name=self._feature_key(feature),
                entity_id=entity_id,
                cause=exc,
            ) from exc

    async def exists(self, feature: Feature, entity_id: str) -> bool:
        return self._entity_path(feature, entity_id).exists()

    async def delete(self, feature: Feature, entity_id: str) -> None:
        path = self._entity_path(feature, entity_id)
        if not path.exists():
            raise KeyError(
                f"No data for feature '{self._feature_key(feature)}', entity '{entity_id}'"
            )
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, path.unlink)
        except Exception as exc:
            raise StoreError(
                store_name=type(self).__name__,
                feature_name=self._feature_key(feature),
                entity_id=entity_id,
                cause=exc,
            ) from exc
