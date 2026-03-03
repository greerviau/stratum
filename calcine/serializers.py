"""Serializers for converting feature values to/from raw bytes.

Used by ``FileStore`` to persist feature values.  You can plug any
``Serializer`` into ``FileStore``::

    store = FileStore("/tmp/features", serializer=JSONSerializer())
"""

from __future__ import annotations

import io
import json
import pickle
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Serializer(ABC):
    """Abstract base class for data serializers.

    Implementations convert arbitrary Python objects to bytes and back.
    """

    @abstractmethod
    def serialize(self, data: Any) -> bytes:
        """Serialize *data* to bytes."""
        ...

    @abstractmethod
    def deserialize(self, raw: bytes) -> Any:
        """Deserialize *raw* bytes back to a Python object."""
        ...


class PickleSerializer(Serializer):
    """Serialize using Python's ``pickle``.

    This is the default serializer.  It handles any picklable Python object
    but produces non-portable bytes (Python-version-dependent).
    """

    def serialize(self, data: Any) -> bytes:
        return pickle.dumps(data)

    def deserialize(self, raw: bytes) -> Any:
        return pickle.loads(raw)  # noqa: S301


class JSONSerializer(Serializer):
    """Serialize using JSON.

    Best suited for dict/list/primitive feature values.  Not compatible
    with numpy arrays or arbitrary Python objects.
    """

    def serialize(self, data: Any) -> bytes:
        return json.dumps(data).encode("utf-8")

    def deserialize(self, raw: bytes) -> Any:
        return json.loads(raw.decode("utf-8"))


class NumpySerializer(Serializer):
    """Serialize numpy arrays using ``np.save`` / ``np.load``.

    Produces compact, portable binary blobs suitable for embedding vectors,
    image tensors, and similar array-valued features.
    """

    def serialize(self, data: Any) -> bytes:
        buf = io.BytesIO()
        np.save(buf, data)
        return buf.getvalue()

    def deserialize(self, raw: bytes) -> Any:
        buf = io.BytesIO(raw)
        return np.load(buf, allow_pickle=False)
