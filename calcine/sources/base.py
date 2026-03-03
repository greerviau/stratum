"""Abstract base class for data sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any


class DataSource(ABC):
    """Contract for all data sources in the calcine pipeline.

    A ``DataSource`` is responsible for fetching raw data for a given entity.
    The raw data format is completely source-defined — it might be a pandas
    DataFrame, raw bytes, a JSON dict, etc.

    The ``Pipeline`` calls ``source.read(entity_id=entity_id)`` for each
    entity during ``generate()``.  Implementations should accept
    ``entity_id`` as a keyword argument and use it to scope the returned
    data.  Additional kwargs allow implementations to define their own
    optional parameters.

    Example implementation::

        class MyDBSource(DataSource):
            def __init__(self, conn):
                self.conn = conn

            async def read(self, entity_id: str, **kwargs) -> list[dict]:
                rows = await self.conn.fetchall(
                    "SELECT * FROM events WHERE user_id = $1", entity_id
                )
                return rows
    """

    @abstractmethod
    async def read(self, **kwargs: Any) -> Any:
        """Read data for a single entity from the source.

        Args:
            **kwargs: Source-specific parameters.  Implementations should
                accept ``entity_id: str`` at minimum.

        Returns:
            Raw data in the source's native format.

        Raises:
            SourceError: If the source cannot fulfill the request.
        """
        ...

    async def stream(self, **kwargs: Any) -> AsyncIterator[Any]:
        """Stream data as an async iterator.

        The default implementation wraps ``read()`` as a single-item stream.
        Override this for large sources where incremental reading is more
        efficient.

        Args:
            **kwargs: Source-specific parameters, forwarded to ``read()``.

        Yields:
            Data items from the source.
        """
        result = await self.read(**kwargs)
        yield result
