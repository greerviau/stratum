"""SourceBundle: compose multiple named DataSources into one."""

from __future__ import annotations

import asyncio
from typing import Any

from .base import DataSource


class SourceBundle(DataSource):
    """Read from multiple named sources and return all results as a dict.

    Sources are read concurrently.  The result passed to ``Feature.extract``
    is a plain ``dict`` keyed by whatever names you choose — no assumptions
    are made about what the sources represent or how they relate to each other.

    Args:
        **sources: Keyword arguments mapping name → ``DataSource``.

    Example — two independent raw sources::

        pipeline = Pipeline(
            source=SourceBundle(
                transactions=TransactionSource(),
                profile=ProfileSource(),
                embeddings=EmbeddingSource(),
            ),
            feature=MyFeature(),
            store=MemoryStore(),
        )

        # Inside MyFeature.extract:
        async def extract(self, raw: dict, context: dict) -> Any:
            txns  = raw["transactions"]
            prof  = raw["profile"]
            emb   = raw["embeddings"]
            ...

    All sources receive the same ``**kwargs`` (including ``entity_id``) that
    ``SourceBundle.read`` is called with.  Sources that don't use a particular
    kwarg should accept ``**kwargs`` and ignore them — all built-in sources
    already do this.
    """

    def __init__(self, **sources: DataSource) -> None:
        if not sources:
            raise ValueError("SourceBundle requires at least one source")
        self.sources: dict[str, DataSource] = sources

    async def read(self, **kwargs: Any) -> dict[str, Any]:
        """Read all sources concurrently and return a named dict of results.

        Args:
            **kwargs: Forwarded verbatim to every source's ``read()`` call.

        Returns:
            ``{source_name: result}`` for each source in the bundle.
        """
        names = list(self.sources.keys())
        results = await asyncio.gather(*[self.sources[name].read(**kwargs) for name in names])
        return dict(zip(names, results, strict=True))
