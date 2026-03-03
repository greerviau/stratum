"""File-based data sources."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from ..exceptions import SourceError
from .base import DataSource


class FileSource(DataSource):
    """Read raw bytes from a single local file.

    Args:
        path: Path to the file to read.

    Example::

        source = FileSource("/data/model_weights.bin")
        raw = await source.read()
    """

    def __init__(self, path: str) -> None:
        self.path = path

    async def read(self, **kwargs: Any) -> bytes:
        """Read the file and return its contents as bytes.

        Args:
            **kwargs: Ignored (accepts ``entity_id`` without error).

        Returns:
            File contents as bytes.

        Raises:
            SourceError: If the file cannot be read.
        """
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, Path(self.path).read_bytes)
        except Exception as exc:
            raise SourceError(
                source_name=type(self).__name__,
                entity_id=kwargs.get("entity_id", self.path),
                cause=exc,
            ) from exc


class DirectorySource(DataSource):
    """Stream bytes from all files matching a glob pattern in a directory.

    Args:
        path: Path to the directory.
        pattern: Glob pattern to match files (default ``"*"``).

    Example::

        source = DirectorySource("/data/images", pattern="*.jpg")
        async for img_bytes in source.stream():
            ...
    """

    def __init__(self, path: str, pattern: str = "*") -> None:
        self.path = path
        self.pattern = pattern

    async def read(self, **kwargs: Any) -> list[bytes]:
        """Read all matching files and return them as a list of byte strings.

        Returns:
            List of file contents, sorted by filename.
        """
        results: list[bytes] = []
        async for chunk in self.stream(**kwargs):
            results.append(chunk)
        return results

    async def stream(self, **kwargs: Any) -> AsyncIterator[bytes]:
        """Yield bytes from each file matching the pattern.

        Files are yielded in sorted path order.

        Raises:
            SourceError: If the directory cannot be listed or a file cannot
                be read.
        """
        directory = Path(self.path)
        loop = asyncio.get_running_loop()

        try:
            paths = sorted(directory.glob(self.pattern))
        except Exception as exc:
            raise SourceError(
                source_name=type(self).__name__,
                entity_id=kwargs.get("entity_id", self.path),
                cause=exc,
            ) from exc

        for file_path in paths:
            if not file_path.is_file():
                continue
            try:
                content = await loop.run_in_executor(None, file_path.read_bytes)
                yield content
            except Exception as exc:
                raise SourceError(
                    source_name=type(self).__name__,
                    entity_id=str(file_path),
                    cause=exc,
                ) from exc
