"""Async HTTP data source backed by httpx."""

from __future__ import annotations

from typing import Any

from ..exceptions import SourceError
from .base import DataSource


class HTTPSource(DataSource):
    """Fetch data via async HTTP GET.

    The URL template may contain a ``{entity_id}`` placeholder that is
    substituted at read time::

        source = HTTPSource("https://api.example.com/users/{entity_id}/profile")
        profile = await source.read(entity_id="u123")

    Args:
        url_template: URL string.  May contain ``{entity_id}`` which is
            formatted with the ``entity_id`` kwarg at read time.
        as_json: If ``True``, parse the response body as JSON and return
            the parsed object.  If ``False`` (default), return raw bytes.
        **request_kwargs: Additional keyword arguments forwarded to
            ``httpx.AsyncClient.get()`` (e.g. ``headers``, ``timeout``).

    Requires the ``[http]`` extra::

        pip install calcine[http]
    """

    def __init__(
        self,
        url_template: str,
        as_json: bool = False,
        **request_kwargs: Any,
    ) -> None:
        self.url_template = url_template
        self.as_json = as_json
        self.request_kwargs = request_kwargs

    async def read(self, entity_id: str | None = None, **kwargs: Any) -> bytes | Any:
        """Perform an async HTTP GET and return the response body.

        Args:
            entity_id: Used to format ``url_template`` if it contains
                the ``{entity_id}`` placeholder.
            **kwargs: Merged into ``request_kwargs`` (overriding on conflict).

        Returns:
            Response bytes, or parsed JSON object if ``as_json=True``.

        Raises:
            ImportError: If ``httpx`` is not installed.
            SourceError: If the HTTP request fails or returns a non-2xx status.
        """
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is required for HTTPSource. Install with: pip install calcine[http]"
            ) from exc

        url = self.url_template
        if entity_id is not None:
            url = url.format(entity_id=entity_id)

        merged = {**self.request_kwargs, **kwargs}
        # Remove entity_id from kwargs forwarded to httpx
        merged.pop("entity_id", None)

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, **merged)
                response.raise_for_status()
                return response.json() if self.as_json else response.content
        except Exception as exc:
            raise SourceError(
                source_name=type(self).__name__,
                entity_id=str(entity_id or url),
                cause=exc,
            ) from exc
