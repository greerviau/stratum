"""Pandas DataFrame data source."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..exceptions import SourceError
from .base import DataSource

if TYPE_CHECKING:
    import pandas as pd


class DataFrameSource(DataSource):
    """Filter a pandas DataFrame by ``entity_id`` and return matching rows.

    The DataFrame is expected to contain an entity column (default
    ``"entity_id"``).  Each ``read`` call filters the DataFrame to rows
    where the entity column equals the supplied ``entity_id``.

    Args:
        df: The source DataFrame.  It is not copied on construction — pass
            a copy if you need isolation from downstream mutations.
        entity_col: Name of the column that contains entity identifiers.

    Example::

        source = DataFrameSource(df, entity_col="user_id")
        rows = await source.read(entity_id="u123")
    """

    def __init__(self, df: pd.DataFrame, entity_col: str = "entity_id") -> None:
        self.df = df
        self.entity_col = entity_col

    async def read(self, entity_id: str | None = None, **kwargs: Any) -> pd.DataFrame:
        """Return a copy of rows whose entity column matches *entity_id*.

        Args:
            entity_id: The entity to filter by.  Required.
            **kwargs: Additional ignored keyword arguments.

        Returns:
            Filtered DataFrame (may be empty if entity has no rows).

        Raises:
            SourceError: If ``entity_id`` is not provided or ``entity_col``
                is missing from the DataFrame.
        """
        try:
            if entity_id is None:
                raise ValueError(f"entity_id is required for {type(self).__name__}")
            if self.entity_col not in self.df.columns:
                raise ValueError(
                    f"Column '{self.entity_col}' not found in DataFrame. "
                    f"Available columns: {list(self.df.columns)}"
                )
            mask = self.df[self.entity_col] == entity_id
            return self.df[mask].copy()
        except Exception as exc:
            raise SourceError(
                source_name=type(self).__name__,
                entity_id=str(entity_id),
                cause=exc,
            ) from exc
