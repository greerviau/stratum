"""ExtractionResult — the universal return type for Feature.extract()."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ExtractionResult:
    """The result of a feature extraction: one or many stored records.

    :meth:`~calcine.Feature.extract` always returns an ``ExtractionResult``.
    For a single-record feature use :meth:`of`; for fan-out (one source entity
    → many sub-entity records) populate *records* directly.

    Attributes:
        records: Mapping from entity ID to extracted value.  Each value is
            validated against ``Feature.schema`` if set.  For single-record
            features, the key is the source ``entity_id``; for fan-out features,
            keys are sub-entity IDs (e.g. ``"recording_001/0"``).
        metadata: Optional parent-level data stored under the source
            ``entity_id``.  Validated against ``Feature.metadata_schema`` if set.
            Only relevant for fan-out features; ``None`` for single-record
            features.

    Examples::

        # Single record
        return ExtractionResult.of(entity_id, {"score": 0.92})

        # Fan-out: one source → many sub-entity records
        return ExtractionResult(
            metadata={"duration_s": total},
            records={f"{entity_id}/{i}": {"rms": r} for i, r in enumerate(segs)},
        )
    """

    records: dict[str, Any]
    metadata: dict[str, Any] | None = None

    @classmethod
    def of(cls, entity_id: str | None, value: Any) -> ExtractionResult:
        """Create a single-record ``ExtractionResult``.

        Shorthand for ``ExtractionResult(records={entity_id: value})``.

        Args:
            entity_id: The entity identifier (the write key).  May be ``None``
                when called outside a ``Pipeline`` context.
            value: The extracted feature value.
        """
        return cls(records={entity_id: value})
