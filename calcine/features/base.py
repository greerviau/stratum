"""Abstract base class for feature extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from ..extraction import ExtractionResult
from ..schema import FeatureSchema


class Feature(ABC):
    """Contract for all feature extractors in the calcine pipeline.

    A ``Feature`` transforms raw data (from a ``DataSource``) into an
    :class:`~calcine.ExtractionResult` containing one or more stored records.

    Subclasses must implement :meth:`extract`.  :meth:`validate` runs
    automatically after extraction if ``schema`` is set.

    ``schema`` validates each record in the result.
    ``metadata_schema`` validates ``ExtractionResult.metadata``.

    Single-record example::

        class PurchaseValueFeature(Feature):
            schema = FeatureSchema({"mean": types.Float64(nullable=False)})

            async def extract(self, raw, context, entity_id=None):
                return ExtractionResult.of(entity_id, {"mean": raw["amount"].mean()})

    Fan-out example::

        class SegmentFeature(Feature):
            metadata_schema = FeatureSchema({"duration_s": types.Float64(nullable=False)})
            schema = FeatureSchema({"rms": types.Float64(nullable=False)})

            async def extract(self, raw, context, entity_id=None):
                segments = split(raw)
                return ExtractionResult(
                    metadata={"duration_s": len(raw) / SR},
                    records={f"{entity_id}/{i}": {"rms": rms(s)} for i, s in enumerate(segments)},
                )
    """

    schema: ClassVar[FeatureSchema | None] = None
    metadata_schema: ClassVar[FeatureSchema | None] = None

    @abstractmethod
    async def extract(
        self, raw: Any, context: dict, entity_id: str | None = None
    ) -> ExtractionResult:
        """Extract the feature value from raw source data.

        Args:
            raw: Raw data returned by the ``DataSource``.  The type depends
                on the source implementation.
            context: Arbitrary dict supplied by the caller (e.g. timestamps,
                model version, experiment flags).
            entity_id: The identifier of the entity being processed.  Use as
                the key in :meth:`ExtractionResult.of` for single-record
                features, or as the prefix for fan-out sub-entity IDs.
                ``None`` when called outside a ``Pipeline`` context.

        Returns:
            :class:`~calcine.ExtractionResult` with one record (single-entity)
            or many records (fan-out).
        """
        ...

    async def extract_batch(
        self,
        raws: list[Any],
        context: dict[str, Any],
        entity_ids: list[str] | None = None,
        entity_contexts: list[dict[str, Any]] | None = None,
    ) -> list[ExtractionResult | BaseException]:
        """Extract features for a batch of entities in a single call.

        Override this to enable vectorised or batch-API computation
        (e.g. ML model inference, bulk database queries, batch embedding
        APIs).  The default implementation calls ``extract()`` for each
        item individually and is therefore equivalent to — but no faster
        than — the per-entity path.

        Return one element per input, in the same order.  Individual items
        may be ``BaseException`` instances to signal per-entity failure
        without aborting the rest of the batch; the pipeline will record
        those entities as failed and continue.

        Args:
            raws: Raw data for each entity in the batch.
            context: Shared context dict forwarded from ``generate()``.
            entity_ids: Entity identifiers corresponding to each item in
                *raws*, in the same order.  ``None`` when called outside
                a ``Pipeline`` context.
            entity_contexts: Per-entity context dicts (already merged with
                the shared *context*), one per item in *raws*.  Present when
                ``generate()`` is called with ``context_fn``.  ``None``
                otherwise — fall back to *context* for all entities.

        Returns:
            List of :class:`~calcine.ExtractionResult` or ``BaseException``
            instances, one per input.
        """
        results: list[ExtractionResult | BaseException] = []
        for i, raw in enumerate(raws):
            eid = entity_ids[i] if entity_ids is not None else None
            ctx = entity_contexts[i] if entity_contexts is not None else context
            try:
                results.append(await self.extract(raw, ctx, entity_id=eid))
            except Exception as exc:  # noqa: BLE001
                results.append(exc)
        return results

    async def validate(self, result: Any) -> list[str]:
        """Validate a single extracted record value.

        Uses ``schema.validate`` if ``schema`` is set; otherwise returns
        an empty list (no validation).  Called once per record in the
        ``ExtractionResult``.

        Args:
            result: Extracted feature value (one record, not the full
                ``ExtractionResult``).

        Returns:
            List of validation error strings.  Empty means valid.
        """
        if self.schema is not None:
            return self.schema.validate(result)
        return []
