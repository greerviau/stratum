"""Abstract base class for feature extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from ..schema import FeatureSchema


class Feature(ABC):
    """Contract for all feature extractors in the stratum pipeline.

    A ``Feature`` transforms raw data (from a ``DataSource``) into a
    structured, typed feature value.  The full extraction lifecycle is::

        raw → pre_extract(raw) → extract(preprocessed, context)
            → post_extract(result) → validate(result)

    Subclasses must implement ``extract``.  All other methods have
    sensible pass-through defaults.

    The ``schema`` class attribute, if set to a ``FeatureSchema``, enables
    automatic validation of the extracted result.

    Example::

        class EmbeddingFeature(Feature):
            schema = FeatureSchema({"vec": types.NDArray(shape=(128,), dtype="float32")})

            async def extract(self, raw: str, context: dict) -> dict:
                return {"vec": encode(raw)}
    """

    schema: ClassVar[FeatureSchema | None] = None

    @abstractmethod
    async def extract(
        self, raw: Any, context: dict, entity_id: str | None = None
    ) -> Any:
        """Extract the feature value from raw source data.

        Args:
            raw: Raw data returned by the ``DataSource``.  The type depends
                on the source implementation.
            context: Arbitrary dict supplied by the caller (e.g. timestamps,
                model version, experiment flags).
            entity_id: The identifier of the entity being processed.  Useful
                for logging, per-entity branching, or keying external lookups.
                ``None`` when called outside a ``Pipeline`` context.

        Returns:
            Extracted feature value.  Should conform to ``schema`` if set.
        """
        ...

    async def pre_extract(self, raw: Any) -> Any:
        """Pre-processing hook executed before ``extract``.

        Override to normalise or filter raw data before the main extraction
        logic.  The default is a pass-through.

        Args:
            raw: Raw data from the source.

        Returns:
            Pre-processed data passed to ``extract``.
        """
        return raw

    async def post_extract(self, result: Any) -> Any:
        """Post-processing hook executed after ``extract``.

        Override to round, clip, cast, or otherwise transform the raw
        extraction output.  The default is a pass-through.

        Args:
            result: Value returned by ``extract``.

        Returns:
            Post-processed result written to the store and validated.
        """
        return result

    async def extract_batch(
        self,
        raws: list[Any],
        context: dict[str, Any],
        entity_ids: list[str] | None = None,
    ) -> list[Any | BaseException]:
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
            raws: Pre-processed raw data for each entity in the batch
                (already passed through ``pre_extract``).
            context: Shared context dict forwarded from ``generate()``.
            entity_ids: Entity identifiers corresponding to each item in
                *raws*, in the same order.  ``None`` when called outside
                a ``Pipeline`` context.

        Returns:
            List of results or ``BaseException`` instances, one per input.
        """
        results: list[Any | BaseException] = []
        for i, raw in enumerate(raws):
            eid = entity_ids[i] if entity_ids is not None else None
            try:
                results.append(await self.extract(raw, context, entity_id=eid))
            except Exception as exc:  # noqa: BLE001
                results.append(exc)
        return results

    async def validate(self, result: Any) -> list[str]:
        """Validate the (post-processed) extraction result.

        Uses ``schema.validate`` if ``schema`` is set; otherwise returns
        an empty list (no validation).

        Args:
            result: Post-processed feature value.

        Returns:
            List of validation error strings.  Empty means valid.
        """
        if self.schema is not None:
            return self.schema.validate(result)
        return []
