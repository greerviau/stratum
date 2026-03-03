"""calcine — A source-agnostic, type-agnostic featurization pipeline framework.

Core abstraction::

    DataSource → Feature → FeatureStore

tied together by::

    Pipeline.generate(entity_ids)  →  GenerationReport
    Pipeline.retrieve(entity_id)   →  Any

Quick start::

    from calcine import Pipeline
    from calcine.sources import DataFrameSource
    from calcine.features.base import Feature
    from calcine.stores import MemoryStore
    from calcine.schema import FeatureSchema, types
    import asyncio

    class MyFeature(Feature):
        schema = FeatureSchema({"score": types.Float64(nullable=False)})

        async def extract(self, raw, context):
            return {"score": raw["value"].mean()}

    pipeline = Pipeline(
        source=DataFrameSource(df),
        feature=MyFeature(),
        store=MemoryStore(),
    )

    report = await pipeline.generate(["e1", "e2"])
    value  = await pipeline.retrieve("e1")
"""

from .exceptions import SchemaViolationError, SourceError, StoreError, CalcineError
from .features.base import Feature
from .pipeline import GenerationReport, Pipeline
from .schema import FeatureSchema, types
from .sources.base import DataSource
from .sources.bundle import SourceBundle
from .stores.base import FeatureStore

__version__ = "0.1.0"

__all__ = [
    # Pipeline
    "Pipeline",
    "GenerationReport",
    # ABCs
    "Feature",
    "DataSource",
    "SourceBundle",
    "FeatureStore",
    # Schema
    "FeatureSchema",
    "types",
    # Exceptions
    "CalcineError",
    "SchemaViolationError",
    "SourceError",
    "StoreError",
]
