"""Specific exception types for calcine.

All exceptions include contextual information (feature name, entity ID, cause)
to make debugging easier.
"""

from __future__ import annotations


class CalcineError(Exception):
    """Base class for all calcine errors."""


class SchemaViolationError(CalcineError):
    """Raised when a feature result fails schema validation.

    Attributes:
        feature_name: The name of the Feature class that produced the result.
        entity_id: The entity whose feature value violated the schema.
        errors: List of validation error strings from FeatureSchema.validate().
    """

    def __init__(self, feature_name: str, entity_id: str, errors: list[str]) -> None:
        self.feature_name = feature_name
        self.entity_id = entity_id
        self.errors = errors
        joined = "; ".join(errors)
        super().__init__(
            f"Schema violation for feature '{feature_name}', entity '{entity_id}': {joined}"
        )


class SourceError(CalcineError):
    """Raised when a DataSource fails to read data.

    Attributes:
        source_name: The name of the DataSource class.
        entity_id: The entity being read when the error occurred.
        cause: The underlying exception.
    """

    def __init__(self, source_name: str, entity_id: str, cause: Exception) -> None:
        self.source_name = source_name
        self.entity_id = entity_id
        self.cause = cause
        super().__init__(
            f"Source '{source_name}' failed for entity '{entity_id}': "
            f"{type(cause).__name__}: {cause}"
        )


class StoreError(CalcineError):
    """Raised when a FeatureStore fails to read or write data.

    Attributes:
        store_name: The name of the FeatureStore class.
        feature_name: The name of the Feature class.
        entity_id: The entity being read/written when the error occurred.
        cause: The underlying exception.
    """

    def __init__(
        self,
        store_name: str,
        feature_name: str,
        entity_id: str,
        cause: Exception,
    ) -> None:
        self.store_name = store_name
        self.feature_name = feature_name
        self.entity_id = entity_id
        self.cause = cause
        super().__init__(
            f"Store '{store_name}' failed for feature '{feature_name}', "
            f"entity '{entity_id}': {type(cause).__name__}: {cause}"
        )
