"""Feature schema system for validating extracted feature values.

Supports both tabular (dict) and non-tabular (numpy arrays, bytes, etc.) data.

Usage::

    from calcine.schema import FeatureSchema, types

    schema = FeatureSchema({
        "embedding": types.NDArray(shape=(None, 128), dtype="float32"),
        "label": types.String(nullable=False),
        "score": types.Float64(nullable=False, default=0.0),
    })

    errors = schema.validate({"embedding": arr, "label": "cat", "score": 0.9})
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Base type
# ---------------------------------------------------------------------------


class FeatureType:
    """Abstract base class for all feature schema types.

    Subclasses implement ``_validate_value`` to provide type-specific checks.
    Nullable and default handling is managed here in ``validate``.
    """

    def __init__(self, nullable: bool = True, default: Any = None) -> None:
        self.nullable = nullable
        self.default = default

    def validate(self, value: Any) -> list[str]:
        """Return a list of error strings for *value*.  Empty means valid."""
        if value is None:
            if not self.nullable:
                return ["Value is None but field is not nullable"]
            return []
        return self._validate_value(value)

    def _validate_value(self, value: Any) -> list[str]:
        """Type-specific validation.  Override in subclasses."""
        return []

    def __repr__(self) -> str:  # pragma: no cover
        return f"{type(self).__name__}(nullable={self.nullable}, default={self.default!r})"


# ---------------------------------------------------------------------------
# Concrete types
# ---------------------------------------------------------------------------


class Float32(FeatureType):
    """32-bit floating-point scalar."""

    def _validate_value(self, value: Any) -> list[str]:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return [f"Expected float-compatible value, got {type(value).__name__}"]
        if math.isnan(v):
            return ["Value is NaN"]
        return []


class Float64(FeatureType):
    """64-bit floating-point scalar."""

    def _validate_value(self, value: Any) -> list[str]:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return [f"Expected float-compatible value, got {type(value).__name__}"]
        if math.isnan(v):
            return ["Value is NaN"]
        return []


class Int32(FeatureType):
    """32-bit integer scalar."""

    def _validate_value(self, value: Any) -> list[str]:
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            return [f"Expected int, got {type(value).__name__}"]
        return []


class Int64(FeatureType):
    """64-bit integer scalar."""

    def _validate_value(self, value: Any) -> list[str]:
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            return [f"Expected int, got {type(value).__name__}"]
        return []


class String(FeatureType):
    """Unicode string."""

    def _validate_value(self, value: Any) -> list[str]:
        if not isinstance(value, str):
            return [f"Expected str, got {type(value).__name__}"]
        return []


class Boolean(FeatureType):
    """Boolean value."""

    def _validate_value(self, value: Any) -> list[str]:
        if not isinstance(value, (bool, np.bool_)):
            return [f"Expected bool, got {type(value).__name__}"]
        return []


class Category(FeatureType):
    """Categorical value restricted to a fixed set of allowed values."""

    def __init__(
        self,
        categories: list[Any],
        nullable: bool = True,
        default: Any = None,
    ) -> None:
        super().__init__(nullable=nullable, default=default)
        self.categories = categories

    def _validate_value(self, value: Any) -> list[str]:
        if value not in self.categories:
            return [f"Value {value!r} not in allowed categories {self.categories}"]
        return []


class NDArray(FeatureType):
    """NumPy ndarray with an expected shape and dtype.

    Use ``None`` for a dimension to allow any size in that axis::

        NDArray(shape=(None, 128), dtype="float32")  # any batch size, 128-d
    """

    def __init__(
        self,
        shape: tuple[int | None, ...],
        dtype: str,
        nullable: bool = True,
        default: Any = None,
    ) -> None:
        super().__init__(nullable=nullable, default=default)
        self.shape = shape
        self.dtype = dtype

    def _validate_value(self, value: Any) -> list[str]:
        if not isinstance(value, np.ndarray):
            return [f"Expected numpy.ndarray, got {type(value).__name__}"]

        errors: list[str] = []

        if value.ndim != len(self.shape):
            errors.append(f"Expected ndim={len(self.shape)}, got ndim={value.ndim}")
        else:
            for i, (expected, actual) in enumerate(zip(self.shape, value.shape, strict=True)):
                if expected is not None and expected != actual:
                    errors.append(f"Dimension {i}: expected {expected}, got {actual}")

        if str(value.dtype) != self.dtype:
            errors.append(f"Expected dtype='{self.dtype}', got dtype='{value.dtype}'")

        return errors


class Bytes(FeatureType):
    """Raw binary data."""

    def _validate_value(self, value: Any) -> list[str]:
        if not isinstance(value, (bytes, bytearray)):
            return [f"Expected bytes, got {type(value).__name__}"]
        return []


class AnyType(FeatureType):
    """No validation — any value passes.  Useful as a passthrough type."""

    def _validate_value(self, value: Any) -> list[str]:
        return []


class List(FeatureType):
    """Typed list where every element must satisfy *item_type*.

    Supports nesting::

        List(item_type=String())
        List(item_type=Dict(key_type=String(), value_type=Float64()))
    """

    def __init__(
        self,
        item_type: FeatureType,
        nullable: bool = True,
        default: Any = None,
    ) -> None:
        super().__init__(nullable=nullable, default=default)
        self.item_type = item_type

    def _validate_value(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return [f"Expected list, got {type(value).__name__}"]
        errors: list[str] = []
        for i, item in enumerate(value):
            for err in self.item_type.validate(item):
                errors.append(f"Index {i}: {err}")
        return errors


class Dict(FeatureType):
    """Typed dict where every key must satisfy *key_type* and every value *value_type*.

    Supports nesting::

        Dict(key_type=String(), value_type=Int64())
        Dict(key_type=String(), value_type=List(item_type=Float64()))
    """

    def __init__(
        self,
        key_type: FeatureType,
        value_type: FeatureType,
        nullable: bool = True,
        default: Any = None,
    ) -> None:
        super().__init__(nullable=nullable, default=default)
        self.key_type = key_type
        self.value_type = value_type

    def _validate_value(self, value: Any) -> list[str]:
        if not isinstance(value, dict):
            return [f"Expected dict, got {type(value).__name__}"]
        errors: list[str] = []
        for k, v in value.items():
            for err in self.key_type.validate(k):
                errors.append(f"Key {k!r} (key): {err}")
            for err in self.value_type.validate(v):
                errors.append(f"Key {k!r} (value): {err}")
        return errors


# ---------------------------------------------------------------------------
# types namespace — mirrors the classes above for ergonomic imports
# ---------------------------------------------------------------------------


class types:
    """Namespace for all built-in feature types.

    Example::

        from calcine.schema import types

        types.Float64(nullable=False, default=0.0)
        types.NDArray(shape=(None, 128), dtype="float32")
        types.Category(categories=["cat", "dog"])
    """

    Float32 = Float32
    Float64 = Float64
    Int32 = Int32
    Int64 = Int64
    String = String
    Boolean = Boolean
    Category = Category
    NDArray = NDArray
    Bytes = Bytes
    Any = AnyType
    List = List
    Dict = Dict


# ---------------------------------------------------------------------------
# FeatureSchema
# ---------------------------------------------------------------------------


class FeatureSchema:
    """Schema for validating feature outputs.

    Pass a mapping of ``{field_name: FeatureType}`` for dict results::

        FeatureSchema({
            "mean_value": types.Float64(nullable=False),
            "count": types.Int64(nullable=False),
        })

    For non-dict results (e.g. raw numpy arrays), use a single-field schema::

        FeatureSchema({"_value": types.NDArray(shape=(128,), dtype="float32")})

    Args:
        fields: Mapping of field names to ``FeatureType`` instances.
    """

    def __init__(self, fields: dict[str, FeatureType]) -> None:
        self.fields = fields

    def validate(self, data: Any) -> list[str]:
        """Validate *data* against the schema.

        For dict results each field is validated individually.
        For non-dict results the schema must have exactly one field, and
        that type is used to validate the whole value.

        Returns:
            List of error strings.  Empty means valid.
        """
        errors: list[str] = []

        if isinstance(data, dict):
            for field_name, field_type in self.fields.items():
                if field_name not in data:
                    # Treat a missing field as None for nullable checking
                    field_errors = field_type.validate(None)
                    for err in field_errors:
                        errors.append(f"Field '{field_name}': {err}")
                    continue
                for err in field_type.validate(data[field_name]):
                    errors.append(f"Field '{field_name}': {err}")
        elif len(self.fields) == 1:
            # Single-field schema for non-dict results
            field_name, field_type = next(iter(self.fields.items()))
            for err in field_type.validate(data):
                errors.append(f"Field '{field_name}': {err}")
        else:
            errors.append(f"Expected dict result for multi-field schema, got {type(data).__name__}")

        return errors
