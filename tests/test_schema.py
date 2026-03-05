"""Tests for FeatureSchema and all built-in types."""

from __future__ import annotations

import numpy as np

from calcine.schema import (
    FeatureSchema,
    types,
)

# ---------------------------------------------------------------------------
# Float types
# ---------------------------------------------------------------------------


class TestFloat64:
    def test_valid_float(self):
        assert types.Float64().validate(3.14) == []

    def test_valid_int_as_float(self):
        # ints are float-compatible
        assert types.Float64().validate(1) == []

    def test_invalid_string(self):
        errors = types.Float64().validate("not a number")
        assert len(errors) > 0

    def test_none_nullable(self):
        assert types.Float64(nullable=True).validate(None) == []

    def test_none_not_nullable(self):
        errors = types.Float64(nullable=False).validate(None)
        assert len(errors) > 0

    def test_default_is_stored(self):
        t = types.Float64(nullable=False, default=0.0)
        assert t.default == 0.0

    def test_nan_is_invalid(self):
        errors = types.Float64().validate(float("nan"))
        assert len(errors) > 0

    def test_nan_is_invalid_not_nullable(self):
        errors = types.Float64(nullable=False).validate(float("nan"))
        assert len(errors) > 0


class TestFloat32:
    def test_valid(self):
        assert types.Float32().validate(0.5) == []

    def test_invalid(self):
        assert len(types.Float32().validate({})) > 0

    def test_nan_is_invalid(self):
        errors = types.Float32().validate(float("nan"))
        assert len(errors) > 0


# ---------------------------------------------------------------------------
# Integer types
# ---------------------------------------------------------------------------


class TestInt64:
    def test_valid_int(self):
        assert types.Int64().validate(42) == []

    def test_valid_numpy_int(self):
        assert types.Int64().validate(np.int64(7)) == []

    def test_invalid_float(self):
        # Floats are NOT valid ints
        assert len(types.Int64().validate(3.14)) > 0

    def test_invalid_bool(self):
        # bool is a subclass of int in Python, but we reject it
        assert len(types.Int64().validate(True)) > 0


class TestInt32:
    def test_valid(self):
        assert types.Int32().validate(0) == []

    def test_numpy_int32(self):
        assert types.Int32().validate(np.int32(5)) == []


# ---------------------------------------------------------------------------
# String
# ---------------------------------------------------------------------------


class TestString:
    def test_valid(self):
        assert types.String().validate("hello") == []

    def test_invalid_int(self):
        assert len(types.String().validate(123)) > 0

    def test_invalid_bytes(self):
        assert len(types.String().validate(b"bytes")) > 0


# ---------------------------------------------------------------------------
# Boolean
# ---------------------------------------------------------------------------


class TestBoolean:
    def test_valid_true(self):
        assert types.Boolean().validate(True) == []

    def test_valid_false(self):
        assert types.Boolean().validate(False) == []

    def test_numpy_bool(self):
        assert types.Boolean().validate(np.bool_(True)) == []

    def test_invalid_int(self):
        # int is not bool even though bool subclasses int
        assert len(types.Boolean().validate(1)) > 0


# ---------------------------------------------------------------------------
# Category
# ---------------------------------------------------------------------------


class TestCategory:
    def test_valid(self):
        t = types.Category(categories=["cat", "dog", "bird"])
        assert t.validate("cat") == []

    def test_invalid_not_in_list(self):
        t = types.Category(categories=["cat", "dog"])
        errors = t.validate("fish")
        assert len(errors) > 0

    def test_none_nullable(self):
        t = types.Category(categories=["a", "b"], nullable=True)
        assert t.validate(None) == []


# ---------------------------------------------------------------------------
# NDArray
# ---------------------------------------------------------------------------


class TestNDArray:
    def test_valid_shape_and_dtype(self):
        t = types.NDArray(shape=(3,), dtype="float32")
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert t.validate(arr) == []

    def test_wrong_dtype(self):
        t = types.NDArray(shape=(3,), dtype="float32")
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        errors = t.validate(arr)
        assert any("dtype" in e for e in errors)

    def test_wrong_shape_size(self):
        t = types.NDArray(shape=(3, 4), dtype="float32")
        arr = np.zeros((3, 5), dtype=np.float32)
        errors = t.validate(arr)
        assert len(errors) > 0

    def test_wrong_ndim(self):
        t = types.NDArray(shape=(3,), dtype="float32")
        arr = np.zeros((3, 4), dtype=np.float32)
        errors = t.validate(arr)
        assert any("ndim" in e for e in errors)

    def test_not_ndarray(self):
        t = types.NDArray(shape=(3,), dtype="float32")
        errors = t.validate([1.0, 2.0, 3.0])
        assert len(errors) > 0

    def test_variable_dim_with_none(self):
        """None in shape means any size in that dimension."""
        t = types.NDArray(shape=(None, 128), dtype="float32")
        arr = np.zeros((10, 128), dtype=np.float32)
        assert t.validate(arr) == []

    def test_variable_dim_wrong_fixed_axis(self):
        t = types.NDArray(shape=(None, 128), dtype="float32")
        arr = np.zeros((10, 64), dtype=np.float32)
        errors = t.validate(arr)
        assert len(errors) > 0


# ---------------------------------------------------------------------------
# Bytes
# ---------------------------------------------------------------------------


class TestBytes:
    def test_valid(self):
        assert types.Bytes().validate(b"data") == []

    def test_bytearray(self):
        assert types.Bytes().validate(bytearray(b"data")) == []

    def test_invalid_string(self):
        assert len(types.Bytes().validate("string")) > 0


# ---------------------------------------------------------------------------
# AnyType
# ---------------------------------------------------------------------------


class TestAnyType:
    def test_passes_everything(self):
        t = types.Any()
        assert t.validate(42) == []
        assert t.validate("hello") == []
        assert t.validate([1, 2, 3]) == []
        assert t.validate({"a": 1}) == []

    def test_none_passes_when_nullable(self):
        t = types.Any(nullable=True)
        assert t.validate(None) == []

    def test_none_fails_when_not_nullable(self):
        t = types.Any(nullable=False)
        errors = t.validate(None)
        assert len(errors) > 0


# ---------------------------------------------------------------------------
# FeatureSchema
# ---------------------------------------------------------------------------


class TestFeatureSchema:
    def test_valid_dict(self):
        schema = FeatureSchema({"value": types.Float64(), "label": types.String()})
        assert schema.validate({"value": 1.0, "label": "pos"}) == []

    def test_missing_required_field(self):
        schema = FeatureSchema({"value": types.Float64(nullable=False)})
        errors = schema.validate({})
        assert len(errors) > 0
        assert any("value" in e for e in errors)

    def test_missing_nullable_field_ok(self):
        schema = FeatureSchema({"value": types.Float64(nullable=True)})
        # Missing field treated as None — allowed if nullable
        assert schema.validate({}) == []

    def test_field_type_error(self):
        schema = FeatureSchema({"score": types.Float64()})
        errors = schema.validate({"score": "bad"})
        assert len(errors) > 0

    def test_single_field_non_dict(self):
        """Single-field schema validates non-dict values directly."""
        schema = FeatureSchema({"_vec": types.NDArray(shape=(4,), dtype="float32")})
        arr = np.zeros(4, dtype=np.float32)
        assert schema.validate(arr) == []

    def test_multi_field_non_dict_error(self):
        """Multi-field schema on a non-dict is an error."""
        schema = FeatureSchema({"a": types.Float64(), "b": types.String()})
        errors = schema.validate(99.9)
        assert len(errors) > 0

    def test_extra_fields_ignored(self):
        """Fields present in data but not in schema are silently ignored."""
        schema = FeatureSchema({"score": types.Float64()})
        assert schema.validate({"score": 1.0, "extra": "ignored"}) == []


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------


class TestList:
    def test_valid_list_of_strings(self):
        t = types.List(item_type=types.String())
        assert t.validate(["a", "b", "c"]) == []

    def test_valid_empty_list(self):
        t = types.List(item_type=types.Int64())
        assert t.validate([]) == []

    def test_invalid_item(self):
        t = types.List(item_type=types.Int64())
        errors = t.validate([1, "oops", 3])
        assert len(errors) > 0
        assert any("Index 1" in e for e in errors)

    def test_not_a_list(self):
        t = types.List(item_type=types.String())
        errors = t.validate("not a list")
        assert len(errors) > 0

    def test_none_nullable(self):
        t = types.List(item_type=types.String(), nullable=True)
        assert t.validate(None) == []

    def test_none_not_nullable(self):
        t = types.List(item_type=types.String(), nullable=False)
        errors = t.validate(None)
        assert len(errors) > 0

    def test_nested_list_of_lists(self):
        t = types.List(item_type=types.List(item_type=types.Float64()))
        assert t.validate([[1.0, 2.0], [3.0]]) == []

    def test_nested_list_of_lists_invalid(self):
        t = types.List(item_type=types.List(item_type=types.Float64()))
        errors = t.validate([[1.0], ["bad"]])
        assert len(errors) > 0

    def test_all_items_validated(self):
        t = types.List(item_type=types.Int64())
        errors = t.validate(["x", "y", "z"])
        assert len(errors) == 3


# ---------------------------------------------------------------------------
# Dict
# ---------------------------------------------------------------------------


class TestDict:
    def test_valid_str_to_int(self):
        t = types.Dict(key_type=types.String(), value_type=types.Int64())
        assert t.validate({"a": 1, "b": 2}) == []

    def test_valid_empty_dict(self):
        t = types.Dict(key_type=types.String(), value_type=types.Float64())
        assert t.validate({}) == []

    def test_invalid_value(self):
        t = types.Dict(key_type=types.String(), value_type=types.Int64())
        errors = t.validate({"a": "not_int"})
        assert len(errors) > 0
        assert any("'a'" in e and "value" in e for e in errors)

    def test_not_a_dict(self):
        t = types.Dict(key_type=types.String(), value_type=types.Int64())
        errors = t.validate([1, 2])
        assert len(errors) > 0

    def test_none_nullable(self):
        t = types.Dict(key_type=types.String(), value_type=types.Int64(), nullable=True)
        assert t.validate(None) == []

    def test_none_not_nullable(self):
        t = types.Dict(key_type=types.String(), value_type=types.Int64(), nullable=False)
        assert len(t.validate(None)) > 0

    def test_nested_dict_of_lists(self):
        t = types.Dict(key_type=types.String(), value_type=types.List(item_type=types.Float64()))
        assert t.validate({"scores": [1.0, 2.5], "weights": [0.3]}) == []

    def test_nested_dict_of_lists_invalid(self):
        t = types.Dict(key_type=types.String(), value_type=types.List(item_type=types.Float64()))
        errors = t.validate({"scores": ["bad"]})
        assert len(errors) > 0

    def test_nested_list_of_dicts(self):
        t = types.List(item_type=types.Dict(key_type=types.String(), value_type=types.Float64()))
        assert t.validate([{"x": 1.0}, {"y": 2.0}]) == []

    def test_multiple_bad_values(self):
        t = types.Dict(key_type=types.String(), value_type=types.Int64())
        errors = t.validate({"a": "bad", "b": "also_bad"})
        assert len(errors) == 2
