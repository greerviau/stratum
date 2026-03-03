# Schema system

calcine's schema system validates feature output before it reaches the
store, catching type mismatches early without requiring a full type-checking
framework.

## Overview

```python
from calcine.schema import FeatureSchema, types

schema = FeatureSchema({
    "score":     types.Float64(nullable=False, default=0.0),
    "label":     types.Category(categories=["low", "med", "high"]),
    "embedding": types.NDArray(shape=(None, 64), dtype="float32"),
})

errors = schema.validate(result)   # returns list[str], never raises
```

`FeatureSchema.validate` returns a list of error strings.  An empty list means
the value is valid.  The pipeline treats any non-empty list as a failure for
that entity, adding it to `report.failed`.

---

## Built-in types

All types accept `nullable: bool = True` and `default: Any = None`.

| Type | Validates |
|------|-----------|
| `types.Float32` | Float-compatible value (`float`, `int`, numpy scalars) |
| `types.Float64` | Float-compatible value |
| `types.Int32` | `int` or numpy integer (booleans rejected) |
| `types.Int64` | `int` or numpy integer (booleans rejected) |
| `types.String` | `str` |
| `types.Boolean` | `bool` or `numpy.bool_` (plain ints rejected) |
| `types.Category(categories=[...])` | Value present in the categories list |
| `types.NDArray(shape=(...), dtype="float32")` | `numpy.ndarray` with matching shape and dtype |
| `types.Bytes` | `bytes` or `bytearray` |
| `types.Any` | Anything — no validation |

### Nullable

`nullable=True` (default) means `None` is a valid value.
`nullable=False` means `None` produces an error.

### Boolean vs. int

`Int32` and `Int64` explicitly reject `bool` values even though `bool` is a
subclass of `int` in Python.  If you intend to accept booleans use
`types.Boolean`.

### NDArray shape wildcards

Use `None` in a shape dimension to accept any size in that axis:

```python
types.NDArray(shape=(None, 128), dtype="float32")
# accepts arrays of shape (1, 128), (32, 128), (1000, 128), etc.
```

---

## Single-field schema for non-dict values

When a feature returns a raw value (not a dict), use a single-field schema.
The field name is arbitrary — the type validator is applied directly to the
value:

```python
# Feature.extract returns a raw ndarray, not a dict
schema = FeatureSchema({"_vec": types.NDArray(shape=(128,), dtype="float32")})
errors = schema.validate(arr)    # validates arr directly
```

A multi-field schema on a non-dict value always returns an error:

```python
schema = FeatureSchema({"a": types.Float64(), "b": types.String()})
errors = schema.validate(42)    # → ["Expected dict result for multi-field schema, got int"]
```

---

## Known limitations

### NaN passes Float64

IEEE 754 `NaN` is a valid float value, so it passes `Float64` validation:

```python
types.Float64(nullable=False).validate(float("nan"))  # → []
```

If your feature can produce NaN (e.g. `DataFrame.mean()` on an empty group),
guard in `post_extract`:

```python
async def post_extract(self, result: dict) -> dict:
    import math
    for k, v in result.items():
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError(f"Non-finite value for field '{k}': {v!r}")
    return result
```

### No cross-field validation

`FeatureSchema` validates each field independently.  If you need cross-field
constraints (e.g. `end_time > start_time`), do it in `post_extract` or
`validate`.

---

## Adding a custom type

Subclass `FeatureType` and implement `_validate_value`:

```python
from calcine.schema import FeatureType

class PositiveFloat(FeatureType):
    def _validate_value(self, value) -> list[str]:
        try:
            f = float(value)
        except (TypeError, ValueError):
            return [f"Expected numeric value, got {type(value).__name__}"]
        return [] if f > 0 else [f"Expected positive float, got {f}"]
```
