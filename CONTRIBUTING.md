# Contributing to calcine

Thank you for your interest in contributing. This guide covers everything you
need to get from zero to a merged pull request.

## Table of contents

- [Development setup](#development-setup)
- [Running tests](#running-tests)
- [Code style](#code-style)
- [Project structure](#project-structure)
- [Adding a new DataSource](#adding-a-new-datasource)
- [Adding a new FeatureStore](#adding-a-new-featurestore)
- [Adding a new schema type](#adding-a-new-schema-type)
- [Pull request process](#pull-request-process)

---

## Development setup

calcine uses [uv](https://docs.astral.sh/uv/) for environment management.

```bash
# Clone and enter the repo
git clone https://github.com/greerviau/calcine.git
cd calcine

# Create a virtual environment and install all dev dependencies
uv venv
uv pip install -e ".[dev]"
```

Activate the environment when working in the shell:

```bash
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows
```

### Generating example data

The large example datasets are excluded from git (see `.gitignore`).
Generate them once before running the examples:

```bash
python examples/data/generate_data.py
```

---

## Running tests

```bash
# All tests
pytest

# A single module
pytest tests/test_pipeline.py -v

# With coverage
pytest --cov=calcine --cov-report=term-missing
```

Tests must pass on **Python 3.10, 3.11, and 3.12**.  The CI matrix runs all
three on every push and pull request.

---

## Code style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
# Check for issues
ruff check calcine/ tests/

# Auto-fix where possible
ruff check --fix calcine/ tests/

# Format
ruff format calcine/ tests/
```

Key conventions:

- Type hints on all public function signatures
- Docstrings on all public classes and methods (Google style)
- `async` throughout — never use sync I/O in `DataSource`, `Feature`, or `FeatureStore` methods
- No `Any` in public API signatures — use the most specific type available
- Prefer `asyncio.get_running_loop()` over `asyncio.get_event_loop()` inside async contexts

---

## Project structure

```
calcine/
├── pipeline.py        # Pipeline + GenerationReport
├── schema.py          # FeatureSchema + type system
├── serializers.py     # Serializer ABC + impls
├── exceptions.py      # Typed exception hierarchy
├── sources/           # DataSource ABC + built-ins
├── features/          # Feature ABC
└── stores/            # FeatureStore ABC + built-ins

tests/                 # mirrors calcine/ structure
examples/              # runnable end-to-end scripts
docs/                  # architecture and extension guides
```

---

## Adding a new DataSource

1. Create `calcine/sources/your_source.py`
2. Subclass `DataSource` and implement `async read(self, **kwargs)`
3. Accept `entity_id` as a keyword argument and scope data to it
4. Raise `SourceError(source_name=..., entity_id=..., cause=...)` on failure
5. Export from `calcine/sources/__init__.py`
6. Add tests in `tests/test_sources.py`

Minimal template:

```python
from calcine.sources.base import DataSource
from calcine.exceptions import SourceError

class MySource(DataSource):
    async def read(self, entity_id: str | None = None, **kwargs) -> Any:
        try:
            ...
        except Exception as exc:
            raise SourceError(
                source_name=type(self).__name__,
                entity_id=str(entity_id),
                cause=exc,
            ) from exc
```

---

## Adding a new FeatureStore

1. Create `calcine/stores/your_store.py`
2. Subclass `FeatureStore` and implement all four abstract methods:
   `write`, `read`, `exists`, `delete`
3. Raise `StoreError(store_name=..., feature_name=..., entity_id=..., cause=...)` on I/O failure
4. Raise `KeyError` (not `StoreError`) when an entity simply doesn't exist
5. Export from `calcine/stores/__init__.py`
6. Add tests in `tests/test_stores.py`

---

## Adding a new schema type

1. Add a class to `calcine/schema.py` that inherits `FeatureType`
2. Implement `_validate_value(self, value) -> list[str]`
3. Add it to the `types` namespace class at the bottom of `schema.py`
4. Add test cases in `tests/test_schema.py`

```python
class MyType(FeatureType):
    def _validate_value(self, value: Any) -> list[str]:
        if not isinstance(value, expected_type):
            return [f"Expected ..., got {type(value).__name__}"]
        return []
```

---

## Pull request process

1. **Fork** the repo and create a branch from `main`
   (`git checkout -b feat/my-new-source`)
2. **Write tests** — new functionality needs test coverage, bug fixes need a
   regression test
3. **Pass CI** — all tests green, ruff passes with no errors
4. **Update docs** — if you're adding a public class or changing behaviour,
   update the relevant docstrings and any affected `docs/` files
5. **Open the PR** — fill out the PR template; link any related issues

PRs that touch the public API (`Pipeline`, `Feature`, `DataSource`,
`FeatureStore`, `FeatureSchema`) should include a brief rationale for the
design choice in the PR description.
