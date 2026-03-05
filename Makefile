.PHONY: test lint format fix install-hooks

test:
	.venv/bin/pytest tests/ -q

lint:
	.venv/bin/ruff check calcine/ tests/
	.venv/bin/ruff format --check calcine/ tests/

format:
	.venv/bin/ruff format calcine/ tests/
	.venv/bin/ruff check --fix calcine/ tests/

# Fix everything then run tests — use this before pushing
fix: format test

# Install git hooks for this repo (run once after cloning)
install-hooks:
	/bin/cp hooks/pre-commit .git/hooks/pre-commit
	/bin/chmod +x .git/hooks/pre-commit
	@echo "Pre-commit hook installed."
