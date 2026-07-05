export SHELL := /bin/bash

test:
	uv run --no-sync pytest -n auto --cov=seaborn --cov=tests --cov-config=pyproject.toml tests

lint:
	uv run --no-sync ruff check seaborn/ tests/

typecheck:
	uv run --no-sync ty check

docs:
	uv run --no-sync make -C doc notebooks html
