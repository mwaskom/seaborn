export SHELL := /bin/bash

NPROC := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)

test:
	uv run --no-sync pytest -n auto --cov=seaborn --cov=tests --cov-config=pyproject.toml tests

lint:
	uv run --no-sync ruff check seaborn/ tests/

typecheck:
	uv run --no-sync ty check

docs:
	uv run --no-sync make -C doc -j$(NPROC) notebooks
	uv run --no-sync make -C doc SPHINXOPTS="-j$(NPROC)" html
