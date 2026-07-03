export SHELL := /bin/bash

install:
	uv sync --extra stats

lock:
	uv lock

test:
	pytest -n auto --cov=seaborn --cov=tests --cov-config=pyproject.toml tests

lint:
	ruff check seaborn/ tests/

typecheck:
	ty check
