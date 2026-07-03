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
	mypy --follow-imports=skip seaborn/_core seaborn/_marks seaborn/_stats
