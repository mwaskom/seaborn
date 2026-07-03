export SHELL := /bin/bash

test:
	pytest -n auto --cov=seaborn --cov=tests --cov-config=pyproject.toml tests

lint:
	ruff check seaborn/ tests/

typecheck:
	mypy --follow-imports=skip seaborn/_core seaborn/_marks seaborn/_stats
