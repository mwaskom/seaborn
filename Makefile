export SHELL := /bin/bash

test:
	pytest -n auto --doctest-modules --cov=seaborn --cov-config=.coveragerc seaborn

unittests:
	pytest -n auto --cov=seaborn --cov-config=.coveragerc seaborn

lint:
	flake8 seaborn

typecheck:
	mypy -p seaborn._core --exclude seaborn._core.orig.py
