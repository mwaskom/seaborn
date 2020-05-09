export SHELL := /bin/bash

test:
	pytest --doctest-modules seaborn

unittests:
	pytest seaborn

coverage:
	pytest --doctest-modules --cov=seaborn --cov-config=.coveragerc seaborn

lint:
	flake8 seaborn