export SHELL := /bin/bash

test:
	pytest --doctest-modules seaborn

unittests:
	pytest seaborn

coverage:
	pytest --doctest-modules --cov=seaborn --cov-config=.coveragerc seaborn

lint:
	flake8 --ignore E121,E123,E126,E226,E24,E704,E741,W503,W504 --exclude seaborn/__init__.py,seaborn/colors/__init__.py,seaborn/cm.py,seaborn/tests,seaborn/external seaborn
