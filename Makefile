export SHELL := /bin/bash

test:
	pytest --doctest-modules seaborn

test-nodoctest:
	pytest seaborn

coverage:
	pytest --cov=seaborn seaborn

lint:
	flake8 --exclude seaborn/__init__.py,seaborn/cm.py,seaborn/tests,seaborn/external seaborn
