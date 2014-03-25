export SHELL := /bin/bash

test:

	nosetests --with-doctest
	python ipnbdoctest.py examples/*.ipynb


coverage:

	nosetests --cover-erase --with-coverage --cover-html --cover-package seaborn

lint:

	pyflakes -x W seaborn
	pep8 --exclude external seaborn

hexstrip:

	make -C examples hexstrip
