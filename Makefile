export SHELL := /bin/bash

test:

	nosetests --with-doctest

test-nodoctest:

	nosetests

coverage:

	nosetests --cover-erase --with-coverage --cover-html --cover-package seaborn

lint:

	pyflakes -x W -X seaborn/external/six.py seaborn
	pep8 --exclude external,cm.py seaborn
