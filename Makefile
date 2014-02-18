export SHELL := /bin/bash

test:

	nosetests
	make -C examples test


coverage:

	nosetests --cover-erase --with-coverage --cover-html --cover-package seaborn

lint:

	pyflakes -x W seaborn
	pep8 seaborn

hexstrip:

	make -C examples hexstrip
