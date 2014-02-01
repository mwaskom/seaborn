export SHELL := /bin/bash

test:

	nosetests
	make -C examples test


coverage:

	nosetests --with-coverage --cover-package seaborn


hexstrip:

	make -C examples hexstrip
