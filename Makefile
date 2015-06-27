export SHELL := /bin/bash

test:

	cp testing/matplotlibrc .
	nosetests --with-doctest
	rm matplotlibrc


coverage:

	cp testing/matplotlibrc .
	nosetests --cover-erase --with-coverage --cover-html --cover-package seaborn
	rm matplotlibrc

lint:

	pyflakes -x W seaborn
	pep8 --exclude external seaborn

hexstrip:

	make -C examples hexstrip
