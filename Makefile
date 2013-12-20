export SHELL := /bin/bash

test:

	make -C examples test
	nosetests


hexstrip:

	make -C examples hexstrip
