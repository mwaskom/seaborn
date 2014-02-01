export SHELL := /bin/bash

test:

	nosetests
	make -C examples test


hexstrip:

	make -C examples hexstrip
