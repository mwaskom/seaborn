test:

	@OLDPYPATH=$$PYTHONPATH
	@export PYTHONPATH=`pwd`/../seaborn:$$OLDPYPATH
	python ipnbdoctest.py *.ipynb
	@export PYTHONPATH=$$OLDPYPATH
