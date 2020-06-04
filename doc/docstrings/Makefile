rst_files := $(patsubst %.ipynb,%.rst,$(wildcard *.ipynb))

docstrings: ${rst_files}

%.rst: %.ipynb
	../tools/nb_to_doc.py $*.ipynb
	cp -r $* ../generated/
