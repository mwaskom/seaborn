rst_files := $(patsubst %.ipynb,%.rst,$(wildcard *.ipynb))

tutorial: ${rst_files}

%.rst: %.ipynb
	../tools/nb_to_doc.py $*
