rst_files := $(patsubst %.ipynb,%.rst,$(wildcard *.ipynb))

tutorial: ${rst_files}

%.rst: %.ipynb
	../tools/nb_to_doc.py $*.ipynb

clean:
	rm -rf *.rst
	rm -rf *_files/
	rm -rf .ipynb_checkpoints/
