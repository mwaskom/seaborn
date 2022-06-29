rst_files := $(patsubst %.ipynb,../tutorial/%.rst,$(wildcard *.ipynb))

tutorial: ${rst_files}

../tutorial/%.rst: %.ipynb
	../tools/nb_to_doc.py $*.ipynb ../tutorial

clean:
	rm -rf *.rst
	rm -rf *_files/
	rm -rf .ipynb_checkpoints/
