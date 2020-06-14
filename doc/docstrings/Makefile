rst_files := $(patsubst %.ipynb,%.rst,$(wildcard *.ipynb))

docstrings: ${rst_files}

%.rst: %.ipynb
	@../tools/nb_to_doc.py $*.ipynb
	@cp -r $*_files ../generated/
	@if [ -f ../generated/seaborn.$*.rst ]; then \
	    touch ../generated/seaborn.$*.rst; \
	 fi

clean:
	rm -rf *.rst
	rm -rf *_files/
	rm -rf .ipynb_checkpoints/
