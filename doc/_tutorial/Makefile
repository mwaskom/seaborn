rst_files := $(patsubst %.ipynb,../tutorial/%.rst,$(wildcard *.ipynb))
export MPLBACKEND := module://matplotlib_inline.backend_inline

tutorial: ${rst_files}

../tutorial/%.rst: %.ipynb
	../tools/nb_to_doc.py $*.ipynb ../tutorial

clean:
	rm -rf ../tutorial
