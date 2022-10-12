Building the seaborn docs
=========================

Building the docs requires additional dependencies; they can be installed with `pip install seaborn[stats,docs]`.

The build process involves conversion of Jupyter notebooks to `rst` files. To facilitate this, you may need to set `NB_KERNEL` environment variable to the name of a kernel on your machine (e.g. `export NB_KERNEL="python3"`). To get a list of available Python kernels, run `jupyter kernelspec list`.

After you're set up, run `make notebooks html` from the `doc` directory to convert all notebooks, generate all gallery examples, and build the documentation itself. The site will live in `_build/html`.

Run `make clean` to delete the built site and all intermediate files. Run `make -C docstrings clean` or `make -C tutorial clean` to remove intermediate files for the API or tutorial components.

If your goal is to obtain an offline copy of the docs for a released version, it may be easier to clone the [website repository](https://github.com/seaborn/seaborn.github.io) or to download a zipfile corresponding to a [specific version](https://github.com/seaborn/seaborn.github.io/tags).
