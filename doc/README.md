Building the seaborn docs
=========================

Building the docs requires additional dependencies; they can be installed with `pip install seaborn[stats,docs]`.

The build process involves conversion of Jupyter notebooks to `rst` files. To facilitate this, you may need to set `NB_KERNEL` environment variable to the name of a kernel on your machine (e.g. `export NB_KERNEL="python3"`). To get a list of available Python kernels, run `jupyter kernelspec list`.

After you're set up, run `make notebooks html` from the `doc` directory to convert all notebooks, generate all gallery examples, and build the documentation itself. The site will live in `_build/html`.

Run `make clean` to delete the built site and all intermediate files. Run `make -C docstrings clean` or `make -C tutorial clean` to remove intermediate files for the API or tutorial components.

If your goal is to obtain an offline copy of the docs for a released version, it may be easier to clone the [website repository](https://github.com/seaborn/seaborn.github.io) or to download a zipfile corresponding to a [specific version](https://github.com/seaborn/seaborn.github.io/tags).

# Commands

```bash
pyenv install 3.12
pyenv global 3.12
python -m venv .venv
. .venv/bin/activate
pip install seaborn[stats,docs]
# sudo dnf install -y python3-ipykernel
python -m ipykernel install --user --name=myenv --display-name "Python (myenv)"
jupyter-kernelspec list
export NB_KERNEL="myenv"
make notebooks latexpdf 
```

Here is an example of the built PDF: https://codeberg.org/jipmelon/t/src/branch/main/seaborn.pdf