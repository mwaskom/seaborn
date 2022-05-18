"""Recursively set the kernel name for all jupyter notebook files."""
import sys
from glob import glob

import nbformat


if __name__ == "__main__":

    _, kernel_name = sys.argv

    nb_paths = glob("./**/*.ipynb", recursive=True)
    for path in nb_paths:

        with open(path) as f:
            nb = nbformat.read(f, as_version=4)

        nb["metadata"]["kernelspec"]["name"] = kernel_name
        nb["metadata"]["kernelspec"]["display_name"] = kernel_name

        with open(path, "w") as f:
            nbformat.write(nb, f)
