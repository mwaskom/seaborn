#! /usr/bin/env python
"""
Convert empty IPython notebook to a sphinx doc page.

"""
import sys
from subprocess import check_call as sh


def convert_nb(nbname):

    # Execute the notebook
    sh(["jupyter", "nbconvert", "--to", "notebook",
        "--execute", "--inplace", nbname])

    # Convert to .rst for Sphinx
    sh(["jupyter", "nbconvert", "--to", "rst", nbname,
        "--TagRemovePreprocessor.remove_cell_tags={'hide'}",
        "--TagRemovePreprocessor.remove_input_tags={'hide-input'}",
        "--TagRemovePreprocessor.remove_all_outputs_tags={'hide-output'}"])

    # Clear notebook output
    sh(["jupyter", "nbconvert", "--to", "notebook", "--inplace",
        "--ClearOutputPreprocessor.enabled=True", nbname])

    # Touch the .rst file so it has a later modify time than the source
    sh(["touch", nbname + ".rst"])


if __name__ == "__main__":

    for nbname in sys.argv[1:]:
        convert_nb(nbname)
