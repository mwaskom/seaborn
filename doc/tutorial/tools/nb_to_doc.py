#! /usr/bin/env python
"""
Convert empty IPython notebook to a sphinx doc page.

"""
import os
import sys


def convert_nb(nbname):

    os.system("runipy --o %s.ipynb --matplotlib --quiet" % nbname)
    os.system("ipython nbconvert --to rst %s.ipynb" % nbname)
    os.system("tools/nbstripout %s.ipynb" % nbname)


if __name__ == "__main__":

    for nbname in sys.argv[1:]:
        convert_nb(nbname)
