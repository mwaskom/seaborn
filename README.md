<img src="https://raw.githubusercontent.com/mwaskom/seaborn/master/doc/_static/logo-wide-lightbg.svg"><br>

--------------------------------------

seaborn: statistical data visualization
=======================================

[![PyPI Version](https://img.shields.io/pypi/v/seaborn.svg)](https://pypi.org/project/seaborn/)
[![License](https://img.shields.io/pypi/l/seaborn.svg)](https://github.com/mwaskom/seaborn/blob/master/LICENSE)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.03021/status.svg)](https://doi.org/10.21105/joss.03021)
[![Tests](https://github.com/mwaskom/seaborn/workflows/CI/badge.svg)](https://github.com/mwaskom/seaborn/actions)
[![Code Coverage](https://codecov.io/gh/mwaskom/seaborn/branch/master/graph/badge.svg)](https://codecov.io/gh/mwaskom/seaborn)

Seaborn is a Python visualization library based on matplotlib. It provides a high-level interface for drawing attractive statistical graphics.


Documentation
-------------

Online documentation is available at [seaborn.pydata.org](https://seaborn.pydata.org).

The docs include a [tutorial](https://seaborn.pydata.org/tutorial.html), [example gallery](https://seaborn.pydata.org/examples/index.html), [API reference](https://seaborn.pydata.org/api.html), and other useful information.

To build the documentation locally, please refer to [`doc/README.md`](doc/README.md).

There is also a [FAQ](https://github.com/mwaskom/seaborn/wiki/Frequently-Asked-Questions-(FAQs)) page, currently hosted on GitHub.

Dependencies
------------

Seaborn supports Python 3.7+ and no longer supports Python 2.

Installation requires [numpy](https://numpy.org/), [pandas](https://pandas.pydata.org/), and [matplotlib](https://matplotlib.org/). Some advanced statistical functionality requires [scipy](https://www.scipy.org/) and/or [statsmodels](https://www.statsmodels.org/).


Installation
------------

The latest stable release (and required dependencies) can be installed from PyPI:

    pip install seaborn

It is also possible to include optional statistical dependencies (only relevant for v0.12+):

    pip install seaborn[stats]

Seaborn can also be installed with conda:

    conda install seaborn

Note that the main anaconda repository lags PyPI in adding new releases, but conda-forge (`-c conda-forge`) typically updates quickly.

Citing
------

A paper describing seaborn has been published in the [Journal of Open Source Software](https://joss.theoj.org/papers/10.21105/joss.03021). The paper provides an introduction to the key features of the library, and it can be used as a citation if seaborn proves integral to a scientific publication.

Testing
-------

Testing seaborn requires installing additional dependencies; they can be installed with the `dev` extra (e.g., `pip install .[dev]`).

To test the code, run `make test` in the source directory. This will exercise both the unit tests and docstring examples (using [pytest](https://docs.pytest.org/)) and generate a coverage report.

The doctests require a network connection (unless all example datasets are cached), but the unit tests can be run offline with `make unittests`.

Code style is enforced with `flake8` using the settings in the [`setup.cfg`](./setup.cfg) file. Run `make lint` to check. Alternately, you can use `pre-commit` to automatically run lint checks on any files you are committing &ndash; just run `pre-commit install` to set it up, and then commit as usual going forward.

Development
-----------

Seaborn development takes place on Github: https://github.com/mwaskom/seaborn

Please submit bugs that you encounter to the [issue tracker](https://github.com/mwaskom/seaborn/issues) with a reproducible example demonstrating the problem. Questions about usage are more at home on StackOverflow, where there is a [seaborn tag](https://stackoverflow.com/questions/tagged/seaborn).
