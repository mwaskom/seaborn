<img src="doc/_static/logo-wide-lightbg.svg"><br>

--------------------------------------

seaborn: statistical data visualization
=======================================

[![PyPI Version](https://img.shields.io/pypi/v/seaborn.svg)](https://pypi.org/project/seaborn/)
[![License](https://img.shields.io/pypi/l/seaborn.svg)](https://github.com/mwaskom/seaborn/blob/master/LICENSE)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.03021/status.svg)](https://doi.org/10.21105/joss.03021)
![Tests](https://github.com/mwaskom/seaborn/workflows/CI/badge.svg)
[![Code Coverage](https://codecov.io/gh/mwaskom/seaborn/branch/master/graph/badge.svg)](https://codecov.io/gh/mwaskom/seaborn)

Seaborn is a Python visualization library based on matplotlib. It provides a high-level interface for drawing attractive statistical graphics.


Documentation
-------------

Online documentation is available at [seaborn.pydata.org](https://seaborn.pydata.org).

The docs include a [tutorial](https://seaborn.pydata.org/tutorial.html), [example gallery](https://seaborn.pydata.org/examples/index.html), [API reference](https://seaborn.pydata.org/api.html), and other useful information.

To build the documentation locally, please refer to [`doc/README.md`](doc/README.md).

Dependencies
------------

Seaborn supports Python 3.6+ and no longer supports Python 2.

Installation requires [numpy](https://numpy.org/), [scipy](https://www.scipy.org/), [pandas](https://pandas.pydata.org/), and [matplotlib](https://matplotlib.org/). Some functions will optionally use [statsmodels](https://www.statsmodels.org/) if it is installed.


Installation
------------

The latest stable release (and older versions) can be installed from PyPI:

    pip install seaborn

You may instead want to use the development version from Github:

    pip install git+https://github.com/mwaskom/seaborn.git#egg=seaborn


Citing
------

A paper describing seaborn has been published in the [Journal of Open Source Software](https://joss.theoj.org/papers/10.21105/joss.03021). The paper provides an introduction to the key features of the library, and it can be used as a citation if seaborn proves integral to a scientific publication.

Testing
-------

Testing seaborn requires installing additional packages listed in `ci/utils.txt`.

To test the code, run `make test` in the source directory. This will exercise both the unit tests and docstring examples (using [pytest](https://docs.pytest.org/)) and generate a coverate report.

The doctests require a network connection (unless all example datasets are cached), but the unit tests can be run offline with `make unittests`.


Code style is enforced with `flake8` using the settings in the [`setup.cfg`](./setup.cfg) file. Run `make lint` to check.
 
Development
-----------

Seaborn development takes place on Github: https://github.com/mwaskom/seaborn

Please submit bugs that you encounter to the [issue tracker](https://github.com/mwaskom/seaborn/issues) with a reproducible example demonstrating the problem. Questions about usage are more at home on StackOverflow, where there is a [seaborn tag](https://stackoverflow.com/questions/tagged/seaborn).

