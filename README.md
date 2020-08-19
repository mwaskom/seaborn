seaborn: statistical data visualization
=======================================

<div class="row">

<a href=https://seaborn.pydata.org/examples/scatterplot_matrix.html>
<img src="https://seaborn.pydata.org/_static/scatterplot_matrix_thumb.png" height="135" width="135">
</a>

<a href=https://seaborn.pydata.org/examples/errorband_lineplots.html>
<img src="https://seaborn.pydata.org/_static/errorband_lineplots_thumb.png" height="135" width="135">
</a>

<a href=https://seaborn.pydata.org/examples/different_scatter_variables.html>
<img src="https://seaborn.pydata.org/_static/different_scatter_variables_thumb.png" height="135" width="135">
</a>

<a href=https://seaborn.pydata.org/examples/many_facets.html>
<img src="https://seaborn.pydata.org/_static/many_facets_thumb.png" height="135" width="135">
</a>

<a href=https://seaborn.pydata.org/examples/structured_heatmap.html>
<img src="https://seaborn.pydata.org/_static/structured_heatmap_thumb.png" height="135" width="135">
</a>

<a href=https://seaborn.pydata.org/examples/horizontal_boxplot.html>
<img src="https://seaborn.pydata.org/_static/horizontal_boxplot_thumb.png" height="135" width="135">
</a>

</div>

--------------------------------------

[![PyPI Version](https://img.shields.io/pypi/v/seaborn.svg)](https://pypi.org/project/seaborn/)
[![License](https://img.shields.io/pypi/l/seaborn.svg)](https://github.com/mwaskom/seaborn/blob/master/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.592845.svg)](https://doi.org/10.5281/zenodo.592845)
[![Build Status](https://travis-ci.org/mwaskom/seaborn.svg?branch=master)](https://travis-ci.org/mwaskom/seaborn)
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

