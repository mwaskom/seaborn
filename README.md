Seaborn: statistical data visualization
=======================================

Seaborn is a library of high-level functions that facilitate making informative
and attractive plots of statistical data using matplotlib. It also provides
concise control over the aesthetics of the plots, improving on matplotlib's
default look.

![](examples/example_plot.png "Example Seaborn Plot")


Documentation
-------------

Online documentation is available [here](http://stanford.edu/~mwaskom/software/seaborn/).

Examples
--------

There are a few tutorial notebooks that offer some thoughts on visualizing
statistical data in a general sense and show how to do it using the tools that
are provided in seaborn. They also serve as the primary test suite for the package.
The notebooks are meant to be fairly, but not completely comprehensive;
hopefully the docstrings for the specific functions will answer any additional
questions.

[Controlling figure aesthetics in seaborn](http://nbviewer.ipython.org/github/mwaskom/seaborn/blob/master/examples/aesthetics.ipynb)

[Plotting complex linear models](http://nbviewer.ipython.org/github/mwaskom/seaborn/blob/master/examples/linear_models.ipynb)

[Visualizing distributions of data](http://nbviewer.ipython.org/github/mwaskom/seaborn/blob/master/examples/plotting_distributions.ipynb)

[Plotting statistical timeseries data](http://nbviewer.ipython.org/github/mwaskom/seaborn/blob/master/examples/timeseries_plots.ipynb)


Dependencies
------------

- Python 2.7

- [numpy](http://www.numpy.org/)

- [scipy](http://www.scipy.org/)

- [matplotlib](http://matplotlib.sourceforge.net)

- [pandas](http://pandas.pydata.org/)

- [statsmodels](http://statsmodels.sourceforge.net/)

- [scikit-learn](http://scikit-learn.org)

- [patsy](http://patsy.readthedocs.org/en/latest/)

- [husl](https://github.com/boronine/pyhusl)

- [moss](http://github.com/mwaskom/moss)

Installing with `pip` will automatically install `patsy`, `husl`, and `moss`, which are the only dependencies not included in Anaconda.


Installation
------------

To install the released version, just do

    pip install seaborn

However, I update the code pretty frequently, so you may want to clone the
github repository and install with

    python setup.py install

from within the source directory.


Testing
-------

To test seaborn, run `make test` in the source directory. This will execute the
example notebooks and compare the outputs of each cell to the data in the
stored versions. There is also a (small) set of unit tests for the utility
functions that can be tested separately with `nosetests`. 


Development
-----------

https://github.com/mwaskom/seaborn

Please submit any bugs you encounter to the Github issue tracker.


Celebrity Endorsements
----------------------

"Those are nice plots" -Hadley Wickham
