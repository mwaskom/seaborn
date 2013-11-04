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

[Plotting complex linear models](http://nbviewer.ipython.org/urls/raw.github.com/mwaskom/seaborn/master/examples/linear_models.ipynb)

[Visualizing distributions of data](http://nbviewer.ipython.org/urls/raw.github.com/mwaskom/seaborn/master/examples/plotting_distributions.ipynb)

[Representing variability in timeseries plots](http://nbviewer.ipython.org/urls/raw.github.com/mwaskom/seaborn/master/examples/timeseries_plots.ipynb)


Dependencies
------------

- Python 2.7

- [numpy](http://www.numpy.org/)

- [scipy](http://www.scipy.org/)

- [matplotlib](matplotlib.sourceforge.net)

- [pandas](http://pandas.pydata.org/)

- [statsmodels](http://statsmodels.sourceforge.net/)

- [patsy](http://patsy.readthedocs.org/en/latest/)

- [scikit-learn](http://scikit-learn.org)

- [husl](https://github.com/boronine/pyhusl)

- [moss](http://github.com/mwaskom/moss)


Installation
------------

To install the released version, just do

    pip install -U seaborn

However, I update the code pretty frequently, so you may want to clone the
github repository and install with

    pip install -r requirements.txt

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
