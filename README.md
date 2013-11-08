Seaborn: statistical data visualization
=======================================

Seaborn is a library of high-level functions that facilitate making informative
and attractive plots of statistical data using matplotlib. It also provides
concise control over the aesthetics of the plots, improving on matplotlib's
default look.

<a href=http://stanford.edu/~mwaskom/software/seaborn/examples/distplot_options.html>
<img src="http://stanford.edu/~mwaskom/software/seaborn/_static/distplot_options_thumb.png" height="135" width="135">
</a>    

<a href=http://stanford.edu/~mwaskom/software/seaborn/examples/regression_marginals.html>
<img src="http://stanford.edu/~mwaskom/software/seaborn/_static/regression_marginals_thumb.png" height="135" width="135">
</a>

<a href=http://stanford.edu/~mwaskom/software/seaborn/examples/violinplots.html>
<img src="http://stanford.edu/~mwaskom/software/seaborn/_static/violinplots_thumb.png" height="135" width="135">
</a>

</div>

Seaborn is a Python visualization library based on matplotlib. It provides a high-level interface for drawing attractive statistical graphics.

<a href=http://stanford.edu/~mwaskom/software/seaborn/examples/regression_marginals.html>
<img src="http://stanford.edu/~mwaskom/software/seaborn/_static/regression_marginals_thumb.png" height="135" width="135">
</a>

<a href=http://stanford.edu/~mwaskom/software/seaborn/examples/violinplots.html>
<img src="http://stanford.edu/~mwaskom/software/seaborn/_static/violinplots_thumb.png" height="135" width="135">
</a>

</div>

Seaborn is a Python visualization library based on matplotlib. It provides a high-level interface for drawing attractive statistical graphics.


Documentation
-------------

Online documentation is available [here](http://stanford.edu/~mwaskom/software/seaborn/).

Documentation
-------------

Online documentation is available [here](http://stanford.edu/~mwaskom/software/seaborn/).

Documentation
-------------

Online documentation is available [here](http://stanford.edu/~mwaskom/software/seaborn/).

Examples
--------

The documentation has an [example gallery](http://stanford.edu/~mwaskom/software/seaborn/examples/index.html) with short scripts showing how to use different parts of the package. You can also check out the example notebooks:

[Controlling figure aesthetics in seaborn](http://nbviewer.ipython.org/urls/raw.github.com/mwaskom/seaborn/master/examples/aesthetics.ipynb)

[Plotting complex linear models](http://nbviewer.ipython.org/urls/raw.github.com/mwaskom/seaborn/master/examples/linear_models.ipynb)

- [Graphical representations of linear models](http://nbviewer.ipython.org/github/mwaskom/seaborn/blob/master/examples/linear_models.ipynb)

- [Visualizing distributions of data](http://nbviewer.ipython.org/github/mwaskom/seaborn/blob/master/examples/plotting_distributions.ipynb)

- [Plotting statistical timeseries data](http://nbviewer.ipython.org/github/mwaskom/seaborn/blob/master/examples/timeseries_plots.ipynb)


Dependencies
------------

- Python 2.7 or 3.3+

### Mandatory

- [numpy](http://www.numpy.org/)

- [scipy](http://www.scipy.org/)

- [matplotlib](http://matplotlib.sourceforge.net)

- [pandas](http://pandas.pydata.org/)

### Recommended

- [patsy](http://patsy.readthedocs.org/en/latest/)

- [scikit-learn](http://scikit-learn.org)

- [husl](https://github.com/boronine/pyhusl)

- [patsy](http://patsy.readthedocs.org/en/latest/)


Installation
------------

To install the released version, just do

    pip install seaborn

You may instead want to use the development version from Github, by running

    pip install git+git://github.com/mwaskom/seaborn.git#egg=seaborn


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

Please [submit](https://github.com/mwaskom/seaborn/issues/new) any bugs you encounter to the Github issue tracker.


Celebrity Endorsements
----------------------

"Those are nice plots" -Hadley Wickham
