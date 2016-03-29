Seaborn: statistical data visualization
=======================================

<div class="row">
<a href=http://stanford.edu/~mwaskom/software/seaborn/examples/anscombes_quartet.html>
<img src="http://stanford.edu/~mwaskom/software/seaborn/_static/anscombes_quartet_thumb.png" height="135" width="135">
</a>

<a href=http://stanford.edu/~mwaskom/software/seaborn/examples/timeseries_from_dataframe.html>
<img src="http://stanford.edu/~mwaskom/software/seaborn/_static/timeseries_from_dataframe_thumb.png" height="135" width="135">
</a>

<a href=http://stanford.edu/~mwaskom/software/seaborn/examples/distplot_options.html>
<img src="http://stanford.edu/~mwaskom/software/seaborn/_static/distplot_options_thumb.png" height="135" width="135">
</a>    

<a href=http://stanford.edu/~mwaskom/software/seaborn/examples/regression_marginals.html>
<img src="http://stanford.edu/~mwaskom/software/seaborn/_static/regression_marginals_thumb.png" height="135" width="135">
</a>

<a href=http://stanford.edu/~mwaskom/software/seaborn/examples/grouped_violinplots.html>
<img src="http://stanford.edu/~mwaskom/software/seaborn/_static/grouped_violinplots_thumb.png" height="135" width="135">
</a>

</div>

Seaborn is a Python visualization library based on matplotlib. It provides a high-level interface for drawing attractive statistical graphics.


Documentation
-------------

Online documentation is available [here](http://stanford.edu/~mwaskom/software/seaborn/). It includes a high-level tutorial, detailed API documentation, and other useful info.

There are docs for the development version [here](http://stanford.edu/~mwaskom/software/seaborn-dev/). These should more or less correspond with the github master branch, but they're not built automatically and thus may fall out of sync at times.

Examples
--------

The documentation has an [example gallery](http://stanford.edu/~mwaskom/software/seaborn/examples/index.html) with short scripts showing how to use different parts of the package.

Citing
------

Seaborn can be cited using a DOI provided through Zenodo: [![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.45133.svg)](http://dx.doi.org/10.5281/zenodo.45133)

Dependencies
------------

- Python 2.7 or 3.3+

### Mandatory

- [numpy](http://www.numpy.org/)

- [scipy](http://www.scipy.org/)

- [matplotlib](http://matplotlib.sourceforge.net)

- [pandas](http://pandas.pydata.org/)

### Recommended

- [statsmodels](http://statsmodels.sourceforge.net/)


Installation
------------

To install the released version, just do

    pip install seaborn

You may instead want to use the development version from Github, by running

    pip install git+git://github.com/mwaskom/seaborn.git#egg=seaborn


Testing
-------

[![Build Status](https://travis-ci.org/mwaskom/seaborn.png?branch=master)](https://travis-ci.org/mwaskom/seaborn)

To test seaborn, run `make test` in the source directory. This will run the
unit-test and doctest suite (using `nose`).

Development
-----------

https://github.com/mwaskom/seaborn

Please [submit](https://github.com/mwaskom/seaborn/issues/new) any bugs you encounter to the Github issue tracker.

License
-------

Released under a BSD (3-clause) license


Celebrity Endorsements
----------------------

"Those are nice plots" -Hadley Wickham
