Seaborn: statistical data visualization
=======================================

<div class="row">
<a href=http://seaborn.pydata.org/examples/anscombes_quartet.html>
<img src="http://seaborn.pydata.org/_static/anscombes_quartet_thumb.png" height="135" width="135">
</a>

<a href=http://seaborn.pydata.org/examples/many_pairwise_correlations.html>
<img src="http://seaborn.pydata.org/_static/many_pairwise_correlations.png" height="135" width="135">
</a>

<a href=http://seaborn.pydata.org/examples/many_facets.html>
<img src="http://seaborn.pydata.org/_static/many_facets.png" height="135" width="135">
</a>    

<a href=http://seaborn.pydata.org/examples/scatterplot_matrix.html>
<img src="http://seaborn.pydata.org/_static/scatterplot_matrix.png" height="135" width="135">
</a>

<a href=http://seaborn.pydata.org/examples/hexbin_marginals.html>
<img src="http://seaborn.pydata.org/_static/hexbin_marginals.png" height="135" width="135">
</a>

</div>

Seaborn is a Python visualization library based on matplotlib. It provides a high-level interface for drawing attractive statistical graphics.


Documentation
-------------

Online documentation is available [here](https://seaborn.github.io/). It includes a high-level tutorial, detailed API documentation, and other useful info.

Examples
--------

The documentation has an [example gallery](https://seaborn.github.io/examples/index.html) with short scripts showing how to use different parts of the package.

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
