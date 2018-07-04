.. _installing:

.. currentmodule:: seaborn

Installing and getting started
------------------------------

.. raw:: html

   <div class="col-md-9">

To install the latest release of seaborn, you can use ``pip``::

    pip install seaborn

It's also possible to install the released version using ``conda``::

    conda install seaborn

Alternatively, you can use ``pip`` to install the development version directly from github::

    pip install git+https://github.com/mwaskom/seaborn.git

Another option would be to to clone the `github repository
<https://github.com/mwaskom/seaborn>`_ and install from your local copy::

    pip install .

Dependencies
~~~~~~~~~~~~

-  Python 2.7 or 3.5+

Mandatory dependencies
^^^^^^^^^^^^^^^^^^^^^^

-  `numpy <http://www.numpy.org/>`__ (>= 1.9.3)

-  `scipy <https://www.scipy.org/>`__ (>= 0.14.0)

-  `matplotlib <https://matplotlib.org>`__ (>= 1.4.3)

-  `pandas <https://pandas.pydata.org/>`__ (>= 0.15.2)

Recommended dependencies
^^^^^^^^^^^^^^^^^^^^^^^^

-  `statsmodels <https://www.statsmodels.org/>`__ (>= 0.5.0)

Testing
~~~~~~~

To test seaborn, run ``make test`` in the root directory of the source
distribution. This runs the unit test suite (using ``pytest``, but many older
tests use ``nose`` asserts). It also runs the example code in function
docstrings to smoke-test a broader and more realistic range of example usage.

The full set of tests requires an internet connection to download the example
datasets (if they haven't been previously cached), but the unit tests should
be possible to run offline.


Bugs
~~~~

Please report any bugs you encounter through the github `issue tracker
<https://github.com/mwaskom/seaborn/issues/new>`_. It will be most helpful to
include a reproducible example on one of the example datasets (accessed through
:func:`load_dataset`). It is difficult debug any issues without knowing the
versions of seaborn and matplotlib you are using, as well as what `matplotlib
backend <https://matplotlib.org/faq/usage_faq.html#what-is-a-backend>`__ you
are using to draw the plots, so please include those in your bug report.

.. raw:: html

   </div>
