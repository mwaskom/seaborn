.. _installing:

Installing and getting started
------------------------------

To install the latest release of seaborn, you can use ``pip``::

    pip install seaborn

It's also possible to install the released version using ``conda``::

    conda install seaborn

Alternatively, you can use ``pip`` to install the development version directly from github::

    pip install git+https://github.com/mwaskom/seaborn.git

Another option would be to to clone the `github repository
<https://github.com/mwaskom/seaborn>`_ install from your local copy::

    pip install .


Dependencies
~~~~~~~~~~~~

-  Python 2.7 or 3.4+

Mandatory dependencies
^^^^^^^^^^^^^^^^^^^^^^

-  `numpy <http://www.numpy.org/>`__

-  `scipy <http://www.scipy.org/>`__

-  `matplotlib <http://matplotlib.org>`__

-  `pandas <http://pandas.pydata.org/>`__

Recommended dependencies
^^^^^^^^^^^^^^^^^^^^^^^^

-  `statsmodels <http://statsmodels.sourceforge.net/>`__

The ``pip`` installation script will attempt to download the mandatory
dependencies only if they do not exist at install-time.

There are not hard minimum version requirements for the dependencies. Unit
tests aim to keep seaborn importable and generally functional on the versions
available through the stable Debian channels, which are relatively old. There
are some known bugs when using older versions of matplotlib (generally meaning
1.3 or earlier), so you should in general try to use a modern version.  For
most use cases, though, older versions of matplotlib will work fine.

Seaborn is also tested on the most recent versions offered through ``conda``.


Testing
~~~~~~~

To test seaborn, run ``make test`` in the root directory of the source
distribution. This runs the unit test suite (using ``pytest``). It also runs
the example code in function docstrings to smoke-test a broader and more
realistic range of example usage.

The full set of tests requires an internet connection to download the example
datasets (if they haven't been previously cached), but the unit tests should
be able to run offline.


Bugs
~~~~

Please report any bugs you encounter through the github `issue tracker
<https://github.com/mwaskom/seaborn/issues/new>`_. It will be most helpful to
include a reproducible example on one of the example datasets (accessed through
:func:`load_dataset`). It is difficult debug any issues without knowing the
versions of seaborn and matplotlib you are using, as well as what matplotlib
backend you are using to draw the plots, so please include those in your bug
report.


Known issues
~~~~~~~~~~~~

An unfortunate consequence of how the matplotlib marker styles work is that
line-art markers (e.g. ``"+"``) or markers with ``facecolor`` set to ``"none"``
will be invisible when the default seaborn style is in effect. This can be
changed by using a different ``markeredgewidth`` (aliased to ``mew``) either in
the function call or globally in the ``rcParams``.
