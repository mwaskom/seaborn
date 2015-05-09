.. _installing:

Installing and getting started
------------------------------

To install the released version of seaborn, you can use ``pip`` (i.e. ``pip install seaborn``). 

Alternatively, you can use ``pip`` to install the development version, with the command ``pip install git+git://github.com/mwaskom/seaborn.git#egg=seaborn``.

Another option would be to to clone the `github repository <https://github.com/mwaskom/seaborn>`_ and install with ``pip install .`` from the source directory. Seaborn itself is pure Python, so installation should be reasonably straightforward.

When using the development version, you may want to refer to the `development docs <http://stanford.edu/~mwaskom/software/seaborn-dev/>`_. Note that these are not built automatically and may at times fall out of sync with the actual master branch on github.

Dependencies 
~~~~~~~~~~~~

We recommend using seaborn with the `Anaconda distribution <https://store.continuum.io/cshop/anaconda/>`_.

-  Python 2.7 or 3.3+

Mandatory dependencies
^^^^^^^^^^^^^^^^^^^^^^

-  `numpy <http://www.numpy.org/>`__

-  `scipy <http://www.scipy.org/>`__

-  `matplotlib <matplotlib.sourceforge.net>`__

-  `pandas <http://pandas.pydata.org/>`__

Recommended dependencies
^^^^^^^^^^^^^^^^^^^^^^^^

-  `statsmodels <http://statsmodels.sourceforge.net/>`__

Version-wise, we make an attempt to keep seaborn working on the stable Debian
channels. There may be cases where some more advanced features only work with
newer versions of these dependencies, although these should be rare. There are
also some known bugs on older versions of matplotlib, so you should in general
try to use a modern version, but for many cases older matplotlibs will work
fine.  Seaborn is tested on the most recent versions offered through ``conda``.

Import conventions
~~~~~~~~~~~~~~~~~~

By convention, ``seaborn`` is abbreviated to ``sns`` on imports.

Testing
~~~~~~~

To test seaborn, run ``make test`` in the root directory of the source
distribution. This runs the unit test suite (which can also be exercised
separately by running ``nosetests``). It also runs the code in the example 
notebooks to smoke-test a broader and more realistic range of example usage.

Bugs
~~~~

Please report any bugs you encounter through the github `issue tracker
<https://github.com/mwaskom/seaborn/issues/new>`_. It will be most helpful to
include a reproducible example on one of the example datasets (accessed through
:func:`load_dataset`) and to include the version of matplotlib that you are
using when you encounter the error.
