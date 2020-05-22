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

-  Python 3.6+

Mandatory dependencies
^^^^^^^^^^^^^^^^^^^^^^

-  `numpy <https://numpy.org/>`__

-  `scipy <https://www.scipy.org/>`__

-  `pandas <https://pandas.pydata.org/>`__

-  `matplotlib <https://matplotlib.org>`__

Recommended dependencies
^^^^^^^^^^^^^^^^^^^^^^^^

-  `statsmodels <https://www.statsmodels.org/>`__

Bugs
~~~~

Please report any bugs you encounter through the github `issue tracker
<https://github.com/mwaskom/seaborn/issues>`_. It will be most helpful to
include a reproducible example on synthetic data or one of the example datasets
(accessed through :func:`load_dataset`). It is difficult to debug any issues
without knowing the versions of seaborn and matplotlib you are using, as well
as what `matplotlib backend
<https://matplotlib.org/faq/usage_faq.html#what-is-a-backend>`__ you are have active, so please include those in your bug report.

.. raw:: html

   </div>
