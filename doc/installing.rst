
Installing seaborn
------------------

To install the released version of seaborn, you can use ``pip`` or
``easy_install``, (i.e. ``pip install seaborn``). Another option would be to to
clone the `github repository <https://github.com/mwaskom/seaborn>`_ and install
with ``pip install``. Seaborn itself is pure Python, so installation should be
reasonably straightforward.

Dependencies 
~~~~~~~~~~~~

Because seaborn is a high-level package, it has a relatively large number of
required dependencies. For the most part these tools are part of the general
scientific Python ecosystem, and almost all can be easily installed with a
distribution such as `Anaconda <https://store.continuum.io/cshop/anaconda/>`_.
The two exceptions are ``husl`` and ``moss``, which provide utility support for the
aesthetics and statistics, respectively. Both are pure Python and easily
installable with ``pip``.

-  Python 2.7

-  `numpy <http://www.numpy.org/>`__

-  `scipy <http://www.scipy.org/>`__

-  `matplotlib <matplotlib.sourceforge.net>`__

-  `pandas <http://pandas.pydata.org/>`__

-  `statsmodels <http://statsmodels.sourceforge.net/>`__

-  `patsy <http://patsy.readthedocs.org/en/latest/>`__

-  `scikit-learn <http://scikit-learn.org>`__

-  `husl <https://github.com/boronine/pyhusl>`__

-  `moss <http://github.com/mwaskom/moss>`__

Testing
~~~~~~~

To test seaborn, run ``make test`` in the root directory of the source
distribution. This runs the unit test suite of the plotting utilities (which
can also be exercised separately by running ``nosetests``). It also runs the code
in the tutorial notebooks, comparing the output to what is stored in the
notebook files.

Bugs
~~~~

Please report any bugs you encounter through the github `issue tracker
<https://github.com/mwaskom/seaborn/issues?state=open>`_.

