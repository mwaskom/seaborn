.. _installing:

.. currentmodule:: seaborn

Installing and getting started
------------------------------

Official releases of seaborn can be installed from `PyPI <https://pypi.org/project/seaborn/>`_::

    pip install seaborn

The basic invocation of `pip` will install seaborn and, if necessary, its mandatory dependencies.
It is possible to include optional dependencies that give access to a few advanced features::

    pip install seaborn[stats]

The library is also included as part of the `Anaconda <https://repo.anaconda.com/>`_ distribution,
and it can be installed with `conda`::

    conda install seaborn

As the main Anaconda repository can be slow to add new releases, you may prefer using the
`conda-forge <https://conda-forge.org/>`_ channel::

    conda install seaborn -c conda-forge

Dependencies
~~~~~~~~~~~~

Supported Python versions
^^^^^^^^^^^^^^^^^^^^^^^^^

- Python 3.8+

Mandatory dependencies
^^^^^^^^^^^^^^^^^^^^^^

- `numpy <https://numpy.org/>`__

- `pandas <https://pandas.pydata.org/>`__

- `matplotlib <https://matplotlib.org>`__

Optional dependencies
^^^^^^^^^^^^^^^^^^^^^

- `statsmodels <https://www.statsmodels.org/>`__, for advanced regression plots

- `scipy <https://www.scipy.org/>`__, for clustering matrices and some advanced options

- `fastcluster <https://pypi.org/project/fastcluster/>`__, faster clustering of large matrices

Quickstart
~~~~~~~~~~

Once you have seaborn installed, you're ready to get started.
To test it out, you could load and plot one of the example datasets::

    import seaborn as sns
    df = sns.load_dataset("penguins")
    sns.pairplot(df, hue="species")

If you're working in a Jupyter notebook or an IPython terminal with
`matplotlib mode <https://ipython.readthedocs.io/en/stable/interactive/plotting.html>`_
enabled, you should immediately see :ref:`the plot <scatterplot_matrix>`.
Otherwise, you may need to explicitly call :func:`matplotlib.pyplot.show`::

    import matplotlib.pyplot as plt
    plt.show()

While you can get pretty far with only seaborn imported, having access to
matplotlib functions is often useful. The tutorials and API documentation
typically assume the following imports::

    import numpy as np
    import pandas as pd

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    import seaborn as sns
    import seaborn.objects as so

Debugging install issues
~~~~~~~~~~~~~~~~~~~~~~~~

The seaborn codebase is pure Python, and the library should generally install
without issue. Occasionally, difficulties will arise because the dependencies
include compiled code and link to system libraries. These difficulties
typically manifest as errors on import with messages such as ``"DLL load
failed"``. To debug such problems, read through the exception trace to
figure out which specific library failed to import, and then consult the
installation docs for that package to see if they have tips for your particular
system.

In some cases, an installation of seaborn will appear to succeed, but trying
to import it will raise an error with the message ``"No module named
seaborn"``. This usually means that you have multiple Python installations on
your system and that your ``pip`` or ``conda`` points towards a different
installation than where your interpreter lives. Resolving this issue
will involve sorting out the paths on your system, but it can sometimes be
avoided by invoking ``pip`` with ``python -m pip install seaborn``.

Getting help
~~~~~~~~~~~~

If you think you've encountered a bug in seaborn, please report it on the
`GitHub issue tracker <https://github.com/mwaskom/seaborn/issues>`_.
To be useful, bug reports must include the following information:

- A reproducible code example that demonstrates the problem
- The output that you are seeing (an image of a plot, or the error message)
- A clear explanation of why you think something is wrong
- The specific versions of seaborn and matplotlib that you are working with

Bug reports are easiest to address if they can be demonstrated using one of the
example datasets from the seaborn docs (i.e. with :func:`load_dataset`).
Otherwise, it is preferable that your example generate synthetic data to
reproduce the problem. If you can only demonstrate the issue with your
actual dataset, you will need to share it, ideally as a csv.

If you've encountered an error, searching the specific text of the message
before opening a new issue can often help you solve the problem quickly and
avoid making a duplicate report.

Because matplotlib handles the actual rendering, errors or incorrect outputs
may be due to a problem in matplotlib rather than one in seaborn. It can save time
if you try to reproduce the issue in an example that uses only matplotlib,
so that you can report it in the right place. But it is alright to skip this
step if it's not obvious how to do it.

General support questions are more at home on `stackoverflow
<https://stackoverflow.com/questions/tagged/seaborn/>`_, where there is a
larger audience of people who will see your post and may be able to offer
assistance. Your chance of getting a quick answer will be higher if you include
`runnable code <https://stackoverflow.com/help/minimal-reproducible-example>`_,
a precise statement of what you are hoping to achieve, and a clear explanation
of the problems that you have encountered.
