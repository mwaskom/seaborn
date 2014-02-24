.. _facet_grid:

.. currentmodule:: seaborn

.. ipython:: python
   :suppress:

   from __future__ import print_function
   import numpy as np
   np.random.seed(sum(map(ord, "facet_grid")))
   np.set_printoptions(precision=4, suppress=True)
   import pandas as pd
   import seaborn as sns
   import matplotlib as mpl
   import matplotlib.pyplot as plt
   plt.close('all')

Plotting on data-aware grids
============================



To use these features, your data has to be in a Pandas DataFrame and it must take the form of what Hadley Whickam calls `"tidy" data <http://vita.had.co.nz/papers/tidy-data.pdf>`_. In brief, that means your dataframe should be structured such that each column is a variable and each row is an observation.

Subsetting data with :class:`FacetGrid`
----------------------------------

The :class:`FacetGrid` class is useful when you want to visualize the distribution of a variable or the relationship between multiple variables separately within subsets of your dataset. A :class:`FacetGrid` can be drawn with up to three dimension: ``row``, ``col``, and ``hue``. The first two have obvious correspondence with the resulting array of axes; think of the hue variable as a third dimension along a depth axis, where different levels are plotted with different colors.

This class is used by initializing a :class:`FacetGrid` object with a dataframe and the names of the variables that will form the row, column, or hue dimensions. These variables should be categorical or discrete, and then the data at each level of the variable will be used for a facet along that axis:

.. ipython:: python

    tips = pd.read_csv("https://raw2.github.com/mwaskom/seaborn-data/master/tips.csv")
    @savefig bare_grid.png
    g = sns.FacetGrid(tips, col="sex")

Initializing the grid like this sets up the matplotlib figure and axes, but doesn't draw anything on them.

The main approach for visualizing data on this grid is with the :meth:`FacetGrid.map` method. Provide it with a plotting function and the name(s) of variable(s) in the dataframe to plot:

.. ipython:: python

    g = sns.FacetGrid(tips, col="sex")
    @savefig tips_hist.png
    g.map(plt.hist, "tip");

This function will draw the figure and annotate the axes, hopefully producing a finished plot in one step. To make a relational plot, just pass multiple variable names. You can also provide keyword arguments, which will be passed to the plotting function:

.. ipython:: python

    g = sns.FacetGrid(tips, col="sex", hue="smoker")
    @savefig tips_scatter.png
    g.map(plt.scatter, "total_bill", "tip", alpha=.7);

There are several options for controlling the look of the grid that can be passed to the class constructor:

.. ipython:: python

    g = sns.FacetGrid(tips, row="sex", col="smoker", margin_titles=True)
    @savefig margin_titles.png
    g.map(sns.regplot, "size", "total_bill", color=".3", fit_reg=False, x_jitter=.1);

The size of the figure is set by providing the height of the facets and the aspect ratio:

.. ipython:: python

    g = sns.FacetGrid(tips, col="day", size=4, aspect=.5)
    @savefig facet_size.png
    g.map(sns.barplot, "sex", "total_bill", palette="Paired");

By default, the facets are plotted in the sorted order of the unique values for each variable, but you can specify an order:

.. ipython:: python

    g = sns.FacetGrid(tips, col="day", size=4, aspect=.5,
                      col_order=["Thur", "Fri", "Sat", "Sun"])
    @savefig facet_order.png
    g.map(sns.barplot, "sex", "size", palette="Pastel1");

If you have many levels of one variable, you can plot it along the columns but "wrap" them so that they span multiple rows. When doing this, you cannot use a ``row`` variable.

.. ipython:: python

    attend = pd.read_csv("https://raw2.github.com/mwaskom/seaborn-data/master/attention.csv").query("subject <= 12")
    g = sns.FacetGrid(attend, col="subject", col_wrap=4, size=2, ylim=(0, 10))
    @savefig column_wrap.png
    g.map(sns.pointplot, "solutions", "score", color=".3", ci=None);

One you've drawn a plot using :meth:`FacetGrid.map` (which can be called multiple times), you may want to adjust some aspects of the plot. You can do this by directly calling methods on the matplotlib `Figure` and `Axes` objects, which are stored as member attributes at ``fig`` and ``axes`` (a two-dimensional array), respectively.

There are also a number of methods on the :class:`FacetGrid` object for manipulating the figure at a higher level of abstraction. For example:

.. ipython:: python

    g = sns.FacetGrid(tips, row="sex", col="smoker", margin_titles=True, size=2.5)
    g.map(plt.scatter, "total_bill", "tip", color="#334488");
    g.set_axis_labels("Total bill (US Dollars)", "Tip (US Dollars)");
    g.set(xticks=[10, 30, 50], yticks=[2, 6, 10]);
    @savefig adjust_facet_grid.png
    g.fig.subplots_adjust(wspace=.02, hspace=.02);

In addition to using :class:`FacetGrid` directly, both the :func:`lmplot` and :func:`factorplot` function use it internally and return the :class:`FacetGrid` object they have plotted on for additional tweaking.
