
Graphical representations of linear models
==========================================


This notebook is intended to provide examples for how four functions in
the `seaborn <https://github.com/mwaskom/seaborn>`__ plotting library,
``regplot``, ``corrplot``, ``lmplot``, and ``coefplot``, can be used to
informatively visualize the relationships between variables in a
dataset. The functions are intended to produce plots that are attractive
and that can be specified without much work. The goal of these
visualizations, which the functions attempt to make achievable, is to
emphasize important comparisons in the dataset and provide supporting
information without distraction.

These functions are a a bit higher-level than the ones covered in the
`distributions <http://nbviewer.ipython.org/urls/raw.github.com/mwaskom/seaborn/master/examples/plotting_distributions.ipynb>`__
and
`timeseries <http://nbviewer.ipython.org/urls/raw.github.com/mwaskom/seaborn/master/examples/timeseries_plots.ipynb>`__
tutorials. Instead of plotting into an existing axis, these functions
expect to have the whole figure to themselves, and they are frequently
composed of multiple axes.

All of the functions covered here are Pandas-aware, and both ``lmplot``
and ``coefplot`` require that the data be in a Pandas dataframe.

.. code:: python

    import seaborn as sns
    import numpy as np
    from numpy import mean
    from numpy.random import randn
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import statsmodels.formula.api as sm
.. code:: python

    import pandas as pd
    from scipy import stats
.. code:: python

    sns.set(palette="Purples_r")
    mpl.rc("figure", figsize=(5, 5))
    np.random.seed(9221999)
Plotting a simple regression: ``regplot``
-----------------------------------------


First, we will show how to visualize the relationship between two
variables. To do so, we'll create a very simple fake data set.

.. code:: python

    x = randn(50)
    y = x + randn(50) 
The ``regplot`` function accepts two arrays of data and draws a
scatterplot of the relationship. It also fits a regression line to the
data, bootstraps the regression to get a confidence interval (95% by
default), and plots the marginal distributions of the two variables. It
does all this with a very simple function call:

.. code:: python

    sns.regplot(x, y)


.. image:: linear_models_files/linear_models_9_0.png


The function also works with pandas ``DataFrame`` objects, in which case
the ``x`` and ``y`` values should be strings. Calling the function this
way gets axis labels for free (otherwise you can use the ``xlabel`` and
``ylabel`` keyword arguments).

.. code:: python

    df = pd.DataFrame(np.transpose([x, y]), columns=["X", "Y"])
    sns.regplot("X", "Y", df)


.. image:: linear_models_files/linear_models_11_0.png


If you like, you can compute a different confidence interval, or even
turn off the confidence bands altogether. You can also change the
general color of the plot at a high level.

.. code:: python

    sns.regplot("X", "Y", df, ci=None, color="slategray")


.. image:: linear_models_files/linear_models_13_0.png


You'll note that the a pearson correlation statistic is automatically
computed and displayed in the scatterplot. If your data are not normally
distributed, you can provide a different function to calculate a
correlation metric; anything that takes two arrays of data and returns a
``stat`` numeric or ``(stat, p)`` tuple will work.

It would probably also be nice to print the intercept and slope of the
regression, which is something I hope to add when I get a chance.

.. code:: python

    sns.regplot("X", "Y", df, corr_func=stats.spearmanr)


.. image:: linear_models_files/linear_models_15_0.png


We hope you find it convenient that the default behavior for the fit
statistic is to use the function name, but sometimes you might want to
use a different string. That's what the ``func_name`` keyword argument
is for.

.. code:: python

    r2 = lambda x, y: stats.pearsonr(x, y)[0] ** 2
    sns.regplot("X", "Y", df, corr_func=r2, func_name="$R^2$", color="seagreen")


.. image:: linear_models_files/linear_models_17_0.png


For finer control over the individual aspects of the plot, you can pass
dictionaries with keyword arguments for the underlying seaborn or
matplotlib functions.

.. code:: python

    c1, c2, c3 = sns.color_palette("bone_r", 3)
    sns.regplot("X", "Y", df, ci=68,
                reg_kws={"color": c2},
                scatter_kws={"marker": "D", "color": c3},
                text_kws={"family": "serif", "size": 12},
                dist_kws={"fit": stats.norm, "kde": False, "color": c1})


.. image:: linear_models_files/linear_models_19_0.png


Plotting linear relationships in complex datasets: ``corrplot`` and ``lmplot``
------------------------------------------------------------------------------


Now let's explore a more complex dataset. We'll use the ``tips`` data
that is provided with R's ``reshape2`` package. This is a good example
dataset in that it provides several quantitative and qualitative
variables in a tidy format, but there aren't actually any interesting
interactions so I am open to other suggestions for different data sets
to use here.

.. code:: python

    tips = pd.read_csv("tips.csv")
    tips["big_tip"] = tips.tip > (.2 * tips.total_bill)
    tips["smoker"] = tips["smoker"] == "Yes"
    tips["female"] = tips["sex"] == "Female"
    mpl.rc("figure", figsize=(7, 7))
Plotting correlation heatmaps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Once you have a tidy dataset, often the first thing you want is a very
high-level summary of the relationships between the variables.
Correlation matrix heatmaps can be very useful for this purpose. The
``corrplot`` function not only plots a color-coded correlation matrix,
but it will also obtain a *p* value for each correlation using a
permutation test to give you some indication of the significance of each
relationship while correcting for multiple comparisons in an intelligent
way.

.. code:: python

    sns.corrplot(tips);


.. image:: linear_models_files/linear_models_25_0.png


Note that if you have a huge dataset, the permutation test will take a
while. Of course, if you have a huge dataset, *p* values will not be
particularly relevant, so you can turn off the significance testing.

.. code:: python

    sns.corrplot(tips, sig_stars=False);


.. image:: linear_models_files/linear_models_27_0.png


You can also choose the colormap and the range it corresponds to, but
choose wisely! Here we might just want a sequential colormap, as the
correlations are mostly positive. By default the colormap is centered on
zero and covers the range of the data (plus a bit), but you can also
manually give the range.

Don't even try using the "Jet" map; you'll get a ``ValueError``.

It's additionally possible to control the direction of the significance
test; in this case, an upper-tail test would be appropriate.

.. code:: python

    sns.corrplot(tips, sig_tail="upper", cmap="PuRd", cmap_range=(-.2, .8));


.. image:: linear_models_files/linear_models_29_0.png


Complex regression scatterplots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


The ``lmplot`` function provides a more general interface for plotting
linear relationships in a complex set of data. In its most basic usage,
it does the same thing as the core of the ``regplot`` function. Note
that ``lmplot`` only works with DataFrames.

.. code:: python

    mpl.rc("figure", figsize=(5, 5))
.. code:: python

    sns.lmplot("total_bill", "tip", tips)


.. image:: linear_models_files/linear_models_33_0.png


The advantage to using ``lmplot`` over ``regplot`` is that you can
visualize linear relationships among subsets of a larger data structure.
There are a few ways to do this; but perhaps the most amenable to direct
comparisons involves separating subgroups by color.

.. code:: python

    sns.lmplot("total_bill", "tip", tips, color="time")


.. image:: linear_models_files/linear_models_35_0.png


The default color palette is ``husl``, but you can use any of the
``seaborn`` color palettes for the color factor.

.. code:: python

    sns.lmplot("total_bill", "tip", tips, color="day", palette="muted", ci=None)


.. image:: linear_models_files/linear_models_37_0.png


It's not actually neccesary to fit a regression line to the data, if you
don't want to. (Although I need to fix things so that the legend shows
up when using color grouping -- this doesn't work at the moment).

.. code:: python

    sns.lmplot("total_bill", "tip", tips, fit_reg=False)


.. image:: linear_models_files/linear_models_39_0.png


Higher-order trends
~~~~~~~~~~~~~~~~~~~


You can also fit higher-order polynomials. Although there is not such a
trend in this dataset, let's invent one to see what that might look
like.

.. code:: python

    tips["tip_sqr"] = tips.tip ** 2
    sns.lmplot("total_bill", "tip_sqr", tips, order=2)


.. image:: linear_models_files/linear_models_42_0.png


Logistic Regression
~~~~~~~~~~~~~~~~~~~


What if we want to fit a model where the response variable is
categorical? (At the moment, it must be binary and numeric, so {0, 1}
and {True, False} both work).

We can use linear regression to get a reasonable estimate of the
influence our predictor variable has. For instance, does group size
influene whether diners leave a relatively "big" tip?

.. code:: python

    sns.lmplot("size", "big_tip", tips)


.. image:: linear_models_files/linear_models_45_0.png


This plot suggets that big groups are relatively less likely to leave a
big tip, but it has a few problems. The first is that (especially in our
case where the predictor varible is discrete) the individual
observations are all plotted on top of each other and it is hard to tell
the joint distributions of observations. We can address this issue by
adding a bit of jitter to the scatter plot.

.. code:: python

    sns.lmplot("size", "big_tip", tips, x_jitter=.3, y_jitter=.075)


.. image:: linear_models_files/linear_models_47_0.png


A more fundamental problem follows from using basic linear regression
with a binary response variable. The regression line implies that the
probabilitiy of a group of 6 diners tipping over 20% is less than 0. Of
course, that doesn't make sense, which is why logistic regression was
invented. ``lmplot`` can likewise plot a logistic curve over the data.
You might want to use fewer bootstrap iterations, as the logistic
regression fit is much more computationally intensive.

.. code:: python

    sns.lmplot("size", "big_tip", tips, x_jitter=.3, y_jitter=.075, logistic=True, n_boot=1000)


.. image:: linear_models_files/linear_models_49_0.png


Faceted plots
~~~~~~~~~~~~~


There are several other ways to visualize fits of the model to
sub-groups in the data.

You can also separate out factors into facet plots on the columns or
rows.

.. code:: python

    sns.lmplot("total_bill", "tip", tips, col="sex")


.. image:: linear_models_files/linear_models_52_0.png


Which doesn't mean you can't keep an association between colors and
factors

.. code:: python

    sns.lmplot("total_bill", "tip", tips, color="sex", col="sex")


.. image:: linear_models_files/linear_models_54_0.png


By default, the same ``x`` and ``y`` axes are used for all facets, but
you can turn this off if you have a big difference in intercepts that
you don't care about.

.. code:: python

    sns.lmplot("total_bill", "tip", tips, col="sex", sharey=False)


.. image:: linear_models_files/linear_models_56_0.png


Plotting with discrete predictor variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Sometimes you will want to plot data where the independent variable is
discrete. Although this works fine out of the box:

.. code:: python

    sns.lmplot("size", "tip", tips)


.. image:: linear_models_files/linear_models_59_0.png


And can be improved with a bit of jitter:

.. code:: python

    sns.lmplot("size", "tip", tips, x_jitter=.15)


.. image:: linear_models_files/linear_models_61_0.png


It might be more informative to estimate the central tendency of each
bin. This is easy to do with the ``x_estimator`` argument. Just pass any
function that aggregates a vector of data into one estimate. The
estimator will be bootstrapped and a confidence interval will be plotted
-- 95% by default, as in other cases within these functions.

.. code:: python

    sns.lmplot("size", "tip", tips, x_estimator=mean)


.. image:: linear_models_files/linear_models_63_0.png


Sometimes you may want to plot binary factors and not extrapolate with
the fitted line beyond your data points. (Here the fitted line doesn't
make all that much sense for extrapolating within the range of the data
either, but it does make the trend more visually obvious). Note that at
the moment the independent variable must be "quantitative" (so,
numerical or boolean typed), but in the future binary factors with
string variables will be implemented.

.. code:: python

    sns.lmplot("smoker", "size", tips, ci=None, x_estimator=mean, x_ci=68, truncate=True)


.. image:: linear_models_files/linear_models_65_0.png


You can plot data on both the rows and columns to compare multiple
factors at once.

.. code:: python

    sns.lmplot("total_bill", "tip", tips, row="sex", col="day", size=4)


.. image:: linear_models_files/linear_models_67_0.png


And, of course, you can compose the color grouping with facets as well
to facilitate comparisons within a complicated model structure.

.. code:: python

    sns.lmplot("total_bill", "tip", tips, col="day", color="sex", size=4)


.. image:: linear_models_files/linear_models_69_0.png


If you have many of levels for some factor (say, your population of
subjects), you may want to "wrap" the levels so that the plot is not too
wide:

.. code:: python

    sns.lmplot("total_bill", "tip", tips, ci=None, col="day", col_wrap=2, color="day", size=4)


.. image:: linear_models_files/linear_models_71_0.png


Plotting partial regressions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Finally, let's create a fake dataset with three variables. We'll
generate two of them by adding noise to the third:

.. code:: python

    df = pd.DataFrame(dict(a=randn(50)))
    df["b"] = df.a + randn(50) / 2
    df["c"] = df.a + randn(50) / 2 + 3
Because of how we generated the data, these two variables are now
related:

.. code:: python

    sns.lmplot("b", "c", df)


.. image:: linear_models_files/linear_models_76_0.png


However, we could remove the influence of the third variable to see if
any residual relationship exists:

.. code:: python

    sns.lmplot("b", "c", df, x_partial="a")


.. image:: linear_models_files/linear_models_78_0.png


Plotting linear model parameters: ``coefplot``
----------------------------------------------


Although the above plots can be very helpful for understanding the
structure of your data, they fail with more than about 4 variables or
with more than one continuous predictor. To visually summarize this kind
of model, it can be helpful to plot the point estimates for each
coefficient along with confidence intervals. The ``coefplot`` function
achieves this by using a
`Patsy <https://patsy.readthedocs.org/en/latest/>`__ formula
specification for the model structure.

.. code:: python

    mpl.rc("figure", figsize=(8, 5))
.. code:: python

    sns.coefplot("tip ~ day + time * size", tips)


.. image:: linear_models_files/linear_models_82_0.png


.. code:: python

    sns.coefplot("total_bill ~ day + time + smoker", tips, ci=68, palette="muted")


.. image:: linear_models_files/linear_models_83_0.png


When you have repeated measures in your dataset (e.g. an experiment
performed with multiple subjects), you can group by the levels of that
variable and plot the model coefficients within each group. Note that
the semantics of the resulting figure changes a little bit from the
example above.

.. code:: python

    sns.coefplot("tip ~ time * sex", tips, "size", intercept=True)


.. image:: linear_models_files/linear_models_85_0.png

