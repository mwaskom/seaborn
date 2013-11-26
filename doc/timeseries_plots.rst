
Plotting statistical timeseries data
====================================


This notebook focuses on the ``tsplot`` function, which can be used to
plot statistical timeseries data. Timeseries data consists of values for
some dependent variable that are observed at several timepoints. In
general, these data are thought of as realizations from some continuous
process, so the ``tsplot`` function makess the visuzalization of that
process simple (though not strictly enforced). Importantly, the function
is concerned with plotting *statistical* timecourses: the expected
dataset is additionally structured into sets of observations for several
sampling *units*, such as subjects or neurons. In the presentation of
these data, the unit dimension is frequently collapsed across to plot
some measure of central tendency along with a representation of the
variability introduced by sampling and measurement error. ``tsplot`` is
capable of representing that variabilty with several sophisticated
methods that are appropriate in different circumstances.

.. code:: python

    import numpy as np
    import pandas as pd
    from scipy import stats, optimize
    import matplotlib.pyplot as plt
    import seaborn as sns
    np.random.seed(9221999)
    sns.set(palette="Set2")
Specifying input data with multidimensional arrays
--------------------------------------------------


We'll use several sets of fake data to demonstrate the ``tsplot``
functionality. The simplest dataset will be noisy observations from an
underlying sine wave model.

.. code:: python

    def sine_wave(n_x, obs_err_sd=1.5, tp_err_sd=.3):
        x = np.linspace(0, (n_x - 1) / 2, n_x)
        y = np.sin(x) + np.random.normal(0, obs_err_sd) + np.random.normal(0, tp_err_sd, n_x)
        return y
.. code:: python

    sines = np.array([sine_wave(31) for _ in range(20)])
``tsplot`` can accept input data from one of two structures. In the
first, as is the case here, a rectangular array-type object with
timepoints in the columns and sampling units in the rows can be passed.

.. code:: python

    sns.tsplot(sines);


.. image:: timeseries_plots_files/timeseries_plots_8_0.png


Note that this reflects a backwards-incompatible change from
``seaborn 0.1``, which requried an ``x`` positional argument. As
elaborated on below, in ``0.2`` and beyond the call

::

    sns.tsplot(x, data)

should be transformed to

::

    sns.tsplot(data, time=x)


Now let's create the second example dataset. Here we'll add an
additional dimension: ``condition``. This dataset will consist of three
random walks with different probabilities of increasing position at each
timepoint.

.. code:: python

    def random_walk(n, start=0, p_inc=.2):
        return start + np.cumsum(np.random.uniform(size=n) < p_inc)
.. code:: python

    starts = np.random.choice(range(4), 10)
    probs = [.1, .3, .5]
    walks = np.dstack([[random_walk(15, s, p) for s in starts] for p in probs])
If the input data are a three dimensional array, the third dimension is
assumed to correspond with condition and the traces are separated out
and separately colored.

.. code:: python

    sns.tsplot(walks);


.. image:: timeseries_plots_files/timeseries_plots_14_0.png


Although using arrays as input objects allows for a very compact
specification of a relatively complex plot, they lack semantic
information about what variables are represented on each dimension. You
can, however, pass that information as seen below. If you use ``Series``
objects, the names will be used to label the axes and legend. However,
any sequence will work.

.. code:: python

    step = pd.Series(range(1, 16), name="step")
    speed = pd.Series(["slow", "average", "fast"], name="speed")
    sns.tsplot(walks, time=step, condition=speed, value="position");


.. image:: timeseries_plots_files/timeseries_plots_16_0.png


For single-condition data, you can set the color of the plot with any
valid matplotlib color spec.

.. code:: python

    sns.tsplot(sines, color="indianred");


.. image:: timeseries_plots_files/timeseries_plots_18_0.png


For multi-condition data, you can additionally use any valid seaborn
palette spec.

.. code:: python

    sns.tsplot(walks, color="husl");


.. image:: timeseries_plots_files/timeseries_plots_20_0.png


If you are providing condition information, you can further use a
dictionary that maps condition names to colors.

.. code:: python

    color_map = dict(slow="indianred", average="darkseagreen", fast="steelblue")
    sns.tsplot(walks, condition=speed, color=color_map);


.. image:: timeseries_plots_files/timeseries_plots_22_0.png


Specifying input data with long-form DataFrames
-----------------------------------------------


There is also a second, substantially different way to pass data into
``tsplot``. If you use a ``DataFrame``, it is expected to be in
"long-form" ("tidy") organization with a single column containing all
observations of the dependent variable and other columns containing
information about the time, sampling unit, and (optionally) condition of
each observation.

Let's make a third dataset with two gamma PDF traces in this format.

.. code:: python

    def gamma_pdf(x, shape, coef, obs_err_sd=.1, tp_err_sd=.001):    
        y = stats.gamma(shape).pdf(x) * coef
        y += np.random.normal(0, obs_err_sd, 1)
        y += np.random.normal(0, tp_err_sd, len(x))
        return y
.. code:: python

    gammas = []
    n_units = 20
    params = [(5, 1), (8, -.5)]
    x = np.linspace(0, 15, 31)
    for s in xrange(n_units):
        for p, (shape, coef) in enumerate(params):
            y = gamma_pdf(x, shape, coef)
            gammas.append(pd.DataFrame(dict(condition=[["pos", "neg"][p]] * len(x),
                                            subj=["subj%d" % s] * len(x),
                                            time=x * 2,
                                            BOLD=y), dtype=np.float))
    gammas = pd.concat(gammas)
When using a DataFrame, you must provide the names for each of the
relevant columns as arguments to ``time=``, ``unit=``, etc.

.. code:: python

    sns.tsplot(gammas, time="time", unit="subj", condition="condition", value="BOLD");


.. image:: timeseries_plots_files/timeseries_plots_28_0.png


Although this is somewhat more verbose, it produces a plot that has rich
semantic information with no additional effort. Everthing else you've
learned so far works with this style, so you can specify the colors in
any way you please.

.. code:: python

    color_map = dict(pos="indianred", neg="steelblue")
    ax = sns.tsplot(gammas, time="time", unit="subj", condition="condition", value="BOLD", color=color_map)
    ax.set_xlabel("time (seconds)");


.. image:: timeseries_plots_files/timeseries_plots_30_0.png


Becaues the confidence intervals are generated with a bootstrapping
procedure, you can pass in any arbitrary estimator to collapse over the
unit dimension. For instance, you may want to use the median instead of
the mean.

.. code:: python

    sns.tsplot(sines, estimator=np.median, color="#F08080");


.. image:: timeseries_plots_files/timeseries_plots_32_0.png


By default, the 68% confidence interval is plotted, which corresponds to
the standard error of the estimator. However, it's easy to change this.

.. code:: python

    pal = sns.dark_palette("cornflowerblue", 3)
    sns.tsplot(walks, ci=95, color=pal);


.. image:: timeseries_plots_files/timeseries_plots_34_0.png


If you want to change other aesthetics of the plot, you can either pass
additional keyword arguments (which get passed to the main call to
``plt.plot()`` that draws the central tendency, or you can provide a
dictionary to the ``err_kws`` parameter, and those arguments will be
used for the error plot.

.. code:: python

    sns.tsplot(gammas, time="time", unit="subj", condition="condition", value="BOLD",
               ci=95, color="muted", linewidth=2.5, err_kws={"alpha": .3});


.. image:: timeseries_plots_files/timeseries_plots_36_0.png


The entire plot is constrained within a single axis, and you can provide
an existing axis to plot into.

.. code:: python

    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    c1, c2 = sns.color_palette("Dark2", 2)
    sns.tsplot(sines, color=c1, ax=ax1)
    sns.tsplot(-sines, color=c2, ax=ax2);


.. image:: timeseries_plots_files/timeseries_plots_38_0.png


Different approaches to representing estimator variability
----------------------------------------------------------


Because of measurement and sampling error, the mean (or other aggregate
value) at each time point is only an estimate of the true value. It is
important to communicate this variability in a way that accurately
represents the precision of your estimate and facilitates comparisons,
for instance, between different conditions or against a baseline value.
``tsplot`` can visualize the uncertainty in a variety of ways. Each has
advantages and disadvantages, so choose the approach (or set of
approaches) that is best suited to what you want to communicate with
each plot.

Visualizing uncertainty at each observation with error bars
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


By default ``tsplot`` draws *confidence bands*, as they help emphaisze
the underyling trend in the data. However, a somewhat more common
approach is to draw an error bar with the width of some confidence
interval at the point of each observation:

.. code:: python

    sns.tsplot(sines, err_style="ci_bars");


.. image:: timeseries_plots_files/timeseries_plots_43_0.png


It's also not actually necessary to plot the linear interpolation
between the central tendency estimates. This is arguably a more pure
approach, although it sacrifices some visual immediacy.

.. code:: python

    ax = sns.tsplot(sines, err_style="ci_bars", interpolate=False);


.. image:: timeseries_plots_files/timeseries_plots_45_0.png


Perhaps the optimal approach is to fit a statistical model to the data
and then plot that on top of point-estimates and confidence bars.

.. code:: python

    ax = sns.tsplot(sines, err_style="ci_bars", interpolate=False)
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, sines.shape[1])
    (a, b), _ = optimize.leastsq(lambda (a, b): sines.mean(0) - (np.sin(x / b) + a), (0, 2))
    xx = np.linspace(xmin, xmax, 100)
    plt.plot(xx, np.sin(xx / b) + a, c="#444444");


.. image:: timeseries_plots_files/timeseries_plots_47_0.png


A problem with the error bars style is that it can become confusing when
you have multiple traces on the same plot, as it is difficult to
visualize the extent of the overlap.

.. code:: python

    sns.tsplot(walks, err_style="ci_bars", ci=95);


.. image:: timeseries_plots_files/timeseries_plots_49_0.png


Drawing comparisons with overlapping error bands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


This kind of comparison lends itself well to the tranlucent error bands,
where it is easy to see the overlap in as the bands get darker.

.. code:: python

    sns.tsplot(walks, err_style="ci_band", ci=95);


.. image:: timeseries_plots_files/timeseries_plots_52_0.png


Representing a distribution with multiple confidence intervals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


This starts to give the impression that there is some region around our
central estimate that we consider to be reliable.

But, it still binarizes "trustworthy" and "untrustworthy", when in
reality we have a continuous distribution giving the likelihood of
observing the central value with repeated samples.

To get a better feel for the shape of this distribution, we can use the
fact that the error bands are translucent and stack several on top of
each other by supplying a list of confidence intervals.

.. code:: python

    cis = np.linspace(95, 10, 4)
    sns.tsplot(sines, err_style="ci_band", ci=cis);


.. image:: timeseries_plots_files/timeseries_plots_55_0.png


This can make for a very informative plot, but it can get cluttered when
you have multiple overlapping traces.

.. code:: python

    sns.tsplot(walks, ci=cis);


.. image:: timeseries_plots_files/timeseries_plots_57_0.png


Representing the distribution of bootstrap resamples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Since the confidence intervals are just measures of the bootstrap
distribution, we may want to try and represent that distribution
directly.

One approach would be to plot the central tendency from each bootstrap
resample directly. By default the function performs a relatively large
number of resamples to obtain a stable estimate of the confidence
intervals. You'll want to use fewer to avoid having to plot thousands of
lines.

.. code:: python

    sns.tsplot(sines, err_style="boot_traces", n_boot=500);


.. image:: timeseries_plots_files/timeseries_plots_60_0.png


It can be somewhat hard to get the parameters right to represent the
uncertainty well, but with tweaking it can do a very good job of
presenting the logic of the bootstrap and helping to characterize the
uncertainty distribution.

.. code:: python

    sns.tsplot(gammas, time="time", unit="subj", condition="condition", value="BOLD",
               ci=95, color="deep", err_style="boot_traces", n_boot=500);


.. image:: timeseries_plots_files/timeseries_plots_62_0.png


Plotting a smooth estimate of the bootstrap density
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Although the overlapping traces give an impression of the density of the
bootstrap distribution, it is not directly encoded.

A related approach uses a kernel density estimate over the bootstrap
distribution and encodes the density at each ``(x, y)`` coordinate in
the plot with an alpha value.

It's not clear that this has any particular advantage over the other
approaches, but some might prefer it.

.. code:: python

    sns.tsplot(gammas, time="time", unit="subj", condition="condition", value="BOLD", err_style="boot_kde");


.. image:: timeseries_plots_files/timeseries_plots_65_0.png


Visualizing the data for each sampling unit
-------------------------------------------


The above methods compress the information in the data to visually
present a statisical inference about the central tendency. It is often
the case, though, that you will want to visualize the data for each
sampling unit at some point in your analysis. Although this does not
present the most informative production graphics, it can be very
important in the early stages as you being to understand the structure
of the data.

.. code:: python

    sns.tsplot(sines, err_style="unit_traces");


.. image:: timeseries_plots_files/timeseries_plots_68_0.png


You may want to make the trace for each unit a different color, to make
the structure more interpretable.

.. code:: python

    sns.tsplot(sines, err_style="unit_traces", err_palette=sns.dark_palette("crimson", len(sines)), color="k");


.. image:: timeseries_plots_files/timeseries_plots_70_0.png


If you pass a list of error styles to ``tsplot``, it will compose them

.. code:: python

    sns.tsplot(gammas[gammas.condition == "pos"], time="time", unit="subj", value="BOLD",
               err_style=["unit_traces", "ci_band"]);


.. image:: timeseries_plots_files/timeseries_plots_72_0.png


Finally, it's also possibly to plot each individual observation with a
point, rather than joining them. This is more useful for the gestalt it
presents than as a quantitative visualization but you may prefer it.

.. code:: python

    sns.tsplot(sines, err_style="unit_points", color="mediumpurple");


.. image:: timeseries_plots_files/timeseries_plots_74_0.png





