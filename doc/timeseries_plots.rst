
Representing variability in timeseries plots
============================================


This notebook demonstrates the functionality of the ``tsplot`` function
that is part of the `seaborn <https://github.com/mwaskom/seaborn>`__
package.

Specifically, it focuses on the many ways to represent the uncertainty
around our estimate of the signal.

.. code:: python

    import numpy as np
    from numpy.random import randn, rand
    import matplotlib.pyplot as plt
    import seaborn as sns
First let's generate some fake data. We'll use an underlying sine wave
model

.. code:: python

    n_pts = 29
    x = np.linspace(0, 14, n_pts)
    true_data = np.sin(x)
Now, "draw" a sample for each subject with different sources of noise

.. code:: python

    np.random.seed(9221999)
    n_subjs = 20
    subj_noise = rand(20) * 2
    subj_data = np.array([true_data +  # real signal
                          randn() +    # subject specific offset from real signal
                          randn(n_pts) * (subj_noise[s])  # sample specific error with subject specific variance
                          for s in range(n_subjs)])
Uncertainty of an estimator
---------------------------


Error bars
~~~~~~~~~~


The most common representation of timeseries data linearly interpolates
between the estimate of central tendency at each timepoint and uses
error bars to represent the variability of this estimate (the standard
error).

.. code:: python

    sns.tsplot(x, subj_data, err_style="ci_bars");


.. image:: timeseries_plots_files/timeseries_plots_10_0.png


Although the linear interpolation makes the underlying trend in the
timeseries visually apparent, some might argue it misrepresents the data
by presenting a model that we have not verified. For a more limited
presentation, you could plot just the point estimate.

.. code:: python

    sns.tsplot(x, subj_data, err_style="ci_bars", interpolate=False);


.. image:: timeseries_plots_files/timeseries_plots_12_0.png


We're actually getting the standard error by bootstrapping and taking
the 68%CI of the boostratrap distribution. We could also show a
different confidence interaval this way.

.. code:: python

    sns.tsplot(x, subj_data, err_style="ci_bars", ci=95);


.. image:: timeseries_plots_files/timeseries_plots_14_0.png


This isn't directly related to the point, but we can bootstrap and plot
different estimators with this function.

.. code:: python

    sns.tsplot(x, subj_data, err_style="ci_bars", label="mean")
    sns.tsplot(x, subj_data, err_style="ci_bars", estimator=np.median, label="median")
    sns.tsplot(x, subj_data, err_style="ci_bars", estimator=np.median, smooth=True, interpolate=False, label="median (smoothed)")
    plt.legend(loc=0);


.. image:: timeseries_plots_files/timeseries_plots_16_0.png


The problem with the error bars approach is that it can become confusing
when you have multiple traces on the same plot.

.. code:: python

    other_data = subj_data + randn(29) / 2 + (x / 4) - 1
    sns.tsplot(x, subj_data, err_style="ci_bars", label="group1")
    sns.tsplot(x, other_data, err_style="ci_bars", label="group2")
    plt.legend(loc=0);


.. image:: timeseries_plots_files/timeseries_plots_18_0.png


Error bands
~~~~~~~~~~~


A better approach here is to use solid error bands that are also
linearly interpolated.

.. code:: python

    sns.tsplot(x, subj_data, err_style="ci_band", label="group1")
    sns.tsplot(x, other_data, err_style="ci_band", label="group2")
    plt.legend(loc=0);


.. image:: timeseries_plots_files/timeseries_plots_21_0.png


Multiple confidence intervals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


This starts to give the impression that there is some region around our
point estimate that we consider to be reliable.

But, it still binarizes "trustworthy" and "untrustworthy" in a way that
doesn't fully represent the variance of our estimate.

To get a better feel for the shape for the distribution, we can use the
fact that the error bands are translucent and stack several on top of
each other by supplying a list of confidence intervals.

.. code:: python

    color = sns.color_palette()[3]
    cis = np.linspace(95, 10, 4)
    sns.tsplot(x, subj_data, err_style="ci_band", ci=cis, color=color);


.. image:: timeseries_plots_files/timeseries_plots_24_0.png


If you want to plot multiple traces this way, it may be best to
intervleave them manually

.. code:: python

    c1, c2 = sns.color_palette("husl", 2)
    for ci in np.linspace(95, 10, 4):
        sns.tsplot(x, subj_data, err_style="ci_band", ci=ci, color=c1)
        sns.tsplot(x, other_data, err_style="ci_band", ci=ci, color=c2)


.. image:: timeseries_plots_files/timeseries_plots_26_0.png


Bootstrap traces
~~~~~~~~~~~~~~~~


Since the confidence intervals are just measures of the bootstrap
distribution, we may want to try and represent that distribution
directly.

One approach would be to plot traces for a random subset of bootstrap
samples. Using a relatively low alpha means areas with higher density
will be more saturated:

.. code:: python

    sns.tsplot(x, subj_data, err_style="boot_traces");


.. image:: timeseries_plots_files/timeseries_plots_29_0.png


It can be somewhat hard to get the parameters right to represent the
uncertainty well, but some people may find this an improvement.

You can also plot multiple traces this way.

.. code:: python

    sns.tsplot(x, subj_data, err_style="boot_traces", label="group1")
    sns.tsplot(x, other_data, err_style="boot_traces", label="group2")
    plt.legend(loc=0);


.. image:: timeseries_plots_files/timeseries_plots_31_0.png


Bootstrap density
~~~~~~~~~~~~~~~~~


Although the alpha values on these traces give the impression of the
density, it is still not directly color-encoded.

To do that, we can use a kernel density estimate over the bootstrap
distribution, and set those values to the alpha channel of an RGB image
with one color.

I do not believe I have perfectly worked out the parameters for how this
density should look, but it still represents the variability
appropropriately.

.. code:: python

    sns.tsplot(x, subj_data, err_style="boot_kde");


.. image:: timeseries_plots_files/timeseries_plots_34_0.png


Because the KDE saturates at the maximum density, you may want to change
the color of the central tendency line.

.. code:: python

    kde_color, line_color = "gray", "black"
    ax = sns.tsplot(x, subj_data, err_style="boot_kde", color=kde_color)
    ax.lines[-1].set_color(line_color);


.. image:: timeseries_plots_files/timeseries_plots_36_0.png


You may also want to use a white background with this error style

.. code:: python

    sns.set(style="whitegrid")
    sns.tsplot(x, subj_data, err_style="boot_kde")
    sns.set(style="darkgrid")


.. image:: timeseries_plots_files/timeseries_plots_38_0.png


Plotting model-based predictions over data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


These methods will also have differing levels of usefulness while
plotting a model over the observed data.

.. code:: python

    # Fit a simple sine wave model to the data
    from scipy import optimize
    sin_func = lambda p, x: np.sin(x) * p[0] + p[1]
    err_func = lambda p, x, y: sin_func(p, x) - y
    p, e = optimize.leastsq(err_func, [1, 0], (x, subj_data.mean(axis=0)))
    model = sin_func(p, x)
.. code:: python

    f, axes = plt.subplots(3, 2, figsize=(14, 14), sharex=True, sharey=True)
    axes = np.ravel(axes)
    for i, style in enumerate(["ci_bars (no interpolation)", "ci_bars", "ci_band",
                               "ci_band (stacked)", "boot_traces", "boot_kde"]):
        ax = axes[i]
        if i == 0:
            sns.tsplot(x, subj_data, err_style="ci_bars", label="data", interpolate=False, ax=ax)
        elif "stacked" in style:
            cis = np.linspace(95, 10, 4)
            sns.tsplot(x, subj_data, err_style="ci_band", ci=cis, label="data", ax=ax)
        else:
            sns.tsplot(x, subj_data, err_style=style, label="data", ax=ax)
        ax.plot(x, model, "black", label="model")
        ax.set_title(style)
        ax.set_xlim(x.min(), x.max())
        ax.legend()
    plt.tight_layout();


.. image:: timeseries_plots_files/timeseries_plots_42_0.png


Visualizing sample variance
---------------------------


Observation traces
~~~~~~~~~~~~~~~~~~


Now that we're plotting all these lines, you might wonder why we're not
looking at the original data.

While you may find that this clutters your production graphics, it is
very important to do while exploring your data.

.. code:: python

    sns.tsplot(x, subj_data, err_style="obs_traces");


.. image:: timeseries_plots_files/timeseries_plots_46_0.png


You may want to make the trace for each observation a different color,
to make the variance structure more obvious.

.. code:: python

    sns.tsplot(x, subj_data, err_style="obs_traces", err_palette="husl", color="#222222");


.. image:: timeseries_plots_files/timeseries_plots_48_0.png


Also, there is no reason we can't use several of these approaches at
once.

.. code:: python

    sns.tsplot(x, subj_data, err_style=["obs_traces", "ci_band"]);


.. image:: timeseries_plots_files/timeseries_plots_50_0.png


Observation points
~~~~~~~~~~~~~~~~~~


Some people do not like the linear interpolation, particularly on the
individual observation plots which are pretty noisy.

So, you might want to plot the individual data points as points.

This is not my favorite style, but some may prefer it.

.. code:: python

    f, axes = plt.subplots(1, 2, figsize=(17, 6))
    sns.tsplot(x, subj_data, err_style="obs_points", ax=axes[0])
    sns.tsplot(x, subj_data, err_style=["obs_points", "ci_band"], ax=axes[1]);


.. image:: timeseries_plots_files/timeseries_plots_53_0.png


Here the ``err_palette`` option could be useful.

.. code:: python

    sns.tsplot(x, subj_data, err_style=["obs_points", "ci_band"],
               err_palette="husl", color="#222222");


.. image:: timeseries_plots_files/timeseries_plots_55_0.png


.. code:: python

    