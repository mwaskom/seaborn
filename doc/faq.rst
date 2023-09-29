.. currentmodule:: seaborn

Frequently asked questions
==========================

This is a collection of answers to questions that are commonly raised about seaborn.

Getting started
---------------

.. _faq_cant_import:

I've installed seaborn, why can't I import it?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*It looks like you successfully installed seaborn by doing* `pip install seaborn` *but it cannot be imported. You get an error like "ModuleNotFoundError: No module named 'seaborn'" when you try.*

This is probably not a `seaborn` problem, *per se*. If you have multiple Python environments on your computer, it is possible that you did `pip install` in one environment and tried to import the library in another. On a unix system, you could check whether the terminal commands `which pip`, `which python`, and (if applicable) `which jupyter` point to the same `bin/` directory. If not, you'll need to sort out the definition of your `$PATH` variable.

Two alternate patterns for installing with `pip` may also be more robust to this problem:

- Invoke `pip` on the command line with `python -m pip install <package>` rather than `pip install <package>`
- Use `%pip install <package>` in a Jupyter notebook to install it in the same place as the kernel

.. _faq_import_fails:

I can't import seaborn, even though it's definitely installed!
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*You've definitely installed seaborn in the right place, but importing it produces a long traceback and a confusing error message, perhaps something like* `ImportError: DLL load failed: The specified module could not be found`.

Such errors usually indicate a problem with the way Python libraries are using compiled resources. Because seaborn is pure Python, it won't directly encounter these problems, but its dependencies (numpy, scipy, matplotlib, and pandas) might. To fix the issue, you'll first need to read through the traceback and figure out which dependency was being imported at the time of the error. Then consult the installation documentation for the relevant package, which might have advice for getting an installation working on your specific system.

The most common culprit of these issues is scipy, which has many compiled components. Starting in seaborn version 0.12, scipy is an optional dependency, which should help to reduce the frequency of these issues.

.. _faq_no_plots:

Why aren't my plots showing up?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*You're calling seaborn functions — maybe in a terminal or IDE with an integrated IPython console — but not seeing any plots.)*

In matplotlib, there is a distinction between *creating* a figure and *showing* it, and in some cases it's necessary to explicitly call :func:`matplotlib.pyplot.show` at the point when you want to see the plot. Because that command blocks by default and is not always desired (for instance, you may be executing a script that saves files to disk) seaborn does not deviate from standard matplotlib practice here.

Yet most of the examples in the seaborn docs do not have this line, because there are multiple ways to avoid needing it. In a Jupyter notebook with the `"inline" <https://ipython.readthedocs.io/en/stable/interactive/plotting.html#id1>`_ (default) or `"widget" <https://github.com/matplotlib/ipympl>`_ backends, :func:`matplotlib.pyplot.show` is automatically called after executing a cell, so any figures will appear in the cell's outputs. You can also activate a more interactive experience by executing `%matplotlib` in any Jupyter or IPython interface or by calling :func:`matplotlib.pyplot.ion` anywhere in Python. Both methods will configure matplotlib to show or update the figure after every plotting command.

.. _faq_repl_output:

Why is something printed after every notebook cell?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*You're using seaborn in a Jupyter notebook, and every cell prints something like <AxesSuplot:> or <seaborn.axisgrid.FacetGrid at 0x7f840e279c10> before showing the plot.*

Jupyter notebooks will show the result of the final statement in the cell as part of its output, and each of seaborn's plotting functions return a reference to the matplotlib or seaborn object that contain the plot. If this is bothersome, you can suppress this output in a few ways:

- Always assign the result of the final statement to a variable (e.g. `ax = sns.histplot(...)`)
- Add a semicolon to the end of the final statement (e.g. `sns.histplot(...);`)
- End every cell with a function that has no return value (e.g. `plt.show()`, which isn't needed but also causes no problems)
- Add `cell metadata tags <https://nbformat.readthedocs.io/en/latest/format_description.html#cell-metadata>`_, if you're converting the notebook to a different representation

.. _faq_inline_dpi:

Why do the plots look fuzzy in a Jupyter notebook?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default "inline" backend (defined by `IPython <https://github.com/ipython/matplotlib-inline>`_) uses an unusually low dpi (`"dots per inch" <https://en.wikipedia.org/wiki/Dots_per_inch>`_) for figure output. This is a space-saving measure: lower dpi figures take up less disk space. (Also, lower dpi inline graphics appear *physically* smaller because they are represented as `PNGs <https://en.wikipedia.org/wiki/Portable_Network_Graphics>`_, which do not exactly have a concept of resolution.) So one faces an economy/quality tradeoff.

You can increase the DPI by resetting the rc parameters through the matplotlib API, using

::

    plt.rcParams.update({"figure.dpi": 96})

Or do it as you activate the seaborn theme::

    sns.set_theme(rc={"figure.dpi": 96})

If you have a high pixel-density monitor, you can make your plots sharper using "retina mode"::

    %config InlineBackend.figure_format = "retina"

This won't change the apparent size of your plots in a Jupyter interface, but they might appear very large in other contexts (i.e. on GitHub). And they will take up 4x the disk space. Alternatively, you can make SVG plots::

    %config InlineBackend.figure_format = "svg"

This will configure matplotlib to emit `vector graphics <https://en.wikipedia.org/wiki/Vector_graphics>`_ with "infinite resolution". The downside is that file size will now scale with the number and complexity of the artists in your plot, and in some cases (e.g., a large scatterplot matrix) the load will impact browser responsiveness.

Tricky concepts
---------------

.. _faq_function_levels:

What do "figure-level" and "axes-level" mean?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*You've encountered the term "figure-level" or "axes-level", maybe in the seaborn docs, StackOverflow answer, or GitHub thread, but you don't understand what it means.*

In brief, all plotting functions in seaborn fall into one of two categories:

- "axes-level" functions, which plot onto a single subplot that may or may not exist at the time the function is called
- "figure-level" functions, which internally create a matplotlib figure, potentially including multiple subplots

This design is intended to satisfy two objectives:

- seaborn should offer functions that are "drop-in" replacements for matplotlib methods
- seaborn should be able to produce figures that show "facets" or marginal distributions on distinct subplots

The figure-level functions always combine one or more axes-level functions with an object that manages the layout. So, for example, :func:`relplot` is a figure-level function that combines either :func:`scatterplot` or :func:`lineplot` with a :class:`FacetGrid`. In contrast, :func:`jointplot` is a figure-level function that can combine multiple different axes-level functions — :func:`scatterplot` and :func:`histplot` by default — with a :class:`JointGrid`.

If all you're doing is creating a plot with a single seaborn function call, this is not something you need to worry too much about. But it becomes relevant when you want to customize at a level beyond what the API of each function offers. It is also the source of various other points of confusion, so it is an important distinction understand (at least broadly) and keep in mind.

This is explained in more detail in the :doc:`tutorial </tutorial/function_overview>` and in `this blog post <https://michaelwaskom.medium.com/three-common-seaborn-difficulties-10fdd0cc2a8b>`_.

.. _faq_categorical_plots:

What is a "categorical plot" or "categorical function"?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next to the figure-level/axes-level distinction, this concept is probably the second biggest source of confusing behavior.

Several :ref:`seaborn functions <categorical_api>` are referred to as "categorical" because they are designed to support a use-case where either the x or y variable in a plot is categorical (that is, the variable takes a finite number of potentially non-numeric values).

At the time these functions were written, matplotlib did not have any direct support for non-numeric data types. So seaborn internally builds a mapping from unique values in the data to 0-based integer indexes, which is what it passes to matplotlib. If your data are strings, that's great, and it more-or-less matches how `matplotlib now handles <https://matplotlib.org/stable/gallery/lines_bars_and_markers/categorical_variables.html>`_ string-typed data.

But a potential gotcha is that these functions *always do this by default*, even if both the x and y variables are numeric. This gives rise to a number of confusing behaviors, especially when mixing categorical and non-categorical plots (e.g., a combo bar-and-line plot).

The v0.13 release added a `native_scale` parameter which provides control over this behavior. It is `False` by default, but setting it to `True` will preserve the original properties of the data used for categorical grouping.

Specifying data
---------------

.. _faq_data_format:

How does my data need to be organized?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To get the most out of seaborn, your data should have a "long-form" or "tidy" representation. In a dataframe, `this means that <https://r4ds.had.co.nz/tidy-data.html#tidy-data>`_ each variable has its own column, each observation has its own row, and each value has its own cell. With long-form data, you can succinctly and exactly specify a visualization by assigning variables in the dataset (columns) to roles in the plot.

Data organization is a common stumbling block for beginners, in part because data are often not collected or stored in a long-form representation. Therefore, it is often necessary to `reshape <https://pandas.pydata.org/pandas-docs/stable/user_guide/reshaping.html>`_ the data using pandas before plotting. Data reshaping can be a complex undertaking, requiring both a solid grasp of dataframe structure and knowledge of the pandas API. Investing some time in developing this skill can pay large dividends.

But while seaborn is *most* powerful when provided with long-form data, nearly every seaborn function will accept and plot "wide-form" data too. You can trigger this by passing an object to seaborn's `data=` parameter without specifying other plot variables (`x`, `y`, ...). You'll be limited when using wide-form data: each function can make only one kind of wide-form plot. In most cases, seaborn tries to match what matplotlib or pandas would do with a dataset of the same structure. Reshaping your data into long-form will give you substantially more flexibility, but it can be helpful to take a quick look at your data very early in the process, and seaborn tries to make this possible.

Understanding how your data should be represented — and how to get it that way if it starts out messy — is very important for making efficient and complete use of seaborn, and it is elaborated on at length in the :doc:`user-guide </tutorial/data_structure>`.

.. _faq_pandas_requirement:

Does seaborn only work with pandas?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generally speaking, no: seaborn is `quite flexible <https://seaborn.pydata.org/tutorial/data_structure.html#options-for-visualizing-long-form-data>`_ about how your dataset needs to be represented.

In most cases, :ref:`long-form data <faq_data_format>` represented by multiple vector-like types can be passed directly to `x`, `y`, or other plotting parameters. Or you can pass a dictionary of vector types to `data` rather than a DataFrame. And when plotting with wide-form data, you can use a 2D numpy array or even nested lists to plot in wide-form mode.

There are a couple older functions (namely, :func:`catplot` and :func:`lmplot`) that do require you to pass a :class:`pandas.DataFrame`. But at this point, they are the exception, and they will gain more flexibility over the next few release cycles.

Layout problems
---------------

.. _faq_figure_size:

How do I change the figure size?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is going to be more complicated than you might hope, in part because there are multiple ways to change the figure size in matplotlib, and in part because of the :ref:`figure-level/axes-level <faq_function_levels>` distinction in seaborn.

In matplotlib, you can usually set the default size for all figures through the `rc parameters <https://matplotlib.org/stable/tutorials/introductory/customizing.html>`_, specifically `figure.figsize`. And you can set the size of an individual figure when you create it (e.g. `plt.subplots(figsize=(w, h))`). If you're using an axes-level seaborn function, both of these will work as expected.

Figure-level functions both ignore the default figure size and :ref:`parameterize the figure size differently <figure_size_tutorial>`. When calling a figure-level function, you can pass values to `height=` and `aspect=` to set (roughly) the size of each *subplot*. The advantage here is that the size of the figure automatically adapts when you add faceting variables. But it can be confusing.

Fortunately, there's a consistent way to set the exact figure size in a function-independent manner. Instead of setting the figure size when the figure is created, modify it after you plot by calling `obj.figure.set_size_inches(...)`, where `obj` is either a matplotlib axes (usually assigned to `ax`) or a seaborn `FacetGrid` (usually assigned to `g`).

Note that :attr:`FacetGrid.figure` exists only on seaborn >= 0.11.2; before that you'll have to access :attr:`FacetGrid.fig`.

Also, if you're making pngs (or in a Jupyter notebook), you can — perhaps surprisingly — scale all your plots up or down by :ref:`changing the dpi <faq_inline_dpi>`.

.. _faq_plot_misplaced:

Why isn't seaborn drawing the plot where I tell it to?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*You've explicitly created a matplotlib figure with one or more subplots and tried to draw a seaborn plot on it, but you end up with an extra figure and a blank subplot. Perhaps your code looks something like*

::

    f, ax = plt.subplots()
    sns.catplot(..., ax=ax)

This is a :ref:`figure-level/axes-level <faq_function_levels>` gotcha. Figure-level functions always create their own figure, so you can't direct them towards an existing axes the way you can with axes-level functions. Most functions will warn you when this happens, suggest the appropriate axes-level function, and ignore the `ax=` parameter. A few older functions might put the plot where you want it (because they internally pass `ax` to their axes-level function) while still creating an extra figure. This latter behavior should be considered a bug, and it is not to be relied on.

The way things currently work, you can either set up the matplotlib figure yourself, or you can use a figure-level function, but you can't do both at the same time.

.. _faq_categorical_line:

Why can't I draw a line over a bar/box/strip/violin plot?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*You're trying to create a single plot using multiple seaborn functions, perhaps by drawing a lineplot or regplot over a barplot or violinplot. You expect the line to go through the mean value for each box (etc.), but it looks to be misalgined, or maybe it's all the way off to the side.*

You are trying to combine a :ref:`"categorical plot" <faq_categorical_plots>` with another plot type. If your `x` variable has numeric values, it seems like this should work. But recall: seaborn's categorical plots map unique values on the categorical axis to integer indexes. So if your data have unique `x` values of 1, 6, 20, 94, the corresponding plot elements will get drawn at 0, 1, 2, 3 (and the tick labels will be changed to represent the actual value).

The line or regression plot doesn't know that this has happened, so it will use the actual numeric values, and the plots won't line up at all.

As of now, there are two ways to work around this. In situations where you want to draw a line, you could use the (somewhat misleadingly named) :func:`pointplot` function, which is also a "categorical" function and will use the same rules for drawing the plot. If this doesn't solve the problem (for one, it's not as visually flexible as :func:`lineplot`, you could implement the mapping from actual values to integer indexes yourself and draw the plot that way::

    unique_xs = sorted(df["x"].unique())
    sns.violinplot(data=df, x="x", y="y")
    sns.lineplot(data=df, x=df["x"].map(unique_xs.index), y="y")

This is something that will be easier in a planned future release, as it will become possible to make the categorical functions treat numeric data as numeric. (As of v0.12, it's possible only in :func:`stripplot` and :func:`swarmplot`, using `native_scale=True`).

How do I move the legend?
~~~~~~~~~~~~~~~~~~~~~~~~~

*When applying a semantic mapping to a plot, seaborn will automatically create a legend and add it to the figure. But the automatic choice of legend position is not always ideal.*

With seaborn v0.11.2 or later, use the :func:`move_legend` function.

On older versions, a common pattern was to call `ax.legend(loc=...)` after plotting. While this appears to move the legend, it actually *replaces* it with a new one, using any labeled artists that happen to be attached to the axes. This does `not consistently work <https://github.com/mwaskom/seaborn/issues/2280>`_ across plot types. And it does not propagate the legend title or positioning tweaks that are used to format a multi-variable legend.

The :func:`move_legend` function is actually more powerful than its name suggests, and it can also be used to modify other `legend parameters <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html>`_ (font size, handle length, etc.) after plotting.

Other customizations
--------------------

.. _faq_figure_customization:

How can I can I change something about the figure?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*You want to make a very specific plot, and seaborn's defaults aren't doing it for you.*

There's basically a four-layer hierarchy to customizing a seaborn figure:

1. Explicit seaborn function parameters
2. Passed-through matplotlib keyword arguments
3. Matplotlib axes methods
4. Matplotlib artist methods

First, read through the API docs for the relevant seaborn function. Each has a lot of parameters (probably too many), and you may be able to accomplish your desired customization using seaborn's own API.

But seaborn does delegate a lot of customization to matplotlib. Most functions have `**kwargs` in their signature, which will catch extra keyword arguments and pass them through to the underlying matplotlib function. For example, :func:`scatterplot` has a number of parameters, but you can also use any valid keyword argument for :meth:`matplotlib.axes.Axes.scatter`, which it calls internally.

Passing through keyword arguments lets you customize the artists that represent data, but often you will want to customize other aspects of the figure, such as labels, ticks, and titles. You can do this by calling methods on the object that seaborn's plotting functions return. Depending on whether you're calling an :ref:`axes-level or figure-level function <faq_function_levels>`, this may be a :class:`matplotlib.axes.Axes` object or a seaborn wrapper (such as :class:`seaborn.FacetGrid`). Both kinds of objects have numerous methods that you can call to customize nearly anything about the figure. The easiest thing is usually to call :meth:`matplotlib.axes.Axes.set` or :meth:`seaborn.FacetGrid.set`, which let you modify multiple attributes at once, e.g.::

    ax = sns.scatterplot(...)
    ax.set(
        xlabel="The x label",
        ylabel="The y label",
        title="The title"
        xlim=(xmin, xmax),
        xticks=[...],
        xticklabels=[...],
    )

Finally, the deepest customization may require you to reach "into" the matplotlib axes and tweak the artists that are stored on it. These will be in artist lists, such as `ax.lines`, `ax.collections`, `ax.patches`, etc.

*Warning:* Neither matplotlib nor seaborn consider the specific artists produced by their plotting functions to be part of stable API. Because it's not possible to gracefully warn about upcoming changes to the artist types or the order in which they are stored, code that interacts with these attributes could break unexpectedly. With that said, seaborn does try hard to avoid making this kind of change.

.. _faq_matplotlib_requirement:

Wait, I need to learn how to use matplotlib too?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It really depends on how much customization you need. You can certainly perform a lot of exploratory data analysis while primarily or exclusively interacting with the seaborn API. But, if you're polishing a figure for a presentation or publication, you'll likely find yourself needing to understand at least a little bit about how matplotlib works. Matplotlib is extremely flexible, and it lets you control literally everything about a figure if you drill down far enough.

Seaborn was originally designed with the idea that it would handle a specific set of well-defined operations through a very high-level API, while letting users "drop down" to matplotlib when they desired additional customization. This can be a pretty powerful combination, and it works reasonably well if you already know how to use matplotlib. But as seaborn as gained more features, it has become more feasible to learn seaborn *first*. In that situation, the need to switch APIs tends to be a bit more confusing / frustrating. This has motivated the development of seaborn's new :doc:`objects interface </tutorial/objects_interface>`, which aims to provide a more cohesive API for both high-level and low-level figure specification. Hopefully, it will alleviate the "two-library problem" as it matures.

With that said, the level of deep control that matplotlib affords really can't be beat, so if you care about doing very specific things, it really is worth learning.

.. _faq_object_oriented:

How do I use seaborn with matplotlib's object-oriented interface?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*You prefer to use matplotlib's explicit or* `"object-oriented" <https://matplotlib.org/stable/users/explain/api_interfaces.html>`_ *interface, because it makes your code easier to reason about and maintain. But the object-orient interface consists of methods on matplotlib objects, whereas seaborn offers you independent functions.*

This is another case where it will be helpful to keep the :ref:`figure-level/axes-level <faq_function_levels>` distinction in mind.

Axes-level functions can be used like any matplotlib axes method, but instead of calling `ax.func(...)`, you call `func(..., ax=ax)`. They also return the axes object (which they may have created, if no figure was currently active in matplotlib's global state). You can use the methods on that object to further customize the plot even if you didn't start with :func:`matplotlib.pyplot.figure` or :func:`matplotlib.pyplot.subplots`::

    ax = sns.histplot(...)
    ax.set(...)

Figure-level functions :ref:`can't be directed towards an existing figure <faq_plot_misplaced>`, but they do store the matplotlib objects on the :class:`FacetGrid` object that they return (which seaborn docs always assign to a variable named `g`).

If your figure-level function created only one subplot, you can access it directly::

    g = sns.displot(...)
    g.ax.set(...)

For multiple subplots, you can either use :attr:`FacetGrid.axes` (which is always a 2D array of axes) or :attr:`FacetGrid.axes_dict` (which maps the row/col keys to the corresponding matplotlib object)::

    g = sns.displot(..., col=...)
    for col, ax in g.axes_dict.items():
        ax.set(...)

But if you're batch-setting attributes on all subplots, use the :meth:`FacetGrid.set` method rather than iterating over the individual axes::

    g = sns.displot(...)
    g.set(...)

To access the underlying matplotlib *figure*, use :attr:`FacetGrid.figure` on seaborn >= 0.11.2 (or :attr:`FacetGrid.fig` on any other version).

.. _faq_bar_annotations:

Can I annotate bar plots with the bar values?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Nothing like this is built into seaborn, but matplotlib v3.4.0 added a convenience function (:meth:`matplotlib.axes.Axes.bar_label`) that makes it relatively easy. Here are a couple of recipes; note that you'll need to use a different approach depending on whether your bars come from a :ref:`figure-level or axes-level function <faq_function_levels>`::

    # Axes-level
    ax = sns.histplot(df, x="x_var")
    for bars in ax.containers:
        ax.bar_label(bars)

    # Figure-level, one subplot
    g = sns.displot(df, x="x_var")
    for bars in g.ax.containers:
        g.ax.bar_label(bars)

    # Figure-level, multiple subplots
    g = sns.displot(df, x="x_var", col="col_var)
    for ax in g.axes.flat:
        for bars in ax.containers:
            ax.bar_label(bars)

.. _faq_dar_mode:

Can I use seaborn in dark mode?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There's no direct support for this in seaborn, but matplotlib has a `"dark_background" <https://matplotlib.org/stable/gallery/style_sheets/dark_background.html>`_ style-sheet that you could use, e.g.::

    sns.set_theme(style="ticks", rc=plt.style.library["dark_background"])

Note that "dark_background" changes the default color palette to "Set2", and that will override any palette you define in :func:`set_theme`. If you'd rather use a different color palette, you'll have to call :func:`sns.set_palette` separately. The default :doc:`seaborn palette </tutorial/color_palettes>` ("deep") has poor contrast against a dark background, so you'd be better off using "muted", "bright", or "pastel".

Statistical inquiries
---------------------

.. _faq_stat_results:

Can I access the results of seaborn's statistical transformations?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because seaborn performs some statistical operations as it builds plots (aggregating, bootstrapping, fitting regression models), some users would like access to the statistics that it computes. This is not possible: it's explicitly considered out of scope for seaborn (a visualization library) to offer an API for interrogating statistical models.

If you simply want to be diligent and verify that seaborn is doing things correctly (or that it matches your own code), it's open-source, so feel free to read the code. Or, because it's Python, you can call into the private methods that calculate the stats (just don't do this in production code). But don't expect seaborn to offer features that are more at home in `scipy <https://scipy.org/>`_ or `statsmodels <https://www.statsmodels.org/>`_.

.. _faq_standard_error:

Can I show standard error instead of a confidence interval?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As of v0.12, this is possible in most places, using the new `errorbar` API (see the :doc:`tutorial </tutorial/error_bars>` for more details).

.. _faq_kde_value:

Why does the y axis for a KDE plot go above 1?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*You've estimated a probability distribution for your data using* :func:`kdeplot`, *but the y axis goes above 1. Aren't probabilities bounded by 1? Is this a bug?*

This is not a bug, but it is a common confusion (about kernel density plots and probability distributions more broadly). A continuous probability distribution is defined by a `probability density function <https://en.wikipedia.org/wiki/Probability_density_function>`_, which :func:`kdeplot` estimates. The probability density function does **not** output *a probability*: a continuous random variable can take an infinite number of values, so the probability of observing any *specific* value is infinitely small. You can only talk meaningfully about the probability of observing a value that falls within some *range*. The probability of observing a value that falls within the complete range of possible values is 1. Likewise, the probability density function is normalized so that the area under it (that is, the integral of the function across its domain) equals 1. If the range of likely values is small, the curve will have to go above 1 to make this possible.

Common curiosities
------------------

.. _faq_import_convention:

Why is seaborn imported as `sns`?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is an obscure reference to the `namesake <https://pbs.twimg.com/media/C3C6q1ZUYAALXX0.jpg>`_ of the library, but you can also think of it as "seaborn name space".

.. _faq_seaborn_sucks:

Why is ggplot so much better than seaborn?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Good question. Probably because you get to use the word "geom" a lot, and it's fun to say. "Geom". "Geeeeeooom".
