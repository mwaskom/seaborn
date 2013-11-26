
Controlling figure aesthetics in seaborn
========================================


Although the aesthetics of a figure are not sufficient to convey useful
statistical information, attractive plots are more pleasant to look at
when trying to understand your own data, and they can draw in your
audience when you present it. Judicious use of stylistic detail, such as
maintaining thematic colors across figures in a presentation, can
support the communication of your ideas without distracting from the
central message. While a beautiful color palette cannot save an
incoherent plot, poor stylistic choices can `obscure or mislead
about <http://blog.visual.ly/subtleties-of-color/>`__ patterns in your
data.

Motivted by these considerations, seaborn tries to make it easy to
control the look of your figures. This notebook walks through the set of
tools that let you manipulate plot styles.

.. code:: python

    import numpy as np
    from scipy import stats
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    np.random.seed(9221999)
Axis Styles
-----------


Let's define a simple function to plot some offset sine waves to help us
see the different stylistic parameters we can tweak.

.. code:: python

    def sinplot(flip=1):
        x = np.linspace(0, 14, 100)
        for i in range(1, 7):
            plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)
This is what the plot looks like with matplotlib defaults:

.. code:: python

    sinplot()


.. image:: aesthetics_files/aesthetics_7_0.png


To switch to seaborn defaults, simply import the package.

.. code:: python

    import seaborn as sns
    sinplot()


.. image:: aesthetics_files/aesthetics_9_0.png


Seaborn plots break from the MATLAB inspired aesthetic of matplotlib to
plot in more muted colors over a light gray background with white grid
lines. We find that the grid aids in the use of figures for conveying
quantitative information -- in almost all cases, figures should be
preferred to tables. The white-on-gray grid that is used by default
avoids being obtrusive. The grid is particularly useful when comparing
across facets of a plot, which is central to some of the more complex
tools in the library:

.. code:: python

    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    plt.subplot(ax1)
    sinplot()
    plt.subplot(ax2)
    sinplot(-1)
    plt.tight_layout()


.. image:: aesthetics_files/aesthetics_11_0.png


There are two other basic styles. One keeps the grid, but plots with a
more traditional white background.

.. code:: python

    sns.set(style="whitegrid")
    sinplot()


.. image:: aesthetics_files/aesthetics_13_0.png


For this kind of plot, where the data are represented with lines, the
gray grid complicates the figure and probably detracts more than it
adds. However, many kinds of statistical plots give more weight to the
foreground and look fine with the whitegrid style:

.. code:: python

    x = np.linspace(0, 14, 100)
    y1 = np.sin(x + .5)
    y2 = np.sin(x + 4 * .5) * 3
    c1, c2 = sns.color_palette("deep", 2)
    plt.plot(x, y1)
    plt.fill_between(x, y1 - .5, y1 + .5, color=c1, alpha=.2)
    plt.plot(x, y2)
    plt.fill_between(x, y2 - .8, y2 + .8, color=c2, alpha=.2);


.. image:: aesthetics_files/aesthetics_15_0.png


.. code:: python

    data = 1 + np.random.randn(20, 6)
    sns.boxplot(data);


.. image:: aesthetics_files/aesthetics_16_0.png


.. code:: python

    pos = np.arange(6) + .6
    h = data.mean(axis=0)
    err = data.std() / np.sqrt(len(data))
    plt.bar(pos, h, yerr=err, color=sns.husl_palette(6, s=.75), ecolor="#333333");


.. image:: aesthetics_files/aesthetics_17_0.png


You can also turn off the grid altogether, which is closest to the
default matplotlib style

.. code:: python

    sns.set(style="nogrid")
    sinplot()


.. image:: aesthetics_files/aesthetics_19_0.png


Because of the way matplotlib figures work, the axis spines cannot be
turned off as part of a default style. However, there is a convenience
function in seaborn for stripping the top and right spines to open up
the plot.

.. code:: python

    sinplot()
    sns.despine()


.. image:: aesthetics_files/aesthetics_21_0.png


.. code:: python

    sns.boxplot(data)
    sns.despine()


.. image:: aesthetics_files/aesthetics_22_0.png


To manipulate the look of more complex figures, you can use the optional
arguments to ``despine``.

.. code:: python

    sns.regplot(*np.random.randn(2, 100))
    main, x_marg, y_marg = plt.gcf().axes
    sns.despine(ax=main)
    sns.despine(ax=x_marg, left=True)
    sns.despine(ax=y_marg, bottom=True)


.. image:: aesthetics_files/aesthetics_24_0.png


Changing style contexts
~~~~~~~~~~~~~~~~~~~~~~~


The seaborn defaults are tailored to make plots that are
well-proportioned for vieweing on your own computer screen. There are a
few other styles that try to set parameters like font sizes to be more
appropriate for other settings, such as at a talk or on a poster:

.. code:: python

    sns.set(style="darkgrid", context="talk")
    sns.boxplot(data)
    plt.title("Score ~ Category");
    sns.axlabel("Category", "Score")


.. image:: aesthetics_files/aesthetics_27_0.png


.. code:: python

    sns.set(style="nogrid", context="poster")
    sns.boxplot(data)
    plt.title("Score ~ Category");
    sns.axlabel("Category", "Score")
    sns.despine()


.. image:: aesthetics_files/aesthetics_28_0.png


I would expect both the specific elements of these styles and the API
for specifying them to change somewhat as the package matures. In
particular, there is not currently a way for seaborn to respect rc
parameters that conflict with those it sets itself. Additionally, there
is no support for custom themes. If you would find these features useful
for your own work, please get in touch.

Seaborn color palettes
----------------------


Let's reset the default styles.

.. code:: python

    sns.set()
Considerable effor has been invested in a simple yet uniform interface
for creating and specifying color palettes, as color is one of the most
important (and also one of the most tricky) aspects of making clear and
informative plots.

The default color scheme is based on the matplotlib default while aiming
to be a bit more pleasant to look at. To grab the current color cycle,
call the ``color_palette`` function with no arguments. This just returns
a list of r, g, b tuples:

.. code:: python

    current_palette = sns.color_palette()
    current_palette



.. parsed-literal::

    [(0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
     (0.3333333333333333, 0.6588235294117647, 0.40784313725490196),
     (0.7686274509803922, 0.3058823529411765, 0.3215686274509804),
     (0.5058823529411764, 0.4470588235294118, 0.6980392156862745),
     (0.8, 0.7254901960784313, 0.4549019607843137),
     (0.39215686274509803, 0.7098039215686275, 0.803921568627451)]



Seaborn has a small function to visualize a palette, which is useful for
documentation and possibly for when you are choosing colors for your own
plots.

.. code:: python

    sns.palplot(current_palette)


.. image:: aesthetics_files/aesthetics_36_0.png


It's also easy to get evenly spaced hues in the ``husl`` or ``hls``
color spaces. The former is preferred for its perceptual uniformity,
although the individual colors can be relatively less attractive than
their brighter versions in the latter.

.. code:: python

    sns.palplot(sns.color_palette("husl", 8))


.. image:: aesthetics_files/aesthetics_38_0.png


.. code:: python

    sns.palplot(sns.color_palette("hls", 8))


.. image:: aesthetics_files/aesthetics_39_0.png


You can also use the name of any matplotlib colormap, and the palette
will return evenly-spaced samples from points near the extremes.

.. code:: python

    sns.palplot(sns.color_palette("coolwarm", 7))


.. image:: aesthetics_files/aesthetics_41_0.png


Palettes can be broadly categorized as *diverging* (as is the palette
above), *sequential*, or *qualitative*. Diverging palettes are useful
when the data has a natural, meaninfgul break-point. Sequential palettes
are better when the data range from "low" to "high" values.

.. code:: python

    sns.palplot(sns.color_palette("YlOrRd_r", 8))


.. image:: aesthetics_files/aesthetics_43_0.png


Categorial data is best represented by a qualitative palette. Seaborn
fixes some problems inherent in the way matplotlib deals with the
qualitative palettes from the `colorbrewer <http://colorbrewer.org>`__
package, although they behave a little differently. If you request more
colors than exist for a given qualitative palette, the colors will
cycle, which is not the case for other matplotlib-based palettes.

.. code:: python

    sns.palplot(sns.color_palette("Set2", 10))


.. image:: aesthetics_files/aesthetics_45_0.png


Finally, you can just pass in a list of color codes to specify a custom
palette.

.. code:: python

    sns.palplot(sns.color_palette(["#8C1515", "#D2C295"], 5))


.. image:: aesthetics_files/aesthetics_47_0.png


Many seaborn functions use the ``color_palette`` function behind the
scenes, and thus accept any of the valid arguments for their ``color``
or ``palette`` parameter.

.. code:: python

    sns.violin(data, inner="points", color="Set3");


.. image:: aesthetics_files/aesthetics_49_0.png


Two other functions allow you to create custom palettes. The first takes
a color and creates a blend to it from a very dark gray.

.. code:: python

    sns.palplot(sns.dark_palette("MediumPurple"))


.. image:: aesthetics_files/aesthetics_51_0.png


Note that the interpolation that is done behind the scenes is not
currently performed in a color space that is compatible with human
perception, so the increments of color in these palettes will not
necessarily appear uniform.

.. code:: python

    sns.palplot(sns.dark_palette("skyblue", 8, reverse=True))


.. image:: aesthetics_files/aesthetics_53_0.png


By default you just get a list of colors, like any other seaborn
palette, but you can also return the palette as a colormap object that
can be passed to matplotlib functions.

.. code:: python

    sample = np.random.multivariate_normal([0, 0], [[1, -.5], [-.5, 1]], size=1000)
    kde2d = stats.gaussian_kde(sample.T)
    xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    density = kde2d([xx.ravel(), yy.ravel()]).reshape(100, 100)
    
    pal = sns.dark_palette("palegreen", as_cmap=True)
    plt.figure(figsize=(6, 6))
    plt.contour(xx, yy, density, 15, cmap=pal);


.. image:: aesthetics_files/aesthetics_55_0.png


A more general function for making custom palettes interpolates between
an arbitrary number of seed points. You could use this to make your own
diverging palette.

.. code:: python

    sns.palplot(sns.blend_palette(["mediumseagreen", "ghostwhite", "#4168B7"], 9))


.. image:: aesthetics_files/aesthetics_57_0.png


Or to create a sequential palette along a saturation scale.

.. code:: python

    sns.palplot(sns.blend_palette([sns.desaturate("#009B76", 0), "#009B76"], 5))


.. image:: aesthetics_files/aesthetics_59_0.png


The resulting palettes can be passed to any seaborn function that can
take a palette as a parameter.

.. code:: python

    pal = sns.blend_palette(["seagreen", "lightblue"])
    sns.boxplot(data, color=pal);


.. image:: aesthetics_files/aesthetics_61_0.png


The ``set_color_palette`` function takes any of these inputs and sets
the persistent axis color cycle.

.. code:: python

    sns.set_color_palette("husl")
    sinplot()


.. image:: aesthetics_files/aesthetics_63_0.png


You can also temporarily set the color cycle by using the
``palette_context`` function, which is a context manager.

.. code:: python

    with sns.palette_context(sns.dark_palette("MediumSeaGreen")):
        sinplot()


.. image:: aesthetics_files/aesthetics_65_0.png


The hope is that these tools will make it easier to create plots that
are beautiful, both for the sake of beauty itself, and for the ways in
which is can enhance the communication of statistical information.
