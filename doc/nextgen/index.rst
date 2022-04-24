Next-generation seaborn interface
=================================

Over the past 8 months, I have been developing an entirely new interface
for making plots with seaborn. This page demonstrates some of its
functionality.

.. note::

    This is very much a work in progress. It is almost certain that code patterns demonstrated here will change before an official release.
    
    I do plan to issue a series of alpha/beta releases so that people can play around with it and give feedback, but it's not at that point yet.

Background and goals
--------------------

This work grew out of long-running efforts to refactor the seaborn
internals so that its functions could rely on common code-paths. At a
certain point, I decided that I was developing an API that would also be
interesting for external users too.

Of course, “write a new interface” quickly turned into “rethink every
aspect of the library.” The current interface has some `pain
points <https://michaelwaskom.medium.com/three-common-seaborn-difficulties-10fdd0cc2a8b>`__
that arise from early constraints and path dependence. By starting
fresh, these can be avoided.

More broadly, seaborn was originally conceived as a toolbox of
domain-specific statistical graphics to be used alongside matplotlib. As
the library (and data science) grew, it became more common to reach for
— or even learn — seaborn first. But one inevitably desires some
customization that is not offered within the (already much-too-long)
list of parameters in seaborn’s functions. Currently, this necessitates
direct use of matplotlib.

I’ve always thought that, if you’re comfortable with both libraries,
this setup offers a powerful blend of convenience and flexibility. But
it can be hard to know which library will let you accomplish some
specific task. And, as seaborn has become more powerful, one has to
write increasing amounts of matpotlib code to recreate what it is doing.

So the goal is to expose seaborn’s core features — integration with
pandas, automatic mapping between data and graphics, statistical
transformations — within an interface that is more compositional,
extensible, and comprehensive.

One will note that the result looks a bit (a lot?) like ggplot. That’s
not unintentional, but the goal is also *not* to “port ggplot2 to
Python”. (If that’s what you’re looking for, check out the very nice
`plotnine <https://plotnine.readthedocs.io/en/stable/>`__ package).
There is an immense amount of wisdom in the grammar of graphics and in
its particular implementation as ggplot2. But I think that, as
languages, R and Python are just too different for idioms from one to
feel natural when translated literally into the other. So while I have
taken much inspiration from ggplot (along with vegalite, and other
declarative visualization libraries), I’ve also made plenty of choices
differently, for better or for worse.

--------------

The basic interface
-------------------

OK enough preamble. What does this look like? The new interface exists
as a set of classes that can be acessed through a single namespace
import:

.. code:: ipython3

    import seaborn.objects as so

This is a clean namespace, and I’m leaning towards recommending
``from seaborn.objects import *`` for interactive usecases. But let’s
not go so far just yet.

Let’s also import the main namespace so we can load our trusty example
datasets.

.. code:: ipython3

    import seaborn
    seaborn.set_theme()

The main object is ``seaborn.objects.Plot``. You instantiate it by
passing data and some assignments from columns in the data to roles in
the plot:

.. code:: ipython3

    tips = seaborn.load_dataset("tips")
    so.Plot(tips, x="total_bill", y="tip")




.. image:: index_files/index_8_0.png
   :width: 489.59999999999997px
   :height: 326.4px



But instantiating the ``Plot`` object doesn’t actually plot anything.
For that you need to add some layers:

.. code:: ipython3

    so.Plot(tips, x="total_bill", y="tip").add(so.Scatter())




.. image:: index_files/index_10_0.png
   :width: 489.59999999999997px
   :height: 326.4px



Variables can be defined globally, or for a specific layer:

.. code:: ipython3

    so.Plot(tips).add(so.Scatter(), x="total_bill", y="tip")




.. image:: index_files/index_12_0.png
   :width: 489.59999999999997px
   :height: 326.4px



Each layer can also have its own data:

.. code:: ipython3

    (
        so.Plot(tips, x="total_bill", y="tip")
        .add(so.Scatter(color=".6"), data=tips.query("size != 2"))
        .add(so.Scatter(), data=tips.query("size == 2"))
    )




.. image:: index_files/index_14_0.png
   :width: 489.59999999999997px
   :height: 326.4px



As in the existing interface, variables can be keys to the ``data``
object or vectors of various kinds:

.. code:: ipython3

    (
        so.Plot(tips.to_dict(), x="total_bill")
        .add(so.Scatter(), y=tips["tip"].to_numpy())
    )




.. image:: index_files/index_16_0.png
   :width: 489.59999999999997px
   :height: 326.4px



The interface also supports semantic mappings between data and plot
variables. But the specification of those mappings uses more explicit
parameter names:

.. code:: ipython3

    so.Plot(tips, x="total_bill", y="tip", color="time").add(so.Scatter())




.. image:: index_files/index_18_0.png
   :width: 489.59999999999997px
   :height: 326.4px



It also offers a wider range of mappable features:

.. code:: ipython3

    (
        so.Plot(tips, x="total_bill", y="tip", color="day", fill="time")
        .add(so.Scatter(fillalpha=.8))
    )




.. image:: index_files/index_20_0.png
   :width: 489.59999999999997px
   :height: 326.4px



--------------

Core components
---------------

Visual representation: the Mark
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each layer needs a ``Mark`` object, which defines how to draw the plot.
There will be marks corresponding to existing seaborn functions and ones
offering new functionality. But not many have been implemented yet:

.. code:: ipython3

    fmri = seaborn.load_dataset("fmri").query("region == 'parietal'")
    so.Plot(fmri, x="timepoint", y="signal").add(so.Line())




.. image:: index_files/index_23_0.png
   :width: 489.59999999999997px
   :height: 326.4px



``Mark`` objects will expose an API to set features directly, rather
than mapping them:

.. code:: ipython3

    so.Plot(tips, y="day", x="total_bill").add(so.Dot(color="#698", alpha=.5))




.. image:: index_files/index_25_0.png
   :width: 489.59999999999997px
   :height: 326.4px



Data transformation: the Stat
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Built-in statistical transformations are one of seaborn’s key features.
But currently, they are tied up with the different visual
representations. E.g., you can aggregate data in ``lineplot``, but not
in ``scatterplot``.

In the new interface, these concerns are separated. Each layer can
accept a ``Stat`` object that applies a data transformation:

.. code:: ipython3

    so.Plot(fmri, x="timepoint", y="signal").add(so.Line(), so.Agg())




.. image:: index_files/index_27_0.png
   :width: 489.59999999999997px
   :height: 326.4px



The ``Stat`` is computed on subsets of data defined by the semantic
mappings:

.. code:: ipython3

    so.Plot(fmri, x="timepoint", y="signal", color="event").add(so.Line(), so.Agg())




.. image:: index_files/index_29_0.png
   :width: 489.59999999999997px
   :height: 326.4px



Each mark also accepts a ``group`` mapping that creates subsets without
altering visual properties:

.. code:: ipython3

    (
        so.Plot(fmri, x="timepoint", y="signal", color="event")
        .add(so.Line(), so.Agg(), group="subject")
    )




.. image:: index_files/index_31_0.png
   :width: 489.59999999999997px
   :height: 326.4px



The ``Mark`` and ``Stat`` objects allow for more compositionality and
customization. There will be guidelines for how to define your own
objects to plug into the broader system:

.. code:: ipython3

    class PeakAnnotation(so.Mark):
        def plot(self, split_generator, scales, orient):
            for keys, data, ax in split_generator():
                ix = data["y"].idxmax()
                ax.annotate(
                    "The peak", data.loc[ix, ["x", "y"]],
                    xytext=(10, -100), textcoords="offset points",
                    va="top", ha="center",
                    arrowprops=dict(arrowstyle="->", color=".2"),
    
                )
    
    (
        so.Plot(fmri, x="timepoint", y="signal")
        .add(so.Line(), so.Agg())
        .add(PeakAnnotation(), so.Agg())
    )




.. image:: index_files/index_33_0.png
   :width: 489.59999999999997px
   :height: 326.4px



The new interface understands not just ``x`` and ``y``, but also range
specifiers; some ``Stat`` objects will output ranges, and some ``Mark``
objects will accept them. (This means that it will finally be possible
to pass pre-defined error-bars into seaborn):

.. code:: ipython3

    (
        fmri
        .groupby("timepoint")
        .signal
        .describe()
        .pipe(so.Plot, x="timepoint")
        .add(so.Line(), y="mean")
        .add(so.Area(alpha=.2), ymin="min", ymax="max")
    )




.. image:: index_files/index_35_0.png
   :width: 489.59999999999997px
   :height: 326.4px



--------------

Overplotting resolution: the Move
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Existing seaborn functions have parameters that allow adjustments for
overplotting, such as ``dodge=`` in several categorical functions,
``jitter=`` in several functions based on scatterplots, and the
``multiple=`` paramter in distribution functions. In the new interface,
those adjustments are abstracted away from the particular visual
representation into the concept of a ``Move``:

.. code:: ipython3

    (
        so.Plot(tips, "day", "total_bill", color="time")
        .add(so.Bar(), so.Agg(), move=so.Dodge())
    )




.. image:: index_files/index_37_0.png
   :width: 489.59999999999997px
   :height: 326.4px



Separating out the positional adjustment makes it possible to add
additional flexibility without overwhelming the signature of a single
function. For example, there will be more options for handling missing
levels when dodging and for fine-tuning the adjustment.

.. code:: ipython3

    (
        so.Plot(tips, "day", "total_bill", color="time")
        .add(so.Bar(), so.Agg(), move=so.Dodge(empty="fill", gap=.1))
    )




.. image:: index_files/index_39_0.png
   :width: 489.59999999999997px
   :height: 326.4px



By default, the ``move`` will resolve all overlapping semantic mappings:

.. code:: ipython3

    (
        so.Plot(tips, "day", "total_bill", color="time", alpha="sex")
        .add(so.Bar(), so.Agg(), move=so.Dodge())
    )




.. image:: index_files/index_41_0.png
   :width: 489.59999999999997px
   :height: 326.4px



But you can specify a subset:

.. code:: ipython3

    (
        so.Plot(tips, "day", "total_bill", color="time", alpha="smoker")
        .add(so.Dot(), move=so.Dodge(by=["color"]))
    )




.. image:: index_files/index_43_0.png
   :width: 489.59999999999997px
   :height: 326.4px



It’s also possible to stack multiple moves or kinds of moves by passing
a list:

.. code:: ipython3

    (
        so.Plot(tips, "day", "total_bill", color="time", alpha="smoker")
        .add(
            so.Dot(),
            move=[so.Dodge(by=["color"]), so.Jitter(.5)]
        )
    )




.. image:: index_files/index_45_0.png
   :width: 489.59999999999997px
   :height: 326.4px



Separating the ``Stat`` and ``Move`` from the visual representation
affords more flexibility, greatly expanding the space of graphics that
can be created.

--------------

Semantic mapping: the Scale
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The declarative interface allows users to represent dataset variables
with visual properites such as position, color or size. A complete plot
can be made without doing anything more defining the mappings: users
need not be concerned with converting their data into units that
matplotlib understands. But what if one wants to alter the mapping that
seaborn chooses? This is accomplished through the concept of a
``Scale``.

The notion of scaling will probably not be unfamiliar; as in matplotlib,
seaborn allows one to apply a mathematical transformation, such as
``log``, to the coordinate variables:

.. code:: ipython3

    planets = seaborn.load_dataset("planets").query("distance < 1000")

.. code:: ipython3

    (
        so.Plot(planets, x="mass", y="distance")
        .scale(x="log", y="log")
        .add(so.Scatter())
    )




.. image:: index_files/index_49_0.png
   :width: 489.59999999999997px
   :height: 326.4px



But the ``Scale`` concept is much more general in seaborn: a scale can
be provided for any mappable property. For example, it is how you
specify the palette used for color variables:

.. code:: ipython3

    (
        so.Plot(planets, x="mass", y="distance", color="orbital_period")
        .scale(x="log", y="log", color="rocket")
        .add(so.Scatter())
    )




.. image:: index_files/index_51_0.png
   :width: 489.59999999999997px
   :height: 326.4px



While there are a number of short-hand “magic” arguments you can provide
for each scale, it is also possible to be more explicit by passing a
``Scale`` object. There are several distinct ``Scale`` classes,
corresponding to the fundamental scale types (nominal, ordinal,
continuous, etc.). Each class exposes a number of relevant parameters
that control the details of the mapping:

.. code:: ipython3

    (
        so.Plot(planets, x="mass", y="distance", color="orbital_period")
        .scale(
            x="log",
            y=so.Continuous(transform="log").tick(at=[3, 10, 30, 100, 300]),
            color=so.Continuous("rocket", transform="log"),
        )
        .add(so.Scatter())
    )




.. image:: index_files/index_53_0.png
   :width: 489.59999999999997px
   :height: 326.4px



There are several different kinds of scales, including scales
appropriate for categorical data:

.. code:: ipython3

    (
        so.Plot(planets, x="year", y="distance", color="method")
        .scale(
            y="log",
            color=so.Nominal(["b", "g"], order=["Radial Velocity", "Transit"])
        )
        .add(so.Scatter())
    )




.. image:: index_files/index_55_0.png
   :width: 489.59999999999997px
   :height: 326.4px



It’s also possible to disable scaling for a variable so that the literal
values in the dataset are passed directly through to matplotlib:

.. code:: ipython3

    (
        so.Plot(planets, x="distance", y="orbital_period", pointsize="mass")
        .scale(x="log", y="log", pointsize=None)
        .add(so.Scatter())
    )




.. image:: index_files/index_57_0.png
   :width: 489.59999999999997px
   :height: 326.4px



Scaling interacts with the ``Stat`` and ``Move`` transformations. When
an axis has a nonlinear scale, any statistical transformations or
adjustments take place in the appropriate space:

.. code:: ipython3

    so.Plot(planets, x="distance").add(so.Bar(), so.Hist()).scale(x="log")




.. image:: index_files/index_59_0.png
   :width: 489.59999999999997px
   :height: 326.4px



This is also true of the ``Move`` transformations:

.. code:: ipython3

    (
        so.Plot(
            planets, x="distance",
            color=(planets["number"] > 1).rename("multiple")
        )
        .add(so.Bar(), so.Hist(), so.Dodge())
        .scale(x="log")
    )




.. image:: index_files/index_61_0.png
   :width: 489.59999999999997px
   :height: 326.4px



--------------

Defining subplot structure
--------------------------

Seaborn’s faceting functionality (drawing subsets of the data on
distinct subplots) is built into the ``Plot`` object and works
interchangably with any ``Mark``/``Stat``/``Move``/``Scale`` spec:

.. code:: ipython3

    (
        so.Plot(tips, x="total_bill", y="tip")
        .facet("time", order=["Dinner", "Lunch"])
        .add(so.Scatter())
    )




.. image:: index_files/index_64_0.png
   :width: 489.59999999999997px
   :height: 326.4px



Unlike the existing ``FacetGrid`` it is simple to *not* facet a layer,
so that a plot is simply replicated across each column (or row):

.. code:: ipython3

    (
        so.Plot(tips, x="total_bill", y="tip")
        .facet(col="day")
        .add(so.Scatter(color=".75"), col=None)
        .add(so.Scatter(), color="day")
        .configure(figsize=(7, 3))
    )




.. image:: index_files/index_66_0.png
   :width: 571.1999999999999px
   :height: 244.79999999999998px



The ``Plot`` object *also* subsumes the ``PairGrid`` functionality:

.. code:: ipython3

    (
        so.Plot(tips, y="day")
        .pair(x=["total_bill", "tip"])
        .add(so.Dot())
    )




.. image:: index_files/index_68_0.png
   :width: 489.59999999999997px
   :height: 326.4px



Pairing and faceting can be combined in the same plot:

.. code:: ipython3

    (
        so.Plot(tips, x="day")
        .facet("sex")
        .pair(y=["total_bill", "tip"])
        .add(so.Dot())
    )




.. image:: index_files/index_70_0.png
   :width: 489.59999999999997px
   :height: 326.4px



Or the ``Plot.pair`` functionality can be used to define unique pairings
between variables:

.. code:: ipython3

    (
        so.Plot(tips)
        .pair(x=["day", "time"], y=["total_bill", "tip"], cartesian=False)
        .add(so.Dot())
    )




.. image:: index_files/index_72_0.png
   :width: 489.59999999999997px
   :height: 326.4px



It’s additionally possible to “pair” with a single variable, for
univariate plots like histograms.

Both faceted and paired plots with subplots along a single dimension can
be “wrapped”, and this works both columwise and rowwise:

.. code:: ipython3

    (
        so.Plot(tips)
        .pair(x=tips.columns, wrap=3)
        .configure(sharey=False)
        .add(so.Bar(), so.Hist())
    )




.. image:: index_files/index_74_0.png
   :width: 489.59999999999997px
   :height: 326.4px



Importantly, there’s no distinction between “axes-level” and
“figure-level” here. Any kind of plot can be faceted or paired by adding
a method call to the ``Plot`` definition, without changing anything else
about how you are creating the figure.

--------------

Iterating and displaying
------------------------

It is possible (and in fact the deafult behavior) to be completely
pyplot-free, and all the drawing is done by directly hooking into
Jupyter’s rich display system. Unlike in normal usage of the inline
backend, writing code in a cell to define a plot is indendent from
showing it:

.. code:: ipython3

    p = so.Plot(fmri, x="timepoint", y="signal").add(so.Line(), so.Agg())

.. code:: ipython3

    p




.. image:: index_files/index_79_0.png
   :width: 489.59999999999997px
   :height: 326.4px



By default, the methods on ``Plot`` do *not* mutate the object they are
called on. This means that you can define a common base specification
and then iterate on different versions of it.

.. code:: ipython3

    p = (
        so.Plot(fmri, x="timepoint", y="signal", color="event")
        .scale(color="crest")
    )

.. code:: ipython3

    p.add(so.Line())




.. image:: index_files/index_82_0.png
   :width: 489.59999999999997px
   :height: 326.4px



.. code:: ipython3

    p.add(so.Line(), group="subject")




.. image:: index_files/index_83_0.png
   :width: 489.59999999999997px
   :height: 326.4px



.. code:: ipython3

    p.add(so.Line(), so.Agg())




.. image:: index_files/index_84_0.png
   :width: 489.59999999999997px
   :height: 326.4px



.. code:: ipython3

    (
        p
        .add(so.Line(linewidth=.5, alpha=.5), group="subject")
        .add(so.Line(linewidth=3), so.Agg())
    )




.. image:: index_files/index_85_0.png
   :width: 489.59999999999997px
   :height: 326.4px



It’s also possible to hook into the ``pyplot`` system by calling
``Plot.show``. (As you might in a terminal interface, or to use a GUI).
Notice how this looks lower-res: that’s because ``Plot`` is generating
“high-DPI” figures internally!

.. code:: ipython3

    (
        p
        .add(so.Line(linewidth=.5, alpha=.5), group="subject")
        .add(so.Line(linewidth=3), so.Agg())
        .show()
    )



.. image:: index_files/index_87_0.png


--------------

Matplotlib integration
----------------------

It’s always been a design aim in seaborn to allow complicated seaborn
plots to coexist within the context of a larger matplotlib figure. This
is acheived within the “axes-level” functions, which accept an ``ax=``
parameter. The ``Plot`` object *will* provide a similar functionality:

.. code:: ipython3

    import matplotlib as mpl
    _, ax = mpl.figure.Figure(constrained_layout=True).subplots(1, 2)
    (
        so.Plot(tips, x="total_bill", y="tip")
        .on(ax)
        .add(so.Scatter())
    )




.. image:: index_files/index_89_0.png
   :width: 489.59999999999997px
   :height: 326.4px



But a limitation has been that the “figure-level” functions, which can
produce multiple subplots, cannot be directed towards an existing
figure. That is no longer the case; ``Plot.on()`` also accepts a
``Figure`` (created either with or without ``pyplot``) object:

.. code:: ipython3

    f = mpl.figure.Figure(constrained_layout=True)
    (
        so.Plot(tips, x="total_bill", y="tip")
        .on(f)
        .add(so.Scatter())
        .facet("time")
    )




.. image:: index_files/index_91_0.png
   :width: 489.59999999999997px
   :height: 326.4px



Providing an existing figure is perhaps only marginally useful. While it
will ease the integration of seaborn with GUI frameworks, seaborn is
still using up the whole figure canvas. But with the introduction of the
``SubFigure`` concept in matplotlib 3.4, it becomes possible to place a
small-multiples plot *within* a larger set of subplots:

.. code:: ipython3

    f = mpl.figure.Figure(constrained_layout=True, figsize=(8, 4))
    sf1, sf2 = f.subfigures(1, 2)
    (
        so.Plot(tips, x="total_bill", y="tip", color="day")
        .add(so.Scatter())
        .on(sf1)
        .plot()
    )
    (
        so.Plot(tips, x="total_bill", y="tip", color="day")
        .facet("day", wrap=2)
        .add(so.Scatter())
        .on(sf2)
        .plot()
    )




.. image:: index_files/index_93_0.png
   :width: 652.8px
   :height: 326.4px



.. toctree::

    api

