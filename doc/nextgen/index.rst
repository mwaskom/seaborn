Next-generation seaborn interface
=================================

Over the past year, I have been developing an entirely new interface for
making plots with seaborn. The new interface is designed to be
declarative, compositional and extensible. If successful, it will
greatly expand the space of plots that can be created with seaborn while
making the experience of using it simpler and more delightful.

To make that concrete, here is a `hello world
example <http://seaborn.pydata.org/introduction.html#our-first-seaborn-plot>`__
with the new interface:

.. code:: ipython3

    import seaborn as sns
    sns.set_theme()
    tips = sns.load_dataset("tips")
    
    import seaborn.objects as so
    (
        so.Plot(
            tips, "total_bill", "tip",
            color="smoker", marker="smoker", pointsize="size",
        )
        .facet("time")
        .add(so.Scatter())
        .configure(figsize=(7, 4))
    )




.. image:: index_files/index_1_0.png
   :width: 632.8249999999999px
   :height: 313.22499999999997px



Testing the pre-release
-----------------------

If you’re interested, please install the pre-release and kick the tires.
It is still a work in progress, so expect some rough edges and
instability! But feedback will be very helpful in pushing this towards a
more stable broad release:

::

   pip install --pre seaborn

Preview components
------------------

The formal documentation is still a work-in-progress, so this preview
serves as an introduction and reference for testing. There’s a
reasonably thorough demo of the main components and some basic API
documentation for the existing classes:

.. toctree::
    :maxdepth: 1

    Demonstration <demo>
    API Overview <api>

Background and goals
--------------------

This work grew out of long-running efforts to refactor the seaborn
internals so that its functions could rely on common code-paths. At a
certain point, I realized that I was developing an API that might also
be interesting for external users.

Of course, “write a new interface” quickly turned into “rethink every
aspect of the library.” The current interface has some `pain
points <https://michaelwaskom.medium.com/three-common-seaborn-difficulties-10fdd0cc2a8b>`__
that arise from early constraints and path dependence. By starting
fresh, these can be avoided.

Originally, seaborn existed as a toolbox of domain-specific statistical
graphics to be used alongside matplotlib. As the library grew, it became
more common to reach for — or even learn — seaborn first. But one
inevitably desires some customization that is not offered within the
(already much-too-long) list of parameters in seaborn’s functions.
Currently, this necessitates direct use of matplotlib. I’ve always
thought that, if you’re comfortable with both libraries, this setup
offers a powerful blend of convenience and flexibility. But it can be
hard to know which library will let you accomplish some specific task.

So the new interface is designed to provide a more comprehensive
experience, such that all of the steps involved in the creation of a
reasonably-customized plot can be accomplished in the same way. And the
compositional nature of the objects provides much more flexibility than
currently exists in seaborn with a similar level of abstraction: this
lets you focus on *what* you want to show rather than *how* to show it.

One will note that the result looks a bit (a lot?) like ggplot. That’s
not unintentional: while the original seaborn interface was never
explicitly intended to implement a grammar of graphics, the redesign is.
But the goal is also *not* to “port ggplot2 to Python”. (If that’s what
you’re looking for, check out the very nice
`plotnine <https://plotnine.readthedocs.io/en/stable/>`__ package). I do
think that ggplot2 gets a lot right, especially its approach to
layering. But, as languages, R and Python are just too different for
idioms from one to feel natural when translated literally into the
other. So while I have taken much inspiration from ggplot (along with
vega-lite, d3, and other great libraries), I’ve also made plenty of
choices differently, for better or for worse.

