.. _introduction:

An introduction to seaborn
==========================

Effective visualization is critical for analyzing data and communicating the
insights of data-driven analyses. The Python data ecosystem is rapidly
maturing, with tools like `Pandas <http://pandas.pydata.org/>`_ and the
`IPython Notebook <http://ipython.org/notebook.html>`_ making Python one of the
best environments for exploring datasets and developing reproducible research
workflows. Matplotlib provides a solid foundation for plotting, and it is
deeply integrated with core scientific Python tools. However, its limitations
are well-known. The default aesthetics, deigned to ease a transition from
MATLAB, are out of place next to alternatives from the R ecosystem or d3-based
web graphics. Although the API is highly flexible, complex plots require a
rather verbose specification. Finally, matplotlib generally does not understand
the semantic metadata encoded within Pandas objects and cannot exploit the
sophisticated reshaping operations offered by those tools.

Seaborn is built on top of matplotlib to integrate with existing data workflows
and draw beautiful, informative plots without much effort. To style all plots
with the default aesthetics, just import the package. The styling works through
on-the-fly manipulation of the matplotlib rc parameters, so seaborn plots can
be reproduced without any additional files defining the aesthetics. You can
also select one of several high-level plotting themes and easily use a diverse
set of color palettes to enhance the presentation of your findings.

Seaborn's core is a library of high-level functions for drawing
statistical graphics. These functions have considerable intelligence about
their domain of visualization and the statistical model corresponding to that
visualization. This is a different approach from that taken by R's `ggplot
<http://ggplot2.org/>`_ and the very promising `Python port
<http://blog.yhathq.com/posts/ggplot-for-python.html>`_ of it. What ggplot
offers is an entirely different way of thinking about programmatic
visualization. While this can be very powerful, it also tends to befuddle both
newcomers and experienced users. Although seaborn is much more limited in what
it can draw, it tries to make it very easy and straightforward to construct
rather sophisticated and complex plots.

Seaborn is tightly integrated with Pandas, and most of the functions will
exploit Pandas attributes and assign names to plot elements. Statistical
datasets are often high-dimensional, and many of the more complex functions
have grouping and reshaping operations built into them to aid the exploration
of these high-dimensional relationships. Because seaborn is intended to aid
statistical visualization, the calculation and representation of measurement
uncertainty is also central to many of the tools.

For more detailed information and copious examples of the syntax and resulting
plots, please see the :ref:`tutorials <tutorials>` and :ref:`API reference <api_ref>`.

