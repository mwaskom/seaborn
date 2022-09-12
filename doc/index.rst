:html_theme.sidebar_secondary.remove:

seaborn: statistical data visualization
=======================================

.. grid:: 6
  :gutter: 1

  .. grid-item::

      .. image:: example_thumbs/scatterplot_matrix_thumb.png
        :target: ./examples/scatterplot_matrix.html

  .. grid-item::

      .. image:: example_thumbs/errorband_lineplots_thumb.png
        :target: examples/errorband_lineplots.html

  .. grid-item::

      .. image:: example_thumbs/scatterplot_sizes_thumb.png
        :target: examples/scatterplot_sizes.html

  .. grid-item::

      .. image:: example_thumbs/timeseries_facets_thumb.png
        :target: examples/timeseries_facets.html

  .. grid-item::

      .. image:: example_thumbs/horizontal_boxplot_thumb.png
        :target: examples/horizontal_boxplot.html

  .. grid-item::

      .. image:: example_thumbs/regression_marginals_thumb.png
        :target: examples/regression_marginals.html

.. grid:: 1 1 3 3

  .. grid-item::
    :columns: 12 12 6 6

    Seaborn is a Python data visualization library based on `matplotlib
    <https://matplotlib.org>`_. It provides a high-level interface for drawing
    attractive and informative statistical graphics.

    For a brief introduction to the ideas behind the library, you can read the
    :doc:`introductory notes <tutorial/introduction>` or the `paper
    <https://joss.theoj.org/papers/10.21105/joss.03021>`_. Visit the
    :doc:`installation page <installing>` to see how you can download the package
    and get started with it. You can browse the :doc:`example gallery
    <examples/index>` to see some of the things that you can do with seaborn,
    and then check out the :doc:`tutorials <tutorial>` or :doc:`API reference <api>`
    to find out how.

    To see the code or report a bug, please visit the `GitHub repository
    <https://github.com/mwaskom/seaborn>`_. General support questions are most at home
    on `stackoverflow <https://stackoverflow.com/questions/tagged/seaborn/>`_, which
    has a dedicated channel for seaborn.

  .. grid-item-card:: Contents
    :columns: 12 12 2 2
    :class-title: sd-fs-5
    :class-body: sd-pl-4

    .. toctree::
      :maxdepth: 1

      Installing <installing>
      Gallery <examples/index>
      Tutorial <tutorial>
      API <api>
      Releases <whatsnew/index>
      Citing <citing>
      FAQ <faq>

  .. grid-item-card:: Features
    :columns: 12 12 4 4
    :class-title: sd-fs-5
    :class-body: sd-pl-3

    * :bdg-secondary:`New` Objects: :ref:`API <objects_api>` | :doc:`Tutorial <tutorial/objects_interface>`
    * Relational plots: :ref:`API <relational_api>` | :doc:`Tutorial <tutorial/relational>`
    * Distribution plots: :ref:`API <distribution_api>` | :doc:`Tutorial <tutorial/distributions>`
    * Categorical plots: :ref:`API <categorical_api>` | :doc:`Tutorial <tutorial/categorical>`
    * Regression plots: :ref:`API <regression_api>` | :doc:`Tutorial <tutorial/regression>`
    * Multi-plot grids: :ref:`API <grid_api>` | :doc:`Tutorial <tutorial/axis_grids>`
    * Figure theming: :ref:`API <style_api>` | :doc:`Tutorial <tutorial/aesthetics>`
    * Color palettes: :ref:`API <palette_api>` | :doc:`Tutorial <tutorial/color_palettes>`
