.. _api_ref:

.. currentmodule:: seaborn

API reference
=============

.. _relational_api:

Relational plots
----------------

.. autosummary::
    :toctree: generated

    relplot
    scatterplot
    lineplot

.. _categorical_api:

Categorical plots
-----------------

.. autosummary::
    :toctree: generated/

    catplot
    stripplot
    swarmplot
    boxplot
    violinplot
    boxenplot
    pointplot
    barplot
    countplot

.. _distribution_api:

Distribution plots
------------------

.. autosummary::
    :toctree: generated/

    distplot
    kdeplot
    rugplot

.. _regression_api:

Regression plots
----------------

.. autosummary::
    :toctree: generated/

    lmplot
    regplot
    residplot

.. _matrix_api:

Matrix plots
------------

.. autosummary::
   :toctree: generated/

    heatmap
    clustermap

.. _grid_api:

Multi-plot grids
----------------

Facet grids
~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

    FacetGrid
    FacetGrid.map
    FacetGrid.map_dataframe

Pair grids
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

    pairplot
    PairGrid
    PairGrid.map
    PairGrid.map_diag
    PairGrid.map_offdiag
    PairGrid.map_lower
    PairGrid.map_upper

Joint grids
~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

    jointplot
    JointGrid
    JointGrid.plot
    JointGrid.plot_joint
    JointGrid.plot_marginals

.. _style_api:

Style control 
-------------

.. autosummary::
    :toctree: generated/

    set
    axes_style
    set_style
    plotting_context
    set_context
    set_color_codes
    reset_defaults
    reset_orig

.. _palette_api:

Color palettes
--------------

.. autosummary::
    :toctree: generated/

    set_palette
    color_palette
    husl_palette
    hls_palette
    cubehelix_palette
    dark_palette
    light_palette
    diverging_palette
    blend_palette
    xkcd_palette
    crayon_palette
    mpl_palette

Palette widgets
---------------

.. autosummary::
    :toctree: generated/

    choose_colorbrewer_palette
    choose_cubehelix_palette
    choose_light_palette
    choose_dark_palette
    choose_diverging_palette


Utility functions
-----------------

.. autosummary::
    :toctree: generated/

    load_dataset
    get_dataset_names
    get_data_home
    despine
    desaturate
    saturate
    set_hls_values
