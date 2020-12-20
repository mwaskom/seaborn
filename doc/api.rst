.. _api_ref:

.. currentmodule:: seaborn

API reference
=============

.. _relational_api:

Relational plots
----------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    relplot
    scatterplot
    lineplot

.. _distribution_api:

Distribution plots
------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    displot
    histplot
    kdeplot
    ecdfplot
    rugplot
    distplot

.. _categorical_api:

Categorical plots
-----------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    catplot
    stripplot
    swarmplot
    boxplot
    violinplot
    boxenplot
    pointplot
    barplot
    countplot

.. _regression_api:

Regression plots
----------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    lmplot
    regplot
    residplot

.. _matrix_api:

Matrix plots
------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    heatmap
    clustermap

.. _grid_api:

Multi-plot grids
----------------

Facet grids
~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :nosignatures:

    FacetGrid
    FacetGrid.map
    FacetGrid.map_dataframe

Pair grids
~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :nosignatures:

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
    :nosignatures:

    jointplot
    JointGrid
    JointGrid.plot
    JointGrid.plot_joint
    JointGrid.plot_marginals

.. _style_api:

Themeing
--------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    set_theme
    axes_style
    set_style
    plotting_context
    set_context
    set_color_codes
    reset_defaults
    reset_orig
    set

.. _palette_api:

Color palettes
--------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

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
    :nosignatures:

    choose_colorbrewer_palette
    choose_cubehelix_palette
    choose_light_palette
    choose_dark_palette
    choose_diverging_palette


Utility functions
-----------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    load_dataset
    get_dataset_names
    get_data_home
    despine
    desaturate
    saturate
    set_hls_values
