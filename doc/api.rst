.. _api_ref:

API reference
=============

.. currentmodule:: seaborn.objects

.. _objects_api:

Objects interface
-----------------

Plot object
~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :template: plot
    :nosignatures:

    Plot

Mark objects
~~~~~~~~~~~~

.. rubric:: Dot marks

.. autosummary::
    :toctree: generated/
    :template: object
    :nosignatures:

    Dot
    Dots

.. rubric:: Line marks

.. autosummary::
    :toctree: generated/
    :template: object
    :nosignatures:

    Dash
    Line
    Lines
    Path
    Paths
    Range

.. rubric:: Bar marks

.. autosummary::
    :toctree: generated/
    :template: object
    :nosignatures:

    Bar
    Bars

.. rubric:: Fill marks

.. autosummary::
    :toctree: generated/
    :template: object
    :nosignatures:

    Area
    Band

.. rubric:: Text marks

.. autosummary::
    :toctree: generated/
    :template: object
    :nosignatures:

    Text

Stat objects
~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :template: object
    :nosignatures:

    Agg
    Est
    Hist
    Perc
    PolyFit

Move objects
~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :template: object
    :nosignatures:

    Dodge
    Jitter
    Norm
    Stack
    Shift

Scale objects
~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :template: scale
    :nosignatures:

    Continuous
    Nominal
    Temporal

Base classes
~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :template: object
    :nosignatures:

    Mark
    Stat
    Move
    Scale

.. currentmodule:: seaborn

Function interface
------------------

.. _relational_api:

Relational plots
~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :nosignatures:

    relplot
    scatterplot
    lineplot

.. _distribution_api:

Distribution plots
~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :nosignatures:

    lmplot
    regplot
    residplot

.. _matrix_api:

Matrix plots
~~~~~~~~~~~~

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

Pair grids
~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :nosignatures:

    pairplot
    PairGrid

Joint grids
~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :nosignatures:

    jointplot
    JointGrid

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
~~~~~~~~~~~~~~~

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

    despine
    move_legend
    saturate
    desaturate
    set_hls_values
    load_dataset
    get_dataset_names
    get_data_home
