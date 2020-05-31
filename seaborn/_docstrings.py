
from types import SimpleNamespace

_core_params = dict(
    data="""\
data : :class:`pandas.DataFrame`, :class:`numpy.ndarray`, mapping, or sequence
    Input data structure. Either a long-form collection of vectors that can be
    assigned to named variables or a wide-form dataset that will be reshaped
    and used *in toto*.
    """,  # TODO add link to user guide narrative when exists
    xy="""\
x, y : vectors or keys in ``data``
    Variables that specify positions on the x and y axes.
    """,
    hue="""\
hue : vector or key in ``data``
    Semantic variable that is mapped to determine the color of plot elements.
    """,
    ax="""\
ax : :class:`matplotlib.axes.Axes`
    Pre-existing axes for the plot. Otherwise, call :func:`matplotlib.pyplot.gca`
    internally.\
    """,  # noqa: E501
    palette="""\
palette : string, list, dict, or :class:`matplotlib.colors.Colormap`
    Method for choosing the colors to use when mapping the ``hue`` semantic.
    String values are passed to :func:`color_palette`. List or dict values
    imply categorical mapping, while a colormap object implies numeric mapping.
    """,  # noqa: E501
    hue_order="""\
hue_order : vector of strings
    Specify the order of processing and plotting for categorical levels of the
    ``hue`` semantic.
    """,
    hue_norm="""
hue_norm : tuple or :class:`matplotlib.colors.Normalize`
    Either a pair of values that set the normalization range in data units
    for numeric ``hue`` mapping. Can also bean object that will map from data
    units into a [0, 1] interval.
    """,

    # TODO add a generic matplotlib color entry with intersphinx link
)


_core_params["hue_mapping"] = "\n".join(
    _core_params[k] for k in ["palette", "hue_order", "hue_norm"]
)


_core_returns = dict(
    ax="""\
ax : :class:`matplotlib.axes.Axes`
    The matplotlib axes containing the plot.
    """,
)


_seealso_blurbs = dict(

    # Distribution plots
    rugplot="""\
rugplot : Plot a tick at each observation value along the x and/or y axes.
    """,

    # Categorical plots
    violinplot="""\
violinplot : Draw an enhanced boxplot using kernel density estimation.
    """,

    # Multiples
    jointplot="""\
jointplot : Draw a bivariate plot with univariate marginal distributions.
    """,
)


_core_docs = dict(
    params=SimpleNamespace(**_core_params),
    returns=SimpleNamespace(**_core_returns),
    seealso=SimpleNamespace(**_seealso_blurbs),
)
