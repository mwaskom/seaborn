import matplotlib as mpl


def axes_style(style):
    """Set the axis style.

    Parameters
    ----------
    style : "ggplot" or "grid"
        Modfile style to look ggplotish or light grid on white

    """
    if style == "ggplot":
        mpl.rcparams.update({"axes.axisbelow": True,
                             "axes.facecolor": "#EAEAF2",
                             "grid.color": "w",
                             "grid.linestyle": "-",
                             "grid.linewidth": 2})
    elif style == "grid":
        mpl.rcparams.update({"axes.axisbelow": True,
                             "axes.facecolor": "white",
                             "grid.color": "k",
                             "grid.linestyle": ":",
                             "grid.linewidth": 1.3})

    else:
        raise ValueError("Style %s not recognized" % style)


def context_setting(context):

    # Possible values:
    # - notebook
    # - talk
    # - paper

    raise NotImplementedError
