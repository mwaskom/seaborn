import matplotlib as mpl


def axes_style(style):
    """Set the axis style.

    Parameters
    ----------
    style : "darkgrid" or "whitegrid"
        Modfile style to look ggplotish or light grid on white

    """
    grid_params = {"axes.grid" : True,
                   "axes.axisbelow": True}

    if style == "darkgrid":
        grid_params.update({"axes.facecolor": "#EAEAF2",
                            "axes.linewidth": 0,
                            "grid.color": "w",
                            "grid.linestyle": "-",
                            "grid.linewidth": 1.5,})
        for axis in ["x", "y"]:
            for step in ["major", "minor"]:
                grid_params["%stick.%s.size" % (axis, step)] = 0
        mpl.rcParams.update(grid_params)

    elif style == "whitegrid":
        grid_params.update({"axes.facecolor": "white",
                            "axes.linewidth": 1,
                            "grid.color": "#222222",
                            "grid.linestyle": ":",
                            "grid.linewidth": .8,})
        mpl.rcParams.update(grid_params)

    else:
        raise ValueError("Style %s not recognized" % style)


def context_setting(context):

    # Possible values:
    # - notebook
    # - talk
    # - paper

    raise NotImplementedError
