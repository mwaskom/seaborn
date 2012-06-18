import matplotlib as mpl


def axes_style(style):
    """Set the axis style.

    Parameters
    ----------
    style : "darkgrid", "whitegrid", or "nogrid"
        Modfile style to look ggplotish or light grid on white

    """
    grid_params = {"axes.grid": True,
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
        _restore_ticks(grid_params)
        mpl.rcParams.update(grid_params)

    elif style == "nogrid":
        params = {"axes.grid": False,
                  "axes.linecolor": 1,
                  "axes.facecolor": "white"}
        _restore_ticks(params)
        mpl.rcParams.update(params)

    else:
        raise ValueError("Style %s not recognized" % style)


def context_setting(context):

    # Possible values:
    # - notebook
    # - talk
    # - paper

    raise NotImplementedError

def _restore_ticks(params):

    for axis in ["x", "y"]:
        for step, size in zip(["major", "minor"], [4, 2]):
            params["%stick.%s.size" % (axis, step)] = size
