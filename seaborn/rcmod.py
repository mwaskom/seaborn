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
                            "grid.linewidth": 1.5})
        _blank_ticks(grid_params)
        mpl.rcParams.update(grid_params)

    elif style == "whitegrid":
        grid_params.update({"axes.facecolor": "white",
                            "axes.linewidth": 1,
                            "grid.color": "#222222",
                            "grid.linestyle": ":",
                            "grid.linewidth": .8})
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


def color_palatte(name):
    """Set the matplotlib color order with one of several palattes."""
    mpl.rcParams["axes.color_cycle"] = get_color_list(name)


def get_color_list(name):
    """Return matplotlib color codes for a given palette."""
    palattes = dict(
        default=["b", "g", "r", "c", "m", "y", "k"],
        pastel=["#92C6FF", "#97F0AA", "#FF9F9A", "#D0BBFF", "#FFFEA3"],
        bright=["#003FFF", "#03ED3A", "#E8000B", "#00D7FF", "#FFB400"],
        muted=["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#C4AD66"],
        deep=["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"],
        dark=["#001C7F", "#017517", "#8C0900", "#7600A1", "#007364"],
        )

    return palattes[name]


def _blank_ticks(params):
    """Turn off x and y ticks in a param dict (but not labels)."""
    for axis in ["x", "y"]:
        for step in ["major", "minor"]:
            params["%stick.%s.size" % (axis, step)] = 0


def _restore_ticks(params):
    """Reset x and y ticks in a param dict to matplotlib defaults."""
    for axis in ["x", "y"]:
        for step, size in zip(["major", "minor"], [4, 2]):
            params["%stick.%s.size" % (axis, step)] = size
