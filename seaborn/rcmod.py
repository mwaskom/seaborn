import matplotlib as mpl


def setup(context="notebook", style="darkgrid", palette="deep"):
    """Set new RC params in one step."""
    context_setting(context)
    axes_style(style)
    color_palatte(palette)
    params = {"figure.figsize": (9, 6.5),
              "lines.linewidth": 1.4,
              "patch.linewidth": .3}
    mpl.rcParams.update(params)


def reset_defaults():
    """Restore all RC params to default settings."""
    mpl.rcParams.update(mpl.rcParamsDefault)


def axes_style(style):
    """Set the axis style.

    Parameters
    ----------
    style : darkgrid | whitegrid | nogrid
        Style of axis background.

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
    """Set some visual parameters based on intended context.

    Currently just changes font sizes

    Parameters
    ----------
    context: notebook | talk | paper
        Intended context for resulting figures.

    """
    if context == "talk":
        params = {"axes.labelsize": 17,
                  "axes.titlesize": 19,
                  "xtick.labelsize": 16,
                  "ytick.labelsize": 16,
                  }

    elif context == "notebook":
        params = {"axes.labelsize": 12,
                  "axes.titlesize": 14,
                  "xtick.labelsize": 11,
                  "ytick.labelsize": 11,
                  }

    elif context == "paper":
        params = {"axes.labelsize": 13,
                  "axes.titlesize": 16,
                  "xtick.labelsize": 12,
                  "ytick.labelsize": 12,
                  }

    else:
        raise ValueError("Context %s is not recognized" % context)

    mpl.rcParams.update(params)


def color_palatte(name):
    """Set the matplotlib color order with one of several palattes."""
    colors = get_color_list(name)
    mpl.rcParams["axes.color_cycle"] = colors
    mpl.rcParams["patch.facecolor"] = colors[0]


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
