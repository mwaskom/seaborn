from __future__ import division
import colorsys
from itertools import cycle

import numpy as np
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

from .external import husl
from .external.six import string_types
from .external.six.moves import range

from .utils import desaturate, set_hls_values
from .xkcd_rgb import xkcd_rgb
from .miscplot import palplot


class _ColorPalette(list):
    """Set the color palette in a with statement, otherwise be a list."""
    def __enter__(self):
        """Open the context."""
        from .rcmod import set_palette
        self._orig_palette = color_palette()
        set_palette(self, len(self))
        return self

    def __exit__(self, *args):
        """Close the context."""
        from .rcmod import set_palette
        set_palette(self._orig_palette, len(self._orig_palette))


def color_palette(name=None, n_colors=6, desat=None):
    """Return a list of colors defining a color palette.

    Availible seaborn palette names:
        deep, muted, bright, pastel, dark, colorblind

    Other options:
        hls, husl, any matplotlib palette

    Matplotlib paletes can be specified as reversed palettes by appending
    "_r" to the name or as dark palettes by appending "_d" to the name.

    This function can also be used in a ``with`` statement to temporarily
    set the color cycle for a plot or set of plots.

    Parameters
    ----------
    name: None, string, or sequence
        Name of palette or None to return current palette. If a
        sequence, input colors are used but possibly cycled and
        desaturated.
    n_colors : int
        Number of colors in the palette. If larger than the number of
        colors in the palette, they will cycle.
    desat : float
        Value to desaturate each color by.

    Returns
    -------
    palette : list of RGB tuples.
        Color palette.

    Examples
    --------
    >>> p = color_palette("muted")

    >>> p = color_palette("Blues_d", 10)

    >>> p = color_palette("Set1", desat=.7)

    >>> import matplotlib.pyplot as plt
    >>> with color_palette("husl", 8):
    ...     f, ax = plt.subplots()
    ...     ax.plot(x, y)                  # doctest: +SKIP

    See Also
    --------
    set_palette : set the default color cycle for all plots.
    axes_style : define parameters to set the style of plots
    plotting_context : define parameters to scale plot elements

    """
    seaborn_palettes = dict(
        deep=["#4C72B0", "#55A868", "#C44E52",
              "#8172B2", "#CCB974", "#64B5CD"],
        muted=["#4878CF", "#6ACC65", "#D65F5F",
               "#B47CC7", "#C4AD66", "#77BEDB"],
        pastel=["#92C6FF", "#97F0AA", "#FF9F9A",
                "#D0BBFF", "#FFFEA3", "#B0E0E6"],
        bright=["#003FFF", "#03ED3A", "#E8000B",
                "#8A2BE2", "#FFC400", "#00D7FF"],
        dark=["#001C7F", "#017517", "#8C0900",
              "#7600A1", "#B8860B", "#006374"],
        colorblind=["#0072B2", "#009E73", "#D55E00",
                    "#CC79A7", "#F0E442", "#56B4E9"],
    )

    if name is None:
        palette = mpl.rcParams["axes.color_cycle"]
    elif not isinstance(name, string_types):
        palette = name
    elif name == "hls":
        palette = hls_palette(n_colors)
    elif name == "husl":
        palette = husl_palette(n_colors)
    elif name.lower() == "jet":
        raise ValueError("No.")
    elif name in seaborn_palettes:
        palette = seaborn_palettes[name]
    elif name in dir(mpl.cm):
        palette = mpl_palette(name, n_colors)
    elif name[:-2] in dir(mpl.cm):
        palette = mpl_palette(name, n_colors)
    else:
        raise ValueError("%s is not a valid palette name" % name)

    if desat is not None:
        palette = [desaturate(c, desat) for c in palette]

    # Always return as many colors as we asked for
    pal_cycle = cycle(palette)
    palette = [next(pal_cycle) for _ in range(n_colors)]

    # Always return in r, g, b tuple format
    try:
        palette = map(mpl.colors.colorConverter.to_rgb, palette)
        palette = _ColorPalette(palette)
    except ValueError:
        raise ValueError("Could not generate a palette for %s" % str(name))

    return palette


def hls_palette(n_colors=6, h=.01, l=.6, s=.65):
    """Get a set of evenly spaced colors in HLS hue space.

    h, l, and s should be between 0 and 1

    Parameters
    ----------

    n_colors : int
        number of colors in the palette
    h : float
        first hue
    l : float
        lightness
    s : float
        saturation

    Returns
    -------
    palette : list of tuples
        color palette

    """
    hues = np.linspace(0, 1, n_colors + 1)[:-1]
    hues += h
    hues %= 1
    hues -= hues.astype(int)
    palette = [colorsys.hls_to_rgb(h_i, l, s) for h_i in hues]
    return palette


def husl_palette(n_colors=6, h=.01, s=.9, l=.65):
    """Get a set of evenly spaced colors in HUSL hue space.

    h, s, and l should be between 0 and 1

    Parameters
    ----------

    n_colors : int
        number of colors in the palette
    h : float
        first hue
    s : float
        saturation
    l : float
        lightness

    Returns
    -------
    palette : list of tuples
        color palette

    """
    hues = np.linspace(0, 1, n_colors + 1)[:-1]
    hues += h
    hues %= 1
    hues *= 359
    s *= 99
    l *= 99
    palette = [husl.husl_to_rgb(h_i, s, l) for h_i in hues]
    return palette


def mpl_palette(name, n_colors=6):
    """Return discrete colors from a matplotlib palette.

    Note that this handles the qualitative colorbrewer palettes
    properly, although if you ask for more colors than a particular
    qualitative palette can provide you will fewer than you are
    expecting.

    If you are using the IPython notebook, you can also use the function
    :func:`choose_colorbrewer_palette` to interactively select palettes.

    Parameters
    ----------
    name : string
        name of the palette
    n_colors : int
        number of colors in the palette

    Returns
    -------
    palette : list of tuples
        palette colors in r, g, b format

    """
    brewer_qual_pals = {"Accent": 8, "Dark2": 8, "Paired": 12,
                        "Pastel1": 9, "Pastel2": 8,
                        "Set1": 9, "Set2": 8, "Set3": 12}

    if name.endswith("_d"):
        pal = ["#333333"]
        pal.extend(color_palette(name.replace("_d", "_r"), 2))
        cmap = blend_palette(pal, n_colors, as_cmap=True)
    else:
        cmap = getattr(mpl.cm, name)
    if name in brewer_qual_pals:
        bins = np.linspace(0, 1, brewer_qual_pals[name])[:n_colors]
    else:
        bins = np.linspace(0, 1, n_colors + 2)[1:-1]
    palette = list(map(tuple, cmap(bins)[:, :3]))

    return palette


def _color_to_rgb(color, input):
    """Add some more flexibility to color choices."""
    if input == "hls":
        color = colorsys.hls_to_rgb(*color)
    elif input == "husl":
        color = husl.husl_to_rgb(*color)
    elif input == "xkcd":
        color = xkcd_rgb[color]
    return color


def dark_palette(color, n_colors=6, reverse=False, as_cmap=False, input="rgb"):
    """Make a sequential palette that blends from dark to ``color``.

    This kind of palette is good for data that range between relatively
    uninteresting low values and interesting high values.

    The ``color`` parameter can be specified in a number of ways, including
    all options for defining a color in matplotlib and several additional
    color spaces that are handled by seaborn. You can also use the database
    of named colors from the XKCD color survey.

    If you are using the IPython notebook, you can also choose this palette
    interactively with the :func:`choose_dark_palette` function.

    Parameters
    ----------
    color : base color for high values
        hex, rgb-tuple, or html color name
    n_colors : int, optional
        number of colors in the palette
    reverse : bool, optional
        if True, reverse the direction of the blend
    as_cmap : bool, optional
        if True, return as a matplotlib colormap instead of list
    input : {'rgb', 'hls', 'husl', xkcd'}
        Color space to interpret the input color. The first three options
        apply to tuple inputs and the latter applies to string inputs.

    Returns
    -------
    palette : list or colormap

    See Also
    --------
    light_palette : Create a sequential palette with bright low values.

    """
    color = _color_to_rgb(color, input)
    gray = "#222222"
    colors = [color, gray] if reverse else [gray, color]
    return blend_palette(colors, n_colors, as_cmap)


def light_palette(color, n_colors=6, reverse=False, as_cmap=False,
                  input="rgb"):
    """Make a sequential palette that blends from light to ``color``.

    This kind of palette is good for data that range between relatively
    uninteresting low values and interesting high values.

    The ``color`` parameter can be specified in a number of ways, including
    all options for defining a color in matplotlib and several additional
    color spaces that are handled by seaborn. You can also use the database
    of named colors from the XKCD color survey.

    If you are using the IPython notebook, you can also choose this palette
    interactively with the :func:`choose_light_palette` function.

    Parameters
    ----------
    color : base color for high values
        hex code, html color name, or tuple in ``input`` space.
    n_colors : int, optional
        number of colors in the palette
    reverse : bool, optional
        if True, reverse the direction of the blend
    as_cmap : bool, optional
        if True, return as a matplotlib colormap instead of list
    input : {'rgb', 'hls', 'husl', xkcd'}
        Color space to interpret the input color. The first three options
        apply to tuple inputs and the latter applies to string inputs.

    Returns
    -------
    palette : list or colormap

    See Also
    --------
    dark_palette : Create a sequential palette with dark low values.

    """
    color = _color_to_rgb(color, input)
    light = set_hls_values(color, l=.95)
    colors = [color, light] if reverse else [light, color]
    return blend_palette(colors, n_colors, as_cmap)


def _flat_palette(color, n_colors=6, reverse=False, as_cmap=False,
                  input="rgb"):
    """Make a sequential palette that blends from gray to ``color``.

    Parameters
    ----------
    color : matplotlib color
        hex, rgb-tuple, or html color name
    n_colors : int, optional
        number of colors in the palette
    reverse : bool, optional
        if True, reverse the direction of the blend
    as_cmap : bool, optional
        if True, return as a matplotlib colormap instead of list

    Returns
    -------
    palette : list or colormap
    dark_palette : Create a sequential palette with dark low values.

    """
    color = _color_to_rgb(color, input)
    flat = desaturate(color, 0)
    colors = [color, flat] if reverse else [flat, color]
    return blend_palette(colors, n_colors, as_cmap)


def diverging_palette(h_neg, h_pos, s=75, l=50, sep=10, n=6, center="light",
                      as_cmap=False):
    """Make a diverging palette between two HUSL colors.

    If you are using the IPython notebook, you can also choose this palette
    interactively with the :func:`choose_diverging_palette` function.

    Parameters
    ----------
    h_neg, h_pos : float in [0, 359]
        Anchor hues for negative and positive extents of the map.
    s : float in [0, 100], optional
        Anchor saturation for both extents of the map.
    l : float in [0, 100], optional
        Anchor lightness for both extents of the map.
    n : int, optional
        Number of colors in the palette (if not returning a cmap)
    center : {"light", "dark"}, optional
        Whether the center of the palette is light or dark
    as_cmap : bool, optional
        If true, return a matplotlib colormap object rather than a
        list of colors.

    Returns
    -------
    palette : array of r, g, b colors or colormap

    """

    palfunc = dark_palette if center == "dark" else light_palette
    neg = palfunc((h_neg, s, l), 128 - (sep / 2), reverse=True, input="husl")
    pos = palfunc((h_pos, s, l), 128 - (sep / 2), input="husl")
    midpoint = dict(light=[(.95, .95, .95, 1.)],
                    dark=[(.133, .133, .133, 1.)])[center]
    mid = midpoint * sep
    pal = blend_palette(np.concatenate([neg, mid,  pos]), n, as_cmap=as_cmap)
    return pal


def blend_palette(colors, n_colors=6, as_cmap=False, input="rgb"):
    """Make a palette that blends between a list of colors.

    Parameters
    ----------
    colors : sequence of matplotlib colors
        hex, rgb-tuple, or html color name
    n_colors : int, optional
        number of colors in the palette
    as_cmap : bool, optional
        if True, return as a matplotlib colormap instead of list

    Returns
    -------
    palette : list or colormap

    """
    colors = [_color_to_rgb(color, input) for color in colors]
    name = "blend"
    pal = mpl.colors.LinearSegmentedColormap.from_list(name, colors)
    if not as_cmap:
        pal = pal(np.linspace(0, 1, n_colors))
    return pal


def xkcd_palette(colors):
    """Make a palette with color names from the xkcd color survey.

    This is just a simple wrapper around the seaborn.xkcd_rbg dictionary.

    See xkcd for the full list of colors: http://xkcd.com/color/rgb/

    """
    palette = [xkcd_rgb[name] for name in colors]
    return color_palette(palette, len(palette))


def cubehelix_palette(n_colors=6, start=0, rot=.4, gamma=1.0, hue=0.8,
                      light=.85, dark=.15, reverse=False, as_cmap=False):
    """Make a sequential palette from the cubehelix system.

    This produces a colormap with linearly-decreasing (or increasing)
    brightness. That means that information will be preserved if printed to
    black and white or viewed by someone who is colorblind.  "cubehelix" is
    also availible as a matplotlib-based palette, but this function gives the
    user more control over the look of the palette and has a different set of
    defaults.

    Parameters
    ----------
    n_colors : int
        Number of colors in the palette.
    start : float, 0 <= start <= 3
        The hue at the start of the helix.
    rot : float
        Rotations around the hue wheel over the range of the palette.
    gamma : float 0 <= gamma
        Gamma factor to emphasize darker (gamma < 1) or lighter (gamma > 1)
        colors.
    hue : float, 0 <= hue <= 1
        Saturation of the colors.
    dark : float 0 <= dark <= 1
        Intensity of the darkest color in the palette.
    light : float 0 <= light <= 1
        Intensity of the lightest color in the palette.
    reverse : bool
        If True, the palette will go from dark to light.
    as_cmap : bool
        If True, return a matplotlib colormap instead of a list of colors.

    Returns
    -------
    palette : list or colormap

    See Also
    --------
    choose_cubehelix_palette : Launch an interactive widget to select cubehelix
                               palette parameters.
    dark_palette : Create a sequential palette with dark low values.
    light_palette : Create a sequential palette with bright low values.

    References
    ----------
    Green, D. A. (2011). "A colour scheme for the display of astronomical
    intensity images". Bulletin of the Astromical Society of India, Vol. 39,
    p. 289-295.

    """
    cdict = mpl._cm.cubehelix(gamma, start, rot, hue)
    cmap = mpl.colors.LinearSegmentedColormap("cubehelix", cdict)

    x = np.linspace(light, dark, n_colors)
    pal = cmap(x)[:, :3].tolist()
    if reverse:
        pal = pal[::-1]

    if as_cmap:
        x_256 = np.linspace(light, dark, 256)
        if reverse:
            x_256 = x_256[::-1]
        pal_256 = cmap(x_256)
        cmap = mpl.colors.ListedColormap(pal_256)
        return cmap
    else:
        return pal


def _init_mutable_colormap():
    """Create a matplotlib colormap that will be updated by the widgets."""
    greys = color_palette("Greys", 256)
    cmap = LinearSegmentedColormap.from_list("interactive", greys)
    cmap._init()
    cmap._set_extremes()
    return cmap


def _update_lut(cmap, colors):
    """Change the LUT values in a matplotlib colormap in-place."""
    cmap._lut[:256] = colors
    cmap._set_extremes()


def choose_colorbrewer_palette(data_type, as_cmap=False):
    """Select a palette from the ColorBrewer set.

    These palettes are built into matplotlib and can be used by name in
    many seaborn functions, or by passing the object returned by this function.

    Parameters
    ----------
    data_type : {'sequential', 'diverging', 'qualitative'}
        This describes the kind of data you want to visualize. See the seaborn
        color palette docs for more information about how to choose this value.
        Note that you can pass substrings (e.g. 'q' for 'qualitative.

    as_cmap : bool
        If True, the return value is a matplotlib colormap rather than a
        list of discrete colors.

    Returns
    -------
    pal or cmap : list of colors or matplotlib colormap
        Object that can be passed to plotting functions.

    See Also
    --------
    dark_palette : Create a sequential palette with dark low values.
    light_palette : Create a sequential palette with bright low values.
    diverging_palette : Create a diverging palette from selected colors.
    cubehelix_palette : Create a sequential palette or colormap using the
                        cubehelix system.


    """
    from IPython.html.widgets import interact, FloatSliderWidget

    if data_type.startswith("q") and as_cmap:
        raise ValueError("Qualitative palettes cannot be colormaps.")

    pal = []
    if as_cmap:
        cmap = _init_mutable_colormap()

    if data_type.startswith("s"):
        opts = ["Greys", "Reds", "Greens", "Blues", "Oranges", "Purples",
                "BuGn", "BuPu", "GnBu", "OrRd", "PuBu", "PuRd", "RdPu", "YlGn",
                "PuBuGn", "YlGnBu", "YlOrBr", "YlOrRd"]
        variants = ["regular", "reverse", "dark"]

        @interact
        def choose_sequential(name=opts, n=(2, 18),
                              desat=FloatSliderWidget(min=0, max=1, value=1),
                              variant=variants):
            if variant == "reverse":
                name += "_r"
            elif variant == "dark":
                name += "_d"

            pal[:] = color_palette(name, n, desat)
            palplot(pal)
            if as_cmap:
                colors = color_palette(name, 256, desat)
                _update_lut(cmap, np.c_[colors, np.ones(256)])

    elif data_type.startswith("d"):
        opts = ["RdBu", "RdGy", "PRGn", "PiYG", "BrBG",
                "RdYlBu", "RdYlGn", "Spectral"]
        variants = ["regular", "reverse"]

        @interact
        def choose_diverging(name=opts, n=(2, 16),
                             desat=FloatSliderWidget(min=0, max=1, value=1),
                             variant=variants):
            if variant == "reverse":
                name += "_r"
            pal[:] = color_palette(name, n, desat)
            palplot(pal)
            if as_cmap:
                colors = color_palette(name, 256, desat)
                _update_lut(cmap, np.c_[colors, np.ones(256)])

    elif data_type.startswith("q"):
        opts = ["Set1", "Set2", "Set3", "Paired", "Accent",
                "Pastel1", "Pastel2", "Dark2"]

        @interact
        def choose_qualitative(name=opts, n=(2, 16),
                               desat=FloatSliderWidget(min=0, max=1, value=1)):
            pal[:] = color_palette(name, n, desat)
            palplot(pal)

    if as_cmap:
        return cmap
    return pal


def choose_dark_palette(input="husl", as_cmap=False):
    """Launch an interactive widget to create a dark sequential palette.

    This corresponds with the :func:`dark_palette` function. This kind
    of palette is good for data that range between relatively uninteresting
    low values and interesting high values.

    Requires IPython 2+ and must be used in the notebook.

    Parameters
    ----------
    input : {'husl', 'hls', 'rgb'}
        Color space for defining the seed value. Note that the default is
        different than the default input for :func:`dark_palette`.
    as_cmap : bool
        If True, the return value is a matplotlib colormap rather than a
        list of discrete colors.

    Returns
    -------
    pal or cmap : list of colors or matplotlib colormap
        Object that can be passed to plotting functions.

    See Also
    --------
    dark_palette : Create a sequential palette with dark low values.
    light_palette : Create a sequential palette with bright low values.
    cubehelix_palette : Create a sequential palette or colormap using the
                        cubehelix system.

    """
    from IPython.html.widgets import interact

    pal = []
    if as_cmap:
        cmap = _init_mutable_colormap()

    if input == "rgb":
        @interact
        def choose_dark_palette_rgb(r=(0., 1.),
                                    g=(0., 1.),
                                    b=(0., 1.),
                                    n=(3, 17)):
            color = r, g, b
            pal[:] = dark_palette(color, n, input="rgb")
            palplot(pal)
            if as_cmap:
                colors = dark_palette(color, 256, input="rgb")
                _update_lut(cmap, colors)

    elif input == "hls":
        @interact
        def choose_dark_palette_hls(h=(0., 1.),
                                    l=(0., 1.),
                                    s=(0., 1.),
                                    n=(3, 17)):
            color = h, l, s
            pal[:] = dark_palette(color, n, input="hls")
            palplot(pal)
            if as_cmap:
                colors = dark_palette(color, 256, input="hls")
                _update_lut(cmap, colors)

    elif input == "husl":
        @interact
        def choose_dark_palette_husl(h=(0, 359),
                                     s=(0, 99),
                                     l=(0, 99),
                                     n=(3, 17)):
            color = h, s, l
            pal[:] = dark_palette(color, n, input="husl")
            palplot(pal)
            if as_cmap:
                colors = dark_palette(color, 256, input="husl")
                _update_lut(cmap, colors)

    if as_cmap:
        return cmap
    return pal


def choose_light_palette(input="husl", as_cmap=False):
    """Launch an interactive widget to create a light sequential palette.

    This corresponds with the :func:`light_palette` function. This kind
    of palette is good for data that range between relatively uninteresting
    low values and interesting high values.

    Requires IPython 2+ and must be used in the notebook.

    Parameters
    ----------
    input : {'husl', 'hls', 'rgb'}
        Color space for defining the seed value. Note that the default is
        different than the default input for :func:`light_palette`.
    as_cmap : bool
        If True, the return value is a matplotlib colormap rather than a
        list of discrete colors.

    Returns
    -------
    pal or cmap : list of colors or matplotlib colormap
        Object that can be passed to plotting functions.

    See Also
    --------
    light_palette : Create a sequential palette with bright low values.
    dark_palette : Create a sequential palette with dark low values.
    cubehelix_palette : Create a sequential palette or colormap using the
                        cubehelix system.

    """
    from IPython.html.widgets import interact

    pal = []
    if as_cmap:
        cmap = _init_mutable_colormap()

    if input == "rgb":
        @interact
        def choose_light_palette_rgb(r=(0., 1.),
                                     g=(0., 1.),
                                     b=(0., 1.),
                                     n=(3, 17)):
            color = r, g, b
            pal[:] = light_palette(color, n, input="rgb")
            palplot(pal)
            if as_cmap:
                colors = light_palette(color, 256, input="husl")
                _update_lut(cmap, colors)

    elif input == "hls":
        @interact
        def choose_light_palette_hls(h=(0., 1.),
                                     l=(0., 1.),
                                     s=(0., 1.),
                                     n=(3, 17)):
            color = h, l, s
            pal[:] = light_palette(color, n, input="hls")
            palplot(pal)
            if as_cmap:
                colors = light_palette(color, 256, input="husl")
                _update_lut(cmap, colors)

    elif input == "husl":
        @interact
        def choose_light_palette_husl(h=(0, 359),
                                      s=(0, 99),
                                      l=(0, 99),
                                      n=(3, 17)):
            color = h, s, l
            pal[:] = light_palette(color, n, input="husl")
            palplot(pal)
            if as_cmap:
                colors = light_palette(color, 256, input="husl")
                _update_lut(cmap, colors)

    if as_cmap:
        return cmap
    return pal


def choose_diverging_palette(as_cmap=False):
    """Launch an interactive widget to choose a diverging color palette.

    This corresponds with the :func:`diverging_palette` function. This kind
    of palette is good for data that range between interesting low values
    and interesting high values with a meaningful midpoint. (For example,
    change scores relative to some baseline value).

    Requires IPython 2+ and must be used in the notebook.

    Parameters
    ----------
    as_cmap : bool
        If True, the return value is a matplotlib colormap rather than a
        list of discrete colors.

    Returns
    -------
    pal or cmap : list of colors or matplotlib colormap
        Object that can be passed to plotting functions.

    See Also
    --------
    diverging_palette : Create a diverging color palette or colormap.
    choose_colorbrewer_palette : Interactively choose palettes from the
                                 colorbrewer set, including diverging palettes.

    """
    from IPython.html.widgets import interact, IntSliderWidget

    pal = []
    if as_cmap:
        cmap = _init_mutable_colormap()

    @interact
    def choose_diverging_palette(h_neg=IntSliderWidget(min=0,
                                                       max=359,
                                                       value=220),
                                 h_pos=IntSliderWidget(min=0,
                                                       max=359,
                                                       value=10),
                                 s=IntSliderWidget(min=0, max=99, value=74),
                                 l=IntSliderWidget(min=0, max=99, value=50),
                                 sep=IntSliderWidget(min=1, max=50, value=10),
                                 n=(2, 16),
                                 center=["light", "dark"]):
        pal[:] = diverging_palette(h_neg, h_pos, s, l, sep, n, center)
        palplot(pal)
        if as_cmap:
            colors = diverging_palette(h_neg, h_pos, s, l, sep, 256, center)
            _update_lut(cmap, colors)

    if as_cmap:
        return cmap
    return pal


def choose_cubehelix_palette(as_cmap=False):
    """Launch an interactive widget to create a sequential cubehelix palette.

    This corresponds with the :func:`cubehelix_palette` function. This kind
    of palette is good for data that range between relatively uninteresting
    low values and interesting high values. The cubehelix system allows the
    palette to have more hue variance across the range, which can be helpful
    for distinguishing a wider range of values.

    Requires IPython 2+ and must be used in the notebook.

    Parameters
    ----------
    as_cmap : bool
        If True, the return value is a matplotlib colormap rather than a
        list of discrete colors.

    Returns
    -------
    pal or cmap : list of colors or matplotlib colormap
        Object that can be passed to plotting functions.

    See Also
    --------
    cubehelix_palette : Create a sequential palette or colormap using the
                        cubehelix system.

    """
    from IPython.html.widgets import (interact,
                                      FloatSliderWidget, IntSliderWidget)

    pal = []
    if as_cmap:
        cmap = _init_mutable_colormap()

    @interact
    def choose_cubehelix(n_colors=IntSliderWidget(min=2, max=16, value=9),
                         start=FloatSliderWidget(min=0, max=3, value=0),
                         rot=FloatSliderWidget(min=-1, max=1, value=.4),
                         gamma=FloatSliderWidget(min=0, max=5, value=1),
                         hue=FloatSliderWidget(min=0, max=1, value=.8),
                         light=FloatSliderWidget(min=0, max=1, value=.85),
                         dark=FloatSliderWidget(min=0, max=1, value=.15),
                         reverse=False):
        pal[:] = cubehelix_palette(n_colors, start, rot, gamma,
                                   hue, light, dark, reverse)
        palplot(pal)
        if as_cmap:
            colors = cubehelix_palette(256, start, rot, gamma,
                                       hue, light, dark, reverse)
            _update_lut(cmap, np.c_[colors, np.ones(256)])

    if as_cmap:
        return cmap
    return pal
