"""Small plotting-related utility functions."""
import colorsys
import os

import numpy as np
from scipy import stats
import pandas as pd
import matplotlib as mpl
import matplotlib.colors as mplcol
import matplotlib.pyplot as plt

import warnings
from urllib.request import urlopen, urlretrieve
from http.client import HTTPException


__all__ = ["desaturate", "saturate", "set_hls_values",
           "despine", "get_dataset_names", "get_data_home", "load_dataset"]


def remove_na(arr):
    """Helper method for removing NA values from array-like.

    Parameters
    ----------
    arr : array-like
        The array-like from which to remove NA values.

    Returns
    -------
    clean_arr : array-like
        The original array with NA values removed.

    """
    return arr[pd.notnull(arr)]


def sort_df(df, *args, **kwargs):
    """Wrapper to handle different pandas sorting API pre/post 0.17."""
    msg = "This function is deprecated and will be removed in a future version"
    warnings.warn(msg)
    try:
        return df.sort_values(*args, **kwargs)
    except AttributeError:
        return df.sort(*args, **kwargs)


def ci_to_errsize(cis, heights):
    """Convert intervals to error arguments relative to plot heights.

    Parameters
    ----------
    cis: 2 x n sequence
        sequence of confidence interval limits
    heights : n sequence
        sequence of plot heights

    Returns
    -------
    errsize : 2 x n array
        sequence of error size relative to height values in correct
        format as argument for plt.bar

    """
    cis = np.atleast_2d(cis).reshape(2, -1)
    heights = np.atleast_1d(heights)
    errsize = []
    for i, (low, high) in enumerate(np.transpose(cis)):
        h = heights[i]
        elow = h - low
        ehigh = high - h
        errsize.append([elow, ehigh])

    errsize = np.asarray(errsize).T
    return errsize


def pmf_hist(a, bins=10):
    """Return arguments to plt.bar for pmf-like histogram of an array.

    DEPRECATED: will be removed in a future version.

    Parameters
    ----------
    a: array-like
        array to make histogram of
    bins: int
        number of bins

    Returns
    -------
    x: array
        left x position of bars
    h: array
        height of bars
    w: float
        width of bars

    """
    msg = "This function is deprecated and will be removed in a future version"
    warnings.warn(msg)
    n, x = np.histogram(a, bins)
    h = n / n.sum()
    w = x[1] - x[0]
    return x[:-1], h, w


def desaturate(color, prop):
    """Decrease the saturation channel of a color by some percent.

    Parameters
    ----------
    color : matplotlib color
        hex, rgb-tuple, or html color name
    prop : float
        saturation channel of color will be multiplied by this value

    Returns
    -------
    new_color : rgb tuple
        desaturated color code in RGB tuple representation

    """
    # Check inputs
    if not 0 <= prop <= 1:
        raise ValueError("prop must be between 0 and 1")

    # Get rgb tuple rep
    rgb = mplcol.colorConverter.to_rgb(color)

    # Convert to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)

    # Desaturate the saturation channel
    s *= prop

    # Convert back to rgb
    new_color = colorsys.hls_to_rgb(h, l, s)

    return new_color


def saturate(color):
    """Return a fully saturated color with the same hue.

    Parameters
    ----------
    color : matplotlib color
        hex, rgb-tuple, or html color name

    Returns
    -------
    new_color : rgb tuple
        saturated color code in RGB tuple representation

    """
    return set_hls_values(color, s=1)


def set_hls_values(color, h=None, l=None, s=None):  # noqa
    """Independently manipulate the h, l, or s channels of a color.

    Parameters
    ----------
    color : matplotlib color
        hex, rgb-tuple, or html color name
    h, l, s : floats between 0 and 1, or None
        new values for each channel in hls space

    Returns
    -------
    new_color : rgb tuple
        new color code in RGB tuple representation

    """
    # Get an RGB tuple representation
    rgb = mplcol.colorConverter.to_rgb(color)
    vals = list(colorsys.rgb_to_hls(*rgb))
    for i, val in enumerate([h, l, s]):
        if val is not None:
            vals[i] = val

    rgb = colorsys.hls_to_rgb(*vals)
    return rgb


def axlabel(xlabel, ylabel, **kwargs):
    """Grab current axis and label it."""
    ax = plt.gca()
    ax.set_xlabel(xlabel, **kwargs)
    ax.set_ylabel(ylabel, **kwargs)


def despine(fig=None, ax=None, top=True, right=True, left=False,
            bottom=False, offset=None, trim=False):
    """Remove the top and right spines from plot(s).

    fig : matplotlib figure, optional
        Figure to despine all axes of, default uses current figure.
    ax : matplotlib axes, optional
        Specific axes object to despine.
    top, right, left, bottom : boolean, optional
        If True, remove that spine.
    offset : int or dict, optional
        Absolute distance, in points, spines should be moved away
        from the axes (negative values move spines inward). A single value
        applies to all spines; a dict can be used to set offset values per
        side.
    trim : bool, optional
        If True, limit spines to the smallest and largest major tick
        on each non-despined axis.

    Returns
    -------
    None

    """
    # Get references to the axes we want
    if fig is None and ax is None:
        axes = plt.gcf().axes
    elif fig is not None:
        axes = fig.axes
    elif ax is not None:
        axes = [ax]

    for ax_i in axes:
        for side in ["top", "right", "left", "bottom"]:
            # Toggle the spine objects
            is_visible = not locals()[side]
            ax_i.spines[side].set_visible(is_visible)
            if offset is not None and is_visible:
                try:
                    val = offset.get(side, 0)
                except AttributeError:
                    val = offset
                ax_i.spines[side].set_position(('outward', val))

        # Potentially move the ticks
        if left and not right:
            maj_on = any(
                t.tick1line.get_visible()
                for t in ax_i.yaxis.majorTicks
            )
            min_on = any(
                t.tick1line.get_visible()
                for t in ax_i.yaxis.minorTicks
            )
            ax_i.yaxis.set_ticks_position("right")
            for t in ax_i.yaxis.majorTicks:
                t.tick2line.set_visible(maj_on)
            for t in ax_i.yaxis.minorTicks:
                t.tick2line.set_visible(min_on)

        if bottom and not top:
            maj_on = any(
                t.tick1line.get_visible()
                for t in ax_i.xaxis.majorTicks
            )
            min_on = any(
                t.tick1line.get_visible()
                for t in ax_i.xaxis.minorTicks
            )
            ax_i.xaxis.set_ticks_position("top")
            for t in ax_i.xaxis.majorTicks:
                t.tick2line.set_visible(maj_on)
            for t in ax_i.xaxis.minorTicks:
                t.tick2line.set_visible(min_on)

        if trim:
            # clip off the parts of the spines that extend past major ticks
            xticks = np.asarray(ax_i.get_xticks())
            if xticks.size:
                firsttick = np.compress(xticks >= min(ax_i.get_xlim()),
                                        xticks)[0]
                lasttick = np.compress(xticks <= max(ax_i.get_xlim()),
                                       xticks)[-1]
                ax_i.spines['bottom'].set_bounds(firsttick, lasttick)
                ax_i.spines['top'].set_bounds(firsttick, lasttick)
                newticks = xticks.compress(xticks <= lasttick)
                newticks = newticks.compress(newticks >= firsttick)
                ax_i.set_xticks(newticks)

            yticks = np.asarray(ax_i.get_yticks())
            if yticks.size:
                firsttick = np.compress(yticks >= min(ax_i.get_ylim()),
                                        yticks)[0]
                lasttick = np.compress(yticks <= max(ax_i.get_ylim()),
                                       yticks)[-1]
                ax_i.spines['left'].set_bounds(firsttick, lasttick)
                ax_i.spines['right'].set_bounds(firsttick, lasttick)
                newticks = yticks.compress(yticks <= lasttick)
                newticks = newticks.compress(newticks >= firsttick)
                ax_i.set_yticks(newticks)


def _kde_support(data, bw, gridsize, cut, clip):
    """Establish support for a kernel density estimate."""
    support_min = max(data.min() - bw * cut, clip[0])
    support_max = min(data.max() + bw * cut, clip[1])
    return np.linspace(support_min, support_max, gridsize)


def percentiles(a, pcts, axis=None):
    """Like scoreatpercentile but can take and return array of percentiles.

    DEPRECATED: will be removed in a future version.

    Parameters
    ----------
    a : array
        data
    pcts : sequence of percentile values
        percentile or percentiles to find score at
    axis : int or None
        if not None, computes scores over this axis

    Returns
    -------
    scores: array
        array of scores at requested percentiles
        first dimension is length of object passed to ``pcts``

    """
    msg = "This function is deprecated and will be removed in a future version"
    warnings.warn(msg)

    scores = []
    try:
        n = len(pcts)
    except TypeError:
        pcts = [pcts]
        n = 0
    for i, p in enumerate(pcts):
        if axis is None:
            score = stats.scoreatpercentile(a.ravel(), p)
        else:
            score = np.apply_along_axis(stats.scoreatpercentile, axis, a, p)
        scores.append(score)
    scores = np.asarray(scores)
    if not n:
        scores = scores.squeeze()
    return scores


def ci(a, which=95, axis=None):
    """Return a percentile range from an array of values."""
    p = 50 - which / 2, 50 + which / 2
    return np.percentile(a, p, axis)


def sig_stars(p):
    """Return a R-style significance string corresponding to p values.

    DEPRECATED: will be removed in a future version.

    """
    msg = "This function is deprecated and will be removed in a future version"
    warnings.warn(msg)

    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    elif p < 0.1:
        return "."
    return ""


def iqr(a):
    """Calculate the IQR for an array of numbers."""
    a = np.asarray(a)
    q1 = stats.scoreatpercentile(a, 25)
    q3 = stats.scoreatpercentile(a, 75)
    return q3 - q1


def get_dataset_names():
    """Report available example datasets, useful for reporting issues."""
    # delayed import to not demand bs4 unless this function is actually used
    from bs4 import BeautifulSoup
    http = urlopen('https://github.com/mwaskom/seaborn-data/')
    gh_list = BeautifulSoup(http)

    return [l.text.replace('.csv', '')
            for l in gh_list.find_all("a", {"class": "js-navigation-open"})
            if l.text.endswith('.csv')]


def get_data_home(data_home=None):
    """Return a path to the cache directory for example datasets.

    This directory is then used by :func:`load_dataset`.

    If the ``data_home`` argument is not specified, it tries to read from the
    ``SEABORN_DATA`` environment variable and defaults to ``~/seaborn-data``.

    """
    if data_home is None:
        data_home = os.environ.get('SEABORN_DATA',
                                   os.path.join('~', 'seaborn-data'))
    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    return data_home


def load_dataset(name, cache=True, data_home=None, **kws):
    """Load an example dataset from the online repository (requires internet).

    This function provides quick access to a small number of example datasets
    that are useful for documenting seaborn or generating reproducible examples
    for bug reports. It is not necessary for normal usage.

    Note that some of the datasets have a small amount of preprocessing applied
    to define a proper ordering for categorical variables.

    Use :func:`get_dataset_names` to see a list of available datasets.

    Parameters
    ----------
    name : str
        Name of the dataset (``{name}.csv`` on
        https://github.com/mwaskom/seaborn-data).
    cache : boolean, optional
        If True, try to load from the local cache first, and save to the cache
        if a download is required.
    data_home : string, optional
        The directory in which to cache data; see :func:`get_data_home`.
    kws : keys and values, optional
        Additional keyword arguments are passed to passed through to
        :func:`pandas.read_csv`.

    Returns
    -------
    df : :class:`pandas.DataFrame`
        Tabular data, possibly with some preprocessing applied.

    """
    path = ("https://raw.githubusercontent.com/"
            "mwaskom/seaborn-data/master/{}.csv")
    full_path = path.format(name)

    if cache:
        cache_path = os.path.join(get_data_home(data_home),
                                  os.path.basename(full_path))
        if not os.path.exists(cache_path):
            urlretrieve(full_path, cache_path)
        full_path = cache_path

    df = pd.read_csv(full_path, **kws)
    if df.iloc[-1].isnull().all():
        df = df.iloc[:-1]

    # Set some columns as a categorical type with ordered levels

    if name == "tips":
        df["day"] = pd.Categorical(df["day"], ["Thur", "Fri", "Sat", "Sun"])
        df["sex"] = pd.Categorical(df["sex"], ["Male", "Female"])
        df["time"] = pd.Categorical(df["time"], ["Lunch", "Dinner"])
        df["smoker"] = pd.Categorical(df["smoker"], ["Yes", "No"])

    if name == "flights":
        df["month"] = pd.Categorical(df["month"], df.month.unique())

    if name == "exercise":
        df["time"] = pd.Categorical(df["time"], ["1 min", "15 min", "30 min"])
        df["kind"] = pd.Categorical(df["kind"], ["rest", "walking", "running"])
        df["diet"] = pd.Categorical(df["diet"], ["no fat", "low fat"])

    if name == "titanic":
        df["class"] = pd.Categorical(df["class"], ["First", "Second", "Third"])
        df["deck"] = pd.Categorical(df["deck"], list("ABCDEFG"))

    return df


def axis_ticklabels_overlap(labels):
    """Return a boolean for whether the list of ticklabels have overlaps.

    Parameters
    ----------
    labels : list of matplotlib ticklabels

    Returns
    -------
    overlap : boolean
        True if any of the labels overlap.

    """
    if not labels:
        return False
    try:
        bboxes = [l.get_window_extent() for l in labels]
        overlaps = [b.count_overlaps(bboxes) for b in bboxes]
        return max(overlaps) > 1
    except RuntimeError:
        # Issue on macos backend raises an error in the above code
        return False


def axes_ticklabels_overlap(ax):
    """Return booleans for whether the x and y ticklabels on an Axes overlap.

    Parameters
    ----------
    ax : matplotlib Axes

    Returns
    -------
    x_overlap, y_overlap : booleans
        True when the labels on that axis overlap.

    """
    return (axis_ticklabels_overlap(ax.get_xticklabels()),
            axis_ticklabels_overlap(ax.get_yticklabels()))


def categorical_order(values, order=None):
    """Return a list of unique data values.

    Determine an ordered list of levels in ``values``.

    Parameters
    ----------
    values : list, array, Categorical, or Series
        Vector of "categorical" values
    order : list-like, optional
        Desired order of category levels to override the order determined
        from the ``values`` object.

    Returns
    -------
    order : list
        Ordered list of category levels not including null values.

    """
    if order is None:
        if hasattr(values, "categories"):
            order = values.categories
        else:
            try:
                order = values.cat.categories
            except (TypeError, AttributeError):
                try:
                    order = values.unique()
                except AttributeError:
                    order = pd.unique(values)
                try:
                    np.asarray(values).astype(np.float)
                    order = np.sort(order)
                except (ValueError, TypeError):
                    order = order
        order = filter(pd.notnull, order)
    return list(order)


def locator_to_legend_entries(locator, limits, dtype):
    """Return levels and formatted levels for brief numeric legends."""
    raw_levels = locator.tick_values(*limits).astype(dtype)

    class dummy_axis:
        def get_view_interval(self):
            return limits

    if isinstance(locator, mpl.ticker.LogLocator):
        formatter = mpl.ticker.LogFormatter()
    else:
        formatter = mpl.ticker.ScalarFormatter()
    formatter.axis = dummy_axis()

    # TODO: The following two lines should be replaced
    # once pinned matplotlib>=3.1.0 with:
    # formatted_levels = formatter.format_ticks(raw_levels)
    formatter.set_locs(raw_levels)
    formatted_levels = [formatter(x) for x in raw_levels]

    return raw_levels, formatted_levels


def get_color_cycle():
    """Return the list of colors in the current matplotlib color cycle

    Parameters
    ----------
    None

    Returns
    -------
    colors : list
        List of matplotlib colors in the current cycle, or dark gray if
        the current color cycle is empty.
    """
    cycler = mpl.rcParams['axes.prop_cycle']
    return cycler.by_key()['color'] if 'color' in cycler.keys else [".15"]


def relative_luminance(color):
    """Calculate the relative luminance of a color according to W3C standards

    Parameters
    ----------
    color : matplotlib color or sequence of matplotlib colors
        Hex code, rgb-tuple, or html color name.

    Returns
    -------
    luminance : float(s) between 0 and 1

    """
    rgb = mpl.colors.colorConverter.to_rgba_array(color)[:, :3]
    rgb = np.where(rgb <= .03928, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    lum = rgb.dot([.2126, .7152, .0722])
    try:
        return lum.item()
    except ValueError:
        return lum


def to_utf8(obj):
    """Return a string representing a Python object.

    Strings (i.e. type ``str``) are returned unchanged.

    Byte strings (i.e. type ``bytes``) are returned as UTF-8-decoded strings.

    For other objects, the method ``__str__()`` is called, and the result is
    returned as a string.

    Parameters
    ----------
    obj : object
        Any Python object

    Returns
    -------
    s : str
        UTF-8-decoded string representation of ``obj``

    """
    if isinstance(obj, str):
        return obj
    try:
        return obj.decode(encoding="utf-8")
    except AttributeError:  # obj is not bytes-like
        return str(obj)


def _network(t=None, url='https://google.com'):
    """
    Decorator that will skip a test if `url` is unreachable.

    Parameters
    ----------
    t : function, optional
    url : str, optional

    """
    import nose

    if t is None:
        return lambda x: _network(x, url=url)

    def wrapper(*args, **kwargs):
        # attempt to connect
        try:
            f = urlopen(url)
        except (IOError, HTTPException):
            raise nose.SkipTest()
        else:
            f.close()
            return t(*args, **kwargs)
    return wrapper
