import warnings
import itertools
import inspect
from functools import partial
from collections.abc import Iterable, Sequence, Mapping
from numbers import Number
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib as mpl


# TODO move to decorators
def share_init_params_with_map(cls):

    map_sig = inspect.signature(cls.map)
    init_sig = inspect.signature(cls.__init__)

    new = [v for k, v in init_sig.parameters.items() if k != "self"]
    new.insert(0, map_sig.parameters["cls"])
    cls.map.__signature__ = map_sig.replace(parameters=new)
    cls.map.__doc__ = cls.__init__.__doc__

    cls.map = classmethod(cls.map)

    return cls


class SemanticMapping:

    def map(cls, plotter, *args, **kwargs):
        method_name = "_{}_map".format(cls.__name__[:-7].lower())
        setattr(plotter, method_name, cls(plotter, *args, **kwargs))
        return plotter

    def __call__(self, data):

        # TODO what about when we need values that aren't in the data
        # (i.e. for legends), we need some sort of continuous lookup

        if isinstance(data, (pd.Series, Sequence)):
            # TODO need to debug why data.map(self.lookup_table) doesn't work
            return [self.lookup_table.get(val) for val in data]
        else:
            return self.lookup_table[data]


@share_init_params_with_map
class HueMapping(SemanticMapping):

    # Default attributes (TODO use data class?)
    map_type = None
    levels = [None]  # TODO better if just None, but then subset_data fails
    limits = None
    norm = None
    cmap = None
    palette = None
    lookup_table = {}

    def __init__(
        self, plotter, palette=None, order=None, norm=None,
    ):

        from .palettes import QUAL_PALETTES  # Avoid circular import

        data = plotter.plot_data["hue"]

        if data.notna().any():

            # Infer the type of mapping to use from the parameters
            if palette in QUAL_PALETTES:
                map_type = "categorical"
            elif norm is not None:
                map_type = "numeric"
            elif isinstance(palette, (Mapping, Sequence)):
                map_type = "categorical"
            elif plotter.input_format == "wide":
                map_type = "categorical"
            else:
                # Otherwise, use the variable type
                # TODO we will likely need to impelement datetime mapping
                if plotter.var_types["hue"] == "numeric":
                    map_type = "numeric"
                else:
                    map_type = "categorical"

            # Our goal is to end up with a dictionary mapping every unique
            # value in `data` to a color. We will also keep track of the
            # metadata about this mapping we will need for, e.g., a legend

            # --- Option 1: numeric mapping with a matplotlib colormap

            if map_type == "numeric":

                data = pd.to_numeric(data)
                levels, lookup_table, cmap, norm = self.numeric_mapping(
                    data, order, palette, norm
                )
                limits = norm.vmin, norm.vmax

            # --- Option 2: categorical mapping using seaborn palette

            else:

                cmap = None
                limits = None
                levels, lookup_table = self.categorical_mapping(
                    # Casting data to list to handle differences in the way
                    # pandas represents numpy datetime64 data
                    list(data), order, palette
                )

            self.map_type = map_type
            self.lookup_table = lookup_table
            self.palette = palette
            self.levels = levels
            self.limits = limits
            self.norm = norm
            self.cmap = cmap

    # TODO why not generic __call__ method that broadcasts?
    def color_vector(self, data):

        # TODO need to debug why data.map(self.palette) doesn't work
        # TODO call this "mapping" and keep palette for the orig var?
        return [self.palette.get(val) for val in data]

    def categorical_mapping(self, data, order, palette):
        """Determine colors when the hue mapping is categorical."""
        # Avoid circular import
        from .palettes import color_palette

        # -- Identify the order and name of the levels

        if order is None:
            levels = categorical_order(data)
        else:
            levels = order
        n_colors = len(levels)

        # -- Identify the set of colors to use

        if isinstance(palette, dict):

            missing = set(levels) - set(palette)
            if any(missing):
                err = "The palette dictionary is missing keys: {}"
                raise ValueError(err.format(missing))

            lookup_table = palette

        else:

            if palette is None:
                if n_colors <= len(get_color_cycle()):
                    colors = color_palette(None, n_colors)
                else:
                    colors = color_palette("husl", n_colors)
            elif isinstance(palette, list):
                if len(palette) != n_colors:
                    err = "The palette list has the wrong number of colors."
                    raise ValueError(err)
                colors = palette
            else:
                colors = color_palette(palette, n_colors)

            lookup_table = dict(zip(levels, colors))

        return levels, lookup_table

    def numeric_mapping(self, data, order, palette, norm):
        """Determine colors when the hue variable is quantitative."""
        levels = list(np.sort(remove_na(data.unique())))

        # TODO do we want to do something complicated to ensure contrast
        # at the extremes of the colormap against the background?

        # Identify the colormap to use
        # Avoid circular import
        from .palettes import cubehelix_palette, _parse_cubehelix_args

        palette = "ch:" if palette is None else palette
        if isinstance(palette, mpl.colors.Colormap):
            cmap = palette
        elif str(palette).startswith("ch:"):
            args, kwargs = _parse_cubehelix_args(palette)
            cmap = cubehelix_palette(0, *args, as_cmap=True, **kwargs)
        elif isinstance(palette, dict):
            colors = [palette[k] for k in sorted(palette)]
            cmap = mpl.colors.ListedColormap(colors)
        else:
            try:
                cmap = mpl.cm.get_cmap(palette)
            except (ValueError, TypeError):
                err = "Palette {} not understood"
                raise ValueError(err)

        if norm is None:
            norm = mpl.colors.Normalize()
        elif isinstance(norm, tuple):
            norm = mpl.colors.Normalize(*norm)
        elif not isinstance(norm, mpl.colors.Normalize):
            err = "``hue_norm`` must be None, tuple, or Normalize object."
            raise ValueError(err)

        if not norm.scaled():
            norm(np.asarray(data.dropna()))

        lookup_table = dict(zip(levels, cmap(norm(levels))))

        return levels, lookup_table, cmap, norm


class _VectorPlotter:
    """Base class for objects underlying *plot functions."""

    _semantic_mappings = {
        "hue": HueMapping,
    }

    semantics = ("x", "y")

    def __init__(self, data=None, variables={}):

        plot_data, variables = self.establish_variables(data, variables)

        for var, cls in self._semantic_mappings.items():
            if var in self.semantics:

                # Create the mapping function
                map_func = partial(cls.map, plotter=self)
                setattr(self, f"map_{var}", map_func)

                # Call the mapping function to initialize with default values
                getattr(self, f"map_{var}")()


    @classmethod
    def get_variables(cls, arguments):
        return {k: arguments[k] for k in cls.semantics}

    # TODO while we're changing names ... call this assign?
    def establish_variables(self, data=None, variables={}):
        """Define plot variables."""
        x = variables.get("x", None)
        y = variables.get("y", None)

        if x is None and y is None:
            self.input_format = "wide"
            plot_data, variables = self.establish_variables_wideform(
                data, **variables,
            )
        else:
            self.input_format = "long"
            plot_data, variables = self.establish_variables_longform(
                data, **variables,
            )

        self.plot_data = plot_data
        self.variables = variables
        self.var_types = {v: variable_type(plot_data[v]) for v in variables}

        # TODO maybe don't return
        return plot_data, variables

    def establish_variables_wideform(self, data=None, **kwargs):
        """Define plot variables given wide-form data.

        Parameters
        ----------
        data : flat vector or collection of vectors
            Data can be a vector or mapping that is coerceable to a Series
            or a sequence- or mapping-based collection of such vectors, or a
            rectangular numpy array, or a Pandas DataFrame.
        kwargs : variable -> data mappings
            Behavior with keyword arguments is currently undefined.

        Returns
        -------
        plot_data : :class:`pandas.DataFrame`
            Long-form data object mapping seaborn variables (x, y, hue, ...)
            to data vectors.
        variables : dict
            Keys are defined seaborn variables; values are names inferred from
            the inputs (or None when no name can be determined).

        """
        # TODO raise here if any kwarg values are not None,
        # # if we decide for "structure-only" wide API

        # First, determine if the data object actually has any data in it
        empty = data is None or not len(data)

        # Then, determine if we have "flat" data (a single vector)
        # TODO extract this into a separate function?
        if isinstance(data, dict):
            values = data.values()
        else:
            values = np.atleast_1d(data)
        flat = not any(
            isinstance(v, Iterable) and not isinstance(v, (str, bytes))
            for v in values
        )

        if empty:

            # Make an object with the structure of plot_data, but empty
            plot_data = pd.DataFrame(columns=self.semantics)
            variables = {}

        elif flat:

            # Coerce the data into a pandas Series such that the values
            # become the y variable and the index becomes the x variable
            # No other semantics are defined.
            # (Could be accomplished with a more general to_series() interface)
            flat_data = pd.Series(data, name="y").copy()
            flat_data.index.name = "x"
            plot_data = flat_data.reset_index().reindex(columns=self.semantics)

            orig_index = getattr(data, "index", None)
            variables = {
                "x": getattr(orig_index, "name", None),
                "y": getattr(data, "name", None)
            }

        else:

            # Otherwise assume we have some collection of vectors.

            # Handle Python sequences such that entries end up in the columns,
            # not in the rows, of the intermediate wide DataFrame.
            # One way to accomplish this is to convert to a dict of Series.
            if isinstance(data, Sequence):
                data_dict = {}
                for i, var in enumerate(data):
                    key = getattr(var, "name", i)
                    # TODO is there a safer/more generic way to ensure Series?
                    # sort of like np.asarray, but for pandas?
                    data_dict[key] = pd.Series(var)

                data = data_dict

            # Pandas requires that dict values either be Series objects
            # or all have the same length, but we want to allow "ragged" inputs
            if isinstance(data, Mapping):
                data = {key: pd.Series(val) for key, val in data.items()}

            # Otherwise, delegate to the pandas DataFrame constructor
            # This is where we'd prefer to use a general interface that says
            # "give me this data as a pandas DataFrame", so we can accept
            # DataFrame objects from other libraries
            wide_data = pd.DataFrame(data, copy=True)

            # At this point we should reduce the dataframe to numeric cols
            numeric_cols = wide_data.apply(variable_type) == "numeric"
            wide_data = wide_data.loc[:, numeric_cols]

            # Now melt the data to long form
            melt_kws = {"var_name": "columns", "value_name": "values"}
            if "index" in self.wide_structure.values():
                melt_kws["id_vars"] = "index"
                wide_data["index"] = wide_data.index.to_series()
            plot_data = wide_data.melt(**melt_kws)

            # Assign names corresponding to plot semantics
            for var, attr in self.wide_structure.items():
                plot_data[var] = plot_data[attr]
            plot_data = plot_data.reindex(columns=self.semantics)

            # Define the variable names
            variables = {}
            for var, attr in self.wide_structure.items():
                obj = getattr(wide_data, attr)
                variables[var] = getattr(obj, "name", None)

        return plot_data, variables

    def establish_variables_longform(self, data=None, **kwargs):
        """Define plot variables given long-form data and/or vector inputs.

        Parameters
        ----------
        data : dict-like collection of vectors
            Input data where variable names map to vector values.
        kwargs : variable -> data mappings
            Keys are seaborn variables (x, y, hue, ...) and values are vectors
            in any format that can construct a :class:`pandas.DataFrame` or
            names of columns or index levels in ``data``.

        Returns
        -------
        plot_data : :class:`pandas.DataFrame`
            Long-form data object mapping seaborn variables (x, y, hue, ...)
            to data vectors.
        variables : dict
            Keys are defined seaborn variables; values are names inferred from
            the inputs (or None when no name can be determined).

        Raises
        ------
        ValueError
            When variables are strings that don't appear in ``data``.

        """
        plot_data = {}
        variables = {}

        # Data is optional; all variables can be defined as vectors
        if data is None:
            data = {}

        # TODO should we try a data.to_dict() or similar here to more
        # generally accept objects with that interface?
        # Note that dict(df) also works for pandas, and gives us what we
        # want, whereas DataFrame.to_dict() gives a nested dict instead of
        # a dict of series.

        # Variables can also be extraced from the index attribute
        # TODO is this the most general way to enable it?
        # There is no index.to_dict on multiindex, unfortunately
        try:
            index = data.index.to_frame()
        except AttributeError:
            index = {}

        # The caller will determine the order of variables in plot_data
        for key, val in kwargs.items():

            if isinstance(val, (str, bytes)):
                # String inputs trigger __getitem__
                if val in data:
                    # First try to get an entry in the data object
                    plot_data[key] = data[val]
                    variables[key] = val
                elif val in index:
                    # Failing that, try to get an entry in the index object
                    plot_data[key] = index[val]
                    variables[key] = val
                else:
                    # We don't know what this name means
                    err = f"Could not interpret input '{val}'"
                    raise ValueError(err)

            else:

                # Otherwise, assume the value is itself a vector of data
                # TODO check for 1D here or let pd.DataFrame raise?
                plot_data[key] = val
                # Try to infer the name of the variable
                variables[key] = getattr(val, "name", None)

        # Construct a tidy plot DataFrame. This will convert a number of
        # types automatically, aligning on index in case of pandas objects
        plot_data = pd.DataFrame(plot_data, columns=self.semantics)

        # Reduce the variables dictionary to fields with valid data
        variables = {
            var: name
            for var, name in variables.items()
            if plot_data[var].notnull().any()
        }

        return plot_data, variables


def variable_type(vector, boolean_type="numeric"):
    """Determine whether a vector contains numeric, categorical, or dateime data.

    This function differs from the pandas typing API in two ways:

    - Python sequences or object-typed PyData objects are considered numeric if
      all of their entries are numeric.
    - String or mixed-type data are considered categorical even if not
      explicitly represented as a :class:pandas.api.types.CategoricalDtype`.

    Parameters
    ----------
    vector : :func:`pandas.Series`, :func:`numpy.ndarray`, or Python sequence
        Input data to test.
    binary_type : 'numeric' or 'categorical'
        Type to use for vectors containing only 0s and 1s (and NAs).

    Returns
    -------
    var_type : 'numeric', 'categorical', or 'datetime'
        Name identifying the type of data in the vector.

    """
    # Special-case all-na data, which is always "numeric"
    if pd.isna(vector).all():
        return "numeric"

    # Special-case binary/boolean data, allow caller to determine
    if np.isin(vector, [0, 1, np.nan]).all():
        return boolean_type

    # Defer to positive pandas tests
    if pd.api.types.is_numeric_dtype(vector):
        return "numeric"

    if pd.api.types.is_categorical_dtype(vector):
        return "categorical"

    if pd.api.types.is_datetime64_dtype(vector):
        return "datetime"

    # --- If we get to here, we need to check the entries

    # Check for a collection where everything is a number

    def all_numeric(x):
        for x_i in x:
            if not isinstance(x_i, Number):
                return False
        return True

    if all_numeric(vector):
        return "numeric"

    # Check for a collection where everything is a datetime

    def all_datetime(x):
        for x_i in x:
            if not isinstance(x_i, (datetime, np.datetime64)):
                return False
        return True

    if all_datetime(vector):
        return "datetime"

    # Otherwise, our final fallback is to consider things categorical

    return "categorical"


def infer_orient(x=None, y=None, orient=None, require_numeric=True):
    """Determine how the plot should be oriented based on the data.

    For historical reasons, the convention is to call a plot "horizontally"
    or "vertically" oriented based on the axis representing its dependent
    variable. Practically, this is used when determining the axis for
    numerical aggregation.

    Paramters
    ---------
    x, y : Vector data or None
        Positional data vectors for the plot.
    orient : string or None
        Specified orientation, which must start with "v" or "h" if not None.
    require_numeric : bool
        If set, raise when the implied dependent variable is not numeric.

    Returns
    -------
    orient : "v" or "h"

    Raises
    ------
    ValueError: When `orient` is not None and does not start with "h" or "v"
    TypeError: When dependant variable is not numeric, with `require_numeric`

    """

    x_type = None if x is None else variable_type(x)
    y_type = None if y is None else variable_type(y)

    nonnumeric_dv_error = "{} orientation requires numeric `{}` variable."
    single_var_warning = "{} orientation ignored with only `{}` specified."

    if x is None:
        if str(orient).startswith("h"):
            warnings.warn(single_var_warning.format("Horizontal", "y"))
        if require_numeric and y_type != "numeric":
            raise TypeError(nonnumeric_dv_error.format("Vertical", "y"))
        return "v"

    elif y is None:
        if str(orient).startswith("v"):
            warnings.warn(single_var_warning.format("Vertical", "x"))
        if require_numeric and x_type != "numeric":
            raise TypeError(nonnumeric_dv_error.format("Horizontal", "x"))
        return "h"

    elif str(orient).startswith("v"):
        if require_numeric and y_type != "numeric":
            raise TypeError(nonnumeric_dv_error.format("Vertical", "y"))
        return "v"

    elif str(orient).startswith("h"):
        if require_numeric and x_type != "numeric":
            raise TypeError(nonnumeric_dv_error.format("Horizontal", "x"))
        return "h"

    elif orient is not None:
        raise ValueError(f"Value for `orient` not understood: {orient}")

    elif x_type != "numeric" and y_type == "numeric":
        return "v"

    elif x_type == "numeric" and y_type != "numeric":
        return "h"

    elif require_numeric and "numeric" not in (x_type, y_type):
        err = "Neither the `x` nor `y` variable appears to be numeric."
        raise TypeError(err)

    else:
        return "v"


def unique_dashes(n):
    """Build an arbitrarily long list of unique dash styles for lines.

    Parameters
    ----------
    n : int
        Number of unique dash specs to generate.

    Returns
    -------
    dashes : list of strings or tuples
        Valid arguments for the ``dashes`` parameter on
        :class:`matplotlib.lines.Line2D`. The first spec is a solid
        line (``""``), the remainder are sequences of long and short
        dashes.

    """
    # Start with dash specs that are well distinguishable
    dashes = [
        "",
        (4, 1.5),
        (1, 1),
        (3, 1.25, 1.5, 1.25),
        (5, 1, 1, 1),
    ]

    # Now programatically build as many as we need
    p = 3
    while len(dashes) < n:

        # Take combinations of long and short dashes
        a = itertools.combinations_with_replacement([3, 1.25], p)
        b = itertools.combinations_with_replacement([4, 1], p)

        # Interleave the combinations, reversing one of the streams
        segment_list = itertools.chain(*zip(
            list(a)[1:-1][::-1],
            list(b)[1:-1]
        ))

        # Now insert the gaps
        for segments in segment_list:
            gap = min(segments)
            spec = tuple(itertools.chain(*((seg, gap) for seg in segments)))
            dashes.append(spec)

        p += 1

    return dashes[:n]


def unique_markers(n):
    """Build an arbitrarily long list of unique marker styles for points.

    Parameters
    ----------
    n : int
        Number of unique marker specs to generate.

    Returns
    -------
    markers : list of string or tuples
        Values for defining :class:`matplotlib.markers.MarkerStyle` objects.
        All markers will be filled.

    """
    # Start with marker specs that are well distinguishable
    markers = [
        "o",
        "X",
        (4, 0, 45),
        "P",
        (4, 0, 0),
        (4, 1, 0),
        "^",
        (4, 1, 45),
        "v",
    ]

    # Now generate more from regular polygons of increasing order
    s = 5
    while len(markers) < n:
        a = 360 / (s + 1) / 2
        markers.extend([
            (s + 1, 1, a),
            (s + 1, 0, a),
            (s, 1, 0),
            (s, 0, 0),
        ])
        s += 1

    # Convert to MarkerStyle object, using only exactly what we need
    # markers = [mpl.markers.MarkerStyle(m) for m in markers[:n]]

    return markers[:n]


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

                if variable_type(values) == "numeric":
                    order = np.sort(order)

        order = filter(pd.notnull, order)
    return list(order)


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
