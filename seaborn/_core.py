import warnings
import itertools
from copy import copy
from functools import partial
from collections.abc import Iterable, Sequence, Mapping
from numbers import Number
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib as mpl

from ._decorators import (
    share_init_params_with_map,
)
from .palettes import (
    QUAL_PALETTES,
)
from .utils import (
    get_color_cycle,
    remove_na,
)


class SemanticMapping:

    def __init__(self, plotter):

        # TODO Putting this here so we can continue to use a lot of the
        # logic that's built into the library, but the idea of this class
        # is to move towards semantic mappings that are agnositic about the
        # kind of plot they're going to be used to draw, which is going to
        # take some rethinking.
        self.plotter = plotter

    def map(cls, plotter, *args, **kwargs):
        method_name = "_{}_map".format(cls.__name__[:-7].lower())
        setattr(plotter, method_name, cls(plotter, *args, **kwargs))
        return plotter

    def _lookup_single(self, key):

        value = self.lookup_table[key]
        return value

    def __call__(self, data):

        if isinstance(data, (list, np.ndarray, pd.Series)):
            # TODO need to debug why data.map(self.lookup_table) doesn't work
            return [self._lookup_single(key) for key in data]
        else:
            return self._lookup_single(data)


@share_init_params_with_map
class HueMapping(SemanticMapping):

    # Default attributes (TODO use data class?)
    # TODO Add docs for what these are (maybe on the base class?)
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

        super().__init__(plotter)

        data = plotter.plot_data["hue"]

        if data.notna().any():

            # Infer the type of mapping to use from the parameters
            if palette in QUAL_PALETTES:
                map_type = "categorical"
            elif norm is not None:
                map_type = "numeric"
            elif isinstance(palette, (dict, list)):
                map_type = "categorical"
            elif plotter.input_format == "wide":
                map_type = "categorical"
            else:
                map_type = plotter.var_types["hue"]

            # Our goal is to end up with a dictionary mapping every unique
            # value in `data` to a color. We will also keep track of the
            # metadata about this mapping we will need for, e.g., a legend

            # --- Option 1: numeric mapping with a matplotlib colormap

            if map_type == "numeric":

                # TODO do we need levels for numeric map?
                data = pd.to_numeric(data)
                levels, lookup_table, norm, cmap = self.numeric_mapping(
                    data, palette, norm,
                )
                # TODO what do we need limits for? Can we get this downstream?
                limits = norm.vmin, norm.vmax

            # --- Option 2: categorical mapping using seaborn palette

            else:

                cmap = limits = norm = None
                levels, lookup_table = self.categorical_mapping(
                    # Casting data to list to handle differences in the way
                    # pandas represents numpy datetime64 data
                    list(data), palette, order,
                )

            # --- Option 3: datetime mapping

            # TODO this needs implementation; currently uses categorical

            self.map_type = map_type
            self.lookup_table = lookup_table
            self.palette = palette
            self.levels = levels
            self.limits = limits  # TODO do we need this
            self.norm = norm
            self.cmap = cmap

    def _lookup_single(self, key):

        try:
            value = self.lookup_table[key]
        except KeyError:
            normed = self.norm(key)
            if np.ma.is_masked(normed):
                normed = np.nan
            value = self.cmap(normed)
        return value

    def categorical_mapping(self, data, palette, order):
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

    def numeric_mapping(self, data, palette, norm):
        """Determine colors when the hue variable is quantitative."""

        # Avoid circular import
        # TODO I think we've decided it's ok for core to import from palettes
        from .palettes import cubehelix_palette, _parse_cubehelix_args

        if isinstance(palette, dict):

            # The presence of a norm object overrides a dictionary of hues
            # in specifying a numeric mapping, so we need to process it here.
            levels = list(np.sort(list(palette)))
            colors = [palette[k] for k in sorted(palette)]
            cmap = mpl.colors.ListedColormap(colors)
            lookup_table = palette.copy()

        else:

            # The levels are the sorted unique values in the data
            levels = list(np.sort(remove_na(data.unique())))

            # --- Sort out the colormap to use from the palette argument

            # Default numeric palette is our default cubehelix palette
            # TODO do we want to do something complicated to ensure contrast?
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

            # Now sort out the data normalization
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

        return levels, lookup_table, norm, cmap


@share_init_params_with_map
class SizeMapping(SemanticMapping):

    scaled_mapping = True  # TODO better name

    map_type = None
    levels = [None]  # TODO needs downstream fix so can just be None
    norm = None
    sizes = None
    lookup_table = {}

    def __init__(
        self, plotter, sizes=None, order=None, norm=None,
    ):

        super().__init__(plotter)

        # TODO putting this here but it needs a home in the docs:
        # "sizes" are in matplotlib artist units, "norm" is in data units

        data = plotter.plot_data["size"]

        if data.notna().any():

            # Infer the type of mapping to use
            if norm is not None:
                map_type = "numeric"
            elif isinstance(sizes, (dict, list)):
                map_type = "categorical"
            else:
                map_type = plotter.var_types["size"]

            # --- Option 1: numeric mapping

            if map_type == "numeric":

                levels, lookup_table, norm = self.numeric_mapping(
                    data, sizes, norm,
                )

            # --- Option 2: categorical mapping

            else:

                levels, lookup_table = self.categorical_mapping(
                    data, sizes, order,
                )

            # --- Option 3: datetime mapping

            # TODO this needs implementation; currently uses categorical

            self.map_type = map_type
            self.levels = levels
            self.norm = norm
            self.sizes = sizes
            self.lookup_table = lookup_table

    def _lookup_single(self, key):

        try:
            value = self.lookup_table[key]
        except KeyError:
            normed = self.norm(key)
            if np.ma.is_masked(normed):
                normed = np.nan
            size_values = self.lookup_table.values()
            size_range = min(size_values), max(size_values)
            value = size_range[0] + normed * np.ptp(size_range)
        return value

    def categorical_mapping(self, data, sizes, order):

        levels = categorical_order(data, order)

        if isinstance(sizes, dict):

            # Dict inputs map existing data values to the size attribute
            missing = set(levels) - set(sizes)
            if any(missing):
                err = f"Missing sizes for the following levels: {missing}"
                raise ValueError(err)
            lookup_table = sizes.copy()

        elif isinstance(sizes, list):

            # List inputs give size values in the same order as the levels
            if len(sizes) != len(levels):
                err = "The `sizes` list has the wrong number of values."
                raise ValueError(err)

            lookup_table = dict(zip(levels, sizes))

        else:

            if isinstance(sizes, tuple):

                # Tuple input sets the min, max size values
                if len(sizes) != 2:
                    err = "A `sizes` tuple must have only 2 values"
                    raise ValueError(err)

            elif sizes is not None:

                err = f"Value for `sizes` not understood: {sizes}"
                raise ValueError(err)

            else:

                # Otherwise, we need to get the min, max size values from
                # the plotter object we are attached to.

                # TODO this is going to cause us trouble later, because we
                # want to restructure things so that the plotter is generic
                # across the visual representation of the data. But at this
                # point, we don't know the visual representation. Likely we
                # want to change the logic of this Mapping so that it gives
                # points on a nornalized range that then gets unnormalized
                # when we know what we're drawing. But given the way the
                # package works now, this way is cleanest.
                sizes = self.plotter._default_size_range

            # For categorical sizes, use regularly-spaced linear steps
            # between the minimum and maximum sizes
            sizes = np.linspace(*sizes, len(levels))
            lookup_table = dict(zip(levels, sizes))

        return levels, lookup_table

    def numeric_mapping(self, data, sizes, norm):

        if isinstance(sizes, dict):
            # The presence of a norm object overrides a dictionary of sizes
            # in specifying a numeric mapping, so we need to process it
            # dictionary here
            levels = list(np.sort(list(sizes)))
            size_values = sizes.values()
            size_range = min(size_values), max(size_values)

        else:

            # The levels here will be the unique values in the data
            levels = list(np.sort(remove_na(data.unique())))

            if isinstance(sizes, tuple):

                # For numeric inputs, the size can be parametrized by
                # the minimum and maximum artist values to map to. The
                # norm object that gets set up next specifies how to
                # do the mapping.

                if len(sizes) != 2:
                    err = "A `sizes` tuple must have only 2 values"
                    raise ValueError(err)

                size_range = sizes

            elif sizes is not None:

                err = f"Value for `sizes` not understood: {sizes}"
                raise ValueError(err)

            else:

                # When not provided, we get the size range from the plotter
                # object we are attached to. See the note in the categorical
                # method about how this is suboptimal for future development.:
                size_range = self.plotter._default_size_range

        # Now that we know the minimum and maximum sizes that will get drawn,
        # we need to map the data values that we have into that range. We will
        # use a matplotlib Normalize class, which is typically used for numeric
        # color mapping but works fine here too. It takes data values and maps
        # them into a [0, 1] interval, potentially nonlinear-ly.

        if norm is None:
            # Default is a linear function between the min and max data values
            norm = mpl.colors.Normalize()
        elif isinstance(norm, tuple):
            # It is also possible to give different limits in data space
            norm = mpl.colors.Normalize(*norm)
        elif not isinstance(norm, mpl.colors.Normalize):
            err = f"Value for size `norm` parameter not understood: {norm}"
            raise ValueError(err)
        else:
            # If provided with Normalize object, copy it so we can modify
            norm = copy(norm)

        # Set the mapping so all output values are in [0, 1]
        norm.clip = True

        # If the input range is not set, use the full range of the data
        if not norm.scaled():
            norm(levels)

        # Map from data values to [0, 1] range
        sizes_scaled = norm(levels)

        # Now map from the scaled range into the artist units
        if isinstance(sizes, dict):
            lookup_table = sizes
        else:
            lo, hi = size_range
            sizes = lo + sizes_scaled * (hi - lo)
            lookup_table = dict(zip(levels, sizes))

        return levels, lookup_table, norm


# =========================================================================== #


class VectorPlotter:
    """Base class for objects underlying *plot functions."""

    _semantic_mappings = {
        "hue": HueMapping,
        "size": SizeMapping,
    }

    semantics = "x", "y", "hue", "size", "style", "units"
    wide_structure = {
        "x": "index", "y": "values", "hue": "columns", "style": "columns",
    }

    _default_size_range = 1, 2  # Unused but needed in tests

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
        self.var_types = {
            v: variable_type(plot_data[v], boolean_type="categorical")
            for v in variables
        }

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
