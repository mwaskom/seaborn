from collections.abc import Iterable, Sequence, Mapping
import numpy as np
import pandas as pd


class _VectorPlotter:
    """Base class for objects underlying *plot functions."""

    semantics = ["x", "y"]

    def establish_variables(self, data=None, **kwargs):
        """Define plot variables."""
        x = kwargs.get("x", None)
        y = kwargs.get("y", None)

        if x is None and y is None:
            self.input_format = "wide"
            plot_data, variables = self.establish_variables_wideform(
                data, **kwargs
            )
        else:
            self.input_format = "long"
            plot_data, variables = self.establish_variables_longform(
                data, **kwargs
            )

        self.plot_data = plot_data
        self.variables = variables

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
        empty = not len(data)

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
            # TODO do we want any control over this?
            wide_data = wide_data.select_dtypes("number")

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
