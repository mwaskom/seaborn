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

        return plot_data  # TODO also return variables?

    def establish_variables_wideform(self, data=None, **kwargs):
        """Define plot variables given wide-form data."""
        raise NotImplementedError

    def establish_variables_longform(self, data=None, **kwargs):
        """Define plot variables given long-form data and/or vector inputs.

        Parameters
        ----------
        data : dict-like collection of vectors
            Input data where variable names map to vector values.
        kwargs : variable -> data mappings
            Keys are seaborn variables (x, y, hue, ...) and values are vectors
            in any format that can construct a :class:`pandas.DataFrame`.

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

        # Variables can also be extraced from the index attribute
        # TODO is this the most general way to enable it? Also it is added in
        # pandas 0.24 so will fail our pinned tests
        try:
            index = data.index.to_frame()
        except AttributeError:
            index = {}

        # The caller will determine the order of variables in plot_data
        for key, val in kwargs.items():

            if isinstance(val, str):
                # String inputs trigger __getitem__, first from data itself
                try:
                    plot_data[key] = data[val]
                    variables[key] = val
                except KeyError:
                    # Failing that, try to get an index level
                    try:
                        plot_data[key] = index[val]
                        variables[key] = val
                    except KeyError:
                        # Raise ValueError for backwards compatability
                        err = f"Could not interpret input '{val}'"
                        raise ValueError(err)

            else:
                # Otherwise, assume the value is the data itself
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
