import pandas as pd


class _Plotter:
    """Base class for objects underlying *plot functions."""
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
            the inputs or None when no name can be determined.

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

        # The caller will determine the order of variables in plot_data
        for key, val in kwargs.items():

            if isinstance(val, str):
                # Use string types to index the data object
                try:
                    plot_data[key] = data[val]
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
        plot_data = pd.DataFrame(plot_data)

        return plot_data, variables
