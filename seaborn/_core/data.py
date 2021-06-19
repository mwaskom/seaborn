"""
Components for parsing variable assignments and internally representing plot data.
"""
from __future__ import annotations

from collections import abc
import pandas as pd

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pandas import DataFrame
    from seaborn._core.typing import DataSource, VariableSpec


# TODO Repetition in the docstrings should be reduced with interpolation tools

class PlotData:
    """
    Data table with plot variable schema and mapping to original names.

    Contains logic for parsing variable specification arguments and updating
    the table with layer-specific data and/or mappings.

    Parameters
    ----------
    data
        Input data where variable names map to vector values.
    variables
        Keys are names of plot variables (x, y, ...) each value is one of:

        - name of a column (or index level, or dictionary entry) in `data`
        - vector in any format that can construct a :class:`pandas.DataFrame`

    Attributes
    ----------
    frame
        Data table with column names having defined plot variables.
    names
        Dictionary mapping plot variable names to names in source data structure(s).

    """
    frame: DataFrame
    names: dict[str, str | None]
    _source: DataSource

    def __init__(
        self,
        data: DataSource,
        variables: dict[str, VariableSpec],
    ):

        frame, names = self._assign_variables(data, variables)

        self.frame = frame
        self.names = names

        self._source_data = data
        self._source_vars = variables

    def __contains__(self, key: str) -> bool:
        """Boolean check on whether a variable is defined in this dataset."""
        return key in self.frame

    def concat(
        self,
        data: DataSource,
        variables: dict[str, VariableSpec] | None,
    ) -> PlotData:
        """Add, replace, or drop variables and return as a new dataset."""
        # Inherit the original source of the upsteam data by default
        if data is None:
            data = self._source_data

        # TODO allow `data` to be a function (that is called on the source data?)

        if variables is None:
            variables = self._source_vars

        # Passing var=None implies that we do not want that variable in this layer
        disinherit = [k for k, v in variables.items() if v is None]

        # Create a new dataset with just the info passed here
        new = PlotData(data, variables)

        # -- Update the inherited DataSource with this new information

        drop_cols = [k for k in self.frame if k in new.frame or k in disinherit]
        frame = pd.concat([self.frame.drop(columns=drop_cols), new.frame], axis=1)

        names = {k: v for k, v in self.names.items() if k not in disinherit}
        names.update(new.names)

        new.frame = frame
        new.names = names

        return new

    def _assign_variables(
        self,
        data: DataSource,
        variables: dict[str, VariableSpec],
    ) -> tuple[DataFrame, dict[str, str | None]]:
        """
        Assign values for plot variables given long-form data and/or vector inputs.

        Parameters
        ----------
        data
            Input data where variable names map to vector values.
        variables
            Keys are names of plot variables (x, y, ...) each value is one of:

            - name of a column (or index level, or dictionary entry) in `data`
            - vector in any format that can construct a :class:`pandas.DataFrame`

        Returns
        -------
        frame
            Table mapping seaborn variables (x, y, hue, ...) to data vectors.
        names
            Keys are defined seaborn variables; values are names inferred from
            the inputs (or None when no name can be determined).

        Raises
        ------
        ValueError
            When variables are strings that don't appear in `data`, or when they are
            non-indexed vector datatypes that have a different length from `data`.

        """
        source_data: dict | DataFrame
        frame: DataFrame
        names: dict[str, str | None]

        plot_data = {}
        names = {}

        given_data = data is not None
        if given_data:
            source_data = data
        else:
            # Data is optional; all variables can be defined as vectors
            # But simplify downstream code by always having a usable source data object
            source_data = {}

        # TODO Generally interested in accepting a generic DataFrame interface
        # Track https://data-apis.org/ for development

        # Variables can also be extracted from the index of a DataFrame
        if isinstance(source_data, pd.DataFrame):
            index = source_data.index.to_frame().to_dict("series")
        else:
            index = {}

        for key, val in variables.items():

            # Simply ignore variables with no specification
            if val is None:
                continue

            # Try to treat the argument as a key for the data collection.
            # But be flexible about what can be used as a key.
            # Usually it will be a string, but allow other hashables when
            # taking from the main data object. Allow only strings to reference
            # fields in the index, because otherwise there is too much ambiguity.

            # TODO this will be rendered unnecessary by the following pandas fix:
            # https://github.com/pandas-dev/pandas/pull/41283
            try:
                hash(val)
                val_is_hashable = True
            except TypeError:
                val_is_hashable = False

            val_as_data_key = (
                # See https://github.com/pandas-dev/pandas/pull/41283
                # (isinstance(val, abc.Hashable) and val in source_data)
                (val_is_hashable and val in source_data)
                or (isinstance(val, str) and val in index)
            )

            if val_as_data_key:

                if val in source_data:
                    plot_data[key] = source_data[val]
                elif val in index:
                    plot_data[key] = index[val]
                names[key] = str(val)

            elif isinstance(val, str):

                # This looks like a column name but, lookup failed.

                err = f"Could not interpret value `{val}` for `{key}`. "
                if not given_data:
                    err += "Value is a string, but `data` was not passed."
                else:
                    err += "An entry with this name does not appear in `data`."
                raise ValueError(err)

            else:

                # Otherwise, assume the value somehow represents data

                # Ignore empty data structures
                if isinstance(val, abc.Sized) and len(val) == 0:
                    continue

                # If vector has no index, it must match length of data table
                if isinstance(data, pd.DataFrame) and not isinstance(val, pd.Series):
                    if isinstance(val, abc.Sized) and len(data) != len(val):
                        val_cls = val.__class__.__name__
                        err = (
                            f"Length of {val_cls} vectors must match length of `data`"
                            f" when both are used, but `data` has length {len(data)}"
                            f" and the vector passed to `{key}` has length {len(val)}."
                        )
                        raise ValueError(err)

                plot_data[key] = val

                # Try to infer the original name using pandas-like metadata
                if hasattr(val, "name"):
                    names[key] = str(val.name)  # type: ignore  # mypy/1424
                else:
                    names[key] = None

        # Construct a tidy plot DataFrame. This will convert a number of
        # types automatically, aligning on index in case of pandas objects
        frame = pd.DataFrame(plot_data)

        return frame, names