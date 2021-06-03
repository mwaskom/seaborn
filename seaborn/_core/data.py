from __future__ import annotations

import pandas as pd

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional, Any
    from collections.abc import Hashable, Mapping, Sized
    from pandas import DataFrame
    from .typing import Vector


class PlotData:  # TODO better name?

    # How to handle wide-form data here, when the dimensional semantics are defined by
    # the mark? (I guess? that will be most consistent with how it currently works.)
    # I think we want to avoid too much deferred execution or else tracebacks are going
    # to be confusing to follow...

    # With wide-form data, should we allow marks with distinct wide_form semantics?
    # I think in most cases that will not make sense? When to check?

    # I guess more generally, what to do when different variables are assigned in
    # different calls to Plot.add()? This has to be possible (otherwise why allow it)?
    # ggplot allows you to do this but only uses the first layer for labels, and only
    # if the scales are compatible.

    # Who owns the existing VectorPlotter.variables, VectorPlotter.var_levels, etc.?

    frame: DataFrame
    names: dict[str, Optional[str]]
    _source: Optional[DataFrame | Mapping]

    def __init__(
        self,
        data: Optional[DataFrame | Mapping],
        variables: Optional[dict[str, Hashable | Vector]],
        # TODO pass in wide semantics?
    ):

        if variables is None:
            variables = {}

        # TODO only specing out with long-form data for now...
        frame, names = self._assign_variables_longform(data, variables)

        self.frame = frame
        self.names = names

        self._source_data = data
        self._source_vars = variables

    def __contains__(self, key: Hashable) -> bool:
        return key in self.frame

    def concat(
        self,
        data: Optional[DataFrame | Mapping],
        variables: Optional[dict[str, Optional[Hashable | Vector]]],
    ) -> PlotData:

        # TODO Note a tricky thing here which is that often x/y will be inherited
        # meaning that the variable specification here will look like "wide-form"

        # Inherit the original source of the upsteam data by default
        if data is None:
            data = self._source_data

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

    def _assign_variables_longform(
        self,
        data: Optional[DataFrame | Mapping],
        variables: dict[str, Optional[Hashable | Vector]]
    ) -> tuple[DataFrame, dict[str, Optional[str]]]:
        """
        Define plot variables given long-form data and/or vector inputs.

        Parameters
        ----------
        data
            Input data where variable names map to vector values.
        variables
            Keys are seaborn variables (x, y, hue, ...) and values are vectors
            in any format that can construct a :class:`pandas.DataFrame` or
            names of columns or index levels in ``data``.

        Returns
        -------
        frame
            Long-form data object mapping seaborn variables (x, y, hue, ...)
            to data vectors.
        names
            Keys are defined seaborn variables; values are names inferred from
            the inputs (or None when no name can be determined).

        Raises
        ------
        ValueError
            When variables are strings that don't appear in ``data``.

        """
        plot_data: dict[str, Vector] = {}
        var_names: dict[str, Optional[str]] = {}

        # Data is optional; all variables can be defined as vectors
        if data is None:
            data = {}

        # TODO Generally interested in accepting a generic DataFrame interface
        # Track https://data-apis.org/ for development

        # Variables can also be extracted from the index of a DataFrame
        index: dict[str, Any]
        if isinstance(data, pd.DataFrame):
            index = data.index.to_frame().to_dict(
                "series")  # type: ignore  # data-sci-types wrong about to_dict return
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
            try:
                val_as_data_key = (
                    val in data
                    or (isinstance(val, str) and val in index)
                )
            except (KeyError, TypeError):
                val_as_data_key = False

            if val_as_data_key:

                if val in data:
                    plot_data[key] = data[val]  # type: ignore # fails on key: Hashable
                elif val in index:
                    plot_data[key] = index[val]  # type: ignore # fails on key: Hashable
                var_names[key] = str(val)

            elif isinstance(val, str):

                # This looks like a column name but we don't know what it means!
                # TODO improve this feedback to distinguish between
                # - "you passed a string, but did not pass data"
                # - "you passed a string, it was not found in data"

                err = f"Could not interpret value `{val}` for parameter `{key}`"
                raise ValueError(err)

            else:

                # Otherwise, assume the value is itself data

                # Raise when data object is present and a vector can't matched
                if isinstance(data, pd.DataFrame) and not isinstance(val, pd.Series):
                    if isinstance(val, Sized) and len(data) != len(val):
                        val_cls = val.__class__.__name__
                        err = (
                            f"Length of {val_cls} vectors must match length of `data`"
                            f" when both are used, but `data` has length {len(data)}"
                            f" and the vector passed to `{key}` has length {len(val)}."
                        )
                        raise ValueError(err)

                plot_data[key] = val  # type: ignore # fails on key: Hashable

                # Try to infer the name of the variable
                var_names[key] = getattr(val, "name", None)

        # Construct a tidy plot DataFrame. This will convert a number of
        # types automatically, aligning on index in case of pandas objects
        frame = pd.DataFrame(plot_data)

        # Reduce the variables dictionary to fields with valid data
        names: dict[str, Optional[str]] = {
            var: name
            for var, name in var_names.items()
            # TODO I am not sure that this is necessary any more
            if frame[var].notnull().any()
        }

        return frame, names
