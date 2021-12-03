from __future__ import annotations
from contextlib import contextmanager

import numpy as np
import pandas as pd
import matplotlib as mpl

from seaborn._core.plot import SEMANTICS

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Literal, Any, Type, Dict, Callable
    from collections.abc import Generator
    from numpy import ndarray
    from pandas import DataFrame
    from matplotlib.axes import Axes
    from seaborn._core.mappings import SemanticMapping, RGBATuple
    from seaborn._stats.base import Stat

    MappingDict = Dict[str, SemanticMapping]


class Feature:
    def __init__(
        self,
        val: Any = None,
        depend: str | None = None,
        rc: str | None = None
    ):
        """Class supporting several default strategies for setting visual features.

        Parameters
        ----------
        val :
            Use this value as the default.
        depend :
            Use the value of this feature as the default.
        rc :
            Use the value of this rcParam as the default.

        """
        if depend is not None:
            assert depend in SEMANTICS
        if rc is not None:
            assert rc in mpl.rcParams

        self._val = val
        self._rc = rc
        self._depend = depend

    def __repr__(self):
        """Nice formatting for when object appears in Mark init signature."""
        if self._val is not None:
            s = f"<{repr(self._val)}>"
        elif self._depend is not None:
            s = f"<depend:{self._depend}>"
        elif self._rc is not None:
            s = f"<rc:{self._rc}>"
        else:
            s = "<undefined>"
        return s

    @property
    def depend(self) -> Any:
        """Return the name of the feature to source a default value from."""
        return self._depend

    @property
    def default(self) -> Any:
        """Get the default value for this feature, or access the relevant rcParam."""
        if self._val is not None:
            return self._val
        return mpl.rcParams.get(self._rc)


class Mark:
    # TODO where to define vars we always group by (col, row, group)
    default_stat: Type[Stat] | None = None
    grouping_vars: list[str] = []
    requires: list[str]  # List of variabes that must be defined
    supports: list[str]  # TODO can probably derive this from Features now, no?
    features: dict[str, Any]

    def __init__(self, **kwargs: Any):
        """Base class for objects that control the actual plotting."""
        self.features = {}
        self._kwargs = kwargs

    @contextmanager
    def use(
        self,
        mappings: dict[str, SemanticMapping],
        orient: Literal["x", "y"]
    ) -> Generator:
        """Temporarily attach a mappings dict and orientation during plotting."""
        # Having this allows us to simplify the number of objects that need to be
        # passed all the way down to where plotting happens while not (permanently)
        # mutating a Mark object that may persist in user-space.
        self.mappings = mappings
        self.orient = orient
        try:
            yield
        finally:
            del self.mappings, self.orient

    def _resolve(
        self,
        data: DataFrame | dict[str, Any],
        name: str,
    ) -> Any:
        """Obtain default, specified, or mapped value for a named feature.

        Parameters
        ----------
        data :
            Container with data values for features that will be semantically mapped.
        name :
            Identity of the feature / semantic.

        Returns
        -------
        value or array of values
            Outer return type depends on whether `data` is a dict (implying that
            we want a single value) or DataFrame (implying that we want an array
            of values with matching length).

        """
        feature = self.features[name]
        standardize = SEMANTICS[name]._standardize_value
        directly_specified = not isinstance(feature, Feature)
        return_array = isinstance(data, pd.DataFrame)

        if directly_specified:
            feature = standardize(feature)
            if return_array:
                feature = np.array([feature] * len(data))
            return feature

        if name in data:
            if name in self.mappings:
                feature = self.mappings[name](data[name])
            else:
                # TODO Might this obviate the identity scale? Just don't add a mapping?
                feature = data[name]
            if return_array:
                feature = np.asarray(feature)
            return feature

        if feature.depend is not None:
            # TODO add source_func or similar to transform the source value?
            # e.g. set linewidth as a proportion of pointsize?
            return self._resolve(data, feature.depend)

        default = standardize(feature.default)
        if return_array:
            default = np.array([default] * len(data))
        return default

    def _resolve_color(
        self,
        data: DataFrame | dict,
        prefix: str = "",
    ) -> RGBATuple | ndarray:
        """
        Obtain a default, specified, or mapped value for a color feature.

        This method exists separately to support the relationship between a
        color and its corresponding alpha. We want to respect alpha values that
        are passed in specified (or mapped) color values but also make use of a
        separate `alpha` variable, which can be mapped. This approach may also
        be extended to support mapping of specific color channels (i.e.
        luminance, chroma) in the future.

        Parameters
        ----------
        data :
            Container with data values for features that will be semantically mapped.
        prefix :
            Support "color", "fillcolor", etc.

        """
        color = self._resolve(data, f"{prefix}color")
        alpha = self._resolve(data, f"{prefix}alpha")

        if isinstance(color, tuple):
            if len(color) == 4:
                return mpl.colors.to_rgba(color)
            return mpl.colors.to_rgba(color, alpha)
        else:
            if color.shape[1] == 4:
                return mpl.colors.to_rgba_array(color)
            return mpl.colors.to_rgba_array(color, alpha)

    def _adjust(
        self,
        df: DataFrame,
    ) -> DataFrame:

        return df

    def _infer_orient(self, scales: dict) -> Literal["x", "y"]:  # TODO type scales

        # TODO The original version of this (in seaborn._oldcore) did more checking.
        # Paring that down here for the prototype to see what restrictions make sense.

        x_type = None if "x" not in scales else scales["x"].scale_type
        y_type = None if "y" not in scales else scales["y"].scale_type

        if x_type is None:
            return "y"

        elif y_type is None:
            return "x"

        elif x_type != "categorical" and y_type == "categorical":
            return "y"

        elif x_type != "numeric" and y_type == "numeric":

            # TODO should we try to orient based on number of unique values?

            return "x"

        elif x_type == "numeric" and y_type != "numeric":
            return "y"

        else:
            return "x"

    def _plot(
        self,
        split_generator: Callable[[], Generator],
    ) -> None:
        """Main interface for creating a plot."""
        axes_cache = set()
        for keys, data, ax in split_generator():
            kws = self._kwargs.copy()
            self._plot_split(keys, data, ax, kws)
            axes_cache.add(ax)

        # TODO what is the best way to do this a minimal number of times?
        # Probably can be moved out to Plot?
        for ax in axes_cache:
            ax.autoscale_view()

        self._finish_plot()

    def _plot_split(
        self,
        keys: dict[str, Any],
        data: DataFrame,
        ax: Axes,
        kws: dict,
    ) -> None:
        """Method that plots specific subsets of data. Must be defined by subclass."""
        raise NotImplementedError()

    def _finish_plot(self) -> None:
        """Method that is called after each data subset has been plotted."""
        pass
