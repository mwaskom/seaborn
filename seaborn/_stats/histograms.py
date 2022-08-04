from __future__ import annotations
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd

from seaborn._core.groupby import GroupBy
from seaborn._stats.base import Stat

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from numpy.typing import ArrayLike


@dataclass
class Hist(Stat):
    """
    Bin observations, count them, and optionally normalize or cumulate.
    """
    stat: str = "count"  # TODO how to do validation on this arg?

    bins: str | int | ArrayLike = "auto"
    binwidth: float | None = None
    binrange: tuple[float, float] | None = None
    common_norm: bool | list[str] = True
    common_bins: bool | list[str] = True
    cumulative: bool = False

    # TODO Require this to be set here or have interface with scale?
    # Q: would Discrete() scale imply binwidth=1 or bins centered on integers?
    discrete: bool = False

    # TODO Note that these methods are mostly copied from _statistics.Histogram,
    # but it only computes univariate histograms. We should reconcile the code.

    def _define_bin_edges(self, vals, weight, bins, binwidth, binrange, discrete):
        """Inner function that takes bin parameters as arguments."""
        vals = vals.dropna()

        if binrange is None:
            start, stop = vals.min(), vals.max()
        else:
            start, stop = binrange

        if discrete:
            bin_edges = np.arange(start - .5, stop + 1.5)
        elif binwidth is not None:
            step = binwidth
            bin_edges = np.arange(start, stop + step, step)
        else:
            bin_edges = np.histogram_bin_edges(vals, bins, binrange, weight)

        # TODO warning or cap on too many bins?

        return bin_edges

    def _define_bin_params(self, data, orient, scale_type):
        """Given data, return numpy.histogram parameters to define bins."""
        vals = data[orient]
        weight = data.get("weight", None)

        # TODO We'll want this for ordinal / discrete scales too
        # (Do we need discrete as a parameter or just infer from scale?)
        discrete = self.discrete or scale_type == "nominal"

        bin_edges = self._define_bin_edges(
            vals, weight, self.bins, self.binwidth, self.binrange, discrete,
        )

        if isinstance(self.bins, (str, int)):
            n_bins = len(bin_edges) - 1
            bin_range = bin_edges.min(), bin_edges.max()
            bin_kws = dict(bins=n_bins, range=bin_range)
        else:
            bin_kws = dict(bins=bin_edges)

        return bin_kws

    def _get_bins_and_eval(self, data, orient, groupby, scale_type):

        bin_kws = self._define_bin_params(data, orient, scale_type)
        return groupby.apply(data, self._eval, orient, bin_kws)

    def _eval(self, data, orient, bin_kws):

        vals = data[orient]
        weight = data.get("weight", None)

        density = self.stat == "density"
        hist, bin_edges = np.histogram(
            vals, **bin_kws, weights=weight, density=density,
        )

        width = np.diff(bin_edges)
        pos = bin_edges[:-1] + width / 2
        other = {"x": "y", "y": "x"}[orient]

        return pd.DataFrame({orient: pos, other: hist, "space": width})

    def _normalize(self, data, orient):

        other = "y" if orient == "x" else "x"
        hist = data[other]

        if self.stat == "probability" or self.stat == "proportion":
            hist = hist.astype(float) / hist.sum()
        elif self.stat == "percent":
            hist = hist.astype(float) / hist.sum() * 100
        elif self.stat == "frequency":
            hist = hist.astype(float) / data["space"]

        if self.cumulative:
            if self.stat in ["density", "frequency"]:
                hist = (hist * data["space"]).cumsum()
            else:
                hist = hist.cumsum()

        return data.assign(**{other: hist})

    def __call__(self, data, groupby, orient, scales):

        # TODO better to do this as an isinstance check?
        # We are only asking about Nominal scales now,
        # but presumably would apply to Ordinal too?
        scale_type = scales[orient].__class__.__name__.lower()
        grouping_vars = [v for v in data if v in groupby.order]
        if not grouping_vars or self.common_bins is True:
            bin_kws = self._define_bin_params(data, orient, scale_type)
            data = groupby.apply(data, self._eval, orient, bin_kws)
        else:
            if self.common_bins is False:
                bin_groupby = GroupBy(grouping_vars)
            else:
                bin_groupby = GroupBy(self.common_bins)
            data = bin_groupby.apply(
                data, self._get_bins_and_eval, orient, groupby, scale_type,
            )

        # TODO Make this an option?
        # (This needs to be tested if enabled, and maybe should be in _eval)
        # other = {"x": "y", "y": "x"}[orient]
        # data = data[data[other] > 0]

        if not grouping_vars or self.common_norm is True:
            data = self._normalize(data, orient)
        else:
            if self.common_norm is False:
                norm_grouper = grouping_vars
            else:
                norm_grouper = self.common_norm
            normalize = partial(self._normalize, orient=orient)
            data = GroupBy(norm_grouper).apply(data, normalize)

        return data
