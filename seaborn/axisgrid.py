from itertools import product
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from . import utils


class FacetGrid(object):

    def __init__(self, data, row=None, col=None, hue=None, col_wrap=None,
                 sharex=True, sharey=True, size=3, aspect=1, palette="husl",
                 dropna=True, legend=True, legend_out=True, despine=True,
                 margin_titles=False, xlim=None, ylim=None):
        """Initialize the plot figure and FacetGrid object.

        Parameters
        ----------
        data : DataFrame
            Tidy (long-form) dataframe where each column is a variable and
            each row is an observation.
        row, col, hue : strings, optional
            Variable (column) names to subset the data for the facets.
        col_wrap : int, optional
            Wrap the column variable at this width. Incompatible with `row`.
        share{x, y}: booleans, optional
            Lock the limits of the vertical and horizontal axes across the
            facets.
        size : scalar, optional
            Height (in inches) of each facet.
        aspect : scalar, optional
            Aspect * size gives the width (in inches) of each facet.
        palette : seaborn color palette
            Set of colors for mapping the `hue` variable.
        dropna : boolean, optional
            Drop missing values from the data before plotting.
        legend : boolean, optional
            Draw a legend for the data when using a `hue` variable.
        legend_out: boolean, optional
            Draw the legend outside the grid of plots.
        despine : boolean, optional
            Remove the top and right spines from the plots.
        margin_titles : boolean, optional
            Write the column and row variable labels on the margins of the
            grid rather than above each plot.
        {x, y}lim: two-tuples, optional
            Limits for each of the axes on each facet when share{x, y} is True.

        Returns
        -------
        FacetGrid

        """
        # Compute the grid shape
        nrow = 1 if row is None else len(data[row].unique())
        ncol = 1 if col is None else len(data[col].unique())

        # Calculate the base figure size
        # This can get stretched later by a legend
        figsize = (ncol * size * aspect, nrow * size)

        subplot_kw = {}
        if xlim is not None:
            subplot_kw["xlim"] = xlim
        if ylim is not None:
            subplot_kw["ylim"] = ylim

        # Initialize the subplot grid
        fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False,
                                 sharex=sharex, sharey=sharey,
                                 subplot_kw=subplot_kw)

        hue_var = hue
        if hue is None:
            hue_names = None
            hue_masks = [np.repeat(True, len(data))]
            colors = None
        else:
            hue_names = np.sort(data[hue].unique())
            if dropna:
                hue_names = list(filter(pd.notnull, hue_names))
            if isinstance(palette, dict):
                palette = [palette[h] for h in hue_names]
            hue_masks = [data[hue] == val for val in hue_names]
            colors = utils.color_palette(palette, len(hue_masks))

        all_na = np.zeros(len(data), np.bool)
        if dropna:
            row_na = all_na if row is None else data[row].isnull()
            col_na = all_na if col is None else data[col].isnull()
            hue_na = all_na if hue is None else data[hue].isnull()
            not_na = ~(row_na & col_na & hue_na)
        else:
            not_na = ~all_na

        self._data = data
        self._fig = fig
        self._axes = axes
        self._nrow = nrow
        self._row_var = row
        if row is None:
            self._row_names = []
        else:
            row_names = sorted(data[row].unique())
            if dropna:
                row_names = list(filter(pd.notnull, row_names))
            self._row_names = row_names

        self._ncol = ncol
        self._col_var = col
        if col is None:
            self._col_names = []
        else:
            col_names = sorted(data[col].unique())
            if dropna:
                col_names = list(filter(pd.notnull, col_names))
            self._col_names = col_names
        self._margin_titles = margin_titles
        self._col_wrap = col_wrap
        self._hue_var = hue_var
        self._hue_names = hue_names
        self._colors = colors
        self._draw_legend = ((hue is not None and hue not in [col, row])
                             and legend)
        self._legend_out = legend_out
        self._legend = None
        self._legend_data = {}
        self._x_var = None
        self._y_var = None
        self._dropna = dropna
        self._not_na = not_na

        fig.tight_layout()
        if despine:
            utils.despine(self._fig)

    def facet_data(self):

        data = self._data

        if self._nrow == 1 or self._col_wrap is not None:
            row_masks = [np.repeat(True, len(self._data))]
        else:
            row_masks = [data[self._row_var] == n for n in self._row_names]

        if self._ncol == 1:
            col_masks = [np.repeat(True, len(self._data))]
        else:
            col_masks = [data[self._col_var] == n for n in self._col_names]

        if len(self._colors) == 1:
            hue_masks = [np.repeat(True, len(self._data))]
        else:
            hue_masks = [data[self._hue_var] == n for n in self._hue_names]

        for (i, row), (j, col), (k, hue) in product(enumerate(row_masks),
                                                    enumerate(col_masks),
                                                    enumerate(hue_masks)):
            data_ijk = data[row & col & hue & self._not_na]
            yield (i, j, k), data_ijk

    def map(self, func, *args, **kwargs):

        for (row_i, col_j, hue_k), data_ijk in self._iter_masks():

            if self._col_wrap is not None:
                f_row = col_j // self._ncol
                f_col = col_j % self._ncol
            else:
                f_row, f_col = row_i, col_j

            ax = self._axes[f_row, f_col]
            plt.sca(ax)

            if not data_ijk.values.tolist():
                continue

            color = self._colors[hue_k]
            if color is not None:
                kwargs["color"]

            if self._hue_var is not None:
                kwargs["label"] = self._hue_names[hue_k]

            plot_data = data_ijk[list(args)]
            if self._dropna:
                plot_data = plot_data.dropna()
            plot_args = [v.values for k, v in plot_data.iteritems()]
            func(*plot_args, **kwargs)

            self._update_legend_data(ax)
            self._clean_axis(ax)

        self._set_axis_labels(*args[:2])

        if self._draw_legend:
            self._make_legend()
        else:
            self._fig.tight_layout()
        self._set_title()

    def set(self, **kwargs):

        for key, val in kwargs.items():
            for ax in self._axes.flat:
                setter = getattr(ax, "set_%s" % key)
                setter(val)

    def despine(self, **kwargs):

        utils.despine(self._fig, **kwargs)

    def set_axis_labels(self, x_var, y_var=None):

        if y_var is not None:
            self._y_var = y_var
            self.set_ylabel(y_var)
        self.set_xlabel(x_var)
        self._x_var = x_var

    def set_xlabels(self, label=None, **kwargs):

        if label is None:
            label = self._x_var
        for ax in self._axes[-1, :]:
            ax.set_xlabel(label, **kwargs)

    def set_ylabels(self, label=None, **kwargs):

        if label is None:
            label = self._y_var
        for ax in self._axes[:, 0]:
            ax.set_ylabel(label, **kwargs)

    def set_xticklabels(self, labels=None, **kwargs):

        for ax in self._axes[-1, :]:
            if labels is None:
                labels = [l.get_text() for l in ax.get_xticklabels()]
            ax.set_xticklabels(labels, **kwargs)

    def set_yticklabels(self, labels=None, **kwargs):

        for ax in self._axes[-1, :]:
            if labels is None:
                labels = [l.get_text() for l in ax.get_yticklabels()]
            ax.set_yticklabels(labels, **kwargs)

    def _clean_axis(self, ax):

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.legend_ = None

    def _set_title(self):

        if self._margin_titles:
            if self._row_names:
                for i, row_name in enumerate(self._row_names):
                    ax = self._axes[i, -1]
                    title = "%s = %s" % (self._row_var, row_name)
                    trans = self._fig.transFigure.inverted()
                    bbox = ax.bbox.transformed(trans)
                    x = bbox.xmax + 0.01
                    y = bbox.ymax - (bbox.height / 2)
                    size = mpl.rcParams["axes.titlesize"]
                    self._fig.text(x, y, title, rotation=270, size=size,
                                   ha="left", va="center")
            if self._col_names:
                for j, col_name in enumerate(self._col_names):
                    title = "%s = %s" % (self._col_var, col_name)
                    self._axes[0, j].set_title(title)

            return

        if self._row_names and self._col_names:
            for i, row_name in enumerate(self._row_names):
                for j, col_name in enumerate(self._col_names):
                    title = "%s = %s | %s = %s" % (self._row_var, row_name,
                                                   self._col_var, col_name)
                    self._axes[i, j].set_title(title)
        elif self._row_names:
            for i, row_name in enumerate(self._row_names):
                title = "%s = %s" % (self._row_var, row_name)
                self._axes[i, 0].set_title(title)
        elif self._col_names:
            for i, col_name in enumerate(self._col_names):
                title = "%s = %s" % (self._col_var, col_name)
                self._axes[0, i].set_title(title)

    def _update_legend_data(self, ax):

        handles, labels = ax.get_legend_handles_labels()
        data = {l: h for h, l in zip(handles, labels)}
        self._legend_data.update(data)

    def _make_legend(self, legend_data=None, title=None):

        legend_data = self._legend_data if legend_data is None else legend_data
        labels = sorted(self._legend_data.keys())
        handles = [legend_data[l] for l in labels]
        title = self._hue_var if title is None else title

        if self._legend_out:

            figlegend = plt.figlegend(handles, labels, "center right",
                                      title=self._hue_var)
            self._legend = figlegend

            plt.draw()
            legend_width = figlegend.get_window_extent().width / self._fig.dpi
            figure_width = self._fig.get_figwidth()
            self._fig.set_figwidth(figure_width + legend_width)

            plt.draw()
            legend_width = figlegend.get_window_extent().width / self._fig.dpi
            space_needed = legend_width / (figure_width + legend_width)
            margin = .04 if self._margin_titles else .01
            self._space_needed = margin + space_needed

            right = 1 - self._space_needed
            self._fig.subplots_adjust(right=right)

        else:

            self._axes[0, 0].legend(handles, labels, loc="best",
                                    title=self._hue_var)
