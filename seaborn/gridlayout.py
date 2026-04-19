"""
Grid Layout Module for Seaborn
===============================

This module provides advanced functionality for combining multiple different
types of plots into a single figure with flexible layout configurations.

Key Features:
-------------
1. GridLayout: A class for creating flexible grid layouts
2. CustomLayout: A class for creating custom position-based layouts
3. LayoutManager: A high-level interface for managing complex layouts
4. Interactivity: Support for shared axes and linked selections

Examples:
--------
>>> import seaborn as sns
>>> from seaborn.gridlayout import GridLayout, LayoutManager
>>> 
>>> # Create a 2x2 grid layout
>>> layout = GridLayout(2, 2, height=3, aspect=1)
>>> 
>>> # Add different plot types to each position
>>> layout.add_plot(0, 0, sns.scatterplot, data=df, x='x', y='y')
>>> layout.add_plot(0, 1, sns.histplot, data=df, x='x')
>>> layout.add_plot(1, 0, sns.boxplot, data=df, x='category', y='value')
>>> layout.add_plot(1, 1, sns.lineplot, data=df, x='x', y='y')
>>> 
>>> # Render the layout
>>> fig = layout.render()

See Also:
--------
FacetGrid : Subplot grid for plotting conditional relationships
PairGrid : Subplot grid for plotting pairwise relationships
JointGrid : Grid for plotting joint and marginal distributions
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections.abc import Generator
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, SubplotSpec
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from seaborn._base import VectorPlotter
from seaborn.axisgrid import _BaseGrid, Grid
from seaborn import utils
from seaborn.utils import _check_argument, _draw_figure, _disable_autolayout


__all__ = [
    "GridLayout",
    "CustomLayout",
    "LayoutManager",
    "plot_grid",
    "combine_plots",
]


class _LayoutBase(_BaseGrid):
    """Base class for all layout classes."""
    
    def __init__(self):
        super().__init__()
        self._plots: List[Dict[str, Any]] = []
        self._axes_dict: Dict[Tuple[int, int], Axes] = {}
        self._figure: Optional[Figure] = None
        self._sharex: Union[bool, str] = False
        self._sharey: Union[bool, str] = False
        self._linked_axes: Dict[str, List[Axes]] = {}
        
    def _init_figure(self, figsize: Tuple[float, float] = None) -> Figure:
        """Initialize the matplotlib figure."""
        if figsize is None:
            figsize = (10, 8)
        
        with _disable_autolayout():
            fig = plt.figure(figsize=figsize)
        
        self._figure = fig
        return fig
    
    def _get_axes_at(self, row: int, col: int) -> Optional[Axes]:
        """Get the axes at the specified position."""
        return self._axes_dict.get((row, col))
    
    def _set_axes_at(self, row: int, col: int, ax: Axes) -> None:
        """Set the axes at the specified position."""
        self._axes_dict[(row, col)] = ax
    
    def share_axes(self, sharex: Union[bool, str] = False, 
                   sharey: Union[bool, str] = False) -> "_LayoutBase":
        """
        Share axes between subplots.
        
        Parameters
        ----------
        sharex : bool or {'col', 'row'}, optional
            If True, share x axes across all subplots.
            If 'col', share x axes within each column.
            If 'row', share x axes within each row.
        sharey : bool or {'col', 'row'}, optional
            If True, share y axes across all subplots.
            If 'col', share y axes within each column.
            If 'row', share y axes within each row.
            
        Returns
        -------
        self : _LayoutBase
            Returns self for method chaining.
        """
        self._sharex = sharex
        self._sharey = sharey
        return self
    
    def link_axes(self, axes_list: List[Axes], axis: str = 'both') -> "_LayoutBase":
        """
        Link multiple axes together for interactive zoom/pan.
        
        Parameters
        ----------
        axes_list : list of Axes
            List of axes to link together.
        axis : {'x', 'y', 'both'}, optional
            Which axis to link. Default is 'both'.
            
        Returns
        -------
        self : _LayoutBase
            Returns self for method chaining.
        """
        if axis not in ['x', 'y', 'both']:
            raise ValueError("axis must be 'x', 'y', or 'both'")
        
        if axis in ['x', 'both']:
            key = f"x_{id(axes_list[0])}"
            self._linked_axes[key] = axes_list
            for i, ax in enumerate(axes_list[1:], 1):
                ax.sharex(axes_list[0])
        
        if axis in ['y', 'both']:
            key = f"y_{id(axes_list[0])}"
            self._linked_axes[key] = axes_list
            for i, ax in enumerate(axes_list[1:], 1):
                ax.sharey(axes_list[0])
        
        return self
    
    def set_title(self, title: str, **kwargs) -> "_LayoutBase":
        """
        Set a main title for the figure.
        
        Parameters
        ----------
        title : str
            The main title text.
        **kwargs
            Additional keyword arguments passed to Figure.suptitle.
            
        Returns
        -------
        self : _LayoutBase
            Returns self for method chaining.
        """
        if self._figure is not None:
            self._figure.suptitle(title, **kwargs)
        return self
    
    def tight_layout(self, **kwargs) -> "_LayoutBase":
        """
        Adjust subplot parameters to give specified padding.
        
        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to Figure.tight_layout.
            
        Returns
        -------
        self : _LayoutBase
            Returns self for method chaining.
        """
        if self._figure is not None:
            self._figure.tight_layout(**kwargs)
        return self
    
    def savefig(self, *args, **kwargs) -> None:
        """
        Save the figure to a file.
        
        This wraps :meth:`matplotlib.figure.Figure.savefig`, using bbox_inches="tight"
        by default. Parameters are passed through to the matplotlib function.
        """
        if self._figure is not None:
            kwargs = kwargs.copy()
            kwargs.setdefault("bbox_inches", "tight")
            self._figure.savefig(*args, **kwargs)
    
    @property
    def figure(self) -> Optional[Figure]:
        """Access the :class:`matplotlib.figure.Figure` object underlying the layout."""
        return self._figure
    
    @property
    def axes(self) -> np.ndarray:
        """An array of the :class:`matplotlib.axes.Axes` objects in the layout."""
        if not self._axes_dict:
            return np.array([])
        
        rows = max(pos[0] for pos in self._axes_dict.keys()) + 1
        cols = max(pos[1] for pos in self._axes_dict.keys()) + 1
        
        axes_array = np.empty((rows, cols), dtype=object)
        for (row, col), ax in self._axes_dict.items():
            axes_array[row, col] = ax
        
        return axes_array


class GridLayout(_LayoutBase):
    """
    A flexible grid layout for combining multiple different types of plots.
    
    GridLayout allows you to create a grid of subplots and add different
    types of plots to each position. It supports flexible sizing, spanning
    multiple rows/columns, and shared axes.
    
    Parameters
    ----------
    nrows : int
        Number of rows in the grid.
    ncols : int
        Number of columns in the grid.
    height : float, optional
        Height (in inches) of each row. Default is 3.
    aspect : float, optional
        Aspect ratio of each subplot (width/height). Default is 1.
    width_ratios : list of floats, optional
        Relative widths of columns. Default is None (equal widths).
    height_ratios : list of floats, optional
        Relative heights of rows. Default is None (equal heights).
    wspace : float, optional
        The amount of width reserved for space between subplots.
    hspace : float, optional
        The amount of height reserved for space between subplots.
    sharex : bool or {'col', 'row'}, optional
        If True, share x axes across all subplots.
    sharey : bool or {'col', 'row'}, optional
        If True, share y axes across all subplots.
        
    Examples
    --------
    >>> import seaborn as sns
    >>> from seaborn.gridlayout import GridLayout
    >>> 
    >>> # Create a 2x2 grid
    >>> layout = GridLayout(2, 2, height=3, aspect=1)
    >>> 
    >>> # Add plots to specific positions
    >>> layout.add_plot(0, 0, sns.scatterplot, data=df, x='x', y='y')
    >>> layout.add_plot(0, 1, sns.histplot, data=df, x='x')
    >>> layout.add_plot(1, 0, sns.boxplot, data=df, x='category', y='value')
    >>> layout.add_plot(1, 1, sns.lineplot, data=df, x='x', y='y')
    >>> 
    >>> # Render and display
    >>> fig = layout.render()
    
    See Also
    --------
    CustomLayout : Layout with custom positions
    LayoutManager : High-level layout manager
    FacetGrid : Subplot grid for conditional relationships
    """
    
    def __init__(
        self,
        nrows: int,
        ncols: int,
        height: float = 3,
        aspect: float = 1,
        width_ratios: Optional[List[float]] = None,
        height_ratios: Optional[List[float]] = None,
        wspace: Optional[float] = None,
        hspace: Optional[float] = None,
        sharex: Union[bool, str] = False,
        sharey: Union[bool, str] = False,
    ):
        super().__init__()
        
        self.nrows = nrows
        self.ncols = ncols
        self.height = height
        self.aspect = aspect
        self.width_ratios = width_ratios
        self.height_ratios = height_ratios
        self.wspace = wspace
        self.hspace = hspace
        self._sharex = sharex
        self._sharey = sharey
        
        self._gridspec: Optional[GridSpec] = None
        self._subplot_specs: Dict[Tuple[int, int], SubplotSpec] = {}
        
    def _create_gridspec(self, fig: Figure) -> GridSpec:
        """Create the GridSpec for the layout."""
        gs_kwargs = {}
        if self.width_ratios is not None:
            gs_kwargs['width_ratios'] = self.width_ratios
        if self.height_ratios is not None:
            gs_kwargs['height_ratios'] = self.height_ratios
        if self.wspace is not None:
            gs_kwargs['wspace'] = self.wspace
        if self.hspace is not None:
            gs_kwargs['hspace'] = self.hspace
            
        gs = GridSpec(self.nrows, self.ncols, figure=fig, **gs_kwargs)
        self._gridspec = gs
        return gs
    
    def _calculate_figsize(self) -> Tuple[float, float]:
        """Calculate the figure size based on grid dimensions."""
        if self.width_ratios is None:
            total_width = self.ncols * self.height * self.aspect
        else:
            total_width = sum(self.width_ratios) * self.height * self.aspect / max(self.width_ratios)
            
        if self.height_ratios is None:
            total_height = self.nrows * self.height
        else:
            total_height = sum(self.height_ratios) * self.height / max(self.height_ratios)
            
        return (total_width, total_height)
    
    def add_plot(
        self,
        row: int,
        col: int,
        plot_func: Callable,
        rowspan: int = 1,
        colspan: int = 1,
        **kwargs
    ) -> "GridLayout":
        """
        Add a plot to the specified position in the grid.
        
        Parameters
        ----------
        row : int
            Row index (0-based).
        col : int
            Column index (0-based).
        plot_func : callable
            The plotting function to use (e.g., sns.scatterplot, sns.histplot).
        rowspan : int, optional
            Number of rows the plot should span. Default is 1.
        colspan : int, optional
            Number of columns the plot should span. Default is 1.
        **kwargs
            Additional keyword arguments passed to the plotting function.
            
        Returns
        -------
        self : GridLayout
            Returns self for method chaining.
            
        Examples
        --------
        >>> layout.add_plot(0, 0, sns.scatterplot, data=df, x='x', y='y')
        >>> # Span multiple columns
        >>> layout.add_plot(1, 0, sns.lineplot, data=df, x='x', y='y', colspan=2)
        """
        if row < 0 or row >= self.nrows:
            raise ValueError(f"Row index {row} out of range [0, {self.nrows-1}]")
        if col < 0 or col >= self.ncols:
            raise ValueError(f"Column index {col} out of range [0, {self.ncols-1}]")
        if row + rowspan > self.nrows:
            raise ValueError(f"Row span {rowspan} exceeds grid rows {self.nrows}")
        if col + colspan > self.ncols:
            raise ValueError(f"Column span {colspan} exceeds grid columns {self.ncols}")
        
        self._plots.append({
            'row': row,
            'col': col,
            'rowspan': rowspan,
            'colspan': colspan,
            'plot_func': plot_func,
            'kwargs': kwargs
        })
        
        return self
    
    def _apply_axis_sharing(self, axes_dict: Dict[Tuple[int, int], Axes]) -> None:
        """Apply axis sharing based on sharex and sharey settings."""
        if not axes_dict:
            return
            
        axes_array = self.axes
        
        # Handle x-axis sharing
        if self._sharex is True:
            # Share x across all axes
            first_ax = None
            for ax in axes_array.flat:
                if ax is not None:
                    if first_ax is None:
                        first_ax = ax
                    else:
                        ax.sharex(first_ax)
        elif self._sharex == 'col':
            # Share x within each column
            for col in range(self.ncols):
                col_axes = [axes_array[row, col] for row in range(self.nrows) 
                           if axes_array[row, col] is not None]
                if len(col_axes) > 1:
                    for ax in col_axes[1:]:
                        ax.sharex(col_axes[0])
        elif self._sharex == 'row':
            # Share x within each row
            for row in range(self.nrows):
                row_axes = [axes_array[row, col] for col in range(self.ncols)
                           if axes_array[row, col] is not None]
                if len(row_axes) > 1:
                    for ax in row_axes[1:]:
                        ax.sharex(row_axes[0])
        
        # Handle y-axis sharing
        if self._sharey is True:
            # Share y across all axes
            first_ax = None
            for ax in axes_array.flat:
                if ax is not None:
                    if first_ax is None:
                        first_ax = ax
                    else:
                        ax.sharey(first_ax)
        elif self._sharey == 'col':
            # Share y within each column
            for col in range(self.ncols):
                col_axes = [axes_array[row, col] for row in range(self.nrows)
                           if axes_array[row, col] is not None]
                if len(col_axes) > 1:
                    for ax in col_axes[1:]:
                        ax.sharey(col_axes[0])
        elif self._sharey == 'row':
            # Share y within each row
            for row in range(self.nrows):
                row_axes = [axes_array[row, col] for col in range(self.ncols)
                           if axes_array[row, col] is not None]
                if len(row_axes) > 1:
                    for ax in row_axes[1:]:
                        ax.sharey(row_axes[0])
    
    def render(self, fig: Optional[Figure] = None) -> Figure:
        """
        Render all plots in the layout.
        
        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            An existing figure to use. If None, a new figure will be created.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The rendered figure.
        """
        if fig is None:
            figsize = self._calculate_figsize()
            fig = self._init_figure(figsize)
        else:
            self._figure = fig
        
        gs = self._create_gridspec(fig)
        
        # Create all axes first
        for plot_info in self._plots:
            row = plot_info['row']
            col = plot_info['col']
            rowspan = plot_info['rowspan']
            colspan = plot_info['colspan']
            
            # Create subplot spec
            if rowspan == 1 and colspan == 1:
                subplot_spec = gs[row, col]
            else:
                subplot_spec = gs[row:row+rowspan, col:col+colspan]
            
            self._subplot_specs[(row, col)] = subplot_spec
            
            # Create axes
            ax = fig.add_subplot(subplot_spec)
            self._set_axes_at(row, col, ax)
        
        # Apply axis sharing
        self._apply_axis_sharing(self._axes_dict)
        
        # Now render all plots
        for plot_info in self._plots:
            row = plot_info['row']
            col = plot_info['col']
            plot_func = plot_info['plot_func']
            kwargs = plot_info['kwargs'].copy()
            
            ax = self._get_axes_at(row, col)
            if ax is None:
                continue
            
            # Add ax to kwargs if the function supports it
            kwargs['ax'] = ax
            
            # Call the plotting function
            try:
                plot_func(**kwargs)
            except TypeError as e:
                # If the function doesn't accept ax, try without it
                if 'ax' in str(e):
                    kwargs.pop('ax')
                    plt.sca(ax)
                    plot_func(**kwargs)
                else:
                    raise
        
        # Final layout adjustments
        fig.tight_layout()
        
        return fig


class CustomLayout(_LayoutBase):
    """
    A layout manager for custom position-based plot arrangements.
    
    CustomLayout allows you to specify exact positions and sizes for each
    subplot using either relative coordinates (0-1) or absolute inches.
    
    Parameters
    ----------
    figsize : tuple of floats, optional
        Figure size in inches (width, height). Default is (10, 8).
    units : {'relative', 'inches'}, optional
        The unit system for positions. 'relative' uses 0-1 coordinates,
        'inches' uses absolute inches. Default is 'relative'.
        
    Examples
    --------
    >>> import seaborn as sns
    >>> from seaborn.gridlayout import CustomLayout
    >>> 
    >>> # Create a custom layout
    >>> layout = CustomLayout(figsize=(12, 8), units='relative')
    >>> 
    >>> # Add plots with custom positions [left, bottom, width, height]
    >>> layout.add_plot([0.1, 0.5, 0.35, 0.4], sns.scatterplot, data=df, x='x', y='y')
    >>> layout.add_plot([0.55, 0.5, 0.35, 0.4], sns.histplot, data=df, x='x')
    >>> layout.add_plot([0.1, 0.1, 0.8, 0.3], sns.lineplot, data=df, x='x', y='y')
    >>> 
    >>> # Render
    >>> fig = layout.render()
    
    See Also
    --------
    GridLayout : Grid-based layout
    LayoutManager : High-level layout manager
    """
    
    def __init__(
        self,
        figsize: Tuple[float, float] = (10, 8),
        units: str = 'relative'
    ):
        super().__init__()
        
        if units not in ['relative', 'inches']:
            raise ValueError("units must be 'relative' or 'inches'")
            
        self.figsize = figsize
        self.units = units
        self._custom_plots: List[Dict[str, Any]] = []
        
    def add_plot(
        self,
        position: List[float],
        plot_func: Callable,
        **kwargs
    ) -> "CustomLayout":
        """
        Add a plot at the specified position.
        
        Parameters
        ----------
        position : list of floats
            Position as [left, bottom, width, height].
            If units='relative', values should be between 0 and 1.
            If units='inches', values are in inches.
        plot_func : callable
            The plotting function to use.
        **kwargs
            Additional keyword arguments passed to the plotting function.
            
        Returns
        -------
        self : CustomLayout
            Returns self for method chaining.
        """
        if len(position) != 4:
            raise ValueError("position must be [left, bottom, width, height]")
            
        self._custom_plots.append({
            'position': position,
            'plot_func': plot_func,
            'kwargs': kwargs
        })
        
        return self
    
    def _to_relative(self, position: List[float]) -> List[float]:
        """Convert position to relative coordinates."""
        if self.units == 'relative':
            return position
        
        # Convert from inches to relative
        fig_width, fig_height = self.figsize
        return [
            position[0] / fig_width,
            position[1] / fig_height,
            position[2] / fig_width,
            position[3] / fig_height
        ]
    
    def render(self, fig: Optional[Figure] = None) -> Figure:
        """
        Render all plots in the custom layout.
        
        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            An existing figure to use. If None, a new figure will be created.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The rendered figure.
        """
        if fig is None:
            fig = self._init_figure(self.figsize)
        else:
            self._figure = fig
        
        # Create and render each plot
        for i, plot_info in enumerate(self._custom_plots):
            position = self._to_relative(plot_info['position'])
            plot_func = plot_info['plot_func']
            kwargs = plot_info['kwargs'].copy()
            
            # Create axes at custom position
            ax = fig.add_axes(position)
            self._set_axes_at(0, i, ax)  # Use (0, i) as unique key
            
            # Add ax to kwargs
            kwargs['ax'] = ax
            
            # Call the plotting function
            try:
                plot_func(**kwargs)
            except TypeError as e:
                if 'ax' in str(e):
                    kwargs.pop('ax')
                    plt.sca(ax)
                    plot_func(**kwargs)
                else:
                    raise
        
        return fig


class LayoutManager:
    """
    High-level interface for managing complex plot layouts.
    
    LayoutManager provides a unified interface for creating and managing
    different types of layouts, with support for saving/loading layouts
    and interactive features.
    
    Parameters
    ----------
    layout_type : {'grid', 'custom'}, optional
        The type of layout to create. Default is 'grid'.
    **kwargs
        Additional arguments passed to the layout constructor.
        
    Examples
    --------
    >>> import seaborn as sns
    >>> from seaborn.gridlayout import LayoutManager
    >>> 
    >>> # Create a grid layout
    >>> manager = LayoutManager('grid', nrows=2, ncols=2, height=3)
    >>> 
    >>> # Add plots using the manager
    >>> manager.add(0, 0, sns.scatterplot, data=df, x='x', y='y')
    >>> manager.add(0, 1, sns.histplot, data=df, x='x')
    >>> 
    >>> # Share axes
    >>> manager.share_axes(sharex=True, sharey=True)
    >>> 
    >>> # Render
    >>> fig = manager.render()
    
    See Also
    --------
    GridLayout : Grid-based layout
    CustomLayout : Custom position layout
    """
    
    def __init__(self, layout_type: str = 'grid', **kwargs):
        if layout_type == 'grid':
            self.layout = GridLayout(**kwargs)
        elif layout_type == 'custom':
            self.layout = CustomLayout(**kwargs)
        else:
            raise ValueError(f"Unknown layout type: {layout_type}")
        
        self.layout_type = layout_type
        
    def add(self, *args, **kwargs) -> "LayoutManager":
        """
        Add a plot to the layout.
        
        This method delegates to the underlying layout's add_plot method.
        
        Parameters
        ----------
        *args
            Positional arguments passed to the layout's add_plot method.
        **kwargs
            Keyword arguments passed to the layout's add_plot method.
            
        Returns
        -------
        self : LayoutManager
            Returns self for method chaining.
        """
        self.layout.add_plot(*args, **kwargs)
        return self
    
    def share_axes(self, sharex: Union[bool, str] = False, 
                   sharey: Union[bool, str] = False) -> "LayoutManager":
        """
        Share axes between subplots.
        
        Parameters
        ----------
        sharex : bool or {'col', 'row'}, optional
            If True, share x axes across all subplots.
        sharey : bool or {'col', 'row'}, optional
            If True, share y axes across all subplots.
            
        Returns
        -------
        self : LayoutManager
            Returns self for method chaining.
        """
        self.layout.share_axes(sharex, sharey)
        return self
    
    def link_axes(self, axes_list: List[Axes], axis: str = 'both') -> "LayoutManager":
        """
        Link multiple axes together for interactive zoom/pan.
        
        Parameters
        ----------
        axes_list : list of Axes
            List of axes to link together.
        axis : {'x', 'y', 'both'}, optional
            Which axis to link. Default is 'both'.
            
        Returns
        -------
        self : LayoutManager
            Returns self for method chaining.
        """
        self.layout.link_axes(axes_list, axis)
        return self
    
    def set_title(self, title: str, **kwargs) -> "LayoutManager":
        """
        Set a main title for the figure.
        
        Parameters
        ----------
        title : str
            The main title text.
        **kwargs
            Additional keyword arguments passed to Figure.suptitle.
            
        Returns
        -------
        self : LayoutManager
            Returns self for method chaining.
        """
        self.layout.set_title(title, **kwargs)
        return self
    
    def render(self, fig: Optional[Figure] = None) -> Figure:
        """
        Render all plots in the layout.
        
        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            An existing figure to use. If None, a new figure will be created.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The rendered figure.
        """
        return self.layout.render(fig)
    
    @property
    def figure(self) -> Optional[Figure]:
        """Access the underlying figure."""
        return self.layout.figure
    
    @property
    def axes(self) -> np.ndarray:
        """Access the underlying axes array."""
        return self.layout.axes


def plot_grid(
    plot_specs: List[Dict[str, Any]],
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    **kwargs
) -> Figure:
    """
    Create a grid of plots from a list of plot specifications.
    
    This is a convenience function for quickly creating a grid of plots
    without explicitly creating a GridLayout object.
    
    Parameters
    ----------
    plot_specs : list of dicts
        List of plot specifications. Each dict should contain:
        - 'plot_func': The plotting function
        - 'row': Row index (optional if nrows/ncols are inferred)
        - 'col': Column index (optional if nrows/ncols are inferred)
        - 'rowspan': Number of rows to span (optional, default 1)
        - 'colspan': Number of columns to span (optional, default 1)
        - Any additional kwargs passed to the plotting function
    nrows : int, optional
        Number of rows. If None, inferred from plot_specs.
    ncols : int, optional
        Number of columns. If None, inferred from plot_specs.
    **kwargs
        Additional arguments passed to GridLayout.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The rendered figure.
        
    Examples
    --------
    >>> import seaborn as sns
    >>> from seaborn.gridlayout import plot_grid
    >>> 
    >>> # Create a grid of plots
    >>> fig = plot_grid([
    ...     {'plot_func': sns.scatterplot, 'data': df, 'x': 'x', 'y': 'y'},
    ...     {'plot_func': sns.histplot, 'data': df, 'x': 'x'},
    ...     {'plot_func': sns.boxplot, 'data': df, 'x': 'category', 'y': 'value'},
    ...     {'plot_func': sns.lineplot, 'data': df, 'x': 'x', 'y': 'y'},
    ... ], nrows=2, ncols=2)
    
    See Also
    --------
    GridLayout : More flexible grid layout class
    combine_plots : Combine plots with custom layout
    """
    # Infer grid dimensions if not provided
    if nrows is None or ncols is None:
        max_row = max(spec.get('row', 0) for spec in plot_specs)
        max_col = max(spec.get('col', i % 2) for i, spec in enumerate(plot_specs))
        
        if nrows is None:
            nrows = max_row + 1
        if ncols is None:
            ncols = max_col + 1
    
    # Create layout
    layout = GridLayout(nrows, ncols, **kwargs)
    
    # Add plots
    for i, spec in enumerate(plot_specs):
        spec = spec.copy()
        plot_func = spec.pop('plot_func')
        row = spec.pop('row', i // ncols if ncols else 0)
        col = spec.pop('col', i % ncols if ncols else i)
        rowspan = spec.pop('rowspan', 1)
        colspan = spec.pop('colspan', 1)
        
        layout.add_plot(row, col, plot_func, rowspan=rowspan, colspan=colspan, **spec)
    
    # Render
    return layout.render()


def combine_plots(
    plots: List[Dict[str, Any]],
    layout: str = 'grid',
    **kwargs
) -> Figure:
    """
    Combine multiple plots into a single figure.
    
    This is a high-level convenience function for combining plots with
    various layout options.
    
    Parameters
    ----------
    plots : list of dicts
        List of plot specifications. The format depends on the layout type:
        - For 'grid' layout: same as plot_grid
        - For 'custom' layout: each dict should have 'position' and 'plot_func'
    layout : {'grid', 'custom'}, optional
        The layout type to use. Default is 'grid'.
    **kwargs
        Additional arguments passed to the layout constructor.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The rendered figure.
        
    Examples
    --------
    >>> import seaborn as sns
    >>> from seaborn.gridlayout import combine_plots
    >>> 
    >>> # Combine plots with grid layout
    >>> fig = combine_plots([
    ...     {'plot_func': sns.scatterplot, 'data': df, 'x': 'x', 'y': 'y'},
    ...     {'plot_func': sns.histplot, 'data': df, 'x': 'x'},
    ... ], layout='grid', nrows=1, ncols=2)
    >>> 
    >>> # Combine plots with custom layout
    >>> fig = combine_plots([
    ...     {'position': [0.1, 0.5, 0.35, 0.4], 'plot_func': sns.scatterplot, 'data': df, 'x': 'x', 'y': 'y'},
    ...     {'position': [0.55, 0.5, 0.35, 0.4], 'plot_func': sns.histplot, 'data': df, 'x': 'x'},
    ...     {'position': [0.1, 0.1, 0.8, 0.3], 'plot_func': sns.lineplot, 'data': df, 'x': 'x', 'y': 'y'},
    ... ], layout='custom', figsize=(12, 8))
    
    See Also
    --------
    plot_grid : Grid-based plot combination
    GridLayout : More flexible grid layout class
    CustomLayout : Custom position layout
    """
    if layout == 'grid':
        return plot_grid(plots, **kwargs)
    elif layout == 'custom':
        custom_layout = CustomLayout(**kwargs)
        for plot_spec in plots:
            plot_spec = plot_spec.copy()
            position = plot_spec.pop('position')
            plot_func = plot_spec.pop('plot_func')
            custom_layout.add_plot(position, plot_func, **plot_spec)
        return custom_layout.render()
    else:
        raise ValueError(f"Unknown layout type: {layout}")
