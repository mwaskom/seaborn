"""
Tests for the gridlayout module.
"""
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns
from seaborn.gridlayout import (
    GridLayout,
    CustomLayout,
    LayoutManager,
    plot_grid,
    combine_plots,
)


class TestGridLayout:
    """Tests for GridLayout class."""
    
    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        self.df = pd.DataFrame({
            'x': np.random.randn(100),
            'y': np.random.randn(100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'value': np.random.randn(100)
        })
        
    def teardown_method(self):
        """Clean up matplotlib figures."""
        plt.close('all')
    
    def test_initialization(self):
        """Test basic initialization."""
        layout = GridLayout(2, 2, height=3, aspect=1)
        assert layout.nrows == 2
        assert layout.ncols == 2
        assert layout.height == 3
        assert layout.aspect == 1
    
    def test_add_plot(self):
        """Test adding plots to the layout."""
        layout = GridLayout(2, 2)
        layout.add_plot(0, 0, sns.scatterplot, data=self.df, x='x', y='y')
        layout.add_plot(0, 1, sns.histplot, data=self.df, x='x')
        layout.add_plot(1, 0, sns.boxplot, data=self.df, x='category', y='value')
        layout.add_plot(1, 1, sns.lineplot, data=self.df, x='x', y='y')
        
        assert len(layout._plots) == 4
    
    def test_add_plot_with_span(self):
        """Test adding plots that span multiple rows/columns."""
        layout = GridLayout(2, 2)
        layout.add_plot(0, 0, sns.scatterplot, data=self.df, x='x', y='y', colspan=2)
        layout.add_plot(1, 0, sns.histplot, data=self.df, x='x')
        layout.add_plot(1, 1, sns.boxplot, data=self.df, x='category', y='value')
        
        assert len(layout._plots) == 3
        assert layout._plots[0]['colspan'] == 2
    
    def test_render_creates_figure(self):
        """Test that render creates a figure with axes."""
        layout = GridLayout(2, 2)
        layout.add_plot(0, 0, sns.scatterplot, data=self.df, x='x', y='y')
        layout.add_plot(0, 1, sns.histplot, data=self.df, x='x')
        
        fig = layout.render()
        
        assert isinstance(fig, mpl.figure.Figure)
        assert len(fig.axes) == 2
    
    def test_share_axes(self):
        """Test axis sharing."""
        layout = GridLayout(2, 2)
        layout.share_axes(sharex=True, sharey=True)
        layout.add_plot(0, 0, sns.scatterplot, data=self.df, x='x', y='y')
        layout.add_plot(0, 1, sns.scatterplot, data=self.df, x='x', y='y')
        layout.add_plot(1, 0, sns.scatterplot, data=self.df, x='x', y='y')
        layout.add_plot(1, 1, sns.scatterplot, data=self.df, x='x', y='y')
        
        fig = layout.render()
        
        # Check that axes are shared
        axes = fig.axes
        assert len(axes) == 4
        
        # The first axis should be the shared reference
        for ax in axes[1:]:
            # Check if x-axis is shared
            assert ax.get_shared_x_axes().joined(axes[0], ax) or \
                   ax.get_shared_x_axes().joined(ax, axes[0])
    
    def test_set_title(self):
        """Test setting main title."""
        layout = GridLayout(1, 1)
        layout.add_plot(0, 0, sns.scatterplot, data=self.df, x='x', y='y')
        fig = layout.render()
        
        layout.set_title("Test Title", fontsize=16)
        
        assert fig._suptitle is not None
        assert fig._suptitle.get_text() == "Test Title"
    
    def test_invalid_row_index(self):
        """Test that invalid row index raises error."""
        layout = GridLayout(2, 2)
        with pytest.raises(ValueError):
            layout.add_plot(5, 0, sns.scatterplot, data=self.df, x='x', y='y')
    
    def test_invalid_col_index(self):
        """Test that invalid column index raises error."""
        layout = GridLayout(2, 2)
        with pytest.raises(ValueError):
            layout.add_plot(0, 5, sns.scatterplot, data=self.df, x='x', y='y')


class TestCustomLayout:
    """Tests for CustomLayout class."""
    
    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        self.df = pd.DataFrame({
            'x': np.random.randn(100),
            'y': np.random.randn(100),
        })
        
    def teardown_method(self):
        """Clean up matplotlib figures."""
        plt.close('all')
    
    def test_initialization(self):
        """Test basic initialization."""
        layout = CustomLayout(figsize=(12, 8), units='relative')
        assert layout.figsize == (12, 8)
        assert layout.units == 'relative'
    
    def test_add_plot(self):
        """Test adding plots with custom positions."""
        layout = CustomLayout(figsize=(12, 8), units='relative')
        layout.add_plot([0.1, 0.5, 0.35, 0.4], sns.scatterplot, data=self.df, x='x', y='y')
        layout.add_plot([0.55, 0.5, 0.35, 0.4], sns.histplot, data=self.df, x='x')
        layout.add_plot([0.1, 0.1, 0.8, 0.3], sns.lineplot, data=self.df, x='x', y='y')
        
        assert len(layout._custom_plots) == 3
    
    def test_render_creates_figure(self):
        """Test that render creates a figure with axes."""
        layout = CustomLayout(figsize=(12, 8), units='relative')
        layout.add_plot([0.1, 0.1, 0.8, 0.8], sns.scatterplot, data=self.df, x='x', y='y')
        
        fig = layout.render()
        
        assert isinstance(fig, mpl.figure.Figure)
        assert len(fig.axes) == 1
    
    def test_invalid_position(self):
        """Test that invalid position raises error."""
        layout = CustomLayout()
        with pytest.raises(ValueError):
            layout.add_plot([0.1, 0.1], sns.scatterplot, data=self.df, x='x', y='y')


class TestLayoutManager:
    """Tests for LayoutManager class."""
    
    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        self.df = pd.DataFrame({
            'x': np.random.randn(100),
            'y': np.random.randn(100),
        })
        
    def teardown_method(self):
        """Clean up matplotlib figures."""
        plt.close('all')
    
    def test_grid_layout_manager(self):
        """Test LayoutManager with grid layout."""
        manager = LayoutManager('grid', nrows=2, ncols=2, height=3)
        manager.add(0, 0, sns.scatterplot, data=self.df, x='x', y='y')
        manager.add(0, 1, sns.histplot, data=self.df, x='x')
        
        fig = manager.render()
        
        assert isinstance(fig, mpl.figure.Figure)
        assert len(fig.axes) == 2
    
    def test_custom_layout_manager(self):
        """Test LayoutManager with custom layout."""
        manager = LayoutManager('custom', figsize=(12, 8), units='relative')
        manager.add([0.1, 0.1, 0.8, 0.8], sns.scatterplot, data=self.df, x='x', y='y')
        
        fig = manager.render()
        
        assert isinstance(fig, mpl.figure.Figure)
        assert len(fig.axes) == 1
    
    def test_invalid_layout_type(self):
        """Test that invalid layout type raises error."""
        with pytest.raises(ValueError):
            LayoutManager('invalid')


class TestConvenienceFunctions:
    """Tests for convenience functions plot_grid and combine_plots."""
    
    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        self.df = pd.DataFrame({
            'x': np.random.randn(100),
            'y': np.random.randn(100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
        })
        
    def teardown_method(self):
        """Clean up matplotlib figures."""
        plt.close('all')
    
    def test_plot_grid(self):
        """Test plot_grid function."""
        fig = plot_grid([
            {'plot_func': sns.scatterplot, 'data': self.df, 'x': 'x', 'y': 'y'},
            {'plot_func': sns.histplot, 'data': self.df, 'x': 'x'},
            {'plot_func': sns.boxplot, 'data': self.df, 'x': 'category', 'y': 'y'},
            {'plot_func': sns.lineplot, 'data': self.df, 'x': 'x', 'y': 'y'},
        ], nrows=2, ncols=2)
        
        assert isinstance(fig, mpl.figure.Figure)
        assert len(fig.axes) == 4
    
    def test_plot_grid_with_positions(self):
        """Test plot_grid with explicit row/col positions."""
        fig = plot_grid([
            {'plot_func': sns.scatterplot, 'row': 0, 'col': 0, 'data': self.df, 'x': 'x', 'y': 'y'},
            {'plot_func': sns.histplot, 'row': 0, 'col': 1, 'data': self.df, 'x': 'x'},
            {'plot_func': sns.boxplot, 'row': 1, 'col': 0, 'data': self.df, 'x': 'category', 'y': 'y'},
        ])
        
        assert isinstance(fig, mpl.figure.Figure)
        assert len(fig.axes) == 3
    
    def test_combine_plots_grid(self):
        """Test combine_plots with grid layout."""
        fig = combine_plots([
            {'plot_func': sns.scatterplot, 'data': self.df, 'x': 'x', 'y': 'y'},
            {'plot_func': sns.histplot, 'data': self.df, 'x': 'x'},
        ], layout='grid', nrows=1, ncols=2)
        
        assert isinstance(fig, mpl.figure.Figure)
        assert len(fig.axes) == 2
    
    def test_combine_plots_custom(self):
        """Test combine_plots with custom layout."""
        fig = combine_plots([
            {'position': [0.1, 0.5, 0.35, 0.4], 'plot_func': sns.scatterplot, 'data': self.df, 'x': 'x', 'y': 'y'},
            {'position': [0.55, 0.5, 0.35, 0.4], 'plot_func': sns.histplot, 'data': self.df, 'x': 'x'},
            {'position': [0.1, 0.1, 0.8, 0.3], 'plot_func': sns.lineplot, 'data': self.df, 'x': 'x', 'y': 'y'},
        ], layout='custom', figsize=(12, 8))
        
        assert isinstance(fig, mpl.figure.Figure)
        assert len(fig.axes) == 3


class TestCompatibility:
    """Tests for compatibility with existing seaborn plot types."""
    
    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        self.df = pd.DataFrame({
            'x': np.random.randn(100),
            'y': np.random.randn(100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'value': np.random.randn(100)
        })
        
    def teardown_method(self):
        """Clean up matplotlib figures."""
        plt.close('all')
    
    def test_relational_plots(self):
        """Test compatibility with relational plots."""
        layout = GridLayout(2, 2)
        layout.add_plot(0, 0, sns.scatterplot, data=self.df, x='x', y='y')
        layout.add_plot(0, 1, sns.lineplot, data=self.df, x='x', y='y')
        
        fig = layout.render()
        
        assert len(fig.axes) == 2
    
    def test_categorical_plots(self):
        """Test compatibility with categorical plots."""
        layout = GridLayout(2, 2)
        layout.add_plot(0, 0, sns.boxplot, data=self.df, x='category', y='value')
        layout.add_plot(0, 1, sns.violinplot, data=self.df, x='category', y='value')
        layout.add_plot(1, 0, sns.barplot, data=self.df, x='category', y='value')
        layout.add_plot(1, 1, sns.pointplot, data=self.df, x='category', y='value')
        
        fig = layout.render()
        
        assert len(fig.axes) == 4
    
    def test_distribution_plots(self):
        """Test compatibility with distribution plots."""
        layout = GridLayout(2, 2)
        layout.add_plot(0, 0, sns.histplot, data=self.df, x='x')
        layout.add_plot(0, 1, sns.kdeplot, data=self.df, x='x', y='y')
        layout.add_plot(1, 0, sns.ecdfplot, data=self.df, x='x')
        layout.add_plot(1, 1, sns.rugplot, data=self.df, x='x')
        
        fig = layout.render()
        
        assert len(fig.axes) == 4
    
    def test_regression_plots(self):
        """Test compatibility with regression plots."""
        layout = GridLayout(1, 2)
        layout.add_plot(0, 0, sns.regplot, data=self.df, x='x', y='y')
        layout.add_plot(0, 1, sns.residplot, data=self.df, x='x', y='y')
        
        fig = layout.render()
        
        assert len(fig.axes) == 2
    
    def test_matrix_plots(self):
        """Test compatibility with matrix plots."""
        # Create correlation matrix
        corr = self.df[['x', 'y', 'value']].corr()
        
        layout = GridLayout(1, 1)
        layout.add_plot(0, 0, sns.heatmap, data=corr, annot=True)
        
        fig = layout.render()
        
        assert len(fig.axes) == 2  # heatmap creates colorbar axis too


class TestInteractivity:
    """Tests for interactive features."""
    
    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        self.df = pd.DataFrame({
            'x': np.random.randn(100),
            'y': np.random.randn(100),
        })
        
    def teardown_method(self):
        """Clean up matplotlib figures."""
        plt.close('all')
    
    def test_link_axes(self):
        """Test axis linking."""
        layout = GridLayout(1, 2)
        layout.add_plot(0, 0, sns.scatterplot, data=self.df, x='x', y='y')
        layout.add_plot(0, 1, sns.scatterplot, data=self.df, x='x', y='y')
        
        fig = layout.render()
        axes = fig.axes
        
        # Link the axes
        layout.link_axes(axes, axis='both')
        
        # Check that axes are linked
        assert len(axes) == 2
        # Both axes should share x and y
        assert axes[0].get_shared_x_axes().joined(axes[0], axes[1])
        assert axes[0].get_shared_y_axes().joined(axes[0], axes[1])
    
    def test_sharex_col(self):
        """Test sharing x-axis within columns."""
        layout = GridLayout(2, 2)
        layout.share_axes(sharex='col')
        layout.add_plot(0, 0, sns.scatterplot, data=self.df, x='x', y='y')
        layout.add_plot(1, 0, sns.scatterplot, data=self.df, x='x', y='y')
        layout.add_plot(0, 1, sns.scatterplot, data=self.df, x='x', y='y')
        layout.add_plot(1, 1, sns.scatterplot, data=self.df, x='x', y='y')
        
        fig = layout.render()
        axes = layout.axes
        
        # Axes in same column should share x
        assert axes[0, 0].get_shared_x_axes().joined(axes[0, 0], axes[1, 0])
        assert axes[0, 1].get_shared_x_axes().joined(axes[0, 1], axes[1, 1])
    
    def test_sharey_row(self):
        """Test sharing y-axis within rows."""
        layout = GridLayout(2, 2)
        layout.share_axes(sharey='row')
        layout.add_plot(0, 0, sns.scatterplot, data=self.df, x='x', y='y')
        layout.add_plot(0, 1, sns.scatterplot, data=self.df, x='x', y='y')
        layout.add_plot(1, 0, sns.scatterplot, data=self.df, x='x', y='y')
        layout.add_plot(1, 1, sns.scatterplot, data=self.df, x='x', y='y')
        
        fig = layout.render()
        axes = layout.axes
        
        # Axes in same row should share y
        assert axes[0, 0].get_shared_y_axes().joined(axes[0, 0], axes[0, 1])
        assert axes[1, 0].get_shared_y_axes().joined(axes[1, 0], axes[1, 1])
