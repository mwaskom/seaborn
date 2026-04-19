"""
Examples for the gridlayout module.

This file demonstrates how to use the new chart combination and layout
functionality in Seaborn.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn.gridlayout import (
    GridLayout,
    CustomLayout,
    LayoutManager,
    plot_grid,
    combine_plots,
)

# Set style
sns.set_theme(style="whitegrid")

# Create sample data
np.random.seed(42)
n = 200
df = pd.DataFrame({
    'x': np.random.randn(n),
    'y': np.random.randn(n) + 0.5 * np.random.randn(n),
    'category': np.random.choice(['A', 'B', 'C'], n),
    'time': np.linspace(0, 10, n),
    'value': np.sin(np.linspace(0, 10, n)) + np.random.randn(n) * 0.2,
})


def example_basic_grid_layout():
    """
    Example 1: Basic GridLayout usage
    
    Create a simple 2x2 grid with different plot types.
    """
    print("Example 1: Basic GridLayout")
    
    # Create a 2x2 grid layout
    layout = GridLayout(2, 2, height=3, aspect=1.2)
    
    # Add different plot types to each position
    layout.add_plot(0, 0, sns.scatterplot, 
                    data=df, x='x', y='y', hue='category')
    layout.add_plot(0, 1, sns.histplot, 
                    data=df, x='x', kde=True, hue='category')
    layout.add_plot(1, 0, sns.boxplot, 
                    data=df, x='category', y='value', palette='Set2')
    layout.add_plot(1, 1, sns.lineplot, 
                    data=df, x='time', y='value')
    
    # Set main title
    layout.set_title("Basic Grid Layout Example", fontsize=14, y=1.02)
    
    # Render the layout
    fig = layout.render()
    
    # Save and show
    plt.savefig("example_basic_grid.png", dpi=150, bbox_inches='tight')
    print("  Saved: example_basic_grid.png")
    plt.close()


def example_grid_with_spanning():
    """
    Example 2: GridLayout with spanning plots
    
    Create a layout where some plots span multiple rows or columns.
    """
    print("Example 2: GridLayout with Spanning")
    
    # Create a 3x2 grid layout
    layout = GridLayout(3, 2, height=2.5, aspect=1.2)
    
    # Add a main plot that spans the entire first row
    layout.add_plot(0, 0, sns.scatterplot, 
                    data=df, x='x', y='y', hue='category', style='category',
                    colspan=2)
    
    # Add distribution plots in the second row
    layout.add_plot(1, 0, sns.histplot, 
                    data=df, x='x', hue='category', element='step')
    layout.add_plot(1, 1, sns.histplot, 
                    data=df, x='y', hue='category', element='step')
    
    # Add a time series plot that spans the entire third row
    layout.add_plot(2, 0, sns.lineplot, 
                    data=df, x='time', y='value', hue='category',
                    colspan=2)
    
    # Set main title
    layout.set_title("Grid Layout with Spanning Plots", fontsize=14, y=1.02)
    
    # Render the layout
    fig = layout.render()
    
    # Save and show
    plt.savefig("example_spanning_grid.png", dpi=150, bbox_inches='tight')
    print("  Saved: example_spanning_grid.png")
    plt.close()


def example_shared_axes():
    """
    Example 3: Shared axes
    
    Create a grid with shared axes for better comparison.
    """
    print("Example 3: Shared Axes")
    
    # Create a 2x2 grid with shared axes
    layout = GridLayout(2, 2, height=3, aspect=1)
    layout.share_axes(sharex=True, sharey=True)
    
    # Add scatter plots for each category
    for i, cat in enumerate(['A', 'B', 'C']):
        row = i // 2
        col = i % 2
        subset = df[df['category'] == cat]
        layout.add_plot(row, col, sns.scatterplot, 
                        data=subset, x='x', y='y', color=f'C{i}')
        layout._plots[-1]['kwargs']['title'] = f'Category {cat}'
    
    # Add a summary plot in the last position
    layout.add_plot(1, 1, sns.kdeplot, 
                    data=df, x='x', y='y', hue='category', fill=True)
    
    # Set main title
    layout.set_title("Grid Layout with Shared Axes", fontsize=14, y=1.02)
    
    # Render the layout
    fig = layout.render()
    
    # Add titles to each subplot
    for ax, cat in zip(fig.axes[:3], ['A', 'B', 'C']):
        ax.set_title(f'Category {cat}')
    fig.axes[3].set_title('All Categories (KDE)')
    
    # Save and show
    plt.savefig("example_shared_axes.png", dpi=150, bbox_inches='tight')
    print("  Saved: example_shared_axes.png")
    plt.close()


def example_custom_layout():
    """
    Example 4: CustomLayout
    
    Create a custom layout with exact positions.
    """
    print("Example 4: Custom Layout")
    
    # Create a custom layout
    layout = CustomLayout(figsize=(14, 10), units='relative')
    
    # Add a large main plot
    layout.add_plot([0.1, 0.35, 0.55, 0.6], 
                    sns.scatterplot, data=df, x='x', y='y', hue='category', s=50)
    
    # Add a smaller distribution plot on the right
    layout.add_plot([0.7, 0.55, 0.25, 0.4], 
                    sns.histplot, data=df, x='x', hue='category', element='step')
    
    # Add a time series plot at the bottom
    layout.add_plot([0.1, 0.1, 0.85, 0.2], 
                    sns.lineplot, data=df, x='time', y='value', hue='category')
    
    # Set main title
    layout.set_title("Custom Layout Example", fontsize=16, y=0.98)
    
    # Render the layout
    fig = layout.render()
    
    # Add titles
    fig.axes[0].set_title('Main Scatter Plot', fontsize=12)
    fig.axes[1].set_title('X Distribution', fontsize=12)
    fig.axes[2].set_title('Time Series', fontsize=12)
    
    # Save and show
    plt.savefig("example_custom_layout.png", dpi=150, bbox_inches='tight')
    print("  Saved: example_custom_layout.png")
    plt.close()


def example_layout_manager():
    """
    Example 5: LayoutManager
    
    Use the high-level LayoutManager interface.
    """
    print("Example 5: LayoutManager")
    
    # Create a grid layout using LayoutManager
    manager = LayoutManager('grid', nrows=2, ncols=2, height=3, aspect=1.2)
    
    # Add plots using the manager
    manager.add(0, 0, sns.scatterplot, data=df, x='x', y='y', hue='category')
    manager.add(0, 1, sns.boxplot, data=df, x='category', y='value')
    manager.add(1, 0, sns.violinplot, data=df, x='category', y='value')
    manager.add(1, 1, sns.pointplot, data=df, x='category', y='value')
    
    # Share axes
    manager.share_axes(sharex='col', sharey='row')
    
    # Set title
    manager.set_title("LayoutManager Example", fontsize=14, y=1.02)
    
    # Render
    fig = manager.render()
    
    # Save
    plt.savefig("example_layout_manager.png", dpi=150, bbox_inches='tight')
    print("  Saved: example_layout_manager.png")
    plt.close()


def example_plot_grid():
    """
    Example 6: plot_grid convenience function
    
    Use the plot_grid function for quick grid creation.
    """
    print("Example 6: plot_grid Convenience Function")
    
    # Create a grid using plot_grid
    fig = plot_grid([
        {
            'plot_func': sns.scatterplot,
            'data': df, 'x': 'x', 'y': 'y', 'hue': 'category',
            'row': 0, 'col': 0
        },
        {
            'plot_func': sns.histplot,
            'data': df, 'x': 'x', 'hue': 'category',
            'row': 0, 'col': 1
        },
        {
            'plot_func': sns.lineplot,
            'data': df, 'x': 'time', 'y': 'value', 'hue': 'category',
            'row': 1, 'col': 0, 'colspan': 2
        },
    ], nrows=2, ncols=2, height=3, aspect=1.2)
    
    # Add main title
    fig.suptitle("plot_grid Convenience Function", fontsize=14, y=1.02)
    
    # Save
    plt.savefig("example_plot_grid.png", dpi=150, bbox_inches='tight')
    print("  Saved: example_plot_grid.png")
    plt.close()


def example_combine_plots():
    """
    Example 7: combine_plots convenience function
    
    Use the combine_plots function for flexible plot combination.
    """
    print("Example 7: combine_plots Convenience Function")
    
    # Combine plots with custom layout
    fig = combine_plots([
        {
            'position': [0.1, 0.4, 0.5, 0.55],
            'plot_func': sns.scatterplot,
            'data': df, 'x': 'x', 'y': 'y', 'hue': 'category'
        },
        {
            'position': [0.65, 0.55, 0.3, 0.4],
            'plot_func': sns.boxplot,
            'data': df, 'x': 'category', 'y': 'value'
        },
        {
            'position': [0.65, 0.1, 0.3, 0.35],
            'plot_func': sns.violinplot,
            'data': df, 'x': 'category', 'y': 'value'
        },
        {
            'position': [0.1, 0.1, 0.5, 0.25],
            'plot_func': sns.lineplot,
            'data': df, 'x': 'time', 'y': 'value', 'hue': 'category'
        },
    ], layout='custom', figsize=(14, 10))
    
    # Add main title
    fig.suptitle("combine_plots Convenience Function", fontsize=16, y=0.98)
    
    # Save
    plt.savefig("example_combine_plots.png", dpi=150, bbox_inches='tight')
    print("  Saved: example_combine_plots.png")
    plt.close()


def example_linked_axes():
    """
    Example 8: Linked axes for interactivity
    
    Create plots with linked axes for interactive zoom/pan.
    """
    print("Example 8: Linked Axes for Interactivity")
    
    # Create a grid layout
    layout = GridLayout(1, 3, height=4, aspect=1)
    
    # Add three scatter plots with different views
    layout.add_plot(0, 0, sns.scatterplot, 
                    data=df, x='x', y='y', hue='category')
    layout.add_plot(0, 1, sns.scatterplot, 
                    data=df, x='x', y='value', hue='category')
    layout.add_plot(0, 2, sns.scatterplot, 
                    data=df, x='y', y='value', hue='category')
    
    # Render first to get axes
    fig = layout.render()
    
    # Link all x-axes together
    layout.link_axes(fig.axes, axis='x')
    
    # Add titles
    for ax, title in zip(fig.axes, ['x vs y', 'x vs value', 'y vs value']):
        ax.set_title(title)
    
    # Set main title
    fig.suptitle("Linked Axes (zoom/pan will affect all plots)", fontsize=14, y=1.02)
    
    # Save
    plt.savefig("example_linked_axes.png", dpi=150, bbox_inches='tight')
    print("  Saved: example_linked_axes.png")
    plt.close()


def example_complex_analysis_dashboard():
    """
    Example 9: Complex analysis dashboard
    
    Create a comprehensive analysis dashboard with multiple plot types.
    """
    print("Example 9: Complex Analysis Dashboard")
    
    # Create a custom layout for a dashboard
    layout = CustomLayout(figsize=(16, 12), units='relative')
    
    # Main scatter plot (large, top-left)
    layout.add_plot([0.05, 0.45, 0.45, 0.5],
                    sns.scatterplot, data=df, x='x', y='y', hue='category',
                    style='category', s=80, alpha=0.7)
    
    # Correlation heatmap (top-right)
    corr = df[['x', 'y', 'value']].corr()
    layout.add_plot([0.55, 0.55, 0.4, 0.4],
                    sns.heatmap, data=corr, annot=True, cmap='coolwarm',
                    vmin=-1, vmax=1, center=0)
    
    # X distribution (middle-right)
    layout.add_plot([0.55, 0.4, 0.4, 0.12],
                    sns.histplot, data=df, x='x', hue='category',
                    element='step', fill=False)
    
    # Y distribution (middle-right, below X)
    layout.add_plot([0.55, 0.25, 0.4, 0.12],
                    sns.histplot, data=df, x='y', hue='category',
                    element='step', fill=False)
    
    # Box plots by category (bottom-left)
    layout.add_plot([0.05, 0.05, 0.3, 0.35],
                    sns.boxplot, data=df, x='category', y='value', palette='Set2')
    
    # Violin plots by category (bottom-middle)
    layout.add_plot([0.38, 0.05, 0.3, 0.35],
                    sns.violinplot, data=df, x='category', y='value', palette='Set2')
    
    # Time series (bottom-right)
    layout.add_plot([0.7, 0.05, 0.25, 0.17],
                    sns.lineplot, data=df, x='time', y='value', hue='category')
    
    # KDE plot (bottom-right, above time series)
    layout.add_plot([0.7, 0.25, 0.25, 0.12],
                    sns.kdeplot, data=df, x='x', y='y', hue='category', fill=True)
    
    # Render
    fig = layout.render()
    
    # Add titles to each subplot
    titles = [
        'Scatter Plot (x vs y)',
        'Correlation Matrix',
        'X Distribution',
        'Y Distribution',
        'Box Plots by Category',
        'Violin Plots by Category',
        'Time Series',
        'KDE Plot'
    ]
    for ax, title in zip(fig.axes, titles):
        ax.set_title(title, fontsize=10)
    
    # Add main title
    fig.suptitle("Comprehensive Analysis Dashboard", fontsize=18, y=0.98)
    
    # Save
    plt.savefig("example_dashboard.png", dpi=150, bbox_inches='tight')
    print("  Saved: example_dashboard.png")
    plt.close()


def run_all_examples():
    """Run all examples."""
    print("=" * 60)
    print("Running GridLayout Examples")
    print("=" * 60)
    print()
    
    example_basic_grid_layout()
    example_grid_with_spanning()
    example_shared_axes()
    example_custom_layout()
    example_layout_manager()
    example_plot_grid()
    example_combine_plots()
    example_linked_axes()
    example_complex_analysis_dashboard()
    
    print()
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_examples()
