"""
Test script for Seaborn GridLayout functionality.

This script demonstrates how to use the new chart combination and layout features.

Usage:
    python test_gridlayout_demo.py
"""
import sys
import os

# Add the current directory to the path to ensure we use the local seaborn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import local seaborn
import seaborn as sns

print("=" * 60)
print("Seaborn GridLayout Demo")
print("=" * 60)
print(f"Seaborn version: {sns.__version__}")
print(f"Seaborn location: {sns.__file__}")
print()

# Check if GridLayout is available
try:
    from seaborn.gridlayout import GridLayout, CustomLayout, LayoutManager, plot_grid, combine_plots
    print("✓ All gridlayout modules imported successfully")
    print()
except ImportError as e:
    print(f"✗ Failed to import gridlayout: {e}")
    sys.exit(1)

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

print("Sample data created:")
print(df.head())
print()

# Test 1: Basic GridLayout
print("Test 1: Basic GridLayout")
print("-" * 40)

try:
    layout = GridLayout(2, 2, height=3, aspect=1.2)
    layout.add_plot(0, 0, sns.scatterplot, data=df, x='x', y='y', hue='category')
    layout.add_plot(0, 1, sns.histplot, data=df, x='x', kde=True, hue='category')
    layout.add_plot(1, 0, sns.boxplot, data=df, x='category', y='value', palette='Set2')
    layout.add_plot(1, 1, sns.lineplot, data=df, x='time', y='value')
    
    layout.set_title("Basic Grid Layout Example", fontsize=14, y=1.02)
    fig = layout.render()
    
    print("✓ Basic GridLayout created and rendered successfully")
    print(f"  - Number of axes: {len(fig.axes)}")
    
    # Save the figure
    fig.savefig("test_basic_grid.png", dpi=150, bbox_inches='tight')
    print("  - Saved: test_basic_grid.png")
    
    plt.close('all')
    print()
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    plt.close('all')

# Test 2: GridLayout with spanning
print("Test 2: GridLayout with Spanning Plots")
print("-" * 40)

try:
    layout = GridLayout(3, 2, height=2.5, aspect=1.2)
    layout.add_plot(0, 0, sns.scatterplot, data=df, x='x', y='y', hue='category', colspan=2)
    layout.add_plot(1, 0, sns.histplot, data=df, x='x', hue='category', element='step')
    layout.add_plot(1, 1, sns.histplot, data=df, x='y', hue='category', element='step')
    layout.add_plot(2, 0, sns.lineplot, data=df, x='time', y='value', hue='category', colspan=2)
    
    layout.set_title("Grid Layout with Spanning Plots", fontsize=14, y=1.02)
    fig = layout.render()
    
    print("✓ GridLayout with spanning created and rendered successfully")
    print(f"  - Number of axes: {len(fig.axes)}")
    
    fig.savefig("test_spanning_grid.png", dpi=150, bbox_inches='tight')
    print("  - Saved: test_spanning_grid.png")
    
    plt.close('all')
    print()
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    plt.close('all')

# Test 3: Shared Axes
print("Test 3: GridLayout with Shared Axes")
print("-" * 40)

try:
    layout = GridLayout(2, 2, height=3, aspect=1)
    layout.share_axes(sharex=True, sharey=True)
    
    for i, cat in enumerate(['A', 'B', 'C']):
        row = i // 2
        col = i % 2
        subset = df[df['category'] == cat]
        layout.add_plot(row, col, sns.scatterplot, data=subset, x='x', y='y', color=f'C{i}')
    
    layout.add_plot(1, 1, sns.kdeplot, data=df, x='x', y='y', hue='category', fill=True)
    
    layout.set_title("Grid Layout with Shared Axes", fontsize=14, y=1.02)
    fig = layout.render()
    
    # Add titles
    for ax, cat in zip(fig.axes[:3], ['A', 'B', 'C']):
        ax.set_title(f'Category {cat}')
    fig.axes[3].set_title('All Categories (KDE)')
    
    print("✓ GridLayout with shared axes created and rendered successfully")
    print(f"  - Number of axes: {len(fig.axes)}")
    
    fig.savefig("test_shared_axes.png", dpi=150, bbox_inches='tight')
    print("  - Saved: test_shared_axes.png")
    
    plt.close('all')
    print()
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    plt.close('all')

# Test 4: CustomLayout
print("Test 4: CustomLayout")
print("-" * 40)

try:
    layout = CustomLayout(figsize=(14, 10), units='relative')
    layout.add_plot([0.1, 0.35, 0.55, 0.6], 
                    sns.scatterplot, data=df, x='x', y='y', hue='category', s=50)
    layout.add_plot([0.7, 0.55, 0.25, 0.4], 
                    sns.histplot, data=df, x='x', hue='category', element='step')
    layout.add_plot([0.1, 0.1, 0.85, 0.2], 
                    sns.lineplot, data=df, x='time', y='value', hue='category')
    
    layout.set_title("Custom Layout Example", fontsize=16, y=0.98)
    fig = layout.render()
    
    # Add titles
    fig.axes[0].set_title('Main Scatter Plot', fontsize=12)
    fig.axes[1].set_title('X Distribution', fontsize=12)
    fig.axes[2].set_title('Time Series', fontsize=12)
    
    print("✓ CustomLayout created and rendered successfully")
    print(f"  - Number of axes: {len(fig.axes)}")
    
    fig.savefig("test_custom_layout.png", dpi=150, bbox_inches='tight')
    print("  - Saved: test_custom_layout.png")
    
    plt.close('all')
    print()
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    plt.close('all')

# Test 5: plot_grid convenience function
print("Test 5: plot_grid Convenience Function")
print("-" * 40)

try:
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
    
    fig.suptitle("plot_grid Convenience Function", fontsize=14, y=1.02)
    
    print("✓ plot_grid function works correctly")
    print(f"  - Number of axes: {len(fig.axes)}")
    
    fig.savefig("test_plot_grid.png", dpi=150, bbox_inches='tight')
    print("  - Saved: test_plot_grid.png")
    
    plt.close('all')
    print()
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    plt.close('all')

# Test 6: LayoutManager
print("Test 6: LayoutManager")
print("-" * 40)

try:
    manager = LayoutManager('grid', nrows=2, ncols=2, height=3, aspect=1.2)
    manager.add(0, 0, sns.scatterplot, data=df, x='x', y='y', hue='category')
    manager.add(0, 1, sns.boxplot, data=df, x='category', y='value')
    manager.add(1, 0, sns.violinplot, data=df, x='category', y='value')
    manager.add(1, 1, sns.pointplot, data=df, x='category', y='value')
    
    manager.share_axes(sharex='col', sharey='row')
    manager.set_title("LayoutManager Example", fontsize=14, y=1.02)
    
    fig = manager.render()
    
    print("✓ LayoutManager works correctly")
    print(f"  - Number of axes: {len(fig.axes)}")
    
    fig.savefig("test_layout_manager.png", dpi=150, bbox_inches='tight')
    print("  - Saved: test_layout_manager.png")
    
    plt.close('all')
    print()
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    plt.close('all')

# Summary
print("=" * 60)
print("Demo Summary")
print("=" * 60)
print()
print("All tests completed! The following files were created:")
print("  - test_basic_grid.png")
print("  - test_spanning_grid.png")
print("  - test_shared_axes.png")
print("  - test_custom_layout.png")
print("  - test_plot_grid.png")
print("  - test_layout_manager.png")
print()
print("You can also run the validation script:")
print("  python validate_gridlayout.py")
print()
print("Or run the pytest tests:")
print("  pytest tests/test_gridlayout.py -v")
print()
print("=" * 60)
