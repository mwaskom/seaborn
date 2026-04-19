"""
Validation script for Seaborn GridLayout functionality.

This script provides methods to verify that the new chart combination and layout
features have been successfully installed and are working correctly.

Usage:
    python validate_gridlayout.py
"""
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


def check_imports():
    """Check if all new modules can be imported correctly."""
    print("=" * 60)
    print("1. Checking Imports")
    print("=" * 60)
    
    try:
        import seaborn as sns
        print("  ✓ seaborn imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import seaborn: {e}")
        return False
    
    try:
        from seaborn.gridlayout import (
            GridLayout,
            CustomLayout,
            LayoutManager,
            plot_grid,
            combine_plots,
        )
        print("  ✓ GridLayout imported successfully")
        print("  ✓ CustomLayout imported successfully")
        print("  ✓ LayoutManager imported successfully")
        print("  ✓ plot_grid imported successfully")
        print("  ✓ combine_plots imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import gridlayout: {e}")
        return False
    
    try:
        # Check that the functions are available in seaborn namespace
        import seaborn as sns
        assert hasattr(sns, 'GridLayout')
        assert hasattr(sns, 'CustomLayout')
        assert hasattr(sns, 'LayoutManager')
        assert hasattr(sns, 'plot_grid')
        assert hasattr(sns, 'combine_plots')
        print("  ✓ All classes/functions available in seaborn namespace")
    except AssertionError as e:
        print(f"  ✗ Not all classes/functions available in seaborn namespace: {e}")
        return False
    
    print()
    return True


def check_gridlayout_basic():
    """Test basic GridLayout functionality."""
    print("=" * 60)
    print("2. Testing Basic GridLayout")
    print("=" * 60)
    
    try:
        import seaborn as sns
        from seaborn.gridlayout import GridLayout
        
        # Create sample data
        np.random.seed(42)
        df = pd.DataFrame({
            'x': np.random.randn(100),
            'y': np.random.randn(100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
        })
        
        # Create a 2x2 grid
        layout = GridLayout(2, 2, height=3, aspect=1)
        print("  ✓ GridLayout created successfully")
        
        # Add plots
        layout.add_plot(0, 0, sns.scatterplot, data=df, x='x', y='y')
        layout.add_plot(0, 1, sns.histplot, data=df, x='x')
        layout.add_plot(1, 0, sns.boxplot, data=df, x='category', y='y')
        layout.add_plot(1, 1, sns.lineplot, data=df, x='x', y='y')
        print("  ✓ Plots added successfully")
        
        # Check that 4 plots were added
        assert len(layout._plots) == 4
        print("  ✓ Correct number of plots added")
        
        # Render the layout
        fig = layout.render()
        print("  ✓ Layout rendered successfully")
        
        # Check that figure was created
        assert isinstance(fig, mpl.figure.Figure)
        print("  ✓ Figure created successfully")
        
        # Check that 4 axes were created
        assert len(fig.axes) == 4
        print("  ✓ Correct number of axes created")
        
        # Clean up
        plt.close('all')
        
        print()
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')
        return False


def check_gridlayout_spanning():
    """Test GridLayout with spanning plots."""
    print("=" * 60)
    print("3. Testing GridLayout with Spanning Plots")
    print("=" * 60)
    
    try:
        import seaborn as sns
        from seaborn.gridlayout import GridLayout
        
        # Create sample data
        np.random.seed(42)
        df = pd.DataFrame({
            'x': np.random.randn(100),
            'y': np.random.randn(100),
        })
        
        # Create a 3x2 grid
        layout = GridLayout(3, 2, height=2.5, aspect=1.2)
        
        # Add spanning plots
        layout.add_plot(0, 0, sns.scatterplot, data=df, x='x', y='y', colspan=2)
        layout.add_plot(1, 0, sns.histplot, data=df, x='x')
        layout.add_plot(1, 1, sns.histplot, data=df, x='y')
        layout.add_plot(2, 0, sns.lineplot, data=df, x='x', y='y', colspan=2)
        
        print("  ✓ Spanning plots added successfully")
        
        # Check spanning
        assert layout._plots[0]['colspan'] == 2
        assert layout._plots[3]['colspan'] == 2
        print("  ✓ Spanning configuration correct")
        
        # Render
        fig = layout.render()
        print("  ✓ Layout with spanning rendered successfully")
        
        # Clean up
        plt.close('all')
        
        print()
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')
        return False


def check_shared_axes():
    """Test shared axes functionality."""
    print("=" * 60)
    print("4. Testing Shared Axes")
    print("=" * 60)
    
    try:
        import seaborn as sns
        from seaborn.gridlayout import GridLayout
        
        # Create sample data
        np.random.seed(42)
        df = pd.DataFrame({
            'x': np.random.randn(100),
            'y': np.random.randn(100),
        })
        
        # Create layout with shared axes
        layout = GridLayout(2, 2, height=3, aspect=1)
        layout.share_axes(sharex=True, sharey=True)
        
        print("  ✓ Shared axes configured")
        
        # Add plots
        layout.add_plot(0, 0, sns.scatterplot, data=df, x='x', y='y')
        layout.add_plot(0, 1, sns.scatterplot, data=df, x='x', y='y')
        layout.add_plot(1, 0, sns.scatterplot, data=df, x='x', y='y')
        layout.add_plot(1, 1, sns.scatterplot, data=df, x='x', y='y')
        
        # Render
        fig = layout.render()
        axes = fig.axes
        
        print("  ✓ Layout with shared axes rendered")
        
        # Check that axes are actually shared
        assert len(axes) == 4
        
        # Check sharing
        for ax in axes[1:]:
            # Check if x-axis is shared with first axis
            x_shared = axes[0].get_shared_x_axes().joined(axes[0], ax)
            y_shared = axes[0].get_shared_y_axes().joined(axes[0], ax)
            assert x_shared or axes[0].get_shared_x_axes().joined(ax, axes[0])
            assert y_shared or axes[0].get_shared_y_axes().joined(ax, axes[0])
        
        print("  ✓ Axes are properly shared")
        
        # Clean up
        plt.close('all')
        
        print()
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')
        return False


def check_custom_layout():
    """Test CustomLayout functionality."""
    print("=" * 60)
    print("5. Testing CustomLayout")
    print("=" * 60)
    
    try:
        import seaborn as sns
        from seaborn.gridlayout import CustomLayout
        
        # Create sample data
        np.random.seed(42)
        df = pd.DataFrame({
            'x': np.random.randn(100),
            'y': np.random.randn(100),
        })
        
        # Create custom layout
        layout = CustomLayout(figsize=(12, 8), units='relative')
        print("  ✓ CustomLayout created successfully")
        
        # Add plots with custom positions
        layout.add_plot([0.1, 0.5, 0.35, 0.4], sns.scatterplot, data=df, x='x', y='y')
        layout.add_plot([0.55, 0.5, 0.35, 0.4], sns.histplot, data=df, x='x')
        layout.add_plot([0.1, 0.1, 0.8, 0.3], sns.lineplot, data=df, x='x', y='y')
        
        print("  ✓ Plots added with custom positions")
        
        # Render
        fig = layout.render()
        print("  ✓ CustomLayout rendered successfully")
        
        # Check
        assert isinstance(fig, mpl.figure.Figure)
        assert len(fig.axes) == 3
        print("  ✓ Correct number of axes created")
        
        # Clean up
        plt.close('all')
        
        print()
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')
        return False


def check_layout_manager():
    """Test LayoutManager functionality."""
    print("=" * 60)
    print("6. Testing LayoutManager")
    print("=" * 60)
    
    try:
        import seaborn as sns
        from seaborn.gridlayout import LayoutManager
        
        # Create sample data
        np.random.seed(42)
        df = pd.DataFrame({
            'x': np.random.randn(100),
            'y': np.random.randn(100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
        })
        
        # Create manager with grid layout
        manager = LayoutManager('grid', nrows=2, ncols=2, height=3)
        print("  ✓ LayoutManager (grid) created successfully")
        
        # Add plots
        manager.add(0, 0, sns.scatterplot, data=df, x='x', y='y', hue='category')
        manager.add(0, 1, sns.boxplot, data=df, x='category', y='y')
        manager.add(1, 0, sns.violinplot, data=df, x='category', y='y')
        manager.add(1, 1, sns.pointplot, data=df, x='category', y='y')
        
        print("  ✓ Plots added via LayoutManager")
        
        # Share axes
        manager.share_axes(sharex='col', sharey='row')
        print("  ✓ Shared axes configured")
        
        # Set title
        manager.set_title("LayoutManager Test")
        print("  ✓ Title set")
        
        # Render
        fig = manager.render()
        print("  ✓ LayoutManager rendered successfully")
        
        # Check
        assert isinstance(fig, mpl.figure.Figure)
        assert len(fig.axes) == 4
        print("  ✓ Correct number of axes created")
        
        # Clean up
        plt.close('all')
        
        print()
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')
        return False


def check_convenience_functions():
    """Test convenience functions plot_grid and combine_plots."""
    print("=" * 60)
    print("7. Testing Convenience Functions")
    print("=" * 60)
    
    try:
        import seaborn as sns
        from seaborn.gridlayout import plot_grid, combine_plots
        
        # Create sample data
        np.random.seed(42)
        df = pd.DataFrame({
            'x': np.random.randn(100),
            'y': np.random.randn(100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
        })
        
        # Test plot_grid
        print("  Testing plot_grid...")
        fig1 = plot_grid([
            {'plot_func': sns.scatterplot, 'data': df, 'x': 'x', 'y': 'y'},
            {'plot_func': sns.histplot, 'data': df, 'x': 'x'},
            {'plot_func': sns.boxplot, 'data': df, 'x': 'category', 'y': 'y'},
            {'plot_func': sns.lineplot, 'data': df, 'x': 'x', 'y': 'y'},
        ], nrows=2, ncols=2)
        
        assert isinstance(fig1, mpl.figure.Figure)
        assert len(fig1.axes) == 4
        print("  ✓ plot_grid works correctly")
        
        plt.close('all')
        
        # Test combine_plots with grid
        print("  Testing combine_plots (grid)...")
        fig2 = combine_plots([
            {'plot_func': sns.scatterplot, 'data': df, 'x': 'x', 'y': 'y'},
            {'plot_func': sns.histplot, 'data': df, 'x': 'x'},
        ], layout='grid', nrows=1, ncols=2)
        
        assert isinstance(fig2, mpl.figure.Figure)
        assert len(fig2.axes) == 2
        print("  ✓ combine_plots (grid) works correctly")
        
        plt.close('all')
        
        # Test combine_plots with custom
        print("  Testing combine_plots (custom)...")
        fig3 = combine_plots([
            {'position': [0.1, 0.1, 0.8, 0.8], 'plot_func': sns.scatterplot, 'data': df, 'x': 'x', 'y': 'y'},
        ], layout='custom', figsize=(10, 8))
        
        assert isinstance(fig3, mpl.figure.Figure)
        assert len(fig3.axes) == 1
        print("  ✓ combine_plots (custom) works correctly")
        
        plt.close('all')
        
        print()
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')
        return False


def check_compatibility():
    """Test compatibility with existing seaborn plot types."""
    print("=" * 60)
    print("8. Testing Compatibility with Existing Plot Types")
    print("=" * 60)
    
    try:
        import seaborn as sns
        from seaborn.gridlayout import GridLayout
        
        # Create sample data
        np.random.seed(42)
        df = pd.DataFrame({
            'x': np.random.randn(100),
            'y': np.random.randn(100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'value': np.random.randn(100),
        })
        
        # Create a layout to test various plot types
        layout = GridLayout(3, 3, height=2.5, aspect=1)
        
        # Relational plots
        layout.add_plot(0, 0, sns.scatterplot, data=df, x='x', y='y')
        layout.add_plot(0, 1, sns.lineplot, data=df, x='x', y='y')
        
        # Categorical plots
        layout.add_plot(0, 2, sns.boxplot, data=df, x='category', y='value')
        layout.add_plot(1, 0, sns.violinplot, data=df, x='category', y='value')
        layout.add_plot(1, 1, sns.barplot, data=df, x='category', y='value')
        layout.add_plot(1, 2, sns.pointplot, data=df, x='category', y='value')
        
        # Distribution plots
        layout.add_plot(2, 0, sns.histplot, data=df, x='x')
        layout.add_plot(2, 1, sns.kdeplot, data=df, x='x', y='y')
        layout.add_plot(2, 2, sns.ecdfplot, data=df, x='x')
        
        print("  ✓ All plot types added successfully")
        
        # Render
        fig = layout.render()
        print("  ✓ All plot types rendered successfully")
        
        # Check
        assert len(fig.axes) == 9
        print("  ✓ Correct number of axes created")
        
        # Clean up
        plt.close('all')
        
        print()
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')
        return False


def check_linked_axes():
    """Test linked axes for interactivity."""
    print("=" * 60)
    print("9. Testing Linked Axes (Interactivity)")
    print("=" * 60)
    
    try:
        import seaborn as sns
        from seaborn.gridlayout import GridLayout
        
        # Create sample data
        np.random.seed(42)
        df = pd.DataFrame({
            'x': np.random.randn(100),
            'y': np.random.randn(100),
            'value': np.random.randn(100),
        })
        
        # Create layout
        layout = GridLayout(1, 3, height=4, aspect=1)
        
        # Add plots
        layout.add_plot(0, 0, sns.scatterplot, data=df, x='x', y='y')
        layout.add_plot(0, 1, sns.scatterplot, data=df, x='x', y='value')
        layout.add_plot(0, 2, sns.scatterplot, data=df, x='y', y='value')
        
        # Render first
        fig = layout.render()
        axes = fig.axes
        
        # Link axes
        layout.link_axes(axes, axis='x')
        print("  ✓ Axes linked successfully")
        
        # Verify linking
        assert len(axes) == 3
        
        # Check that all axes share x with first axis
        for ax in axes[1:]:
            x_shared = axes[0].get_shared_x_axes().joined(axes[0], ax)
            assert x_shared or axes[0].get_shared_x_axes().joined(ax, axes[0])
        
        print("  ✓ Axes are properly linked")
        
        # Clean up
        plt.close('all')
        
        print()
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')
        return False


def run_all_checks():
    """Run all validation checks."""
    print("\n" + "=" * 60)
    print("Seaborn GridLayout Validation")
    print("=" * 60 + "\n")
    
    results = []
    
    # Run all checks
    results.append(("Imports", check_imports()))
    results.append(("Basic GridLayout", check_gridlayout_basic()))
    results.append(("Spanning Plots", check_gridlayout_spanning()))
    results.append(("Shared Axes", check_shared_axes()))
    results.append(("CustomLayout", check_custom_layout()))
    results.append(("LayoutManager", check_layout_manager()))
    results.append(("Convenience Functions", check_convenience_functions()))
    results.append(("Compatibility", check_compatibility()))
    results.append(("Linked Axes", check_linked_axes()))
    
    # Summary
    print("=" * 60)
    print("Validation Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n  Passed: {passed}/{total}")
    print(f"  Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n  ✓ All checks passed! GridLayout functionality is working correctly.")
    else:
        print("\n  ✗ Some checks failed. Please review the errors above.")
        
        # Show which tests failed
        for name, result in results:
            if not result:
                print(f"    - {name}")
    
    print("\n" + "=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_checks()
    sys.exit(0 if success else 1)
