---
title: 'seaborn: statistical data visualization'
tags:
  - Python
  - data visualization
  - data science
authors:
  - name: Michael L. Waskom
    affiliation: 1
affiliations:
  - name: Center for Neural Science, New York University
    index: 1
date: 1 February 2021
bibliography: paper.bib
---

# Summary

`seaborn` is a library for making statistical graphics in Python. It provides a high-level interface to `matplotlib` [CITE] and integrates closely with `numpy` [CITE] and `pandas` [CITE] data structures. Functions in the `seaborn` library expose a declarative, dataset-oriented API that makes it easy to translate questions about data into graphics that can answer them. When given a dataset and a specification of the plot to make, `seaborn` automatically maps the data values to visual attributes such as color, size, or style, internally computes statistical transformations, and decorates the plot with informative axis labels and a legend. Many `seaborn` functions can generate figures with multiple panels that elicit comparisons between conditional subsets of data or across different pairings of variables in a dataset. `seaborn` is designed to be useful throughout the lifecycle of a scientific project. By producing complete graphics from a single function call with minimal arguments, `seaborn` facilitates rapid exploratory data analysis. And by offering extensive options for customization, along with exposing the underlying `matplotlib` objects, it can be used to create polished, publication-quality figures.

# Statement of need

Data visualization is an indispensible part of the scientific process. Effective visualizations will allow a scientist both to understand their own data and to communicate their insights to others. These goals can be aided by tools that provide a good balance between efficiency and flexibility in the specification of a graph. Within the scientific Python ecosystem, the `matplotlib` [CITE] project is very well established, having been under continuous development for nearly two decades. It is highly flexible, offering fine-grained control over the placement and visual appearance of objects in a plot. It can be used interactively through GUI applications and, it can output graphics to a wide range of static formats. Yet its relatively low-level API can make some common tasks cumbersome to perform. For example, creating a scatter plot where the marker size represents a numeric variable and the marker shape represents a categorical variable requires one to transform the size values to graphical units and to loop over the categorical levels, separately invoking a plotting function for each marker type.

BRIEF MISSION STATEMENT

# Overview

Users interface with seaborn through a collection of plotting functions that share a common API for plot specification and offer many more specific options for customization. These functions range from basic plot types such as scatter and line plots to functions that apply various transformations and abstractions, such as histogram binning, kernel density estimation, or regression model fitting. Functions in `seaborn` are divided into the classes of "axes-level" and "figure-level". Axes-level functions behave like most plotting functions in matplotlib's `pyplot` namespace. By default they hook into the state machine that tracks a "current" figure and add a layer to it, but they can also accept a matplotlib axes object to control where the plot is drawn, similar to using matplotlib's "object-oriented" interface. Figure-level functions create their own figure when invoked, allowing them to "facet" the dataset by creating multiple conditional subplots, along with conveniences such as putting the legend outside the space of the plot by default. Each figure-level function corresponds to several axes-level functions that serve similar purposes, with a single parameter selecting the kind of plot to make. The figure-level functions also make use of a `seaborn` class that controls the layout of the figure. These classes are part of the public API and can be used directly for advanced applications.

`seaborn` aims to be flexible about the format of its input data. The most convenient usage pattern provides a `pandas` dataframe with variables encoded in a long-form or "tidy" [CITE] format. With this format, columns in the dataframe can be explicitly assigned to roles in the plot, such as specifying the x and y positions of a scatterplot and providing data to be mapped to the size and shape of each point. This format supports very efficient exploration and prototyping because variables can be assigned different roles in the plot without modifying anything about the original dataset. But most `seaborn` functions can also consume and visualize "wide-form" data, typically producing similar output to how `matplotlib` would interpret a 2D array (e.g., producing a box plot where each box represents a column in the dataframe) but making use of the index and column names to label the graph. While using the label information in a `pandas` object can help make plots that are interpretable without further tweaking and reduce errors of interpretation, `seaborn` also accepts data from a variety of more basic formats, including `numpy` arrays and simple Python dictionaries and lists.

SEMANTIC MAPPING

STATISTICS AND AGGREGATION

`seaborn` also offers multiple built-in themes that one can select to modify the visual appearance of the plots. The themes make use of matplotlib's `rcParams` system, meaning that they will take effect for any figure created using matplotlib, not just those made by `seaborn`. The themes are defined by two disjoint sets of parameters that separately control the style of the figure and the scaling of the elements (such as line widths and font sizes), along with a default color palette. As color is particularly important in data visualization and no one set of defaults is universally appropriate, every plotting function makes it easy to choose an alternate palette or gradient mapping that is well-suited for the particular dataset and plot type. The seaborn documentation contains a tutorial on the use of color in data visualization to help users make this important decision.

# Acknowledgements

M.L.W. has been supported by the National Science Foundation IGERT program (0801700) and by the Simons Foundation as a Junior Fellow in the Simons Society of Fellows (527794).

# References

TODO cite tidy data, tukey, clevelend?