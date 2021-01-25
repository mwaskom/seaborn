---
title: 'seaborn: statistical data visualization'
tags:
  - Python
  - data visualization
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

`seaborn` is a library for making statistical graphics in Python. It provides a high-level interface to `matplotlib` [CITE] and integrates closely with `numpy` [CITE] and `pandas` [CITE] data structures. Functions in the `seaborn` library expose a declarative, dataset-oriented API that makes it easy to translate questions about data into graphics that can answer them. When given a dataset and a specification of the plot to make, `seaborn` automatically maps the data values to visual attributes such as color, size, or style, internally computes aggregate statistics and model fits where appropriate, and decorates the plot with informative axis labels and a legend. Many `seaborn` functions can generate figures with multiple panels that elicit comparisons between subsets of data or across different variable pairings. `seaborn` is designed to be useful throughout the lifecycle of a scientific project. By producing complete graphics from a single function call with minimal arguments, `seaborn` facilitates rapid exploratory data analysis. And by offering extensive options for customization, along with exposing the underlying `matplotlib` objects, it can be used to create polished, publication-quality figures.

# Statement of need

Data visualization is a fundamental part of the scientific process. Effective visualizations will allow a scientist to understand their own data and to communicate their insights to others. These goals can be aided by tools that provide a good balance between efficiency and flexibility. Within the scientific Python ecosystem, the `matplotlib` [CITE] project is very well established, having been under continuous development for nearly two decades. It is highly flexible, offering fine-grained control over the placement and visual appearance of objects in a plot. It can be used interactively through GUI applications and it can output graphics to a wide range of static formats. Yet its relatively low-level API for specifying a plot can make some common tasks cumbersome to perform. For example, creating a scatter plot where the marker size represents a numeric variable and the marker shape represents a categorical variable requires one to transform the size values to graphical units and to loop over the categorical levels, separately invoking a plotting function for each marker type.


# Implementatione

# Acknowledgements

M.L.W. has been supported by the National Science Foundation IGERT program (0801700) and by the Simons Foundation as a Junior Fellow in the Simons Society of Fellows (527794).

# References

TODO cite tidy data, tukey, clevelend?