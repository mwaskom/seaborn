"""
Example of alternative map_diag histograms
=========================

"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style="white", palette="muted", color_codes=True)
rs = np.random.RandomState(10)

# make some test data with a boolean label of unequal sample size
df = pd.DataFrame({'data1':np.random.randn(2000),
                   'data2':np.random.randn(2000),
                   'label': np.random.randn(2000) > 0})

# Default barstacked histogram
g1 = sns.PairGrid(df, x_vars= ['data1', 'data2'],
                 y_vars= ['data1', 'data2'],
                 hue = "label", diag_sharey = False,
                 palette = "Set1")

g1.map_offdiag(plt.scatter)
g1.map_diag(plt.hist, normed = False, bins = np.arange(-5,5,.25))

# Updated custom histtype, and also illustrate how normalization should appear
g2 = sns.PairGrid(df, x_vars= ['data1', 'data2'],
                 y_vars= ['data1', 'data2'],
                 hue = "label", diag_sharey = False,
                 palette = "Set1")

g2.map_offdiag(plt.scatter)
g2.map_diag(plt.hist, normed = True,
                     alpha = 0.8,
                     histtype = 'step',
                     linewidth = 3,
                     bins = np.arange(-5,5,.25))
