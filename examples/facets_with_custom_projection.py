"""
FacetGrid with custom projection
================================

_thumb: .38, .5

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

r = np.linspace(0, 10, num=100)
df = pd.DataFrame({'r': r, 'slow': r, 'medium': 2 * r, 'fast': 4 * r})
df = pd.melt(df, id_vars=['r'], var_name='speed', value_name='theta')

g = sns.FacetGrid(df, col="speed", hue="speed",
                  subplot_kws=dict(projection='polar'), size=4,
                  sharex=False, sharey=False, despine=False)
g.map(plt.scatter, "theta", "r")
