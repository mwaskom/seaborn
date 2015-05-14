"""
Horizontal boxplot with log axis
================================

_thumb: .6, .5
"""
import numpy as np
import seaborn as sns
sns.set(style="ticks", palette="muted", color_codes=True)

# Load the example planets dataset
planets = sns.load_dataset("planets")

# Plot the orbital period with horizontal boxes
ax = sns.boxplot(x="orbital_period", y="method", data=planets,
                 whis=np.inf, color="c")

# Make the quantitative axis logarithmic
ax.set_xscale("log")
sns.despine(left=True)
