"""
Horizontal boxplot with observations
====================================

_thumb: .7, .45
"""
import numpy as np
import seaborn as sns
sns.set(style="ticks", palette="muted", color_codes=True)

# Load the example planets dataset
planets = sns.load_dataset("planets")

# Plot the orbital period with horizontal boxes
ax = sns.boxplot(x="distance", y="method", data=planets,
                 whis=np.inf, color="c")

# Add in points to show each observation
sns.stripplot(x="distance", y="method", data=planets,
              jitter=True, size=3, color=".3", linewidth=0)


# Make the quantitative axis logarithmic
ax.set_xscale("log")
sns.despine(trim=True)
