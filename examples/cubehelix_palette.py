"""
Different cubehelix palettes
============================

_thumb: .4, .65
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="dark")

f, axes = plt.subplots(9, figsize=(8, 8))
for ax, s in zip(axes, np.linspace(0, 3, 10)):
    pal = sns.cubehelix_palette(10, s)
    ax.imshow([pal], interpolation="nearest")
    ax.set(yticks=[], xticklabels=[])
f.tight_layout()
