"""
Barplot timeseries
==================

_thumb: .6, .4
"""
import numpy as np
import seaborn as sns

sns.set(style="white")
planets = sns.load_dataset("planets")
years = np.arange(2000, 2015)
g = sns.factorplot("year", data=planets, palette="BuPu",
                   aspect=1.5, x_order=years)
g.set_xticklabels(step=2)
