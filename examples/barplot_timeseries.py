"""
Barplot timeseries
==================

"""
import numpy as np
import seaborn as sns

sns.set(style="nogrid")
planets = sns.load_dataset("planets")
years = np.arange(2000, 2015)
sns.factorplot("year", data=planets, palette="BuPu", aspect=1.5, x_order=years)
