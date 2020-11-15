"""
Scatterplot Matrix
==================

_thumb: .3, .2
"""
import seaborn as sns
sns.set_theme(style="ticks")

df = sns.load_dataset("penguins")
sns.pairplot(df, hue="species")
