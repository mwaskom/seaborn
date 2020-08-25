"""
Scatterplot Matrix
==================

_thumb: .2, .2
"""
import seaborn as sns
sns.set(style="ticks")

df = sns.load_dataset("penguins")
sns.pairplot(df, hue="species")
