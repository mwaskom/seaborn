"""
Scatterplot Matrix
==================

_thumb: .5, .43
"""
import seaborn as sns
sns.set(style="ticks")

df = sns.load_dataset("iris")
sns.pairplot(df, hue="species")
