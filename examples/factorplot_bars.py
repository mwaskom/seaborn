"""
Grouped barplots
================

_thumb: .5, .4
"""
import seaborn as sns
sns.set(style="whitegrid")

titanic = sns.load_dataset("titanic")

g = sns.factorplot("class", "survived", "sex",
                    data=titanic, kind="bar",
                    size=6, palette="muted")
g.despine(left=True)
g.set_ylabels("survival probability")
