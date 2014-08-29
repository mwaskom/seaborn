"""
Paired discrete plots
=====================

"""
import seaborn as sns
sns.set(style="whitegrid")

titanic = sns.load_dataset("titanic")

g = sns.PairGrid(titanic, y_vars="survived",
                 x_vars=["class", "sex", "who", "alone"],
                 size=5, aspect=.5)
g.map(sns.pointplot, color=sns.xkcd_rgb["plum"])
g.set(ylim=(0, 1))
sns.despine(left=True)
