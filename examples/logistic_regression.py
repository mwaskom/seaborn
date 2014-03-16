"""
Faceted logistic regression
===========================

_thumb: .58, .5
"""
import seaborn as sns
sns.set(style="darkgrid")

df = sns.load_dataset("titanic")

pal = dict(male="#6495ED", female="#F08080")
g = sns.lmplot("age", "survived", col="sex", hue="sex", data=df,
               palette=pal, y_jitter=.02, logistic=True)
g.set(xlim=(0, 80), ylim=(-.05, 1.05))
