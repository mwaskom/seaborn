"""
Plotting a three-way ANOVA
==========================

_thumb: .42, .5
"""
import seaborn as sns
sns.set_theme(style="whitegrid")

# Load the example exercise dataset
exercise = sns.load_dataset("exercise")

# Draw a pointplot to show pulse as a function of three categorical factors
g = sns.catplot(
    data=exercise, x="time", y="pulse", hue="kind", col="diet",
    capsize=.2, palette="YlGnBu_d", errorbar="se",
    kind="point", height=6, aspect=.75,
)
g.despine(left=True)
