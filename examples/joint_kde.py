"""
Joint kernel density estimate
=============================

_thumb: .6, .4
"""
import seaborn as sns
sns.set(style="ticks")

# Load the penguins dataset
penguins = sns.load_dataset("penguins")

# Show the joint distribution using kernel density estimation
g = sns.jointplot(
    data=penguins,
    x="culmen_length_mm", y="culmen_depth_mm", hue="species",
    kind="kde",
)