"""
Conditional kernel density estimate
===================================

_thumb: .4, .5
"""
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")

# Load the diamonds dataset
diamonds = sns.load_dataset("diamonds")

# Plot the distribution of clarity ratings, conditional on carat
f, ax = plt.subplots(figsize=(7, 7))
sns.kdeplot(
    data=diamonds,
    x="carat",
    hue="cut",
    multiple="fill",
    clip=(0, None),
    palette="ch:rot=-.25,hue=1,light=.75",
)
