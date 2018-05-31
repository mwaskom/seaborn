"""
Grouped violinplots with split violins
======================================

_thumb: .5, .47
"""
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid", palette="pastel", color_codes=True)

# Load the example tips dataset
tips = sns.load_dataset("tips")

# Draw a nested violinplot and split the violins for easier comparison
sns.violinplot(x="day", y="total_bill", hue="sex", data=tips, split=True,
               inner="pointquart", palette={"Male": "b", "Female": "y"})
sns.despine(left=True)
plt.show()
