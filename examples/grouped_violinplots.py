"""
Grouped violinplots with split violins
======================================

"""
import seaborn as sns
sns.set(style="darkgrid", palette="pastel", color_codes=True)

# Load the example tips dataset
tips = sns.load_dataset("tips")

# Draw a nested violinplot and split the violins for easier comparison
sns.violinplot(x="day", y="total_bill", hue="sex", data=tips,
               inner="quart", split=True, palette={"Male": "r", "Female": "y"})
sns.despine(offset=10, trim=True)
