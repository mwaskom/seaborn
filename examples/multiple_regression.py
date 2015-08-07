"""
Multiple linear regression
==========================

"""
import seaborn as sns
sns.set(style="ticks", context="talk")

# Load the example tips dataset
tips = sns.load_dataset("tips")

# Make a custom sequential palette using the cubehelix system
pal = sns.cubehelix_palette(4, 1.5, .75, light=.6, dark=.2)

# Plot tip as a function of toal bill across days
g = sns.lmplot(x="total_bill", y="tip", hue="day", data=tips,
               palette=pal, size=7)

# Use more informative axis labels than are provided by default
g.set_axis_labels("Total bill ($)", "Tip ($)")
