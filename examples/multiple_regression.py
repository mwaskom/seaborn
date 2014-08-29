"""
Multiple linear regression
==========================

"""
import seaborn as sns
sns.set(style="ticks", context="talk")

tips = sns.load_dataset("tips")

days = ["Thur", "Fri", "Sat", "Sun"]
pal = sns.cubehelix_palette(4, 1.5, .75, light=.6, dark=.2)
g = sns.lmplot("total_bill", "tip", hue="day", data=tips,
               hue_order=days, palette=pal, size=6)
g.set_axis_labels("Total bill ($)", "Tip ($)")
