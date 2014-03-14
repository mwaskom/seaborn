"""
Grouped boxplots
================

"""
import seaborn as sns
sns.set(style="whitegrid")

tips = sns.load_dataset("tips")
days = ["Thur", "Fri", "Sat", "Sun"]

g = sns.factorplot("day", "total_bill", "sex", tips, kind="box",
                   palette="PRGn", aspect=1.25, x_order=days)
g.despine(left=True)
g.set_axis_labels("Day", "Total Bill")
