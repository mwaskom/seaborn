"""
Grouped boxplots
================

"""
import seaborn as sns
sns.set(style="ticks")

tips = sns.load_dataset("tips")
days = ["Thur", "Fri", "Sat", "Sun"]

sns.boxplot("day", "total_bill", "sex", tips, palette="PRGn")
sns.despine(offset=10, trim=True)
