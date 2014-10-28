"""
Cortical networks correlation matrix
====================================

"""
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(context="paper", font="monospace")

df = sns.load_dataset("brain_networks", header=[0, 1, 2], index_col=0)
corrmat = df.corr()

f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, linewidths=0, square=True)

networks = corrmat.columns.get_level_values("network")
for i, network in enumerate(networks):
    if i and network != networks[i - 1]:
        ax.axhline(len(networks) - i, c="w")
        ax.axvline(i, c="w")

f.tight_layout()
