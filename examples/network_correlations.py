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


networks = corrmat.columns.get_level_values("network").astype(int).values

start, end = ax.get_ylim()
rect_kws = dict(facecolor="none", edgecolor=".2",
                linewidth=1.5, capstyle="projecting")

for n in range(1, 18):
    n_nodes = (networks == n).sum()
    rect = plt.Rectangle((start, end), n_nodes, -n_nodes, **rect_kws)                         
    start += n_nodes
    end -= n_nodes
    ax.add_artist(rect)

f.tight_layout()
