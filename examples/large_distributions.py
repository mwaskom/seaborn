"""
Plotting large distributions
============================

"""
import seaborn as sns
sns.set(style="whitegrid")

networks = sns.load_dataset("brain_networks", index_col=0, header=[0, 1, 2])
networks = networks.T.groupby(level="network").mean().T
order = networks.std().sort_values().index

sns.lvplot(data=networks, order=order, scale="linear", palette="mako")
