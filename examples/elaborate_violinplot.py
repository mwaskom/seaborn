"""
Violinplots showing observations
================================

_thumb: .6, .45
"""
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")

df = sns.load_dataset("brain_networks", header=[0, 1, 2], index_col=0)
used_networks = [1, 3, 4, 5, 6, 7, 8, 11, 12, 13, 16, 17]
used_columns = (df.columns.get_level_values("network")
                          .astype(int)
                          .isin(used_networks))
df = df.loc[:, used_columns]

corr_df = df.corr().groupby(level="network").mean()
corr_df.index = corr_df.index.astype(int)
corr_df = corr_df.sort_index().T

f, ax = plt.subplots(figsize=(11, 6))
sns.violinplot(corr_df, color="Set3", bw=.2, cut=.6,
               lw=.5, inner="points", inner_kws={"ms": 6})
ax.set(ylim=(-.7, 1.05))
sns.despine(left=True, bottom=True)
