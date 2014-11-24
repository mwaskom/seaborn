"""
Discovering structure in heatmap data
=====================================

_thumb: .4, .2
"""
import pandas as pd
import seaborn as sns
sns.set(font="monospace")

df = sns.load_dataset("brain_networks", header=[0, 1, 2], index_col=0)
used_networks = [1, 5, 6, 7, 8, 11, 12, 13, 16, 17]
used_columns = (df.columns.get_level_values("network")
                          .astype(int)
                          .isin(used_networks))
df = df.loc[:, used_columns]

network_pal = sns.cubehelix_palette(len(used_networks),
                                    light=.9, dark=.1, reverse=True,
                                    start=1, rot=-2)
network_lut = dict(zip(map(str, used_networks), network_pal))

networks = df.columns.get_level_values("network")
network_colors = pd.Series(networks).map(network_lut)

cmap = sns.diverging_palette(h_neg=210, h_pos=350, s=90, l=30, as_cmap=True)

sns.clustermap(df.corr(), row_colors=network_colors, method="average",
               col_colors=network_colors, figsize=(13, 13), cmap=cmap)
