"""
Large heatmap with divergent colormap
=====================================

"""
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(context="paper", font="monospace")

# Load the datset of correlations between cortical brain networks
df = sns.load_dataset("brain_networks", header=[0, 1, 2], index_col=0)

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 6))

# Draw the heatmap using seaborn
sns.heatmap(df.T, center=0, robust=True)
