import seaborn as sns 
sns.set_theme(context="paper")
fmri = sns.load_dataset("fmri")
g = sns.relplot(
    data=fmri, kind="line",
    x="timepoint", y="signal",
    hue="event", style="event", col="region",
    height=3.5, aspect=.8,
)
g.savefig("paper_demo.pdf")