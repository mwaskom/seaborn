"""
Line plots on multiple facets
=============================

_thumb: .45, .42

"""
import seaborn as sns
sns.set(style="ticks")

dots = sns.load_dataset("dots")

# Define a palette to ensure that colors will be
# shared across the facets
palette = dict(zip(dots.coherence.unique(),
                   sns.color_palette("rocket_r", 6)))

# Set up the FacetGrid with independent x axes
g = sns.FacetGrid(dots, col="align",
                  sharex=False, size=5, aspect=.75)

# Draw the lineplot on each facet
g.map_dataframe(sns.lineplot, "time", "firing_rate",
                hue="coherence", size="choice",
                size_order=["T1", "T2"],
                palette=palette)
g.add_legend()
