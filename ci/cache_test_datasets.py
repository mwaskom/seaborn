"""
Cache test datasets before running test suites to avoid
race conditions to due tests parallelization
"""
import seaborn as sns

datasets = (
    "anscombe",
    "attention",
    "dots",
    "exercise",
    "flights",
    "fmri",
    "iris",
    "penguins",
    "planets",
    "tips",
    "titanic"
)
list(map(sns.load_dataset, datasets))
