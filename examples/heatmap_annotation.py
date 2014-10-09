"""
Annotated heatmaps
==================

"""
import seaborn as sns
sns.set()

flights_long = sns.load_dataset("flights")
flights = flights_long.pivot("month", "year", "passengers")
flights = flights.reindex(flights_long.iloc[:12].month)

sns.heatmap(flights, annot=True, fmt="d")
