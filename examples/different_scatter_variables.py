"""
Scatterplot with categorical and continuous semantics
=====================================================

_thumb: .55, .5

"""
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white")

# Load the example iris dataset
iris = sns.load_dataset("iris")

# Draw a scatter plot while assigning point colors and sizes
# to different variables in the dataset
f, ax = plt.subplots(figsize=(6.5, 6.5))
ax = sns.scatterplot(x="sepal_length", y="sepal_width",
                     hue="species", size="petal_width",
                     sizes=(50, 200), alpha=.75,
                     palette="tab10",
                     data=iris)
