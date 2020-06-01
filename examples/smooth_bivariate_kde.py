"""
Smooth bivariate kernel density

_thumb: .5, .45
"""
import seaborn as sns
sns.set(style="dark")

iris = sns.load_dataset("iris")

sns.kdeplot(
    data=iris,
    x="sepal_length",
    y="sepal_width",
    fill=True,
    thresh=0,
    levels=100,
    cmap="rocket",
)
