"""
Bivariate plot with multiple elements
=====================================


"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="dark")

# Simulate data from a bivariate Gaussian
n = 10000
mean = [0, 0]
cov = [(2, .4), (.4, .2)]
rng = np.random.RandomState(0)
x, y = rng.multivariate_normal(mean, cov, n).T

# Draw a combo histogram and scatterplot with density contours
f, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(x=x, y=y, s=5, color=".2")
sns.histplot(x=x, y=y, bins=50, pthresh=.1, color="seagreen")
sns.kdeplot(x=x, y=y, levels=5, color="w", linewidths=1)
