"""
Continuous interactions
=======================

"""
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style="darkgrid")

# Generate a random dataset with strong simple effects and an interaction
n = 80
rs = np.random.RandomState(11)
x1 = rs.randn(n)
x2 = x1 / 5 + rs.randn(n)
b0, b1, b2, b3 = .5, .25, -1, 2
y = b0  + b1 * x1 + b2 * x2 + b3 * x1 * x2 + rs.randn(n)
df = pd.DataFrame(np.c_[x1, x2, y], columns=["x1", "x2", "y"])

# Show a scatterplot of the predictors with the estimated model surface
sns.interactplot("x1", "x2", "y", df)
