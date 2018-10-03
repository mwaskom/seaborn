#!/usr/bin/env pythonw
# -*- coding: utf8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style='whitegrid')


# Generate som artificial wine rating data
np.random.seed(42)
ratings = 2*np.random.randn(250, 3, ) + 3 + np.array([2, -1, 1])
ratings[(ratings > 5) | (ratings < 0)] = np.NaN  # set outliers to NaN
df = pd.DataFrame(ratings, columns=['Red', 'White', 'Rose'])

# Define colors in dict, with keys correspinding to DataFrame column names
color_dict = {'Red': '#920e16',
              'White': '#ebdeb4',
              'Rose': (0.988, 0.482, 0.498)}  # rgb and rgba is also supported
sns.catplot(data=df, palette=color_dict)
plt.show()
