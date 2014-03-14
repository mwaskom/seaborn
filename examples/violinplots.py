"""
Violinplots
===========

"""
import numpy as np
import seaborn as sns

sns.set(style="whitegrid")

rs = np.random.RandomState(0)

n, p = 40, 8
d = rs.normal(0, 1, (n, p))
d += np.log(np.arange(1, p + 1)) * -5 + 10

sns.violinplot(d)
sns.despine(left=True)
