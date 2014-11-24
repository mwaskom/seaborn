"""
Simple Violinplots
==================

"""
import numpy as np
import seaborn as sns

sns.set()

rs = np.random.RandomState(0)

n, p = 40, 8
d = rs.normal(0, 1, (n, p))
d += np.log(np.arange(1, p + 1)) * -5 + 10

pal = sns.cubehelix_palette(p, rot=-.5, dark=.3)
sns.violinplot(d, color=pal)
