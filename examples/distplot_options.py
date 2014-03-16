"""
Distribution plot options
=========================

"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="nogrid", palette="muted")
f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)
sns.despine(left=True)

rs = np.random.RandomState(10)

b, g, r, p = sns.color_palette("muted", 4)

d = rs.normal(size=100)

sns.distplot(d, kde=False, color=b, ax=axes[0, 0])
sns.distplot(d, hist=False, rug=True, color=r, ax=axes[0, 1])
sns.distplot(d, hist=False, color=g, kde_kws={"shade": True}, ax=axes[1, 0])
sns.distplot(d, color=p, ax=axes[1, 1])

plt.setp(axes, yticks=[])
plt.tight_layout()
