"""
Color palette choices
=====================

"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", context="talk")
rs = np.random.RandomState(7)

x = np.array(list("ABCDEFGHI"))

f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

y1 = np.arange(1, 10)
sns.barplot(x, y1, ci=None, palette="BuGn_d", hline=.1, ax=ax1)
ax1.set_ylabel("Sequential")

y2 = y1 - 5
sns.barplot(x, y2, ci=None, palette="coolwarm", hline=0, ax=ax2)
ax2.set_ylabel("Diverging")

y3 = rs.choice(y1, 9, replace=False)
sns.barplot(x, y3, ci=None, palette="Paired", hline=.1, ax=ax3)
ax3.set_ylabel("Qualitative")

sns.despine(bottom=True)
plt.setp(f.axes, yticks=[])
plt.tight_layout(h_pad=3)
