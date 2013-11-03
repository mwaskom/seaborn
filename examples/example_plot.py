import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import moss

f = plt.figure(figsize=(10, 8))
gs = plt.GridSpec(4, 4)

# Linear regression
# -----------------

ax = plt.subplot(gs[:2, :2])
plt.title("lmplot()")

n = 80
c = "#222222"
rs = np.random.RandomState(5)
x = rs.normal(4, 1, n)
y = 2 + 1.5 * x + rs.normal(0, 3, n)
ax.plot(x, y, "o", c=c, alpha=.8)

xx = np.linspace(1, 7, 100)
lmpred = lambda x, y: np.polyval(np.polyfit(x, y, 1), xx)
yy = lmpred(x, y)
ax.plot(xx, yy, c=c)
boots = moss.bootstrap(x, y, func=lmpred, n_boot=100)
ci = moss.percentiles(boots, [2.5, 97.5], 0)
ax.fill_between(xx, *ci, alpha=.15, color=c)

# Timeseries plot
# ---------------

ax = plt.subplot(gs[:2, 2:])
plt.title("tsplot()")

n = 20
t = 10
ax.set_xlim(0, t)
x = np.linspace(0, t, 100)
s = np.array([stats.gamma.pdf(x, a) for a in [3, 5, 7]])
d = s[:, np.newaxis, :]
rs = np.random.RandomState(24)

d = d * np.array([1, -1])[rs.binomial(1, .3, 3)][:, np.newaxis, np.newaxis]
d = d + rs.normal(0, .15, (3, n))[:, :, np.newaxis]
d = d + rs.uniform(0, .25, 3)[:, np.newaxis, np.newaxis]
d *= 10

for d_i in d:
    sns.tsplot(x, d_i, ax=ax)

# Violin plots
# ------------
ax = plt.subplot(gs[2:, :2])
plt.title("violin()")

n = 40
p = 8
rs = np.random.RandomState(8)
d = rs.normal(0, 1, (n, p))
d += np.log(np.arange(1, p + 1)) * -5 + 10

sns.violin(d, inner="points", ax=ax, inner_kws={"marker": "."})

# Beta distributions
# ------------------

ax = plt.subplot(gs[2, 2:])
plt.title("distplot()")
plt.xlim(0, 1)
ax.set_xticklabels([])

b, _, r = sns.color_palette("dark", desat=.4)[:3]
n = 1000
rs = np.random.RandomState(0)
d = rs.beta(8, 13, n)

sns.distplot(d, color=r, fit=stats.beta, ax=ax)

ax = plt.subplot(gs[3, 2:], sharey=ax)
plt.xlim(0, 1)

rs = np.random.RandomState(0)
d = rs.beta(30, 25, n)

sns.distplot(d, color=b, fit=stats.beta, ax=ax)

f.tight_layout()
f.savefig("%s/example_plot.png" % os.path.dirname(__file__))
